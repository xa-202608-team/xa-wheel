#!/usr/bin/env python
"""scripts/basilisk_b2/build_split.py

BASILISK-B2 §2 —— trajectory-level split, 冻结 IDs 与 split hash。

150 条轨迹 = event-observed 71 + right-censored 79。
train 30% / val 20% / test 50%。

分层键的选择 (这是本阶段唯一的设计判断, 写在这里而非事后解释):
    首要分层键 = event_observed / censored, **不是** EOL 分位。
    理由: 79 条右删失轨迹根本没有 EOL, 无法参与 EOL 分位分层。若强行按
    "EOL 或末端时刻"混合排序, censored 的末端时刻会被当成 EOL 使用 ——
    那正是伪造 EOL, §4 明确禁止。
    因此: 先按 event/censored 分成两组各自配额, 再在 event 组内部按 EOL
    分位做二级分层 (n_eol_strata_within_event), censored 组内不再分层
    (它们没有可分层的结局量, 只有观测截止长度)。

不变量 (由 tests/basilisk_b2 钉死):
    * trajectory-level: 同一条轨迹的所有时间点只出现在一个 split 里
    * event/censored 在三个 split 里都按比例出现
    * 给定 split_seed 完全确定; 不读任何模型结果
    * train/val/test 三者两两不相交, 并集 = 全部 150 条

用法:
    python scripts/basilisk_b2/build_split.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.basilisk_b11.calibrate_degradation import (  # noqa: E402
    load_b11_config as load_cfg,
)

CFG_PATH = "configs/wheel_basilisk_b2.yaml"

SPLIT_PURPOSE = (
    "B2 §2: 在 B1.9 冻结的 150 条轨迹上做 trajectory-level 的 30/20/50 划分。"
    "首要分层键是 event_observed/censored 而非 EOL 分位, 因为 79 条右删失轨迹"
    "没有 EOL —— 拿观测截止长度当 EOL 用就是伪造结局。event 组内部再按 EOL "
    "分位二级分层。划分只依赖 split_seed 与轨迹属性, 不依赖任何模型输出。"
)


def read_traj_meta(h5_path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """读每条轨迹的 (tid, event_observed, eol_or_len)。

    eol_or_len 只用于 event 组内部的二级分层; censored 条目在那里不参与,
    因此它们的值不会被当成 EOL 使用。
    """
    tids: list[str] = []
    ev: list[bool] = []
    key: list[float] = []
    with h5py.File(h5_path, "r") as f:
        for tid in sorted(f.keys()):
            g = f[tid]
            e = bool(int(g.attrs["event_observed"]))
            tids.append(tid)
            ev.append(e)
            key.append(float(g.attrs["eol_idx"]) if e
                       else float(g["x_T"].shape[0]))
    return tids, np.asarray(ev, bool), np.asarray(key, float)


def _quota(n: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    """与 src/transfer/train_transfer.split_trajectories 同一取整规则。"""
    n_tr = int(round(ratios[0] * n))
    n_va = int(round(ratios[1] * n))
    n_te = n - n_tr - n_va
    return n_tr, n_va, n_te


def _assign_block(idx: np.ndarray, ratios, rng) -> tuple[list, list, list]:
    """把一个同层的 index 块按配额随机分到三个 split。"""
    perm = rng.permutation(idx.size)
    shuffled = idx[perm]
    n_tr, n_va, _ = _quota(idx.size, ratios)
    return (shuffled[:n_tr].tolist(),
            shuffled[n_tr:n_tr + n_va].tolist(),
            shuffled[n_tr + n_va:].tolist())


def build_split(cfg: dict) -> dict:
    sc = cfg["b2"]["split"]
    ratios = (float(sc["train"]), float(sc["val"]), float(sc["test"]))
    seed = int(sc["split_seed"])
    n_sub = int(sc["n_eol_strata_within_event"])
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise SystemExit(f"!! split 比例之和 != 1: {ratios}")

    h5_path = ROOT / cfg["transfer"]["target_feature_path"]
    tids, ev, key = read_traj_meta(h5_path)
    n = len(tids)

    # 局部 rng, 绝不用全局 np.random.seed()
    rng = np.random.default_rng(seed)
    tr: list[int] = []
    va: list[int] = []
    te: list[int] = []

    # --- event 组: 按 EOL 分位二级分层 ---
    ev_idx = np.flatnonzero(ev)
    order = ev_idx[np.argsort(key[ev_idx], kind="stable")]
    for block in np.array_split(order, min(n_sub, order.size)):
        if block.size == 0:
            continue
        a, b, c = _assign_block(block, ratios, rng)
        tr += a
        va += b
        te += c

    # --- censored 组: 没有结局量, 不做二级分层 ---
    cen_idx = np.flatnonzero(~ev)
    a, b, c = _assign_block(cen_idx, ratios, rng)
    tr += a
    va += b
    te += c

    tr, va, te = sorted(tr), sorted(va), sorted(te)

    # --- 硬校验: 不相交 + 完备 ---
    s_tr, s_va, s_te = set(tr), set(va), set(te)
    if s_tr & s_va or s_tr & s_te or s_va & s_te:
        raise SystemExit("!! split 重叠 —— 这会造成轨迹级泄漏")
    if len(s_tr | s_va | s_te) != n:
        raise SystemExit(f"!! split 不完备: {len(s_tr | s_va | s_te)} != {n}")

    def _stat(ix: list[int]) -> dict:
        e = int(ev[ix].sum())
        return {"n": len(ix), "n_event": e, "n_censored": len(ix) - e,
                "event_frac": (e / len(ix)) if ix else float("nan"),
                "tids": [tids[i] for i in ix]}

    splits = {"train": _stat(tr), "val": _stat(va), "test": _stat(te)}

    # split hash: 只覆盖 (协议参数 + 三个 tid 列表), 与容器顺序无关
    payload = json.dumps({
        "ratios": list(ratios), "split_seed": seed,
        "stratify_by": sc["stratify_by"],
        "n_eol_strata_within_event": n_sub, "level": sc["level"],
        "train": splits["train"]["tids"],
        "val": splits["val"]["tids"],
        "test": splits["test"]["tids"],
    }, sort_keys=True, ensure_ascii=False)
    split_sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    return {
        "stage": "BASILISK_B2",
        "section": "§2 trajectory split",
        "split_purpose": SPLIT_PURPOSE,
        "feature_h5": cfg["transfer"]["target_feature_path"],
        "n_traj": n,
        "n_event_observed": int(ev.sum()),
        "n_censored": int((~ev).sum()),
        "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "split_seed": seed,
        "stratify_by": sc["stratify_by"],
        "n_eol_strata_within_event": n_sub,
        "level": sc["level"],
        "stratification_note": (
            "首要键 event_observed/censored; event 组内按 eol_idx 分位二级分层; "
            "censored 组不分层 (无结局量, 其观测长度不得当作 EOL)"),
        "splits": splits,
        "split_sha256": split_sha,
        "disjoint": True,
        "covers_all": True,
    }


def main() -> int:
    cfg = load_cfg(CFG_PATH)
    out = build_split(cfg)
    p = ROOT / cfg["paths"]["split_json"]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n",
                 encoding="utf-8")

    print(f"[B2 §2] split 已冻结 -> {p.relative_to(ROOT)}")
    print(f"  n_traj={out['n_traj']} "
          f"event={out['n_event_observed']} censored={out['n_censored']}")
    for k in ("train", "val", "test"):
        s = out["splits"][k]
        print(f"  {k:5s} n={s['n']:3d} event={s['n_event']:3d} "
              f"censored={s['n_censored']:3d} event_frac={s['event_frac']:.4f}")
    print(f"  split_sha256={out['split_sha256']}")
    print("[B2 §2] B2_SPLIT_FROZEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
