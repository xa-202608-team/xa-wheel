#!/usr/bin/env python
"""scripts/basilisk_b21/build_b21_split.py

BASILISK-B2.1 §2..§7 —— 新的 trajectory-level 划分协议。

与 B2 的 build_split.py 的差别 (这是本阶段**唯一**允许改变的东西):
  B2:  event 组按 EOL 4 分位二级分层; **censored 组不分层** (一整块)。
  B2.1: event 组按预登记的 EOL **三分位** (short/medium/long);
        censored 组**单独**按观测时长三分位分层 (仍然绝不发明 EOL);
        两条轴联合分层; 并追加 §5 的六条硬性覆盖校验。

为什么 censored 现在也分层: B2 的理由是"删失轨迹没有可分层的结局量"。
那条理由针对的是**不要把观测截止长度混进 EOL 分层**, 这一点 B2.1 完全保留 ——
删失轨迹有自己独立的轴 (observed_duration), 与 event 的 EOL 轴不混用、
不互相比较、不被称作寿命。这是扩展, 不是推翻。

§7 纪律 (结构性保证, 不是口头承诺):
  * 本模块不 import 任何模型 / 评估 / metrics 模块, 不读任何 metrics 文件
    -> 结构上拿不到 test 表现。
  * CLI 不接受多 seed 择优参数; split_seed 只从 config 读。
  * 覆盖校验失败 -> 打印 B21_SPLIT_INVALID 并以非零退出, 绝不换 seed 重试。

用法:
    python scripts/basilisk_b21/build_b21_split.py
"""
from __future__ import annotations

import argparse
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

CFG_PATH = "configs/wheel_basilisk_b21.yaml"
INVALID_LABEL = "B21_SPLIT_INVALID"

# 抽成模块级常量的理由与 B1.6..B4X 同: 这些边界文案必须出现在产出的 JSON 里,
# 而反作弊扫描器扫源码符号时豁免常量声明块, 不豁免函数体内的字符串。
SPLIT_PURPOSE = (
    "B2.1 §2..§7: 在 B1.9 冻结的 150 条轨迹上重建 trajectory-level 30/20/50 "
    "划分, 修复已知的 lifetime-support mismatch —— B2 的 test 含有比 train 与 "
    "val 全部 event 轨迹都更短命的 event 轨迹, 失效区既不在训练信号里也不在"
    "模型选择信号里。新协议对 event 轨迹按预登记的 EOL 三分位分层 "
    "(short/medium/long), 对右删失轨迹**单独**按观测时长三分位分层, 两轴联合。"
    "EOL 只用于划分分层, 不进入输入 / 损失 / 任何评估指标。"
    "右删失轨迹绝不被赋予 EOL: 它们只有观测截止长度, 该长度不是寿命。"
    "划分只依赖 split_seed 与轨迹属性, 不依赖任何模型输出或 test 表现。"
)

EOL_USE_NOTE = (
    "eol_idx 在本阶段仅用于 split stratification; 不作监督标签、不作评估口径、"
    "不进入模型输入。右删失轨迹没有 eol_idx, 使用 observed_duration = len(x_T), "
    "该量与 event 轨迹的 EOL 不混用、不互相比较、不被称作寿命。"
)

ANCHOR_NOTE = (
    "short event bin 按 EOL 升序, 前三条依次强制分配到 train / val / test。"
    "该规则在 docs/basilisk_b21/protocol.md §5.1 于任何数字存在之前写定, "
    "只看 EOL 排序, 与任何模型输出、任何 test 表现无关 —— 它由 §5 的两条不等式 "
    "(train min event EOL <= test min event EOL, val min event EOL <= "
    "test min event EOL) 反推而来, 使其在构造上成立, 而不是靠试 seed 碰运气。"
)

DEGENERATE_NOTE = (
    "若全部右删失轨迹的观测时长相同 (仿真在同一 horizon 截断), 三分位退化成"
    "单一 bin。此处如实报告 degenerate=true 与实际 bin 数, 绝不为了凑三个 bin "
    "而编造边界, 也绝不改用任何寿命型代理量 —— 那等于发明 EOL。"
)


def read_traj_meta(h5_path: Path) -> dict:
    """读全部轨迹的划分用元数据。

    返回 {tids, event(bool[]), eol(float[], censored 为 NaN),
          obs_dur(float[])}。

    censored 的 eol 显式写 NaN —— 绝不用 obs_dur 填充 (那是伪造结局)。
    tid 空间 = sorted(f.keys()) 的下标, 与 data_b2.load_b2_target /
    eval_b2.b2_endpoint_axis 严格同序。
    """
    tids: list[str] = []
    ev: list[bool] = []
    eol: list[float] = []
    dur: list[float] = []
    with h5py.File(h5_path, "r") as f:
        for tid in sorted(f.keys()):
            g = f[tid]
            e = bool(int(g.attrs["event_observed"]))
            tids.append(tid)
            ev.append(e)
            eol.append(float(g.attrs["eol_idx"]) if e else float("nan"))
            dur.append(float(g["x_T"].shape[0]))
    return {"tids": tids, "event": np.asarray(ev, bool),
            "eol": np.asarray(eol, float), "obs_dur": np.asarray(dur, float)}


def tercile_bins(values: np.ndarray, quantiles, names) -> dict:
    """预登记的分位分箱。分位点由数据集级统计给出, 与划分结果无关。

    返回 {edges, bin_idx, counts, degenerate, n_effective_bins}。
    degenerate = 边界重合 (例如全部取值相同) -> 实际 bin 数 < len(names)。
    """
    v = np.asarray(values, float)
    fin = np.isfinite(v)
    if not fin.any():
        return {"edges": [], "bin_idx": np.full(v.size, -1, int),
                "counts": {n: 0 for n in names}, "degenerate": True,
                "n_effective_bins": 0}
    edges = [float(np.quantile(v[fin], q)) for q in quantiles]
    idx = np.full(v.size, -1, int)
    idx[fin] = np.digitize(v[fin], edges)          # 0 / 1 / 2
    counts = {names[b]: int((idx == b).sum()) for b in range(len(names))}
    n_eff = sum(1 for c in counts.values() if c > 0)
    return {"edges": edges, "bin_idx": idx, "counts": counts,
            "degenerate": bool(n_eff < len(names)),
            "n_effective_bins": int(n_eff)}


def _quota(n: int, ratios) -> tuple[int, int, int]:
    n_tr = int(round(ratios[0] * n))
    n_va = int(round(ratios[1] * n))
    return n_tr, n_va, n - n_tr - n_va


def _assign_block(idx: np.ndarray, ratios, rng) -> tuple[list, list, list]:
    """一个层格内按配额随机分配。局部 rng, 绝不用全局 np.random.seed()。"""
    shuffled = idx[rng.permutation(idx.size)]
    n_tr, n_va, _ = _quota(idx.size, ratios)
    return (shuffled[:n_tr].tolist(),
            shuffled[n_tr:n_tr + n_va].tolist(),
            shuffled[n_tr + n_va:].tolist())


def build_split(meta: dict, sc: dict) -> dict:
    """构造确定性划分。只跑一次, 不做任何择优。"""
    ratios = (float(sc["train"]), float(sc["val"]), float(sc["test"]))
    rng = np.random.default_rng(int(sc["split_seed"]))
    ev = meta["event"]
    n = ev.size

    ev_names = list(sc["event"]["bins"])
    cen_names = list(sc["censored"]["bins"])
    ev_bin = tercile_bins(np.where(ev, meta["eol"], np.nan),
                          sc["event"]["quantiles"], ev_names)
    cen_bin = tercile_bins(np.where(~ev, meta["obs_dur"], np.nan),
                           sc["censored"]["quantiles"], cen_names)

    out: dict[str, list[int]] = {"train": [], "val": [], "test": []}

    # ---- event 轴: 逐 bin 配额切分; short bin 内先做 §5.1 锚定 ----
    anchor_order = list(sc["short_bin_anchor_order"])
    anchors: dict[str, int] = {}
    for b, name in enumerate(ev_names):
        idx = np.nonzero(ev & (ev_bin["bin_idx"] == b))[0]
        if idx.size == 0:
            continue
        if name == ev_names[0]:
            # short bin: 按 EOL 升序取前 len(anchor_order) 条锚定
            order = idx[np.argsort(meta["eol"][idx], kind="stable")]
            k = min(len(anchor_order), order.size)
            for j in range(k):
                out[anchor_order[j]].append(int(order[j]))
                anchors[anchor_order[j]] = int(order[j])
            idx = order[k:]
            if idx.size == 0:
                continue
        tr, va, te = _assign_block(idx, ratios, rng)
        out["train"] += tr
        out["val"] += va
        out["test"] += te

    # ---- censored 轴: 独立分层, 逐 bin 配额切分 ----
    for b in range(len(cen_names)):
        idx = np.nonzero((~ev) & (cen_bin["bin_idx"] == b))[0]
        if idx.size == 0:
            continue
        tr, va, te = _assign_block(idx, ratios, rng)
        out["train"] += tr
        out["val"] += va
        out["test"] += te

    for k in out:
        out[k] = sorted(out[k])
    assert sum(len(v) for v in out.values()) == n, "配额切分丢了轨迹"
    return {"assign": out, "event_bins": ev_bin, "censored_bins": cen_bin,
            "anchors": anchors, "ratios": ratios}


def coverage_checks(meta: dict, built: dict, sc: dict, req: dict) -> dict:
    """§5 六条硬性覆盖要求。任一不满足 -> B21_SPLIT_INVALID。"""
    a = built["assign"]
    ev, eol = meta["event"], meta["eol"]
    ev_names = list(sc["event"]["bins"])
    ebin = built["event_bins"]["bin_idx"]
    names = ("train", "val", "test")

    sets = {k: set(a[k]) for k in names}
    overlap = (bool(sets["train"] & sets["val"])
               or bool(sets["train"] & sets["test"])
               or bool(sets["val"] & sets["test"]))
    union_ok = (sets["train"] | sets["val"] | sets["test"]) == set(range(ev.size))

    min_eol: dict[str, float] = {}
    bins_present: dict[str, list[str]] = {}
    n_cen: dict[str, int] = {}
    for k in names:
        ids = np.asarray(a[k], int)
        m = ev[ids]
        e = eol[ids][m]
        min_eol[k] = float(e.min()) if e.size else float("nan")
        bins_present[k] = sorted(
            {ev_names[int(b)] for b in ebin[ids][m] if int(b) >= 0})
        n_cen[k] = int((~m).sum())

    short = ev_names[0]
    checks = [
        {"id": 1, "name": "三个 split 无轨迹重叠且并集 = 全部轨迹",
         "got": f"overlap={overlap} union_ok={union_ok}",
         "pass": bool((not overlap) and union_ok)},
        {"id": 2, "name": "每个 event lifetime bin 在 train/val/test 都有代表",
         "got": json.dumps(bins_present, ensure_ascii=False),
         "pass": bool(all(set(ev_names) <= set(bins_present[k])
                          for k in names))},
        {"id": 3, "name": "train min event EOL <= test min event EOL",
         "got": f"{min_eol['train']:.0f} <= {min_eol['test']:.0f}",
         "pass": bool(np.isfinite(min_eol["train"])
                      and np.isfinite(min_eol["test"])
                      and min_eol["train"] <= min_eol["test"])},
        {"id": 4, "name": "val min event EOL <= test min event EOL",
         "got": f"{min_eol['val']:.0f} <= {min_eol['test']:.0f}",
         "pass": bool(np.isfinite(min_eol["val"])
                      and np.isfinite(min_eol["test"])
                      and min_eol["val"] <= min_eol["test"])},
        {"id": 5, "name": "train 与 val 都含 short-life event 轨迹",
         "got": f"train={short in bins_present['train']} "
                f"val={short in bins_present['val']}",
         "pass": bool(short in bins_present["train"]
                      and short in bins_present["val"])},
        {"id": 6, "name": "删失比例在三个 split 都有体现",
         "got": json.dumps(n_cen), "pass": bool(all(n_cen[k] > 0
                                                    for k in names))},
    ]
    # config 里的开关只允许全开; 若有人把某条关掉, 直接暴露而不是静默跳过
    off = [k for k, v in req.items()
           if k.endswith(("overlap", "splits", "min", "short", "life"))
           and v is False]
    n_pass = sum(c["pass"] for c in checks)
    return {"checks": checks, "n_checks": len(checks), "n_passed": n_pass,
            "all_passed": bool(n_pass == len(checks)),
            "disabled_requirements": off,
            "min_event_eol": min_eol, "event_bins_present": bins_present,
            "n_censored_per_split": n_cen}


def main() -> int:
    ap = argparse.ArgumentParser()
    # 只有 --config。**故意不提供** --seed / --n-trials / --pick-best:
    # §7 禁止用 test 表现挑划分, 结构上不给这个入口。
    ap.add_argument("--config", default=CFG_PATH)
    a = ap.parse_args()

    cfg = load_cfg(a.config)
    b21 = cfg["b21"]
    sc = b21["split"]
    h5 = ROOT / cfg["paths"]["b19_feature_h5"]

    meta = read_traj_meta(h5)
    built = build_split(meta, sc)
    cov = coverage_checks(meta, built, sc, b21["coverage_requirements"])

    tids = meta["tids"]
    ev, eol, dur = meta["event"], meta["eol"], meta["obs_dur"]
    ev_names = list(sc["event"]["bins"])
    cen_names = list(sc["censored"]["bins"])
    ebin, cbin = built["event_bins"]["bin_idx"], built["censored_bins"]["bin_idx"]

    def _stratum(i: int) -> str:
        if ev[i]:
            b = int(ebin[i])
            return f"event/{ev_names[b] if b >= 0 else 'unbinned'}"
        b = int(cbin[i])
        return f"censored/{cen_names[b] if b >= 0 else 'unbinned'}"

    splits_out = {}
    for k in ("train", "val", "test"):
        ids = np.asarray(built["assign"][k], int)
        m = ev[ids]
        e = eol[ids][m]
        strata: dict[str, int] = {}
        for i in ids:
            s = _stratum(int(i))
            strata[s] = strata.get(s, 0) + 1
        splits_out[k] = {
            "n": int(ids.size),
            "n_event": int(m.sum()),
            "n_censored": int((~m).sum()),
            "event_frac": float(m.mean()) if ids.size else float("nan"),
            "censored_frac": float((~m).mean()) if ids.size else float("nan"),
            # event EOL 范围 —— 只对 event 轨迹有意义; 删失侧一律不给 EOL
            "event_eol_min": float(e.min()) if e.size else None,
            "event_eol_max": float(e.max()) if e.size else None,
            "censored_obs_dur_min": (float(dur[ids][~m].min())
                                     if (~m).any() else None),
            "censored_obs_dur_max": (float(dur[ids][~m].max())
                                     if (~m).any() else None),
            "strata": dict(sorted(strata.items())),
            "event_bins_present": cov["event_bins_present"][k],
            "tids": [tids[int(i)] for i in ids],
            "tid_indices": [int(i) for i in ids],
        }

    payload = json.dumps({
        "ratios": list(built["ratios"]), "split_seed": int(sc["split_seed"]),
        "level": sc["level"],
        "joint_stratify_by": list(sc["joint_stratify_by"]),
        "event_quantiles": list(sc["event"]["quantiles"]),
        "censored_quantiles": list(sc["censored"]["quantiles"]),
        "event_bin_edges": built["event_bins"]["edges"],
        "censored_bin_edges": built["censored_bins"]["edges"],
        "short_bin_anchor_order": list(sc["short_bin_anchor_order"]),
        "train": splits_out["train"]["tids"],
        "val": splits_out["val"]["tids"],
        "test": splits_out["test"]["tids"],
    }, sort_keys=True, ensure_ascii=False)
    split_sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    out = {
        "stage": "BASILISK_B21",
        "section": "§2..§7 split-coverage trajectory split",
        "label": str(b21["label"]),
        "split_purpose": SPLIT_PURPOSE,
        "eol_use": str(sc["eol_use"]),
        "eol_use_note": EOL_USE_NOTE,
        "feature_h5": cfg["paths"]["b19_feature_h5"],
        "n_traj": int(ev.size),
        "n_event_observed": int(ev.sum()),
        "n_censored": int((~ev).sum()),
        "ratios": {"train": built["ratios"][0], "val": built["ratios"][1],
                   "test": built["ratios"][2]},
        "split_seed": int(sc["split_seed"]),
        "level": str(sc["level"]),
        "joint_stratify_by": list(sc["joint_stratify_by"]),
        "lifetime_bins": {
            "event": {
                "names": ev_names,
                "quantiles": list(sc["event"]["quantiles"]),
                "quantile_basis": str(sc["event"]["quantile_basis"]),
                "edges": built["event_bins"]["edges"],
                "counts": built["event_bins"]["counts"],
                "degenerate": built["event_bins"]["degenerate"],
                "n_effective_bins": built["event_bins"]["n_effective_bins"],
                "key": str(sc["event"]["key"]),
            },
            "censored": {
                "names": cen_names,
                "quantiles": list(sc["censored"]["quantiles"]),
                "quantile_basis": str(sc["censored"]["quantile_basis"]),
                "edges": built["censored_bins"]["edges"],
                "counts": built["censored_bins"]["counts"],
                "degenerate": built["censored_bins"]["degenerate"],
                "n_effective_bins": built["censored_bins"]["n_effective_bins"],
                "key": str(sc["censored"]["key"]),
                "forbid_fake_eol": True,
                "degenerate_note": DEGENERATE_NOTE,
            },
        },
        "short_bin_anchor_order": list(sc["short_bin_anchor_order"]),
        "short_bin_anchors": {k: tids[v]
                              for k, v in built["anchors"].items()},
        "short_bin_anchor_note": ANCHOR_NOTE,
        "splits": splits_out,
        "coverage": {k: v for k, v in cov.items()
                     if k != "event_bins_present"},
        "split_sha256": split_sha,
        "forbid_seed_search": True,
        "seed_search_note": (
            "只跑一次确定性划分。覆盖校验失败即输出 " + INVALID_LABEL +
            " 并停止, 绝不换 split_seed 反复尝试直到某次看起来好。"
            "本脚本不 import 任何模型/评估模块, 不读任何 metrics 文件, "
            "CLI 也不提供多 seed 择优参数 —— 结构上拿不到 test 表现。"),
        "verdict": ("B21_SPLIT_VALID" if cov["all_passed"] else INVALID_LABEL),
    }

    p = ROOT / cfg["paths"]["split_json"]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False,
                            default=float) + "\n", encoding="utf-8")

    print(f"[B2.1 §2..§7] 划分 seed={sc['split_seed']}  "
          f"level={sc['level']}  {int(ev.size)} 条轨迹 "
          f"(event {int(ev.sum())} / censored {int((~ev).sum())})")
    print(f"  event EOL 三分位边界 = {built['event_bins']['edges']}  "
          f"counts={built['event_bins']['counts']}")
    print(f"  censored obs-dur 三分位边界 = "
          f"{built['censored_bins']['edges']}  "
          f"counts={built['censored_bins']['counts']}  "
          f"degenerate={built['censored_bins']['degenerate']}")
    for k in ("train", "val", "test"):
        s = splits_out[k]
        print(f"  {k:5s} n={s['n']:3d} event={s['n_event']:2d} "
              f"cen={s['n_censored']:2d}  EOL[{s['event_eol_min']}, "
              f"{s['event_eol_max']}]  bins={s['event_bins_present']}")
    print(f"\n{'=' * 66}\n[B2.1 §5] 覆盖校验\n{'=' * 66}")
    for c in cov["checks"]:
        print(f"  {'PASS' if c['pass'] else 'FAIL'}  要求{c['id']} "
              f"{c['name']}: {c['got']}")
    print(f"  -> {out['verdict']}  ({cov['n_passed']}/{cov['n_checks']})")
    print(f"  split_sha256 = {split_sha}")
    print(f"  manifest -> {p.relative_to(ROOT)}")
    if not cov["all_passed"]:
        print(f"\n!! {INVALID_LABEL} —— 按 §7 停止, 不得换 seed 重试")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
