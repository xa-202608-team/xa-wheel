#!/usr/bin/env python
"""scripts/basilisk_b6/build_label_subsets.py

BASILISK-B6 §6/§7/§24-3 —— 生成失效标签稀缺档的 train 轨迹清单。

规则 (全部来自已冻结的 protocol_hash.json, 不从命令行接收任何可调项):
  * 稀缺轴 = event-observed (失效标签) 轨迹条数, 取值 {3,5,10,21};
  * 每档**始终保留 B2.1 train 中全部 24 条 censored 轨迹** —— 稀缺的是失效标签,
    不是数据量。若把稀缺定义成 n_train 总量下降, 就会重新制造 B2 的
    coverage mismatch;
  * 未被选中的 event 轨迹**完全从该档 train 移除**: 不改标成 censored (那是伪造
    删失 —— 我们已知其真实 EOL), 不作为 unlabelled 输入偷偷带回;
  * 选取按 B2.1 冻结的 lifetime bin (edges 取自 manifest, 不重算) 做 coverage-aware
    配额, 各 bin 内用固定 subset_seed 的确定性洗牌后取前 k 条;
  * val / test 永不改动。

各 bin 的洗牌只做一次, 各档从同一洗牌顺序取前 k —— 于是各档是**嵌套**的
(n=3 ⊂ n=5 ⊂ n=10 ⊂ n=21), 标签变多只会增加轨迹, 不会换掉已有轨迹。
这样"标签越少增益越大"的趋势才不会被"换了一批不同轨迹"混淆。

用法:
    python scripts/basilisk_b6/build_label_subsets.py --config configs/wheel_basilisk_b6.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.basilisk_b11.calibrate_degradation import load_b11_config  # noqa: E402
from scripts.basilisk_b21.run_b21_gate import _strata_of  # noqa: E402

INVALID = "B6_INVALID"


def _sha_obj(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def build(cfg: dict) -> dict:
    paths = cfg["paths"]
    b6 = cfg["b6"]
    ph_path = ROOT / cfg["protocol"]["hash_path"]
    if not ph_path.exists():
        raise SystemExit(f"!! {ph_path.relative_to(ROOT)} 缺失 —— 必须先 freeze_protocol")
    ph = json.loads(ph_path.read_text("utf-8"))
    if not bool(ph.get("frozen_before_any_b6_number")):
        raise SystemExit(f"!! protocol 未标记 frozen_before_any_b6_number —— {INVALID}")

    man_path = ROOT / paths["b21_split_json"]
    split = json.loads(man_path.read_text("utf-8"))
    if str(split["split_sha256"]) != str(ph["split"]["sha256"]):
        raise SystemExit(f"!! B2.1 split hash 与冻结值不符 —— {INVALID}")

    # 各档定义与配额一律取自 protocol_hash (冻结件), 不取自 live config ——
    # 若两者不一致, 说明出数字前后有人改过, 下面会交叉校验并拒绝。
    levels = [int(x) for x in ph["label_scarcity"]["levels"]]
    n_cen_req = int(ph["label_scarcity"]["n_censored_train_always"])
    quota = {int(k): {str(b): int(v) for b, v in q.items()}
             for k, q in ph["subset_selection"]["per_level_bin_quota"].items()}
    bin_names = list(ph["subset_selection"]["bin_names"])
    edges = [float(x) for x in ph["subset_selection"]["bin_edges"]]
    subset_seed = int(ph["subset_selection"]["subset_seed"])

    live_levels = [int(x) for x in b6["label_scarcity"]["levels"]]
    live_quota = {int(k): {str(b): int(v) for b, v in q.items()}
                  for k, q in b6["subset_selection"]["per_level_bin_quota"].items()}
    if live_levels != levels or live_quota != quota \
            or int(b6["subset_selection"]["subset_seed"]) != subset_seed:
        raise SystemExit(f"!! live config 与冻结 protocol 的档位/配额/种子不一致 —— {INVALID}")
    if edges != [float(x) for x in split["lifetime_bins"]["event"]["edges"]]:
        raise SystemExit(f"!! bin edges 与 B2.1 manifest 不一致 —— {INVALID}")

    # --- 逐条重建 train 的 strata 归属 (event bin edges 来自 manifest) ---
    tids = list(split["splits"]["train"]["tids"])
    strata = _strata_of(split, "train")
    if len(strata) != len(tids):
        raise SystemExit(f"!! strata 重建长度 {len(strata)} != tids {len(tids)}")

    event_by_bin: dict[str, list[str]] = {b: [] for b in bin_names}
    censored: list[str] = []
    for tid, s in zip(tids, strata):
        kind, _, bn = str(s).partition("/")
        if kind == "event":
            if bn not in event_by_bin:
                raise SystemExit(f"!! 未知 event bin {bn} (tid={tid})")
            event_by_bin[bn].append(tid)
        else:
            censored.append(tid)

    n_event_total = sum(len(v) for v in event_by_bin.values())
    if n_event_total != int(ph["split"]["n_train_event"]):
        raise SystemExit(f"!! train event 条数 {n_event_total} != 冻结值 "
                         f"{ph['split']['n_train_event']} —— {INVALID}")
    if len(censored) != n_cen_req:
        raise SystemExit(f"!! train censored 条数 {len(censored)} != 要求 "
                         f"{n_cen_req} —— {INVALID}")

    # --- 各 bin 内确定性洗牌 (局部 rng, 绝不动全局 np.random) ---
    order: dict[str, list[str]] = {}
    for i, b in enumerate(bin_names):
        rng = np.random.default_rng([subset_seed, i])
        pool = sorted(event_by_bin[b])          # 先排序 -> 与 manifest 书写顺序解耦
        idx = rng.permutation(len(pool))
        order[b] = [pool[int(j)] for j in idx]

    # --- 读 EOL, 仅用于清单登记与人工核对 (不参与选取) ---
    import h5py
    eol: dict[str, float] = {}
    with h5py.File(ROOT / split["feature_h5"], "r") as f:
        for tid in tids:
            g = f[tid]
            eol[tid] = (float(g.attrs["eol_idx"])
                        if bool(int(g.attrs["event_observed"]))
                        else float("nan"))

    levels_out: dict[str, dict] = {}
    for n in levels:
        q = quota[n]
        sel: list[str] = []
        per_bin: dict[str, list[str]] = {}
        for b in bin_names:
            k = q[b]
            if k > len(order[b]):
                raise SystemExit(f"!! n={n} 的 {b} 配额 {k} > 可用 "
                                 f"{len(order[b])} —— {INVALID}")
            per_bin[b] = list(order[b][:k])
            sel.extend(per_bin[b])
        if len(sel) != n or len(set(sel)) != n:
            raise SystemExit(f"!! n={n} 选中 event 条数 {len(sel)} 不等于 {n}")
        removed = sorted(set(t for v in event_by_bin.values() for t in v) - set(sel))
        train_tids = sorted(set(sel) | set(censored))
        if len(train_tids) != n + n_cen_req:
            raise SystemExit(f"!! n={n} train 总数 {len(train_tids)} != {n + n_cen_req}")
        levels_out[str(n)] = {
            "n_event_labeled": n,
            "role": ("PRIMARY" if n == int(ph["primary"]["n_event_labeled"])
                     else "SECONDARY"),
            "event_tids": sorted(sel),
            "event_tids_by_bin": {b: sorted(per_bin[b]) for b in bin_names},
            "event_bin_counts": {b: len(per_bin[b]) for b in bin_names},
            "event_eol": {t: eol[t] for t in sorted(sel)},
            "censored_tids": sorted(censored),
            "n_censored": len(censored),
            "n_train_total": len(train_tids),
            "train_tids": train_tids,
            "removed_event_tids": removed,
            "n_removed_event": len(removed),
            "removed_handling": "fully_removed_not_relabelled_not_unlabelled_input",
        }

    # --- 嵌套性自检: 标签变多只应新增轨迹 ---
    nesting = []
    for a, bb in zip(levels[:-1], levels[1:]):
        sa = set(levels_out[str(a)]["event_tids"])
        sb = set(levels_out[str(bb)]["event_tids"])
        ok = sa.issubset(sb)
        nesting.append({"lower": a, "upper": bb, "subset": bool(ok),
                        "added": sorted(sb - sa)})
        if not ok:
            raise SystemExit(f"!! 嵌套性破坏: n={a} 不是 n={bb} 的子集 —— {INVALID}")
    if set(levels_out[str(levels[-1])]["event_tids"]) != \
            set(t for v in event_by_bin.values() for t in v):
        raise SystemExit(f"!! n={levels[-1]} 应为全部 train event 轨迹 —— {INVALID}")

    core = {
        "levels": levels_out,
        "subset_seed": subset_seed,
        "bin_names": bin_names,
        "bin_edges": edges,
        "per_level_bin_quota": {str(n): quota[n] for n in levels},
    }
    rec = {
        "stage": "BASILISK_B6",
        "label": "LABEL_SUBSET_MANIFEST",
        "purpose": "失效标签稀缺档的 train 轨迹清单; 在任何训练之前生成并冻结。",
        "scarcity_axis": str(ph["scarcity_axis"]),
        "scarcity_axis_is_not": str(ph["scarcity_axis_is_not"]),
        "protocol_sha256": str(ph["protocol_sha256"]),
        "config_sha256": str(ph["config_sha256"]),
        "split_sha256": str(split["split_sha256"]),
        "split_source": str(paths["b21_split_json"]),
        "bin_edges_source": str(ph["subset_selection"]["bin_edges_source"]),
        "forbid_recompute_edges_from_b6_test": True,
        "selection_method": (
            "各 bin 内 sorted(tids) 后用 np.random.default_rng([subset_seed, bin_index]) "
            "做一次确定性置换; 各档从同一置换取前 k 条 -> 各档嵌套。"),
        "generated_before_any_training": True,
        "available_event_per_bin": {b: len(event_by_bin[b]) for b in bin_names},
        "shuffled_order_per_bin": order,
        "censored_train_tids": sorted(censored),
        "n_censored_train_always": n_cen_req,
        "always_keep_all_censored_train": True,
        "forbid_relabel_known_eol_as_censored": True,
        "forbid_smuggling_as_unlabelled_input": True,
        "val_test_never_touched": True,
        "primary_n_event_labeled": int(ph["primary"]["n_event_labeled"]),
        "secondary_levels": [int(x) for x in ph["secondary"]["levels"]],
        "nesting_check": nesting,
        **core,
        "subset_manifest_sha256": _sha_obj(core),
        "forbid_change_after_written": True,
        "invalid_label": INVALID,
    }
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wheel_basilisk_b6.yaml")
    a = ap.parse_args()
    cfg = load_b11_config(a.config)
    out = ROOT / cfg["paths"]["subset_manifest_json"]

    rec = build(cfg)
    if out.exists():
        # §7: 一旦写出即禁止更改。允许幂等重跑, 但内容必须逐字节等价。
        old = json.loads(out.read_text("utf-8"))
        if str(old.get("subset_manifest_sha256")) != str(rec["subset_manifest_sha256"]):
            raise SystemExit(
                f"!! {out.relative_to(ROOT)} 已存在且哈希不同 "
                f"({old.get('subset_manifest_sha256')} -> "
                f"{rec['subset_manifest_sha256']}) —— 子集清单写出后禁止更改, {INVALID}")
        print(f"[subset] 清单已存在且哈希一致, 幂等跳过写入")
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n",
                       encoding="utf-8")
        print(f"[subset] 已写出 -> {out.relative_to(ROOT)}")

    print(f"[subset] subset_manifest_sha256 = {rec['subset_manifest_sha256']}")
    print(f"[subset] subset_seed = {rec['subset_seed']}, "
          f"bin edges = {rec['bin_edges']} (取自 B2.1)")
    for n in [int(x) for x in rec["levels"].keys()]:
        L = rec["levels"][str(n)]
        print(f"[subset] n={n:>2} ({L['role']}): event {L['n_event_labeled']} + "
              f"censored {L['n_censored']} = {L['n_train_total']}; "
              f"bins {L['event_bin_counts']}; 移除 event {L['n_removed_event']} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
