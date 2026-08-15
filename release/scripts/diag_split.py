"""scripts/diag_split.py — S0 诊断 2: 训练集是否已看过失效

问题假设:
  transfer.split = train 0.15 / val 0.20 / test 0.65, 且 load_target(observed_only=True)
  只保留已观测失效轨迹。若 train 划分里的每条轨迹都完整包含失效段 (HI_B→1、
  label_fail=1 的样本占比可观), 则模型在训练时已直接见过"失效长什么样",
  target_only 不需要任何源域先验就能学到失效模式 → 迁移增益天然被压到 0。

做法 (纯读取, 不训练):
  读 target_features.h5, 用 run_groups 同一 split_trajectories(seed) 复现划分,
  对 train 划分逐条打印 event_observed / HI_B 最大值 / label_fail==1 样本占比,
  最后统计 train 集"含失效段轨迹数 / 总轨迹数"。

S1.5 扩展 (任务 2, 同样纯只读):
  HI 与失效判据一致性核查 —— 对全部 100 条轨迹 (不限已观测) 输出
  2x2 交叉表 (HI_B 是否达到 1.0 × event_observed 1/0), 不一致轨迹逐条明细
  (traj_id / HI 首达 1.0 的 index / EOL index / 两者之差 / b0_hat 相对偏差),
  并澄清 "HI_B max~0.8378" 的口径 (逐轨迹最大值的均值 vs 全局 max, 两者都打印)。

用法:
  python scripts/diag_split.py --config configs/wheel.yaml --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.transfer.train_transfer import (                            # noqa: E402
    split_trajectories, split_trajectories_from_cfg, describe_eol_split)
from scripts.diag_common import load_cfg, hyper, observed_traj_keys   # noqa: E402


def describe_split(target_h5: Path, keys: list[str], tids, name: str, hi_key: str):
    """对一个划分逐条打印失效相关统计, 返回 (含失效段轨迹数, 总数, 汇总行)。"""
    rows = []
    n_with_fail = 0
    with h5py.File(target_h5, "r") as f:
        for tid in sorted(int(t) for t in tids):
            key = keys[tid]
            g = f[key]
            hi = np.asarray(g[hi_key][:], dtype=float)
            lf = np.asarray(g["label_fail"][:], dtype=int)
            observed = int(g.attrs.get("event_observed", int(lf.any())))
            eol = int(g.attrs.get("eol_idx", -1))
            fail_ratio = float(lf.sum()) / max(len(lf), 1)
            has_fail = bool(lf.any())
            n_with_fail += int(has_fail)
            rows.append({
                "tid": tid, "key": key, "observed": observed,
                "hi_max": float(hi.max()), "hi_min": float(hi.min()),
                "fail_ratio": fail_ratio, "n": len(lf), "eol": eol,
                "has_fail": has_fail,
            })
    print(f"\n===== {name} 划分逐条明细 ({len(rows)} 条轨迹) =====")
    print(f"{'tid':>4} {'h5_key':>10} {'event_obs':>10} {'HI_B_max':>9} "
          f"{'HI_B_min':>9} {'fail占比':>9} {'样本数':>7} {'EOL_idx':>8}")
    for r in rows:
        print(f"{r['tid']:>4} {r['key']:>10} {r['observed']:>10} {r['hi_max']:>9.4f} "
              f"{r['hi_min']:>9.4f} {r['fail_ratio']:>9.4f} {r['n']:>7} {r['eol']:>8}")
    return n_with_fail, len(rows), rows


def consistency_check(target_h5: Path, hi_key: str, reach_tol: float):
    """S1.5 任务 2: HI 与失效判据一致性核查 (全部轨迹, 纯只读)。

    输出 2x2 交叉表 (HI 是否达 1.0 × event_observed) + 不一致轨迹逐条明细
    + HI max 的两种口径 (逐轨迹最大值的均值 vs 全局 max)。
    reach_tol 由 config sim.hi.reach_tol 传入 (与 build_hi / tests 同一值, 不硬编码)。
    """
    HI_REACH_TOL = float(reach_tol)
    rows = []
    with h5py.File(target_h5, "r") as f:
        for key in sorted(f.keys()):
            g = f[key]
            hi = np.asarray(g[hi_key][:], dtype=float)
            lf = np.asarray(g["label_fail"][:], dtype=int)
            observed = int(g.attrs.get("event_observed", int(lf.any())))
            eol = int(g.attrs.get("eol_idx", -1))
            reached = bool(hi.max() >= 1.0 - HI_REACH_TOL)
            first_reach = int(np.argmax(hi >= 1.0 - HI_REACH_TOL)) if reached else -1
            # 自校准 b0_hat 相对偏差 (build_hi 写入 attrs; 真值来自仿真 h5 不可得,
            # 这里用 b_true 首值作参考 —— b_true 已在 target h5 供离线校核)
            b0_dev = np.nan
            if "b0_hat" in g.attrs and "b_true" in g:
                b0_true = float(np.asarray(g["b_true"][:])[0])
                if abs(b0_true) > 0:
                    b0_dev = (float(g.attrs["b0_hat"]) - b0_true) / b0_true
            rows.append(dict(
                key=key, observed=observed, reached=reached,
                first_reach=first_reach, eol=eol,
                diff=(first_reach - eol) if reached else None,
                hi_max=float(hi.max()), b0_dev=b0_dev, n=len(hi),
            ))

    n_all = len(rows)
    # ---- 2x2 交叉表 ----
    cell = {}
    for r_flag in (True, False):
        for o_flag in (1, 0):
            cell[(r_flag, o_flag)] = [r for r in rows
                                      if r["reached"] == r_flag and r["observed"] == o_flag]
    print(f"\n===== S1.5 任务 2: HI 与失效判据一致性 2x2 交叉表 (全部 {n_all} 条) =====")
    print(f"HI 达到 1.0 的判定容差 = {HI_REACH_TOL} (即 HI_max >= {1.0 - HI_REACH_TOL})")
    print(f"\n{'':<22} {'event_observed=1':>18} {'event_observed=0':>18} {'行合计':>9}")
    for r_flag, label in [(True, f"HI_{hi_key} 达到 1.0"), (False, f"HI_{hi_key} 未达 1.0")]:
        n1 = len(cell[(r_flag, 1)])
        n0 = len(cell[(r_flag, 0)])
        print(f"{label:<22} {n1:>18} {n0:>18} {n1 + n0:>9}")
    c1 = len(cell[(True, 1)]) + len(cell[(False, 1)])
    c0 = len(cell[(True, 0)]) + len(cell[(False, 0)])
    print(f"{'列合计':<22} {c1:>18} {c0:>18} {n_all:>9}")

    n_consistent = len(cell[(True, 1)]) + len(cell[(False, 0)])
    print(f"\n一致 (达1.0&已观测 + 未达1.0&未观测) = {n_consistent}/{n_all} "
          f"({100.0 * n_consistent / max(n_all, 1):.1f}%)")
    print(f"不一致 = {n_all - n_consistent}/{n_all}")
    print(f"  类型 I  达到 1.0 但 event_observed=0 (HI 报警早于/无失效判据) = "
          f"{len(cell[(True, 0)])}")
    print(f"  类型 II 未达 1.0 但 event_observed=1 (已失效但 HI 未饱和)     = "
          f"{len(cell[(False, 1)])}")

    # ---- 不一致轨迹逐条明细 ----
    incons = cell[(True, 0)] + cell[(False, 1)]
    if incons:
        print(f"\n===== 不一致轨迹逐条明细 ({len(incons)} 条) =====")
        print(f"{'traj_id':>10} {'类型':>6} {'HI首达1.0_idx':>14} {'EOL_idx':>9} "
              f"{'差(首达-EOL)':>13} {'HI_max':>8} {'b0_hat相对偏差':>15} {'样本数':>7}")
        for r in sorted(incons, key=lambda x: x["key"]):
            typ = "I" if (r["reached"] and r["observed"] == 0) else "II"
            fr = str(r["first_reach"]) if r["first_reach"] >= 0 else "未达"
            df_ = f"{r['diff']:+d}" if r["diff"] is not None else "n/a"
            bd = f"{r['b0_dev']:+.4f}" if np.isfinite(r["b0_dev"]) else "n/a"
            print(f"{r['key']:>10} {typ:>6} {fr:>14} {r['eol']:>9} "
                  f"{df_:>13} {r['hi_max']:>8.4f} {bd:>15} {r['n']:>7}")
    else:
        print("\n>> 无不一致轨迹")

    # ---- 一致组的时序对齐情况 (达1.0 且已观测: 首达 vs EOL 相差多少) ----
    ok = cell[(True, 1)]
    if ok:
        d = np.array([r["diff"] for r in ok], dtype=float)
        print(f"\n===== 一致组 (达1.0 & 已观测, {len(ok)} 条) 的时序对齐 =====")
        print(f"HI 首达 1.0 相对 EOL 的偏移 (负=HI 先饱和, 正=HI 后饱和):")
        print(f"  mean={d.mean():+.1f}  median={np.median(d):+.1f}  "
              f"p5={np.quantile(d, 0.05):+.1f}  p95={np.quantile(d, 0.95):+.1f}  "
              f"min={d.min():+.0f}  max={d.max():+.0f}")
        print(f"  HI 早于 EOL 饱和的轨迹数 = {int((d < 0).sum())}/{len(ok)}")

    # ---- HI max 口径澄清 ----
    hi_maxs = np.array([r["hi_max"] for r in rows])
    print(f"\n===== HI_{hi_key} max 口径澄清 =====")
    print(f"口径 1 「各轨迹 HI 最大值的均值」 mean(max_t HI) = {hi_maxs.mean():.4f}")
    print(f"        ^^ build_hi --report 打印的 'HI_B 范围 max~0.8378' 就是这个口径")
    print(f"口径 2 「全局最大值」            max over all      = {hi_maxs.max():.4f}")
    print(f"        全局 min over all (各轨迹 max 的最小值)   = {hi_maxs.min():.4f}")
    print(f"        达到 1.0 的轨迹数 = {int((hi_maxs >= 1.0 - HI_REACH_TOL).sum())}/{n_all}")
    print("==========================")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wheel.yaml")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    seed = args.seed if args.seed is not None else int(cfg["seed"])
    h = hyper(cfg)
    target_h5 = h["target_h5"]
    if not target_h5.exists():
        print(f"!! 缺 {target_h5}; 先 python -m src.sim.build_hi --report")
        sys.exit(1)

    hi_key = "hi_array" if h["has_nodes"] else "hi_b"
    keys = observed_traj_keys(target_h5)          # load_target(observed_only=True) 的保留序列
    n_traj = len(keys)

    with h5py.File(target_h5, "r") as f:
        n_all = len(list(f.keys()))
    print("===== S0 诊断 2: 训练集是否已看过失效 =====")
    print(f"config              = {args.config}")
    print(f"target h5           = {target_h5}")
    print(f"h5 内轨迹总数       = {n_all}")
    print(f"已观测失效轨迹      = {n_traj} (load_target observed_only=True 只用这些)")
    print(f"HI 字段             = {hi_key}")
    sc = cfg["transfer"]["split"]
    print(f"split 比例          = train {h['split_ratios'][0]} / "
          f"val {h['split_ratios'][1]} / test {h['split_ratios'][2]}")
    print(f"split_trajectories seed = {seed}")
    print(f"HI 口径             = sim.failure.hi_source="
          f"{cfg['sim']['failure']['hi_source']}  "
          f"(Tf_window={cfg['sim']['hi']['Tf_window']}, "
          f"Tf_fail_Kt_source={cfg['sim']['failure']['Tf_fail_Kt_source']})")
    print(f"EOL 分层            = stratify_by_eol={sc.get('stratify_by_eol')} "
          f"n_eol_strata={sc.get('n_eol_strata')}")

    tr, va, te, eol_by_tid = split_trajectories_from_cfg(
        target_h5, n_traj, cfg["transfer"], seed, observed_only=True)
    print(f"划分结果            = train {len(tr)} / val {len(va)} / test {len(te)} 条")
    describe_eol_split(eol_by_tid, tr, va, te)
    # 对照: 同 seed 的纯随机划分 EOL 分布 (说明分层确实改善了难度可比性)
    if eol_by_tid is not None:
        rtr, rva, rte = split_trajectories(n_traj, h["split_ratios"], seed)
        print("  -- 对照: 同 seed 纯随机划分的 EOL 分位数 --")
        describe_eol_split(eol_by_tid, rtr, rva, rte)

    n_fail_tr, n_tr, rows_tr = describe_split(target_h5, keys, tr, "TRAIN", hi_key)

    print(f"\n===== 诊断 2 结论数据 =====")
    print(f"train 集含失效段的轨迹数 / 总轨迹数 = {n_fail_tr} / {n_tr} "
          f"({100.0 * n_fail_tr / max(n_tr, 1):.1f}%)")
    fr = np.array([r["fail_ratio"] for r in rows_tr])
    hm = np.array([r["hi_max"] for r in rows_tr])
    print(f"train label_fail==1 样本占比: min={fr.min():.4f} max={fr.max():.4f} "
          f"mean={fr.mean():.4f}")
    print(f"train HI_B 最大值:           min={hm.min():.4f} max={hm.max():.4f} "
          f"mean={hm.mean():.4f}")
    print(f"train 全部 event_observed==1 = "
          f"{all(r['observed'] == 1 for r in rows_tr)}")
    # 参考: val/test 是否同样含失效 (说明这不是 train 独有)
    n_fail_va = sum(1 for t in va
                    if rows_lookup(target_h5, keys, int(t)))
    n_fail_te = sum(1 for t in te
                    if rows_lookup(target_h5, keys, int(t)))
    print(f"对照 val 含失效段  = {n_fail_va} / {len(va)}")
    print(f"对照 test 含失效段 = {n_fail_te} / {len(te)}")
    print("==========================")

    # ---- S1.5 任务 2: HI 与失效判据一致性核查 (全部 100 条, 不限已观测) ----
    consistency_check(target_h5, hi_key, float(cfg["sim"]["hi"]["reach_tol"]))


def rows_lookup(target_h5: Path, keys: list[str], tid: int) -> bool:
    """单条轨迹是否含失效段 (label_fail 任一为 1)。"""
    with h5py.File(target_h5, "r") as f:
        return bool(np.asarray(f[keys[tid]]["label_fail"][:]).any())


if __name__ == "__main__":
    main()
