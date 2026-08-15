"""baselines/trivial.py — 平凡基线对照 (S2' 任务 1, 后续所有阶段的前置判据)

存在意义: target_only 在主口径 (info, 0<RUL<1) 上的 test RMSE 约 0.23~0.26, 而 info 区
RUL 近似均匀分布, 常数均值预测器的理论 RMSE ≈ 1/sqrt(12) = 0.2887 —— 即模型在唯一有预测
价值的区间里 R² 仅 0.17~0.37。在继续任何结构性改动 (S3 截断等) 之前, 必须先确认模型显著
优于"什么都不学"的平凡预测器, 否则后续所有增益都是在噪声上做优化。

三个平凡预测器 (全部与 physical_extrap / target_only 同划分、同口径、同指标、同取点):
  (a) const_mean       输出 train 集**全样本** RUL 均值 (含 RUL=0 段, 故被拉低到 ~0.13)
      const_mean_info  输出 train 集**info 区** RUL 均值 —— 与理论锚点 1/sqrt(12) 可比的
                       强常数基线; 仍只用 train, 不看 test
  (b) hi_extrap        由当前 HI 与滑窗 HI 斜率线性外推到 HI=1, 得剩余步数 → 归一 RUL
  (c) persistence      输出"上一可见时刻"的真实 RUL。**用真值, 不可部署**, 仅作乐观上界:
                       它的 RMSE ≈ stride/rul_scale, 量化了"RUL 标签在时间上有多平滑",
                       即任何能定位轨迹内相对时间的方法能达到的下界。

METHODS = 打印与汇总的方法顺序。

用法:
  python -m src.baselines.trivial --config configs/wheel.yaml
  python -m src.baselines.trivial --config configs/wheel.yaml --seeds 42,43,44
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils import load_config, set_seed                              # noqa: E402
from src.transfer.train_transfer import (                                # noqa: E402
    split_trajectories_from_cfg, describe_eol_split)
from src.baselines.physical_extrap import (                              # noqa: E402
    baseline_hyper, caliber_metrics, caliber_mask, format_calibers,
    clip_to_label_domain, eval_point_indices, observed_traj_keys,
    evaluate_physical, PRIMARY_CALIBER, PRIMARY_STAT)

# info 区 RUL 近似均匀分布下常数预测器的理论 RMSE = 1/sqrt(12) (用户提供的判据锚点)
UNIFORM_CONST_RMSE = float(1.0 / np.sqrt(12.0))

# 打印顺序; const_mean_info 是与理论锚点可比的那一档
METHODS = ("const_mean", "const_mean_info", "hi_extrap", "persistence")

# **ORACLE-LIKE / NOT DEPLOYABLE** —— 这些方法读了未来/真值, 在轨无法实现。
# 它们只用于给出"任何方法能达到的乐观下界", 严禁作为对比实验的可部署基线,
# 严禁参与 S2.5 泛化闸门 (scripts/run_s25_gate.py 用 GATE_BASELINE 显式排除)。
ORACLE_LIKE = ("persistence",)

# S2.5 泛化闸门唯一使用的基线: 最强的**可部署**平凡基线。
# 常数值只由 train 划分的 info 区真实 RUL 求均值 —— 绝不看 val/test
# (tests/test_generalization_gate.py::test_const_mean_info_uses_train_only 钉死)。
GATE_BASELINE = "const_mean_info"


def _hi_key(cfg: dict) -> str:
    """HI 列名: 相控阵 hi_array / 飞轮 hi_b (与 train_transfer.load_target 同口径)。"""
    return "hi_array" if bool(cfg["transfer"].get("target_has_nodes", False)) else "hi_b"


def hi_extrap_rul(hi: np.ndarray, window: int, hi_target: float = 1.0) -> np.ndarray:
    """由当前 HI 与滑窗 HI 斜率线性外推到 HI=hi_target, 返回剩余步数 (绝对时间单位)。

    对每个 t: 用 [t-window, t] 的 HI 线性拟合斜率 s, RUL = (hi_target - HI[t]) / s。
    s <= 0 (HI 未上升) → NaN, 由调用方前向填充; HI 已达 hi_target → 0。
    """
    n = len(hi)
    t = np.arange(n, dtype=float)
    out = np.full(n, np.nan)
    w = max(2, int(window))
    for i in range(n):
        lo = max(0, i - w)
        if i - lo + 1 < 5:
            continue
        s = float(np.polyfit(t[lo:i + 1], hi[lo:i + 1], 1)[0])
        if s <= 1e-12:
            continue
        gap = float(hi_target) - float(hi[i])
        out[i] = max(0.0, gap / s)
    return out


def _fill_forward(v: np.ndarray, fallback: float) -> np.ndarray:
    v = np.asarray(v, dtype=float).copy()
    nm = np.isnan(v)
    if nm.any():
        idx = np.maximum.accumulate(np.where(~nm, np.arange(len(v)), -1))
        v = np.where(idx >= 0, v[np.clip(idx, 0, None)], float(fallback))
    return v


def evaluate_trivial(cfg: dict, seed: int, return_raw: bool = False) -> dict | None:
    """三个平凡预测器的三档口径评估。返回 {name: caliber_metrics dict}。

    return_raw=True 时额外返回 `_raw = {"true": t, "tids": tids, "pred": {name: arr}}`,
    供 S2.5 闸门在**与模型完全相同的 test 取点**上算 PSR / macro_corr。
    """
    set_seed(seed, cfg["reproducibility"]["deterministic"])
    h = baseline_hyper(cfg)
    hi_key = _hi_key(cfg)
    if not h["target_h5"].exists():
        print(f"!! 缺 {h['target_h5']}; 先 python -m src.sim.build_hi --report")
        return None

    with h5py.File(h["target_h5"], "r") as f:
        keys = observed_traj_keys(f)
        if not keys:
            return None
        tr, va, te, eol = split_trajectories_from_cfg(
            h["target_h5"], len(keys), cfg["transfer"], seed, observed_only=True)
        print(f">> 轨迹划分 train {len(tr)} / val {len(va)} / test {len(te)}")
        describe_eol_split(eol, tr, va, te)

        # ---- rul_scale 与 train 集 RUL 均值: 只用 train, 与 run_groups 同口径 ----
        rul_scale = max(float(np.max([np.nanmax(f[keys[int(i)]]["rul"][:]) for i in tr])), 1.0)
        # 两档常数值都在 train 的**评估取点**上算 (与 test 侧取点方式一致):
        #   const_val      = 全样本均值 (被 RUL=0 段拉低)
        #   const_val_info = info 区 (0<RUL<1-eps) 均值, 与理论锚点 1/sqrt(12) 可比
        tr_vals = []
        for i in tr:
            g = f[keys[int(i)]]
            r = g["rul"][:].astype(float) / rul_scale
            pts = eval_point_indices(len(r), h["L"], h["stride"])
            if len(pts):
                tr_vals.append(r[pts])
        tr_all = np.concatenate(tr_vals) if tr_vals else np.zeros(1)
        const_val = float(np.mean(tr_all))
        _im = caliber_mask(tr_all, "info", h["cap_eps"])
        const_val_info = float(np.mean(tr_all[_im])) if _im.any() else const_val

        preds = {k: [] for k in METHODS}
        T, TID = [], []
        for tid in te:
            g = f[keys[int(tid)]]
            r_true = g["rul"][:].astype(float) / rul_scale
            hi = np.asarray(g[hi_key][:], dtype=float)
            pts = eval_point_indices(len(r_true), h["L"], h["stride"])
            if not len(pts):
                continue
            # (a) 常数均值 (两档)
            preds["const_mean"].append(np.full(len(pts), const_val))
            preds["const_mean_info"].append(np.full(len(pts), const_val_info))
            # (b) HI 线性外推到 HI=1 (绝对步数 → 同 rul_scale 归一)
            hx = hi_extrap_rul(hi, h["extrap_window"], h["hi_target"]) / rul_scale
            preds["hi_extrap"].append(clip_to_label_domain(_fill_forward(hx[pts], 1.0)))
            # (c) persistence: 上一个评估时刻的真实 RUL (首点用自身, 无更早可见值)
            prev = np.concatenate([[r_true[pts[0]]], r_true[pts[:-1]]])
            preds["persistence"].append(clip_to_label_domain(prev))
            T.append(r_true[pts])
            TID.append(np.full(len(pts), int(tid), dtype=np.int64))

    if not T:
        return None
    t = np.concatenate(T)
    tids = np.concatenate(TID)
    out = {k: caliber_metrics(np.concatenate(v), t, tids, cap_eps=h["cap_eps"])
           for k, v in preds.items()}
    out["_meta"] = {"rul_scale": rul_scale, "const_val": const_val,
                    "const_val_info": const_val_info,
                    "n_test_traj": int(len(te)), "seed": int(seed),
                    "n_points": int(len(t)),
                    "stride_over_scale": float(h["stride"]) / rul_scale,
                    "oracle_like": list(ORACLE_LIKE)}
    if return_raw:
        out["_raw"] = {"true": t, "tids": tids,
                       "pred": {k: np.concatenate(v) for k, v in preds.items()}}
    return out


def _row(name, m) -> str:
    c = m["calibers"]
    flag = "  <- ORACLE-LIKE / NOT DEPLOYABLE" if name in ORACLE_LIKE else ""
    return (f"  {name:<16} {c['full']['pooled']['rmse']:>10.4f} "
            f"{c['valid']['pooled']['rmse']:>11.4f} {c['info']['pooled']['rmse']:>10.4f} "
            f"{c['info']['macro']['rmse']:>11.4f} {c['info']['pooled']['mae']:>10.4f} "
            f"{c['info']['pooled']['phm']:>12.2f}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wheel.yaml")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--seeds", default=None, help="多种子, 逗号分隔 (报 mean±std)")
    ap.add_argument("--out", default="checkpoints/trivial_metrics.json")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else [args.seed if args.seed is not None else cfg["seed"]])
    _bh = baseline_hyper(cfg)
    h_L, h_stride = _bh["L"], _bh["stride"]

    all_runs = {}
    for s in seeds:
        print(f"\n===== 平凡基线 seed {s} =====")
        tv = evaluate_trivial(cfg, s)
        if tv is None:
            return
        ph = evaluate_physical(cfg, s)
        meta = tv.pop("_meta")
        print(f">> rul_scale={meta['rul_scale']:.1f}  const_mean={meta['const_val']:.4f}  "
              f"const_mean_info={meta['const_val_info']:.4f}  评估点数={meta['n_points']}")
        print(f"\n  {'方法':<14} {'full RMSE':>10} {'valid RMSE':>11} "
              f"{'info RMSE':>10} {'info macro':>11} {'info MAE':>10} {'info PHM':>12}")
        for k in METHODS:
            print(_row(k, tv[k]))
        if ph:
            print(_row("physical_extrap", ph))
        print(f"\n  理论锚点: info 区 RUL 若均匀分布, 常数预测器 RMSE = "
              f"1/sqrt(12) = {UNIFORM_CONST_RMSE:.4f}")
        print(f"  persistence 下界解释: stride/rul_scale = {meta['stride_over_scale']:.4f} "
              f"—— **ORACLE-LIKE / NOT DEPLOYABLE**, 仅量化 RUL 标签的时间平滑度")
        print(f"  主口径 = {PRIMARY_CALIBER}/{PRIMARY_STAT}; "
              f"S2.5 闸门基线 = {GATE_BASELINE} (可部署), 排除 {list(ORACLE_LIKE)}")
        print("\n  -- const_mean_info 三档明细 --")
        print(format_calibers(tv["const_mean_info"]))
        # ---- S2.5 §14 口径统一核查 (打印可核对的事实, 不做隐式断言) ----
        _ci = tv[GATE_BASELINE]["calibers"]["info"]
        print("\n  -- S2.5 §14 口径统一核查 --")
        print(f"     const_mean_info 常数值 {meta['const_val_info']:.4f} "
              f"= mean(train 划分 info 区真实 RUL); train/val/test 划分由 "
              f"split_trajectories_from_cfg(seed={s}) 给出, 与 target_only 同一函数同一 seed")
        print(f"     test 轨迹 {meta['n_test_traj']} 条 / 评估点 {meta['n_points']} 个, "
              f"取点 = eval_point_indices(L={h_L}, stride={h_stride}) 与模型评估 loader 同点")
        print(f"     info 口径: pooled n={_ci['n']} (占比 {_ci['share']:.3f}) / "
              f"macro n_traj={_ci['macro']['n_traj']} (轨迹等权, 无样本数加权)")
        print(f"     可部署基线 = {[m for m in METHODS if m not in ORACLE_LIKE]}; "
              f"不可部署 = {list(ORACLE_LIKE)} (不进 S2.5 gate)")
        all_runs[str(s)] = {k: {"info_pooled_rmse": v["rmse"],
                                "info_macro_rmse": v["calibers"]["info"]["macro"]["rmse"],
                                "info_mae": v["mae"], "info_phm": v["phm"]}
                            for k, v in list(tv.items()) +
                            ([("physical_extrap", ph)] if ph else [])}

    if len(seeds) > 1:
        print(f"\n===== 多种子汇总 (info/pooled RMSE, mean±std, {len(seeds)} seeds) =====")
        names = list(all_runs[str(seeds[0])].keys())
        for n in names:
            v = np.array([all_runs[str(s)][n]["info_pooled_rmse"] for s in seeds])
            print(f"  {n:<16} {v.mean():.4f} ± {v.std(ddof=1):.4f}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"seeds": seeds, "runs": all_runs,
                               "uniform_const_rmse": UNIFORM_CONST_RMSE},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n>> 平凡基线结果 -> {out}")


if __name__ == "__main__":
    main()
