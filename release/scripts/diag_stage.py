"""scripts/diag_stage.py — S0 诊断 3: 增益是否被晚期样本淹没

问题假设:
  test 集里晚期样本 (HI_B 高、临近失效、RUL→0) 占比可能很大。这些样本 RUL 标签接近 0、
  几乎无预测难度, 任何模型都能压低误差; 若它们主导 test RMSE 的分母, 则迁移方法在
  真正有价值的早/中期 (HI_B 低、需要外推退化趋势) 上的增益会被整体 RMSE 平均掉,
  表现为"迁移增益≈0"。

做法:
  加载 (或现训) target_only 与 source_mmd_finetune 两个模型, 在 test 集上按 config
  transfer.hi_bins ([0,0.2) [0.2,0.5) [0.5,0.8) [0.8,1.0]) 分箱, 分别算 RMSE / MAE /
  样本数与占比, 打印两个模型的分箱对照表。

用法:
  python scripts/diag_stage.py --config configs/wheel.yaml --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diag_common import (                                    # noqa: E402
    load_cfg, hyper, prepare_target, prepare_source, train_target_only,
    train_source_mmd, collect_predictions, CKPT_DIR)
from src.experiments.run_groups import eval_test                     # noqa: E402


def bin_metrics(pred, true, hi, bins):
    """按 HI 分箱算 RMSE / MAE / 样本数。最后一箱右闭 (含 HI=1.0)。"""
    rows = []
    n_all = len(hi)
    for i, (lo, hi_edge) in enumerate(bins):
        last = (i == len(bins) - 1)
        m = (hi >= lo) & ((hi <= hi_edge) if last else (hi < hi_edge))
        n = int(m.sum())
        if n == 0:
            rows.append({"lo": lo, "hi": hi_edge, "n": 0, "ratio": 0.0,
                         "rmse": float("nan"), "mae": float("nan"),
                         "closed": last})
            continue
        d = pred[m] - true[m]
        rows.append({
            "lo": lo, "hi": hi_edge, "n": n, "ratio": n / max(n_all, 1),
            "rmse": float(np.sqrt(np.mean(d ** 2))),
            "mae": float(np.mean(np.abs(d))),
            "true_mean": float(np.mean(true[m])),
            "closed": last,
        })
    return rows


def fmt_bin(r):
    return f"[{r['lo']:.1f},{r['hi']:.1f}{']' if r['closed'] else ')'}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wheel.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    h = hyper(cfg)
    if not h["target_h5"].exists():
        print(f"!! 缺 {h['target_h5']}; 先 python -m src.sim.build_hi --report")
        sys.exit(1)
    if not h["source_h5"].exists():
        print(f"!! 缺 {h['source_h5']}; 先 python -m src.data.preprocess.wheel_features --report")
        sys.exit(1)

    print("===== S0 诊断 3: 增益是否被晚期样本淹没 =====")
    print(f"config     = {args.config}")
    print(f"seed       = {args.seed}")
    print(f"device     = {h['device']}  encoder = {h['encoder']}")
    print(f"HI 分箱    = {h['bins']} (取自 config transfer.hi_bins)")

    featsS, loader_S = prepare_source(cfg)
    n_features = featsS.shape[1]
    # 飞轮组件 ckpt 无后缀 (run_groups: suffix='' when component=='wheel')
    ckpt = CKPT_DIR / f"source_{h['encoder']}_pretrain.pt"
    print(f"源域 ckpt  = {ckpt} (exists={ckpt.exists()})")
    print(f"源域 n_features = {n_features}")

    # 两个模型各自独立准备数据 (同 seed → 同划分/同归一, 与 run_groups 逐组重建一致)
    print(f"\n---------- 训练 target_only (seed={args.seed}) ----------")
    data_t = prepare_target(cfg, args.seed, epochs=args.epochs)
    hh = data_t["h"]
    print(f"  轨迹划分 = train {len(data_t['tr'])} / val {len(data_t['va'])} "
          f"/ test {len(data_t['te'])}; 样本块 = {data_t['sizes']}")
    print(f"  epochs   = {hh['epochs']}  L={hh['L']} K={hh['K']} stride={hh['target_stride']}")
    model_t = train_target_only(cfg, data_t, n_features, args.seed, "target_only")
    m_t = eval_test(model_t, data_t["lte"], hh["device"])
    print(f"  >> target_only TEST RMSE={m_t['rmse']:.4f} MAE={m_t['mae']:.4f} PHM={m_t['phm']:.2f}")

    print(f"\n---------- 训练 source_mmd_finetune (seed={args.seed}) ----------")
    data_s = prepare_target(cfg, args.seed, epochs=args.epochs)
    model_s = train_source_mmd(cfg, data_s, n_features, loader_S,
                               args.seed, "source_mmd_finetune", ckpt)
    m_s = eval_test(model_s, data_s["lte"], hh["device"])
    print(f"  >> source_mmd_finetune TEST RMSE={m_s['rmse']:.4f} MAE={m_s['mae']:.4f} "
          f"PHM={m_s['phm']:.2f}")

    # ---- 分箱对照 ----
    Pt, Tt, Ht = collect_predictions(model_t, data_t["lte"], hh["device"])
    Ps, Ts, Hs = collect_predictions(model_s, data_s["lte"], hh["device"])
    assert np.allclose(Tt, Ts) and np.allclose(Ht, Hs), \
        "两模型 test 集标签/HI 不一致 — 划分或归一化不同步, 分箱不可比"

    rows_t = bin_metrics(Pt, Tt, Ht, h["bins"])
    rows_s = bin_metrics(Ps, Ts, Hs, h["bins"])

    print(f"\n===== test 集 HI_B 分箱对照 (共 {len(Ht)} 个窗末样本) =====")
    print(f"{'HI_B 箱':<12} {'样本数':>8} {'占比':>8} {'RUL真值均值':>12} "
          f"{'target RMSE':>12} {'MMD RMSE':>10} {'ΔRMSE':>9} "
          f"{'target MAE':>11} {'MMD MAE':>9} {'ΔMAE':>9}")
    for rt, rs in zip(rows_t, rows_s):
        if rt["n"] == 0:
            print(f"{fmt_bin(rt):<12} {0:>8} {'0.0%':>8} {'—':>12} "
                  f"{'—':>12} {'—':>10} {'—':>9} {'—':>11} {'—':>9} {'—':>9}")
            continue
        d_rmse = rt["rmse"] - rs["rmse"]        # >0 表示迁移更好
        d_mae = rt["mae"] - rs["mae"]
        print(f"{fmt_bin(rt):<12} {rt['n']:>8} {100 * rt['ratio']:>7.1f}% "
              f"{rt['true_mean']:>12.4f} {rt['rmse']:>12.4f} {rs['rmse']:>10.4f} "
              f"{d_rmse:>+9.4f} {rt['mae']:>11.4f} {rs['mae']:>9.4f} {d_mae:>+9.4f}")

    print(f"\n===== 诊断 3 结论数据 =====")
    print(f"整体 test RMSE: target_only={m_t['rmse']:.4f}  "
          f"source_mmd_finetune={m_s['rmse']:.4f}  "
          f"迁移增益={m_t['rmse'] - m_s['rmse']:+.4f}")
    late = [r for r in rows_t if r["lo"] >= 0.5]
    early = [r for r in rows_t if r["lo"] < 0.5]
    late_ratio = sum(r["ratio"] for r in late)
    early_ratio = sum(r["ratio"] for r in early)
    print(f"晚期样本 (HI_B>=0.5) 占比 = {100 * late_ratio:.1f}%")
    print(f"早中期样本 (HI_B<0.5) 占比 = {100 * early_ratio:.1f}%")
    for rt, rs in zip(rows_t, rows_s):
        if rt["n"] == 0:
            continue
        d = rt["rmse"] - rs["rmse"]
        print(f"  箱 {fmt_bin(rt):<10} 占比 {100 * rt['ratio']:>5.1f}%  "
              f"迁移增益 ΔRMSE={d:+.4f}  ({'迁移更好' if d > 0 else '迁移更差'})")
    print("==========================")


if __name__ == "__main__":
    main()
