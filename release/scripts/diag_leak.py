"""scripts/diag_leak.py — 诊断 1: x_T 特征列的贡献 (S0 初版 / S1.5 修噪后重跑)

背景演进:
  S0 初版假设: 第 6 维 b_hat 由 identify_b_hat 直接吃仿真真值 params['Kt']/['Tc'] 反算,
    而 HI_B 又是 b_hat 的固定尺度归一 → b_hat 与监督目标近乎同源 (泄漏);
    第 5 维 Im_over_omega 与 b_hat 信息近似等价 (同属 b≈(Kt·Im-Tcmd-Tc·sgn)/ω)。
    S0 实测: 去 b_hat 反而改善 13.1%, 去两列改善 18.6% → 泄漏假设不成立。
  S0 归因错误 (S1 更正): S0 曾把"去列反而改善"归因于量级失配, 该归因错误 ——
    本脚本是在**逐列 z-score 之后**置零的, 原始量级早已被消除。真实原因是
    wheel_sim.py 的量测噪声未传 size、返回标量, 导致每条轨迹的 omega 与 I_m
    各带一个常值直流偏置, 使 b_hat 携带逐轨迹随机常偏 (nuisance 变量)。
  S1 已修: 噪声逐点采样 + 偏置单独建模; build_hi 改健康段自校准 (不读真值);
    x_T 第 5 维 Im_over_omega → sigma_Im (I_m 滑窗标准差)。
  S1.5 (上一轮): 修噪后重跑, 回答两问 —— b_hat 是否仍为干扰项 / sigma_Im 有益还是有害。
  S2' (本次): **不再用于列选择** (列选择决策已关闭: 保留全部 8 维)。
    本次唯一目的 = 验证早停修复 —— 贴出四配置的 val(info) 与 test(info),
    检查 val 与 test 的排序是否已恢复一致。
    S1.5 病症: full 口径下 65%~95.6% 样本 RUL 恒为 0, val 与 test 难度不可比,
      方法排序反相关 (B: val 0.0928/test 0.1373 最好, C: val 0.0853/test 0.1525)。
    S2' 三项修复: HI 改挂总摩擦力矩 / 按 EOL 分层划分 / 评估口径三档 (主口径 info)。

四种配置 (数据加载后、z-score 与训练前置零):
  A 原始 (8 维全用)
  B 去 b_hat                    → 隔离 b_hat 的净贡献
  C 去 b_hat + 去 sigma_Im      → 与 S0 配置 C 位置对应 (第 5+6 维同时去)
  D 只去 sigma_Im               → 隔离 sigma_Im 的净贡献 (S1.5 新增)

其余超参与 S0 完全一致 (seed=42, epochs=20, L=64, K=8, stride=50)。

用法:
  python scripts/diag_leak.py --config configs/wheel.yaml --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sim.build_hi import XT_COLS                                  # noqa: E402
from scripts.diag_common import (                                     # noqa: E402
    load_cfg, hyper, xt_col_index, prepare_target, prepare_source,
    train_target_only)
from src.experiments.run_groups import (                              # noqa: E402
    eval_test, format_calibers, PRIMARY_CALIBER, PRIMARY_STAT)

# 受检列名集中定义 (唯一字面量出处; 索引一律经 xt_col_index 按名查找, 不硬编码数字)
COL_BHAT = "b_hat"
COL_SIGMA_IM = "sigma_Im"

# S0 旧数字 (2026-08-06 实测, 修噪前 + 第 5 维为 Im_over_omega), 作对照列
S0_BASELINE = {
    "A": 0.1388,
    "B": 0.1206,
    "C": 0.1130,
    "D": None,      # S0 无此配置
}

# S1.5 实测 (修噪后, 随机划分 + full 口径早停) —— 排序反相关的病症原始数字
S15_BASELINE = {"A": (0.1029, 0.1451), "B": (0.0928, 0.1373),
                "C": (0.0853, 0.1525), "D": (0.0975, 0.1442)}   # (val_full, test_full)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wheel.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None,
                    help="覆盖 epochs (默认用 config transfer.epochs_s2 正式值)")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    h = hyper(cfg)
    if not h["target_h5"].exists():
        print(f"!! 缺 {h['target_h5']}; 先 python -m src.sim.build_hi --report")
        sys.exit(1)
    if not h["source_h5"].exists():
        print(f"!! 缺 {h['source_h5']}; 先 python -m src.data.preprocess.wheel_features --report")
        sys.exit(1)

    # 按名查找, 列不存在时 xt_col_index 抛 KeyError (禁止硬编码索引)
    i_bhat = xt_col_index(COL_BHAT)
    i_sig = xt_col_index(COL_SIGMA_IM)

    _tcfg = cfg.get("training", {})
    pw = float(_tcfg.get("post_eol_weight", 1.0))
    cw = float(_tcfg.get("capped_weight", 1.0))

    print("===== 诊断 1: S2' 早停修复验证 (val/test 排序一致性) =====")
    print(f"config       = {args.config}")
    print(f"x_T 列       = {XT_COLS}")
    print(f"{COL_BHAT} 列索引       = {i_bhat} (第 {i_bhat + 1} 维)")
    print(f"{COL_SIGMA_IM} 列索引    = {i_sig} (第 {i_sig + 1} 维)")
    print(f"seed         = {args.seed}")
    print(f"device       = {h['device']}  encoder = {h['encoder']}")
    print(f"HI 口径      = sim.failure.hi_source={cfg['sim']['failure']['hi_source']}")
    print(f"划分         = stratify_by_eol={cfg['transfer']['split'].get('stratify_by_eol')} "
          f"n_eol_strata={cfg['transfer']['split'].get('n_eol_strata')}")
    print(f"主口径       = {PRIMARY_CALIBER}/{PRIMARY_STAT}   "
          f"post_eol_weight={pw} capped_weight={cw}")
    print(f"(S2' 任务 5: 分层划分与损失权重已直接落在 diag_common.py, 不再模块注入)")

    # 源域只为拿 n_features (target_only 组不用源 loader, 但 _build_model 需要该维度)
    featsS, _ = prepare_source(cfg)
    n_features = featsS.shape[1]
    print(f"源域 n_features = {n_features} (供 _build_model; target_only 不加载源 ckpt)")

    configs = [
        ("A", "A 原始 (8 维全用)", None),
        ("B", f"B 去 {COL_BHAT}", [i_bhat]),
        ("C", f"C 去 {COL_BHAT} + 去 {COL_SIGMA_IM}", [i_bhat, i_sig]),
        ("D", f"D 只去 {COL_SIGMA_IM}", [i_sig]),
    ]

    results = []
    for tag, name, zero_cols in configs:
        print(f"\n---------- 配置: {name} ----------")
        data = prepare_target(cfg, args.seed, zero_cols=zero_cols, epochs=args.epochs)
        hh = data["h"]
        print(f"  置零列        = {data['zeroed'] or '无'}")
        print(f"  轨迹划分      = train {len(data['tr'])} / val {len(data['va'])} "
              f"/ test {len(data['te'])} (共 {data['n_traj']} 条已观测; "
              f"EOL 分位数见 prepare_target 输出)")
        print(f"  样本块数      = train {data['sizes'][0]} / val {data['sizes'][1]} "
              f"/ test {data['sizes'][2]}")
        print(f"  epochs        = {hh['epochs']}  L={hh['L']} K={hh['K']} "
              f"stride={hh['target_stride']} bs={hh['batch_size']} lr={hh['finetune_lr']}")
        print(f"  rul_scale     = {data['rul_scale']:.1f}")
        model = train_target_only(cfg, data, n_features, args.seed, f"target_only[{tag}]")
        mv = eval_test(model, data["lva"], hh["device"])
        mt = eval_test(model, data["lte"], hh["device"])
        print(f"  -- VAL 三档口径 --")
        print(format_calibers(mv, indent="     "))
        print(f"  -- TEST 三档口径 --")
        print(format_calibers(mt, indent="     "))
        results.append((tag, name, data["zeroed"], mv, mt))

    # ---------- val / test 排序一致性 (本阶段关键判据) ----------
    def _g(m, cal, stat="pooled"):
        return m["calibers"][cal][stat]["rmse"]

    print(f"\n===== S2' 关键判据: val 与 test 的排序是否一致 (seed={args.seed}) =====")
    print(f"{'配置':<8} {'val(info)':>10} {'test(info)':>11} {'val(full)':>10} "
          f"{'test(full)':>11} {'S1.5 val(full)':>15} {'S1.5 test(full)':>16}")
    for tag, name, _, mv, mt in results:
        s15 = S15_BASELINE.get(tag) or (float('nan'), float('nan'))
        print(f"{tag:<8} {_g(mv, 'info'):>10.4f} {_g(mt, 'info'):>11.4f} "
              f"{_g(mv, 'full'):>10.4f} {_g(mt, 'full'):>11.4f} "
              f"{s15[0]:>15.4f} {s15[1]:>16.4f}")

    for cal in ("info", "full"):
        order_v = [t for t, *_ in sorted(results, key=lambda r: _g(r[3], cal))]
        order_t = [t for t, *_ in sorted(results, key=lambda r: _g(r[4], cal))]
        vv = [_g(m, cal) for _, _, _, m, _ in results]
        tt = [_g(m, cal) for _, _, _, _, m in results]
        rho = float(np.corrcoef(vv, tt)[0, 1]) if len(vv) > 1 else float("nan")
        same_best = order_v[0] == order_t[0]
        print(f"\n  [{cal} 口径] val 排序 = {' < '.join(order_v)}")
        print(f"  [{cal} 口径] test 排序 = {' < '.join(order_t)}")
        print(f"  [{cal} 口径] 最优配置一致: {same_best} (val 最优 {order_v[0]}, "
              f"test 最优 {order_t[0]});  val-test RMSE 相关系数 rho = {rho:+.3f}")
        print(f"  [{cal} 口径] 排序完全一致: {order_v == order_t}")
        ratio = [t / max(v, 1e-12) for v, t in zip(vv, tt)]
        print(f"  [{cal} 口径] test/val 比值 = "
              f"[{', '.join(f'{r:.2f}' for r in ratio)}] "
              f"(S1.5 full 口径为 1.5~1.8 倍)")

    print(f"\n  判据说明: 主口径 {PRIMARY_CALIBER} 下 rho>0 且 val 最优 == test 最优 "
          f"→ 早停信号恢复有效 (S1.5 该口径下 rho 为负)。")
    print("==========================")


if __name__ == "__main__":
    main()
