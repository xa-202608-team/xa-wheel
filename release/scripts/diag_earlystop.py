"""scripts/diag_earlystop.py — 早停直接检验 + 损失权重消融 (S2' 任务 2 & 3)

背景: S2' 判据 4 试图用四配置 (A/B/C/D) 的 val/test 排序一致性**间接**检验早停信号,
但 train 仅 9 条轨迹时四配置本无可分辨差异 (实测 test(full) 极差 0.0004), 该判据作废。
本脚本改为**直接**检验: 固定配置 A, 逐 epoch 同时记 val(info) 与 test(info), 算

    regret = test_info(argmin_e val_info) - min_e test_info(e)

即"照 val 选 epoch"相对"上帝视角选最优 epoch"多付的 test 误差。
判据: regret 中位数 < info 区 test RMSE 的 10% (--regret-tol)。
同时输出两条曲线的 Spearman 相关 (正且大 = val 是 test 的可用代理)。

任务 2 消融: (post_eol_weight, capped_weight) ∈ {(1,1), (0.1,1), (0.1,0.1), (0,0)}
× 配置 A × N 个种子, 报 info 口径 test RMSE 的 mean±std, 并与 info 区常数预测器
的理论 RMSE 1/sqrt(12)=0.2887 对比。

用法:
  python scripts/diag_earlystop.py --config configs/wheel.yaml                 # 全跑
  python scripts/diag_earlystop.py --seeds 42,43,44 --mode earlystop           # 只跑任务 3
  python scripts/diag_earlystop.py --seeds 42,43,44 --mode ablation            # 只跑任务 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diag_common import (                                        # noqa: E402
    load_cfg, prepare_target, train_kwargs, CONFIG_DEFAULT)
from src.experiments.run_groups import (                                # noqa: E402
    _build_model, eval_test, _train_with_early_stop)
from src.baselines.physical_extrap import (                             # noqa: E402
    PRIMARY_CALIBER, PRIMARY_STAT, format_calibers)
from src.baselines.trivial import UNIFORM_CONST_RMSE                    # noqa: E402

# 任务 2 的四种权重组合 (post_eol_weight, capped_weight)
ABLATION_GRID = ((1.0, 1.0), (0.1, 1.0), (0.1, 0.1), (0.0, 0.0))
DEFAULT_SEEDS = (42, 43, 44)


def spearman(a, b) -> float:
    """Spearman 秩相关 (不引 scipy.stats; 平均秩处理并列)。"""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 3:
        return float("nan")

    def rank(v):
        order = np.argsort(v, kind="stable")
        r = np.empty(len(v), dtype=float)
        r[order] = np.arange(1, len(v) + 1, dtype=float)
        # 并列取平均秩
        for u in np.unique(v):
            m = v == u
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r

    ra, rb = rank(a), rank(b)
    sa, sb = ra.std(), rb.std()
    if sa < 1e-12 or sb < 1e-12:
        return float("nan")
    return float(np.mean((ra - ra.mean()) * (rb - rb.mean())) / (sa * sb))


def _train_traced(cfg, data, seed, tag, weights=None):
    """训一次 target_only, 逐 epoch 记 (val_info, test_info)。返回 (曲线, 末态指标)。

    weights=(post_eol_weight, capped_weight) 时覆盖 config 值 (消融用), None = 用 config。
    走 run_groups._train_with_early_stop 的 trace 钩子, 训练/早停逻辑与正式路径完全一致。
    """
    h = data["h"]
    kw = train_kwargs(data)
    if weights is not None:
        kw["post_eol_weight"], kw["capped_weight"] = float(weights[0]), float(weights[1])
    model = _build_model(cfg, data["n_target"], data["n_target"], h["device"])
    model.freeze_encoder(False)
    opt = torch.optim.Adam(model.parameters(), lr=h["finetune_lr"])
    huber, mse = nn.HuberLoss(delta=h["huber_delta"]), nn.MSELoss()

    curve = {"val": [], "test": []}

    def trace(ep, vm, m):
        tm = eval_test(m, data["lte"], h["device"], cap_eps=kw["cap_eps"])
        curve["val"].append(float(vm["rmse"]))
        curve["test"].append(float(tm["rmse"]))

    _train_with_early_stop(model, data["ltr"], data["lva"], opt, h["device"],
                           huber, mse, h["lam"], h["epochs"], tag,
                           trace=trace, **kw)
    # 早停已把 model 恢复到 best-val 权重 → 此处 test 指标即"按 val 选出的模型"的表现
    return curve, eval_test(model, data["lte"], h["device"], cap_eps=kw["cap_eps"])


def run_earlystop(cfg, seeds, regret_tol):
    """任务 3: 逐 epoch regret + Spearman。"""
    print("\n" + "=" * 78)
    print(f"任务 3  早停直接检验 (配置 A 全 8 维 x_T, 主口径 {PRIMARY_CALIBER}/{PRIMARY_STAT})")
    print("=" * 78)
    rows = []
    for s in seeds:
        print(f"\n----- seed {s} -----")
        data = prepare_target(cfg, s)
        curve, m_es = _train_traced(cfg, data, s, f"A seed{s}")
        v, t = np.asarray(curve["val"]), np.asarray(curve["test"])
        e_val = int(np.argmin(v))
        regret = float(t[e_val] - t.min())
        rho = spearman(v, t)
        rows.append({"seed": int(s), "e_val": e_val, "e_test_best": int(np.argmin(t)),
                     "val_best": float(v.min()), "test_at_e_val": float(t[e_val]),
                     "test_best": float(t.min()), "regret": regret, "rho": rho,
                     "test_es_model": float(m_es["rmse"]),
                     "curve_val": v.tolist(), "curve_test": t.tolist()})
        print(f"  epoch 曲线 ({len(v)} epoch):")
        print("    ep " + " ".join(f"{i:>6d}" for i in range(len(v))))
        print("    val" + " ".join(f"{x:>6.3f}" for x in v))
        print("    tst" + " ".join(f"{x:>6.3f}" for x in t))
        print(f"  argmin val = ep{e_val} (val={v.min():.4f}) → test={t[e_val]:.4f}")
        print(f"  argmin test= ep{int(np.argmin(t))} (test={t.min():.4f})")
        print(f"  regret = {regret:.4f}   Spearman(val,test) = {rho:+.3f}")
        print(f"  早停恢复权重后 test 三档:")
        print(format_calibers(m_es, indent="      "))

    reg = np.array([r["regret"] for r in rows])
    tst = np.array([r["test_at_e_val"] for r in rows])
    rho = np.array([r["rho"] for r in rows])
    med_reg, med_tst = float(np.median(reg)), float(np.median(tst))
    tol = float(regret_tol) * med_tst
    passed = med_reg < tol
    print(f"\n>> 汇总 ({len(seeds)} seeds)")
    print(f"   regret        中位数 {med_reg:.4f}  (各 seed {np.round(reg, 4).tolist()})")
    print(f"   test@argmin_val 中位数 {med_tst:.4f}  mean±std "
          f"{tst.mean():.4f} ± {tst.std(ddof=1):.4f}")
    print(f"   Spearman(val,test) 中位数 {float(np.median(rho)):+.3f}  "
          f"(各 seed {np.round(rho, 3).tolist()})")
    print(f"   判据 regret 中位数 < {regret_tol:.0%} × test RMSE = {tol:.4f}  →  "
          f"{'通过' if passed else '未通过'}")
    return {"rows": rows, "median_regret": med_reg, "median_test": med_tst,
            "tol": tol, "passed": bool(passed),
            "median_rho": float(np.median(rho))}


def run_ablation(cfg, seeds):
    """任务 2: (post_eol_weight, capped_weight) 四配置 × N seed 消融。"""
    print("\n" + "=" * 78)
    print("任务 2  损失权重消融 (post_eol_weight, capped_weight) × 配置 A")
    print("=" * 78)
    res = {}
    for pw, cw in ABLATION_GRID:
        key = f"pw={pw}/cw={cw}"
        vals, mets = [], []
        for s in seeds:
            data = prepare_target(cfg, s, verbose=False)
            _, m = _train_traced(cfg, data, s, f"{key} seed{s}", weights=(pw, cw))
            vals.append(float(m["rmse"]))
            mets.append(m)
            print(f"  {key:<18} seed {s}: info test RMSE = {m['rmse']:.4f}  "
                  f"(full {m['calibers']['full']['pooled']['rmse']:.4f}, "
                  f"info macro {m['calibers']['info']['macro']['rmse']:.4f})")
        v = np.array(vals)
        res[key] = {"pw": pw, "cw": cw, "vals": vals,
                    "mean": float(v.mean()),
                    "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                    "full_mean": float(np.mean(
                        [m["calibers"]["full"]["pooled"]["rmse"] for m in mets])),
                    "macro_mean": float(np.mean(
                        [m["calibers"]["info"]["macro"]["rmse"] for m in mets]))}

    print(f"\n>> 消融汇总 (info/pooled test RMSE, {len(seeds)} seeds mean±std)")
    print(f"   {'(pw, cw)':<14}{'info RMSE':>18}{'info macro':>12}{'full RMSE':>11}"
          f"{'vs 常数':>12}")
    for d in res.values():
        gain = (UNIFORM_CONST_RMSE - d["mean"]) / UNIFORM_CONST_RMSE
        label = f"({d['pw']}, {d['cw']})"
        print(f"   {label:<14}{d['mean']:>10.4f} ± {d['std']:<5.4f}"
              f"{d['macro_mean']:>12.4f}{d['full_mean']:>11.4f}{gain:>11.1%}")
    print(f"   注: 末列 = 相对 info 区常数预测器理论 RMSE {UNIFORM_CONST_RMSE:.4f} 的相对改善;"
          f" <=0 表示不如常数预测器")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CONFIG_DEFAULT)
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--mode", choices=["all", "earlystop", "ablation"], default="all")
    ap.add_argument("--regret-tol", type=float, default=0.10,
                    help="判据: regret 中位数 < 此比例 × test RMSE")
    ap.add_argument("--out", default=None,
                    help="结果 json (缺省 checkpoints/diag_earlystop_<mode>.json; "
                         "按 mode 区分以免单跑某一项时覆盖另一项的结果)")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    seeds = [int(s) for s in args.seeds.split(",")]
    print(f">> config={args.config}  seeds={seeds}  mode={args.mode}")

    out = {"seeds": seeds, "mode": args.mode}
    if args.mode in ("all", "earlystop"):
        out["earlystop"] = run_earlystop(cfg, seeds, args.regret_tol)
    if args.mode in ("all", "ablation"):
        out["ablation"] = run_ablation(cfg, seeds)

    p = ROOT / (args.out or f"checkpoints/diag_earlystop_{args.mode}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n>> 结果 -> {p}")


if __name__ == "__main__":
    main()
