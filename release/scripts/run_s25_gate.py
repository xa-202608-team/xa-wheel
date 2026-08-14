"""scripts/run_s25_gate.py — S2.5 最终泛化闸门 (5 个全新 gate seed)

协议: docs/diagnostics_s25_protocol.md (§9–§12)。必须在 --phase select 冻结候选之后跑。

做什么:
  对 5 个 **gate** seed (= cfg.seed + s25.gate_seed_offsets, 与调参 seed 不相交),
  每个 seed 只训练**冻结后**的 target_only, 并在完全相同的划分/取点上评
  const_mean_info, 算 paired gain, 做 seed-level paired bootstrap, 按 §12 的
  8 条判据出唯一判词 S25_PASS / S25_STOP_GENERALIZATION。

主指标 = test info **轨迹宏平均** RMSE (s25.gate_metric)。不改全局 evaluation.primary_stat。
基线 = trivial.GATE_BASELINE = const_mean_info (可部署); persistence 是 ORACLE-LIKE,
被显式排除, 绝不进 gate。

用法:
  python scripts/run_s25_gate.py --config configs/wheel.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import load_config                                          # noqa: E402
from src.baselines.physical_extrap import caliber_mask                     # noqa: E402
from src.baselines.trivial import (                                        # noqa: E402
    evaluate_trivial, GATE_BASELINE, ORACLE_LIKE)
from src.experiments.run_groups import eval_test, training_hyper           # noqa: E402
from scripts.diag_generalization import (                                  # noqa: E402
    CANDIDATE_KEYS, CONFIG_DEFAULT, candidate_signature, collect_pred,
    config_hash, gate_seeds, prepare, protocol_hash, s25_cfg, shape_stats,
    train_candidate, tuning_seeds)

# gate 主指标名 → 从 caliber_metrics 结构取值 (只有 test 侧 gate 用, 不是早停指标)
GATE_METRICS = {
    "info_macro_rmse":  lambda c: c["info"]["macro"]["rmse"],
    "info_pooled_rmse": lambda c: c["info"]["pooled"]["rmse"],
}


def _report(c: dict) -> dict:
    """§10 要求同时报告的一组指标 (主指标之外)。"""
    return {
        "info_macro_rmse": float(c["info"]["macro"]["rmse"]),
        "info_pooled_rmse": float(c["info"]["pooled"]["rmse"]),
        "full_macro_rmse": float(c["full"]["macro"]["rmse"]),
        "info_macro_mae": float(c["info"]["macro"]["mae"]),
        "info_pooled_mae": float(c["info"]["pooled"]["mae"]),
        "info_n": int(c["info"]["n"]), "info_n_traj": int(c["info"]["macro"]["n_traj"]),
    }


def paired_bootstrap(gains: np.ndarray, n_boot: int, seed: int):
    """**seed-level** paired gain 的 bootstrap 均值分布 + 95% CI。

    重采样单位 = 每个 seed 的一个配对差值 (5 个数里有放回抽 5 个求均值)。
    绝不把时间点当独立样本 —— 那会把 n 从 5 放大到上万, 人为制造置信度。
    """
    g = np.asarray(gains, dtype=float)
    draws = np.random.default_rng(int(seed)).choice(
        g, size=(int(n_boot), len(g)), replace=True).mean(axis=1)
    lo, hi = (float(x) for x in np.quantile(draws, [0.025, 0.975]))
    return {"n_units": int(len(g)), "n_boot": int(n_boot), "bootstrap_seed": int(seed),
            "mean": float(g.mean()), "ci95": [lo, hi],
            "draws_mean": float(draws.mean())}


def run_gate(cfg: dict, cfg_path: Path) -> dict:
    sc = s25_cfg(cfg)
    sel = sc.get("selected_candidate")
    if not sel:
        raise RuntimeError("s25.selected_candidate 为 null: 候选尚未冻结; "
                           "先跑 scripts/diag_generalization.py --phase select")
    gm_name = str(sc["gate_metric"])
    if gm_name not in GATE_METRICS:
        raise KeyError(f"未知 gate_metric {gm_name}; 可选 {sorted(GATE_METRICS)}")
    gm = GATE_METRICS[gm_name]
    th = training_hyper(cfg)
    sig = candidate_signature(cfg)
    seeds, tseeds = gate_seeds(cfg), tuning_seeds(cfg)
    if set(seeds) & set(tseeds):
        raise RuntimeError(f"gate seeds {seeds} 与调参 seeds {tseeds} 相交")
    n_boot = int(cfg["experiments"]["bootstrap_samples"])

    print("=" * 92)
    print("S2.5 最终泛化闸门 (Generalization Gate)")
    print("=" * 92)
    print(f">> protocol sha256 = {protocol_hash()}")
    print(f">> config   sha256 = {config_hash(cfg_path)}")
    print(f">> 冻结候选 = {sel}  " +
          "  ".join(f"{k}={th[k]}" for k in CANDIDATE_KEYS) + f"  sig={sig}")
    print(f">> gate seeds = {seeds}  (调参 seeds {tseeds}, 不相交)")
    print(f">> 主指标 = test {gm_name} (info 轨迹宏平均);  基线 = {GATE_BASELINE} "
          f"(可部署);  排除 {list(ORACLE_LIKE)} (ORACLE-LIKE / NOT DEPLOYABLE)")
    print(f">> paired bootstrap: 单位=seed-level gain, reps={n_boot}, "
          f"seed={sc['bootstrap_seed']}")

    rows = []
    for s in seeds:
        print(f"\n{'-' * 92}\ngate seed {s}\n{'-' * 92}")
        data = prepare(cfg, s, with_test=True, verbose=True)
        model, hist, _ = train_candidate(cfg, data, s, f"{sel} gate seed{s}")
        # ---- target_only 在 test 上评 (best checkpoint 已由 val 指标恢复) ----
        tm = eval_test(model, data["lte"], data["device"], cap_eps=data["cap_eps"])
        pt, tt, it = collect_pred(model, data["lte"], data["device"])
        ss = shape_stats(pt, tt, it, data["cap_eps"])
        # ---- const_mean_info 在**完全相同的 test 划分与取点**上评 ----
        tv = evaluate_trivial(cfg, s, return_raw=True)
        raw = tv.pop("_raw")
        meta = tv.pop("_meta")
        bm = tv[GATE_BASELINE]
        bs_ = shape_stats(raw["pred"][GATE_BASELINE], raw["true"], raw["tids"],
                          data["cap_eps"])
        assert bm["calibers"]["info"]["macro"]["n_traj"] == \
               tm["calibers"]["info"]["macro"]["n_traj"], "基线与模型 test 轨迹数不一致"

        r_t, r_b = gm(tm["calibers"]), gm(bm["calibers"])
        best_ep = int(np.argmin([h["es_value"] for h in hist]))
        row = {
            "seed": int(s), "candidate": sel, "sig": sig,
            "es_metric": th["early_stop_metric"], "best_epoch": best_ep,
            "n_epochs_run": len(hist),
            "val_info_macro_at_best": float(hist[best_ep]["val_info_macro_rmse"]),
            "target_only": {**_report(tm["calibers"]), **ss},
            "baseline": {**_report(bm["calibers"]), **bs_,
                         "const_val_info": float(meta["const_val_info"])},
            "gain": float(r_b - r_t),
            "target_metric": float(r_t), "baseline_metric": float(r_b),
        }
        rows.append(row)
        print(f"  [seed {s}] test {gm_name}: target_only={r_t:.4f}  "
              f"{GATE_BASELINE}={r_b:.4f}  paired_gain={row['gain']:+.4f}")
        print(f"            info pooled {tm['calibers']['info']['pooled']['rmse']:.4f} vs "
              f"{bm['calibers']['info']['pooled']['rmse']:.4f} | full macro "
              f"{tm['calibers']['full']['macro']['rmse']:.4f} | "
              f"macro_corr={ss['macro_corr']:+.3f} pooled_corr={ss['pooled_corr']:+.3f} "
              f"PSR={ss['psr']:.3f} (pred_std {ss['pred_std']:.4f} / "
              f"true_std {ss['true_std']:.4f})")
        print(f"            best_epoch={best_ep}/{len(hist)} (由 val_{th['early_stop_metric']} 选择)")

    # ---------------- 统计 ----------------
    g = np.array([r["gain"] for r in rows])
    t_v = np.array([r["target_metric"] for r in rows])
    b_v = np.array([r["baseline_metric"] for r in rows])
    corr = np.array([r["target_only"]["macro_corr"] for r in rows])
    psr = np.array([r["target_only"]["psr"] for r in rows])
    boot = paired_bootstrap(g, n_boot, int(sc["bootstrap_seed"]))

    print("\n" + "=" * 92)
    print(f"表 2  5-seed gate (主指标 = test {gm_name})")
    print("=" * 92)
    print(f"  {'seed':<6}{'target_only':>13}{'const_mean_info':>17}{'gain':>10}"
          f"{'macro_corr':>12}{'PSR':>9}{'best_ep':>9}")
    for r in rows:
        print(f"  {r['seed']:<6}{r['target_metric']:>13.4f}{r['baseline_metric']:>17.4f}"
              f"{r['gain']:>+10.4f}{r['target_only']['macro_corr']:>+12.3f}"
              f"{r['target_only']['psr']:>9.3f}{r['best_epoch']:>9d}")

    print("\n" + "=" * 92)
    print("表 3  总体统计")
    print("=" * 92)
    print(f"  target_only     {t_v.mean():.4f} ± {t_v.std(ddof=1):.4f}")
    print(f"  {GATE_BASELINE:<15} {b_v.mean():.4f} ± {b_v.std(ddof=1):.4f}")
    print(f"  paired gain     {g.mean():+.4f} ± {g.std(ddof=1):.4f}   "
          f"median {float(np.median(g)):+.4f}   worst {g.min():+.4f}")
    print(f"  improve_count   {int((g > 0).sum())} / {len(g)}")
    print(f"  bootstrap 95% CI [{boot['ci95'][0]:+.4f}, {boot['ci95'][1]:+.4f}]  "
          f"(单位=seed-level paired gain, n_units={boot['n_units']}, "
          f"reps={boot['n_boot']}, seed={boot['bootstrap_seed']})")
    print(f"  macro_corr      {corr.mean():+.3f} ± {corr.std(ddof=1):.3f}  "
          f"(>0 的 seed 数 {int((corr > 0).sum())}/{len(corr)})")
    print(f"  PSR             {psr.mean():.3f} ± {psr.std(ddof=1):.3f}  "
          f"(>= {float(sc['psr_min'])} 的 seed 数 "
          f"{int((psr >= float(sc['psr_min'])).sum())}/{len(psr)})")

    # ---------------- §12 八条判据 ----------------
    es_ok = all(r["es_metric"] == "info_macro_rmse" for r in rows)
    checks = [
        (1, f"至少 {sc['min_improve_count']}/5 seed 优于 {GATE_BASELINE}",
         int((g > 0).sum()) >= int(sc["min_improve_count"]),
         f"{int((g > 0).sum())}/5"),
        (2, "mean paired gain > 0", bool(g.mean() > 0), f"{g.mean():+.4f}"),
        (3, "paired bootstrap 95% CI 下界 > 0", bool(boot["ci95"][0] > 0),
         f"{boot['ci95'][0]:+.4f}"),
        (4, f"至少 {sc['min_corr_count']}/5 seed macro_corr > 0",
         int((corr > 0).sum()) >= int(sc["min_corr_count"]),
         f"{int((corr > 0).sum())}/5"),
        (5, f"至少 {sc['min_psr_count']}/5 seed PSR >= {sc['psr_min']}",
         int((psr >= float(sc["psr_min"])).sum()) >= int(sc["min_psr_count"]),
         f"{int((psr >= float(sc['psr_min'])).sum())}/5"),
        (6, "所有 checkpoint 由 val_info_macro_rmse 选择", es_ok,
         f"es_metric={sorted({r['es_metric'] for r in rows})}"),
        (7, "test 从未参与候选配置选择", True,
         "tune phase 不构造 test loader; 早停无 test 入口 (测试钉死)"),
        (8, "5 个 gate seed 使用同一冻结配置",
         len({r["sig"] for r in rows}) == 1, f"sig={sorted({r['sig'] for r in rows})}"),
    ]
    print("\n" + "=" * 92)
    print("协议 §12 判据 (在看到 gate 结果之前冻结)")
    print("=" * 92)
    for i, desc, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {i}. {desc:<48} {detail}")
    verdict = "S25_PASS" if all(c[2] for c in checks) else "S25_STOP_GENERALIZATION"
    print("\n" + "=" * 92)
    print(f"  阶段判词: {verdict}")
    print("=" * 92)
    if verdict != "S25_PASS":
        print("  -> 不进入 S3。失败判据见上表; 下一步方向由用户裁定。")
    else:
        print("  -> 允许进入 S3, 但本会话不自动执行, 等用户下一条指令。")

    return {
        "protocol_sha256": protocol_hash(), "config_sha256": config_hash(cfg_path),
        "selected_candidate": sel, "signature": sig,
        "training": {k: th[k] for k in CANDIDATE_KEYS},
        "gate_metric": gm_name, "baseline": GATE_BASELINE,
        "oracle_like_excluded": list(ORACLE_LIKE),
        "gate_seeds": seeds, "tuning_seeds": tseeds, "rows": rows,
        "stats": {
            "target_mean": float(t_v.mean()), "target_std": float(t_v.std(ddof=1)),
            "baseline_mean": float(b_v.mean()), "baseline_std": float(b_v.std(ddof=1)),
            "gain_mean": float(g.mean()), "gain_std": float(g.std(ddof=1)),
            "gain_median": float(np.median(g)), "worst_seed_gain": float(g.min()),
            "improve_count": int((g > 0).sum()), "n_seeds": int(len(g)),
            "macro_corr_mean": float(corr.mean()),
            "macro_corr_positive": int((corr > 0).sum()),
            "psr_mean": float(psr.mean()),
            "psr_pass": int((psr >= float(sc["psr_min"])).sum()),
            "bootstrap": boot,
        },
        "checks": [{"id": i, "desc": d, "passed": bool(o), "detail": t}
                   for i, d, o, t in checks],
        "verdict": verdict,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CONFIG_DEFAULT)
    ap.add_argument("--out", default="checkpoints/s25_gate.json")
    args = ap.parse_args()
    cfg_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    out = run_gate(load_config(str(cfg_path)), cfg_path)
    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f">> 结果 -> {p}")


if __name__ == "__main__":
    main()
