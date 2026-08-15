#!/usr/bin/env python
"""scripts/basilisk_b2/run_gate.py

BASILISK-B2 §8 —— 开发 Gate。

5 个 seed [72,73,74,75,76], 每 seed 跑 target_only + const_mean_info,
主指标 = **test info trajectory-macro RMSE**。

B2_GENERALIZATION_PASS 的 7 个条件 (逐字来自任务书 §8):
  1. target_only 至少 4/5 个 seed 优于 const_mean_info
  2. mean paired gain > 0
  3. bootstrap 95% CI lower > 0
  4. macro corr > 0 至少 4/5
  5. PSR >= 0.30 至少 4/5
  6. warning / pre-EOL 指标有效
  7. checkpoint 只按 val 选

bootstrap 纪律: 配对单位 = **一个 seed 一个差值** (5 个数), 有放回重采样 2000 次。
绝不把时间点当独立样本 —— 78825 个 endpoint 来自 75 条轨迹, 把它们当独立样本会
把 CI 缩小两个数量级, 造出虚假显著性。

split 固定 (split_seed=20260810), 5 个 seed 只改模型初始化与 batch 顺序 ——
因此 target_only 与 const_mean_info 在**完全相同的划分与取点**上比较。

用法:
    python scripts/basilisk_b2/run_gate.py
    python scripts/basilisk_b2/run_gate.py --seeds 72 --fast   # 冒烟
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.basilisk_b11.calibrate_degradation import (  # noqa: E402
    load_b11_config as load_cfg,
)
from scripts.basilisk_b2.baselines_b2 import (  # noqa: E402
    EXCLUDED_ORACLE_METHODS,
    const_mean_info,
    const_pred_on,
    damage_extrapolation,
)
from scripts.basilisk_b2.data_b2 import prepare_b2  # noqa: E402
from scripts.basilisk_b2.eval_b2 import (  # noqa: E402
    b2_endpoint_axis,
    evaluate_arrays_b2,
    evaluate_b2,
    paired_warning,
    strip_private,
)
from scripts.basilisk_b2.train_b2 import train_target_only  # noqa: E402
from scripts.diag_generalization import summarize_history  # noqa: E402

CFG_PATH = "configs/wheel_basilisk_b2.yaml"

GATE_PURPOSE = (
    "B2 §8: 在 B1.9 冻结数据上判断 target_only 是否稳定优于最强可部署平凡基线 "
    "const_mean_info。这是**开发 Gate**, 不是最终结论: 它只回答"
    "'在完整目标域训练数据下, 模型能否稳定学会新的 RUL 任务'。"
    "不做 S3 truncation, 不跑 source transfer, 不重新调参。"
)


def paired_bootstrap(diffs, n_rep: int, seed: int) -> dict:
    """seed 级配对差值的 bootstrap 95% CI。

    单位 = 一个 seed 一个差值。n=5 很小, CI 必然宽 —— 这是诚实的代价,
    把时间点当独立样本能把 CI 做窄, 但那是错的。
    """
    d = np.asarray([x for x in diffs if np.isfinite(x)], float)
    if d.size == 0:
        return {"n": 0, "mean": float("nan"), "ci_lower": float("nan"),
                "ci_upper": float("nan"), "n_rep": int(n_rep),
                "unit": "one_paired_difference_per_seed"}
    rng = np.random.default_rng(int(seed))       # 局部 rng, 不用全局 seed
    means = np.array([rng.choice(d, size=d.size, replace=True).mean()
                      for _ in range(int(n_rep))])
    return {"n": int(d.size), "mean": float(d.mean()),
            "ci_lower": float(np.percentile(means, 2.5)),
            "ci_upper": float(np.percentile(means, 97.5)),
            "n_rep": int(n_rep),
            "unit": "one_paired_difference_per_seed",
            "note": "绝不把时间点当独立样本"}


def warning_valid(w: dict) -> tuple[bool, dict]:
    """§8 条件 6: warning / pre-EOL 指标是否有效。

    "有效"= 分母非空且 whole-risk 三项都算得出来 (不是 NaN 占位), 并且
    coverage_before_eol 与 miss_rate_before_eol 互补。**不要求 coverage 高** ——
    条件 6 问的是指标可用, 不是模型报警报得好; 把它写成"coverage 要高"
    就变成了给闸门放水。
    """
    need = ("warning_coverage", "miss_rate", "coverage_before_eol",
            "miss_rate_before_eol", "false_alarm_rate")
    have = {k: w.get(k) for k in need}
    ok_finite = all(np.isfinite(float(v)) if v is not None else False
                    for v in have.values())
    n_obs, n_cen = int(w.get("n_observed", 0)), int(w.get("n_censored", 0))
    comp = (np.isfinite(float(w.get("coverage_before_eol", np.nan)))
            and abs(float(w["coverage_before_eol"])
                    + float(w["miss_rate_before_eol"]) - 1.0) < 1e-9)
    ok = bool(ok_finite and n_obs > 0 and n_cen > 0 and comp)
    return ok, {**have, "n_observed": n_obs, "n_censored": n_cen,
                "complementary": bool(comp), "valid": ok}


def run_seed(cfg: dict, split: dict, axis: dict, seed: int,
             dmg_cache: dict | None = None) -> dict:
    """一个 seed: target_only 训练 + 三个方法在同一 test 取点上评估。

    dmg_cache: damage_extrapolation 不含任何随机性、也不依赖 seed (划分固定,
    无模型), 逐 seed 重算只是浪费 ~3 分钟。缓存的是**同一份**预测数组,
    因此各 seed 看到的 B 基线数字必然完全一致。
    """
    print(f"\n{'=' * 66}\n[B2 §8] seed {seed}\n{'=' * 66}")
    t0 = time.time()
    data = prepare_b2(cfg, split, seed, with_test=True, verbose=True)

    # --- C. target_only ---
    model, history, th, budget = train_target_only(
        cfg, data, seed, f"b2_target_only_s{seed}")
    m_te = evaluate_b2(model, data["lte"], cfg, axis, data,
                       tag=f"target_only_s{seed}")

    # --- A. const_mean_info (常数只由 train 算) ---
    cval = const_mean_info(cfg, axis, data)
    pa, ta, ia = const_pred_on(axis, data["te"], cval, cfg, data)
    m_const = evaluate_arrays_b2(pa, ta, ia, cfg, axis, data,
                                 tag=f"const_mean_info_s{seed}")

    # --- B. damage_extrapolation (非学习, 与 seed 无关, 但同 test 取点) ---
    if dmg_cache is not None and "pred" in dmg_cache:
        pb, tb, ib = dmg_cache["pred"], dmg_cache["true"], dmg_cache["tids"]
    else:
        pb, tb, ib = damage_extrapolation(cfg, axis, data, data["te"])
        if dmg_cache is not None:
            dmg_cache.update({"pred": pb, "true": tb, "tids": ib})
    m_dmg = evaluate_arrays_b2(pb, tb, ib, cfg, axis, data,
                               tag=f"damage_extrapolation_s{seed}")

    # 取点必须逐点一致, 否则指标不可比 —— 直接报错, 不静默对齐
    for nm, arr in (("const", ia), ("damage", ib)):
        if not np.array_equal(arr, m_te["_raw"]["tids"]):
            raise SystemExit(f"!! {nm} 与模型 test 取点不一致, 拒绝比较")

    gain = float(m_const["info_macro_rmse"]) - float(m_te["info_macro_rmse"])
    pw = paired_warning(m_te, m_const)
    wok, wdet = warning_valid(m_te["warning"])

    row = {
        "seed": int(seed),
        "const_mean_info_value": float(cval),
        "target_only": {k: v for k, v in strip_private(m_te).items()
                        if k != "calibers"},
        "const_mean_info": {k: v for k, v in strip_private(m_const).items()
                            if k != "calibers"},
        "damage_extrapolation": {k: v for k, v
                                 in strip_private(m_dmg).items()
                                 if k != "calibers"},
        "paired": {
            "metric": "test_info_macro_rmse",
            "target_only": float(m_te["info_macro_rmse"]),
            "const_mean_info": float(m_const["info_macro_rmse"]),
            "damage_extrapolation": float(m_dmg["info_macro_rmse"]),
            "gain_vs_const": gain,
            "better_than_const": bool(gain > 0),
        },
        "shape": m_te["shape"],
        "warning_validity": wdet,
        "warning_valid": wok,
        "paired_lead_vs_const": pw,
        "history": history,
        "history_summary": summarize_history(history, th["early_stop_metric"]),
        "hyper": th,
        "budget": budget,
        "checkpoint_selection": "val_only",
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(f"  [seed {seed}] target_only {m_te['info_macro_rmse']:.4f} vs "
          f"const {m_const['info_macro_rmse']:.4f} "
          f"(gain {gain:+.4f})  PSR {m_te['shape']['psr']:.3f}  "
          f"macro_corr {m_te['shape']['macro_corr']:.3f}  "
          f"[{row['elapsed_s']}s]")
    return row


def decide(rows: list, gc: dict) -> dict:
    """§8 七条件判定。任一不满足 -> B2_GENERALIZATION_FAIL。"""
    n = len(rows)
    diffs = [r["paired"]["gain_vs_const"] for r in rows]
    n_better = sum(bool(r["paired"]["better_than_const"]) for r in rows)
    corrs = [r["shape"]["macro_corr"] for r in rows]
    psrs = [r["shape"]["psr"] for r in rows]
    n_corr = sum(bool(np.isfinite(c) and c > 0) for c in corrs)
    psr_min = float(gc["psr_min"])
    n_psr = sum(bool(np.isfinite(p) and p >= psr_min) for p in psrs)
    n_warn = sum(bool(r["warning_valid"]) for r in rows)
    n_valonly = sum(r["checkpoint_selection"] == "val_only" for r in rows)
    bs = paired_bootstrap(diffs, int(gc["bootstrap_samples"]),
                          int(gc["bootstrap_seed"]))
    mean_gain = float(np.mean([d for d in diffs if np.isfinite(d)])) \
        if any(np.isfinite(d) for d in diffs) else float("nan")

    conds = [
        {"id": 1, "name": "target_only 优于 const_mean_info 的 seed 数",
         "need": f">= {gc['min_improve_count']}/{n}",
         "got": f"{n_better}/{n}",
         "pass": bool(n_better >= int(gc["min_improve_count"]))},
        {"id": 2, "name": "mean paired gain > 0",
         "need": "> 0", "got": f"{mean_gain:.6f}",
         "pass": bool(np.isfinite(mean_gain) and mean_gain > 0)},
        {"id": 3, "name": "bootstrap 95% CI lower > 0",
         "need": "> 0", "got": f"{bs['ci_lower']:.6f}",
         "pass": bool(np.isfinite(bs["ci_lower"]) and bs["ci_lower"] > 0)},
        {"id": 4, "name": "macro corr > 0 的 seed 数",
         "need": f">= {gc['min_corr_count']}/{n}", "got": f"{n_corr}/{n}",
         "pass": bool(n_corr >= int(gc["min_corr_count"]))},
        {"id": 5, "name": f"PSR >= {psr_min} 的 seed 数",
         "need": f">= {gc['min_psr_count']}/{n}", "got": f"{n_psr}/{n}",
         "pass": bool(n_psr >= int(gc["min_psr_count"]))},
        {"id": 6, "name": "warning / pre-EOL 指标有效",
         "need": f"{n}/{n}", "got": f"{n_warn}/{n}",
         "pass": bool(n_warn == n)},
        {"id": 7, "name": "checkpoint 只按 val 选",
         "need": f"{n}/{n}", "got": f"{n_valonly}/{n}",
         "pass": bool(n_valonly == n)},
    ]
    n_pass = sum(c["pass"] for c in conds)
    verdict = ("B2_GENERALIZATION_PASS" if n_pass == len(conds)
               else "B2_GENERALIZATION_FAIL")
    return {
        "conditions": conds, "n_conditions": len(conds), "n_passed": n_pass,
        "verdict": verdict,
        "primary_metric": "test_info_trajectory_macro_rmse",
        "per_seed_gain": diffs, "mean_paired_gain": mean_gain,
        "bootstrap": bs,
        "macro_corr_per_seed": corrs, "psr_per_seed": psrs,
        "excluded_oracle_methods": list(EXCLUDED_ORACLE_METHODS),
        "source_transfer_run": False,
        "truncation_applied": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--fast", action="store_true",
                    help="仅冒烟: max_epochs=1 (结果不得用于判定)")
    a = ap.parse_args()

    cfg = load_cfg(CFG_PATH)
    gc = cfg["b2"]["gate"]
    seeds = a.seeds if a.seeds else [int(s) for s in gc["seeds"]]
    if a.fast:
        cfg["training"]["max_epochs"] = 1
        cfg["b2_frozen_training_expected"]["max_epochs"] = 1

    split = json.loads((ROOT / cfg["paths"]["split_json"]).read_text(
        encoding="utf-8"))
    axis = b2_endpoint_axis(cfg)

    print(f"[B2 §8] Gate: seeds={seeds}  主指标={gc['metric']}  "
          f"基线={gc['baseline']}")
    print(f"        split_sha256={split['split_sha256']}")
    dmg_cache: dict = {}
    rows = [run_seed(cfg, split, axis, s, dmg_cache) for s in seeds]
    dec = decide(rows, gc)

    out = {
        "stage": "BASILISK_B2",
        "section": "§8 generalization gate",
        "gate_purpose": GATE_PURPOSE,
        "feature_h5": cfg["transfer"]["target_feature_path"],
        "hi_key": cfg["transfer"]["target_hi_key"],
        "split_sha256": split["split_sha256"],
        "split_seed": split["split_seed"],
        "seeds": seeds,
        "fast_smoke": bool(a.fast),
        "frozen_training": cfg["b2_frozen_training_expected"],
        "per_seed": rows,
        "decision": dec,
        "verdict": dec["verdict"],
    }
    p = ROOT / cfg["paths"]["metrics_json"]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False,
                            default=float) + "\n", encoding="utf-8")

    print(f"\n{'=' * 66}\n[B2 §8] 判定\n{'=' * 66}")
    for c in dec["conditions"]:
        print(f"  {'PASS' if c['pass'] else 'FAIL'}  条件{c['id']} "
              f"{c['name']}: 需 {c['need']}, 实得 {c['got']}")
    bs = dec["bootstrap"]
    print(f"  mean paired gain = {dec['mean_paired_gain']:+.6f}  "
          f"95% CI [{bs['ci_lower']:+.6f}, {bs['ci_upper']:+.6f}]  "
          f"(n={bs['n']} seed 级配对, {bs['n_rep']} 次重采样)")
    print(f"  -> {dec['verdict']}  ({dec['n_passed']}/{dec['n_conditions']})")
    print(f"  metrics -> {p.relative_to(ROOT)}")
    if a.fast:
        print("  !! --fast 冒烟结果, 不得用于正式判定")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
