#!/usr/bin/env python
"""scripts/basilisk_b3x/summarize_b3x.py

BASILISK-B3X §8/§9 —— 诊断问题 A–D 的回答与探索性分类。

`POST_B2_FAIL_EXPLORATORY` / `NOT_FORMAL_EVIDENCE`

只允许两个标签 (§9):
    B3X_STABILIZING_SIGNAL      —— 四条判据同时成立
    B3X_NO_STABILIZING_SIGNAL   —— 否则
正式标签 B3_MISSION_FEATURES_USEFUL **禁止使用**, 由 config 的
forbidden_labels 与 tests 双重把关。

四条判据的阈值全部在 protocol 冻结 (跑数之前写定), 本脚本只做比对,
不做任何"看到结果再定标准"的事:
  1. short-EOL 相对下降 >= short_eol_improve_min_rel 的 seed 数 >= 3/5
  2. overall mean gain > 0
  3. 崩塌 seed {72,74,76} 中 overall RMSE <= recovered_threshold_value 的 >= 2
  4. warning miss 不恶化 (容差 warning_miss_worse_tol)

bootstrap: 配对单位 = **一个 seed 一个差值** (5 个数)。绝不把 endpoint
当独立样本 —— 那会把 CI 缩小两个数量级, 造出虚假显著性。CI 是**描述性**的,
不作显著性断言。

用法:
    python scripts/basilisk_b3x/summarize_b3x.py --config configs/wheel_basilisk_b3x.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.basilisk_b11.calibrate_degradation import (  # noqa: E402
    load_b11_config as load_cfg,
)

CFG_PATH = "configs/wheel_basilisk_b3x.yaml"
NAN = float("nan")

ALLOWED_LABELS = ("B3X_STABILIZING_SIGNAL", "B3X_NO_STABILIZING_SIGNAL")

SUMMARY_PURPOSE = (
    "B3X §8/§9: 回答 A–D 四个诊断问题并给出探索性分类。"
    "本阶段不是正式 B3, 不产生迁移结论, 不宣称 mission features 有效。"
    "B2_GENERALIZATION_FAIL 保持终局, 不被本阶段覆盖或重新解释。"
    "结论只能是 B3X_STABILIZING_SIGNAL 或 B3X_NO_STABILIZING_SIGNAL。"
)

# §10: B4X 输入 schema 规则, 在 B3X 出数之前已写入 protocol。
# 这里只是把 protocol 的规则机器化执行, 不允许按哪个好看来选。
B4X_SCHEMA_RULE = {
    "B3X_STABILIZING_SIGNAL": "core_plus_mission",
    "B3X_NO_STABILIZING_SIGNAL": "core_only",
}


def paired_bootstrap(diffs, n_rep: int, seed: int) -> dict:
    """seed 级配对差值的描述性 bootstrap 区间。"""
    d = np.asarray([x for x in diffs if np.isfinite(x)], float)
    if d.size == 0:
        return {"n": 0, "mean": NAN, "ci_lower": NAN, "ci_upper": NAN,
                "n_rep": int(n_rep),
                "unit": "one_paired_difference_per_seed"}
    rng = np.random.default_rng(int(seed))     # 局部 rng, 不用全局 seed
    means = np.array([rng.choice(d, size=d.size, replace=True).mean()
                      for _ in range(int(n_rep))])
    return {"n": int(d.size), "mean": float(d.mean()),
            "ci_lower": float(np.percentile(means, 2.5)),
            "ci_upper": float(np.percentile(means, 97.5)),
            "n_rep": int(n_rep),
            "unit": "one_paired_difference_per_seed",
            "interpretation": "descriptive_only_not_significance_claim",
            "note": "绝不把 endpoint 当独立样本"}


def _stat(v) -> dict:
    x = np.asarray([q for q in v if np.isfinite(q)], float)
    if x.size == 0:
        return {"n": 0, "mean": NAN, "median": NAN, "std": NAN}
    return {"n": int(x.size), "mean": float(x.mean()),
            "median": float(np.median(x)),
            "std": float(x.std(ddof=1)) if x.size > 1 else NAN}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CFG_PATH)
    a = ap.parse_args()

    cfg = load_cfg(a.config)
    b3 = cfg["b3x"]
    vc = b3["verdict"]

    dp = ROOT / cfg["paths"]["short_eol_json"]
    if not dp.exists():
        raise SystemExit(f"!! 先跑 diagnose_short_eol.py: 缺 {dp}")
    diag = json.loads(dp.read_text(encoding="utf-8"))
    abl = json.loads(
        (ROOT / cfg["paths"]["metrics_json"]).read_text(encoding="utf-8"))

    seeds = [int(r["seed"]) for r in diag["per_seed"]]
    by = {int(r["seed"]): r for r in diag["per_seed"]}
    collapsed = [int(s) for s in b3["collapsed_seeds"]]
    normal_seeds = [int(s) for s in b3["normal_seeds"]]

    rel_min = float(vc["short_eol_improve_min_rel"])
    rec_thr = float(vc["recovered_threshold_value"])
    wm_tol = float(vc["warning_miss_worse_tol"])

    per_seed: list[dict] = []
    for s in seeds:
        r = by[s]
        A, B = r["core_only"], r["core_plus_mission"]
        g = r["gain"]
        wmA = A["overall"]["warning"]["miss_rate_before_eol"]
        wmB = B["overall"]["warning"]["miss_rate_before_eol"]
        wmA_s = A["short_eol"]["warning"]["miss_rate_before_eol"]
        wmB_s = B["short_eol"]["warning"]["miss_rate_before_eol"]
        per_seed.append({
            "seed": s,
            "is_collapsed_in_b2": s in collapsed,
            "overall": {
                "rmse_core": A["overall"]["rmse"],
                "rmse_mission": B["overall"]["rmse"],
                "gain": g["overall"]["gain"],
                "rel_gain": g["overall"]["rel_gain"],
                "corr_core": A["overall"]["corr"],
                "corr_mission": B["overall"]["corr"],
                "psr_core": A["overall"]["psr"],
                "psr_mission": B["overall"]["psr"],
                "pred_std_core": A["overall"]["pred_std"],
                "pred_std_mission": B["overall"]["pred_std"],
                "pred_mean_core": A["overall"]["pred_mean"],
                "pred_mean_mission": B["overall"]["pred_mean"],
                "true_std": A["overall"]["true_std"],
            },
            "short_eol": {
                "rmse_core": A["short_eol"]["rmse"],
                "rmse_mission": B["short_eol"]["rmse"],
                "gain": g["short_eol"]["gain"],
                "rel_gain": g["short_eol"]["rel_gain"],
                "mae_core": A["short_eol"]["mae"],
                "mae_mission": B["short_eol"]["mae"],
                "corr_core": A["short_eol"]["corr"],
                "corr_mission": B["short_eol"]["corr"],
                "psr_core": A["short_eol"]["psr"],
                "psr_mission": B["short_eol"]["psr"],
                "pred_std_core": A["short_eol"]["pred_std"],
                "pred_std_mission": B["short_eol"]["pred_std"],
                "true_std": A["short_eol"]["true_std"],
                "improved": bool(np.isfinite(g["short_eol"]["rel_gain"])
                                 and g["short_eol"]["rel_gain"] >= rel_min),
            },
            "normal_eol": {
                "rmse_core": A["normal_eol"]["rmse"],
                "rmse_mission": B["normal_eol"]["rmse"],
                "gain": g["normal_eol"]["gain"],
                "rel_gain": g["normal_eol"]["rel_gain"],
                "corr_core": A["normal_eol"]["corr"],
                "corr_mission": B["normal_eol"]["corr"],
            },
            "warning": {
                "overall_miss_core": wmA,
                "overall_miss_mission": wmB,
                "overall_delta": (float(wmB) - float(wmA)
                                  if np.isfinite(wmA) and np.isfinite(wmB)
                                  else NAN),
                "short_eol_miss_core": wmA_s,
                "short_eol_miss_mission": wmB_s,
                "false_alarm_core": A["overall"]["warning"]["false_alarm_rate"],
                "false_alarm_mission":
                    B["overall"]["warning"]["false_alarm_rate"],
            },
            # §9 判据 3: "恢复"用 protocol 冻结的绝对阈值
            "recovered": bool(np.isfinite(B["overall"]["rmse"])
                              and B["overall"]["rmse"] <= rec_thr),
        })

    g_over = [r["overall"]["gain"] for r in per_seed]
    g_short = [r["short_eol"]["gain"] for r in per_seed]
    g_norm = [r["normal_eol"]["gain"] for r in per_seed]
    ci = b3["descriptive_ci"]

    # ---------- §8 A–D ----------
    recovered = [r["seed"] for r in per_seed
                 if r["is_collapsed_in_b2"] and r["recovered"]]
    improved_short = [r["seed"] for r in per_seed if r["short_eol"]["improved"]]
    wm_deltas = [r["warning"]["overall_delta"] for r in per_seed]
    wm_worse = [r["seed"] for r in per_seed
                if np.isfinite(r["warning"]["overall_delta"])
                and r["warning"]["overall_delta"] > wm_tol]

    st_over, st_short, st_norm = _stat(g_over), _stat(g_short), _stat(g_norm)

    # C: 只降方差 vs 改变方向。判据材料 = PSR/pred_std 变化 与 corr 变化的对比。
    d_psr = [r["overall"]["psr_mission"] - r["overall"]["psr_core"]
             for r in per_seed
             if np.isfinite(r["overall"]["psr_mission"])
             and np.isfinite(r["overall"]["psr_core"])]
    d_corr = [r["overall"]["corr_mission"] - r["overall"]["corr_core"]
              for r in per_seed
              if np.isfinite(r["overall"]["corr_mission"])
              and np.isfinite(r["overall"]["corr_core"])]
    answers = {
        "A_collapsed_seeds_recovered": {
            "collapsed_seeds": collapsed,
            "recovered_seeds": recovered,
            "n_recovered": len(recovered),
            "threshold": rec_thr,
            "threshold_source": str(vc["recovered_threshold_from"]),
            "per_seed_overall_rmse_mission": {
                str(r["seed"]): r["overall"]["rmse_mission"]
                for r in per_seed if r["is_collapsed_in_b2"]},
        },
        "B_where_the_gain_comes_from": {
            "short_eol_gain": st_short,
            "normal_eol_gain": st_norm,
            "short_eol_improved_seeds": improved_short,
            "n_short_eol_improved": len(improved_short),
            "improve_threshold_rel": rel_min,
        },
        "C_variance_reduction_or_direction_change": {
            "delta_psr": _stat(d_psr),
            "delta_macro_corr": _stat(d_corr),
            "delta_pred_std": _stat(
                [r["overall"]["pred_std_mission"] - r["overall"]["pred_std_core"]
                 for r in per_seed]),
            "reading_rule": (
                "PSR/pred_std 明显下降而 macro corr 基本不动 -> 主要是降方差; "
                "macro corr 明显上升 -> 预测方向也发生了变化"),
        },
        "D_gain_mission": {
            "definition": "RMSE_core_only - RMSE_core_plus_mission",
            "per_seed": {str(r["seed"]): r["overall"]["gain"]
                         for r in per_seed},
            "overall": st_over,
            "improve_count": int(sum(1 for x in g_over
                                     if np.isfinite(x) and x > 0)),
            "n_seeds": len(per_seed),
            "descriptive_ci_overall": paired_bootstrap(
                g_over, int(ci["bootstrap_samples"]),
                int(ci["bootstrap_seed"])),
            "descriptive_ci_short_eol": paired_bootstrap(
                g_short, int(ci["bootstrap_samples"]),
                int(ci["bootstrap_seed"])),
        },
    }

    # ---------- §9 分类 ----------
    conds = [
        {"id": 1,
         "text": f"short-EOL 相对下降 >= {rel_min:.0%} 的 seed 数 >= "
                 f"{int(vc['min_short_eol_improved_seeds'])}/{len(per_seed)}",
         "value": len(improved_short),
         "threshold": int(vc["min_short_eol_improved_seeds"]),
         "passed": bool(len(improved_short)
                        >= int(vc["min_short_eol_improved_seeds"]))},
        {"id": 2, "text": "overall mean gain > 0",
         "value": st_over["mean"], "threshold": 0.0,
         "passed": bool(np.isfinite(st_over["mean"]) and st_over["mean"] > 0)},
        {"id": 3,
         "text": f"崩塌 seed 中 overall RMSE <= {rec_thr:.6f} 的数量 >= "
                 f"{int(vc['min_collapsed_recovered'])}",
         "value": len(recovered),
         "threshold": int(vc["min_collapsed_recovered"]),
         "passed": bool(len(recovered)
                        >= int(vc["min_collapsed_recovered"]))},
        {"id": 4,
         "text": f"warning miss (before EOL) 不恶化 (容差 {wm_tol})",
         "value": wm_worse, "threshold": wm_tol,
         "passed": bool(not wm_worse)},
    ]
    n_pass = sum(1 for c in conds if c["passed"])
    label = ("B3X_STABILIZING_SIGNAL" if n_pass == len(conds)
             else "B3X_NO_STABILIZING_SIGNAL")
    if label not in ALLOWED_LABELS:
        raise SystemExit(f"!! 非法标签 {label}")
    forbidden = [str(x) for x in vc.get("forbidden_labels", [])]
    if label in forbidden:
        raise SystemExit(f"!! 标签 {label} 在 forbidden_labels 中")

    out = {
        "stage": "BASILISK_B3X",
        "label_discipline": str(b3["label"]),
        "formal_stage": False,
        "not_formal_evidence": True,
        "b2_verdict_unchanged": "B2_GENERALIZATION_FAIL",
        "b2_verdict_status": "B2_GENERALIZATION_FAIL_REMAINS_FINAL",
        "purpose": SUMMARY_PURPOSE,
        "forbidden_formal_label": forbidden,
        "seeds": seeds,
        "collapsed_seeds": collapsed,
        "normal_seeds": normal_seeds,
        "short_eol_subset": list(diag["short_eol_subset"]),
        "n_normal_eol": len(diag["normal_eol_subset"]),
        "arms": list(diag["arms"]),
        "input_cols": dict(abl["input_cols"]),
        "comparability_note": str(abl["comparability_note"]),
        "per_seed": per_seed,
        "answers": answers,
        "warning_miss_delta_per_seed": {
            str(r["seed"]): r["warning"]["overall_delta"] for r in per_seed},
        "warning_miss_delta_stat": _stat(wm_deltas),
        "decision": {
            "conditions": conds,
            "n_conditions": len(conds),
            "n_passed": n_pass,
            "all_required": True,
            "thresholds_frozen_before_run": True,
            "threshold_source": "docs/basilisk_b3x/protocol.md §8 + "
                                "configs/wheel_basilisk_b3x.yaml::b3x.verdict",
        },
        "verdict": label,
        # §10: 规则在跑 B4X 前已写入 protocol, 此处只是执行
        "b4x_input_schema": B4X_SCHEMA_RULE[label],
        "b4x_input_schema_rule": dict(B4X_SCHEMA_RULE),
        "b4x_allowed_regardless": True,
    }
    sp = ROOT / cfg["paths"]["summary_json"]
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(out, indent=2, ensure_ascii=False,
                             default=float) + "\n", encoding="utf-8")

    # protocol hash 记录 (B3X 自己的产物, 不进 baseline 契约)
    ph = {"stage": "BASILISK_B3X", "label": str(b3["label"]),
          "verdict": label, "b2_verdict_unchanged": "B2_GENERALIZATION_FAIL"}
    for k, rel in (("b3x_protocol", cfg["protocol"]["path"]),
                   ("b3x_config", a.config),
                   ("b3x_metrics", cfg["paths"]["metrics_json"]),
                   ("b3x_short_eol", cfg["paths"]["short_eol_json"]),
                   ("b3x_summary", cfg["paths"]["summary_json"])):
        p = ROOT / rel
        ph[k] = (hashlib.sha256(p.read_bytes()).hexdigest()
                 if p.exists() else "MISSING")
    hp = ROOT / cfg["protocol"]["hash_path"]
    hp.parent.mkdir(parents=True, exist_ok=True)
    hp.write_text(json.dumps(ph, indent=2, ensure_ascii=False) + "\n",
                  encoding="utf-8")

    print(f"\n{'=' * 66}")
    for c in conds:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] 判据 {c['id']}: "
              f"{c['text']}  -> {c['value']}")
    print(f"  {n_pass}/{len(conds)} 通过  =>  {label}")
    print(f"  B2 判定不变: B2_GENERALIZATION_FAIL")
    print(f"  §10 B4X 输入 schema = {out['b4x_input_schema']}")
    print(f"{'=' * 66}")
    print(f">> 已写 {sp.relative_to(ROOT)} / {hp.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
