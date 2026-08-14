#!/usr/bin/env python
"""scripts/basilisk_b21/summarize_b21.py

BASILISK-B2.1 §11..§14 —— 汇总与出口。

只做三件事:
  1. 把 §11 的指标集从 gate_metrics.json 抽成便于阅读的表 (不重算, 不改口径);
  2. 复述 §12 的七条件判定 (从 conditions 重算 n_passed, 不信任已写入的 verdict);
  3. 按 §13/§14 写出口: FAIL -> 停止不跑迁移; PASS -> 停止并写下 B5 计划。

**不自动运行 B5。** 本脚本不 import 任何迁移训练入口, 也不写任何
checkpoints/basilisk_b5 路径 —— 结构上跑不了 B5。

用法:
    python scripts/basilisk_b21/summarize_b21.py
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
from scripts.basilisk_b3x.summarize_b3x import _stat  # noqa: E402

CFG_PATH = "configs/wheel_basilisk_b21.yaml"
NAN = float("nan")

ALLOWED_LABELS = ("B21_GENERALIZATION_PASS", "B21_GENERALIZATION_FAIL")

SUMMARY_PURPOSE = (
    "B2.1 §11..§14: 汇总新划分下的五 seed Target-only Gate 结果, 复述七条件判定, "
    "并按出口规则写下下一步。本阶段只改划分协议, 不跑任何迁移方法。"
    "B2_GENERALIZATION_FAIL 保持终局; B3X_NO_STABILIZING_SIGNAL 不变; "
    "B4X_TRANSFER_STABILIZATION_SIGNAL 仍然只是 EXPLORATORY_ONLY。"
    "不存在任何正式的 positive-transfer 结论, 本阶段也不产生迁移结论。"
)

NEXT_STAGE_ON_PASS = "B5 formal transfer on the frozen B2.1 split"

EXIT_DISCIPLINE = (
    "B21_GENERALIZATION_FAIL -> 停止, 不跑迁移。"
    "B21_GENERALIZATION_PASS -> 停止, 只写下下一阶段计划 " + NEXT_STAGE_ON_PASS +
    "。两种情形都不得自动运行 B5: 本脚本不 import 任何迁移训练入口, "
    "也不创建任何 checkpoints/basilisk_b5 产物, 且 baseline 契约把 B5 的产物"
    "钉为必须缺席。"
)

PASS_MEANING = (
    "PASS 不等于 B2 判错了, 也不追溯修改 B2 的判定。它只说明: 在覆盖完整的划分上"
    "该结论不再成立, 于是'划分覆盖不足'作为 B2 失败的一个候选解释被确认为"
    "实质因素。B2_GENERALIZATION_FAIL 在其自身划分上依然成立。"
)

FAIL_MEANING = (
    "FAIL 意味着修好 lifetime-support mismatch 之后问题依旧, 失败原因在别处 "
    "(已知的无界 softplus RUL 头 + 单边删失 hinge 在外推区失控)。"
    "此时按 §13 停止, 不跑迁移。"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CFG_PATH)
    a = ap.parse_args()

    cfg = load_cfg(a.config)
    b21 = cfg["b21"]
    gm = json.loads((ROOT / cfg["paths"]["metrics_json"]).read_text(
        encoding="utf-8"))
    split = json.loads((ROOT / cfg["paths"]["split_json"]).read_text(
        encoding="utf-8"))
    if bool(gm.get("fast_smoke")):
        raise SystemExit("!! gate_metrics 是 --fast 冒烟结果, 不得用于判定")

    rows = gm["per_seed"]
    dec = gm["decision"]
    bins = list(b21["split"]["event"]["bins"])
    short = bins[0]

    # §12: 从 conditions 重算, 不信任已写入的 verdict (防止手改 JSON 改结论)
    n_pass = sum(bool(c["pass"]) for c in dec["conditions"])
    verdict = (str(b21["gate"]["pass_label"])
               if n_pass == len(dec["conditions"])
               else str(b21["gate"]["fail_label"]))
    if verdict != str(gm["verdict"]):
        raise SystemExit(f"!! verdict 不自洽: 重算 {verdict} != "
                         f"写入 {gm['verdict']} —— 拒绝汇总")
    if verdict not in ALLOWED_LABELS:
        raise SystemExit(f"!! 非法标签 {verdict}")

    # §11 指标表
    per_seed = []
    for r in rows:
        lt = r["lifetime_subsets"]["target_only"]
        per_seed.append({
            "seed": r["seed"],
            "target_only_rmse": r["paired"]["target_only"],
            "const_mean_info_rmse": r["paired"]["const_mean_info"],
            "damage_extrapolation_rmse": r["paired"]["damage_extrapolation"],
            "gain_vs_const": r["paired"]["gain_vs_const"],
            "better_than_const": r["paired"]["better_than_const"],
            "macro_corr": r["shape"]["macro_corr"],
            "psr": r["shape"]["psr"],
            "warning_coverage": r["target_only"]["warning"][
                "warning_coverage"],
            "warning_miss": r["target_only"]["warning"]["miss_rate"],
            "warning_coverage_before_eol": r["target_only"]["warning"][
                "coverage_before_eol"],
            "warning_miss_before_eol": r["target_only"]["warning"][
                "miss_rate_before_eol"],
            "warning_valid": r["warning_valid"],
            **{f"{n}_life_rmse": lt[n]["macro_rmse"] for n in bins},
            **{f"{n}_life_n_traj": lt[n]["n_traj_evaluable"] for n in bins},
            "catastrophic_threshold": r["catastrophic_threshold"]["threshold"],
            "n_catastrophic_overall": r["catastrophic_overall"][
                "n_catastrophic"],
            "n_catastrophic_short_life": r["catastrophic_short_life"][
                "n_catastrophic"],
            "no_catastrophic_short_life": r["no_catastrophic_short_life"],
            "censored_lb_violation_rate": r["censored_lower_bound"][
                "violation_rate"],
            "censored_lb_n_evaluable": r["censored_lower_bound"][
                "n_evaluable"],
            "checkpoint_selection": r["checkpoint_selection"],
        })

    def _col(key):
        return [x[key] for x in per_seed]

    # 三方法在三个 lifetime 子集上的对照 (同口径, 便于看 short-life 是谁在崩)
    by_method_lifetime = {}
    for meth in ("target_only", "const_mean_info", "damage_extrapolation"):
        by_method_lifetime[meth] = {
            n: _stat([r["lifetime_subsets"][meth][n]["macro_rmse"]
                      for r in rows]) for n in bins}

    out = {
        "stage": "BASILISK_B21",
        "label": str(b21["label"]),
        "purpose": SUMMARY_PURPOSE,
        "frozen_facts": dict(b21["frozen_facts"]),
        "transfer_run": False,
        "forbid_auto_run_b5": True,
        "allowed_labels": list(ALLOWED_LABELS),
        "split": {
            "manifest": cfg["paths"]["split_json"],
            "split_sha256": split["split_sha256"],
            "split_seed": split["split_seed"],
            "split_verdict": str(split["verdict"]),
            "n_traj": split["n_traj"],
            "n_event_observed": split["n_event_observed"],
            "n_censored": split["n_censored"],
            "event_bin_edges": split["lifetime_bins"]["event"]["edges"],
            "censored_bin_edges": split["lifetime_bins"]["censored"]["edges"],
            "censored_bins_degenerate":
                split["lifetime_bins"]["censored"]["degenerate"],
            "min_event_eol": split["coverage"]["min_event_eol"],
            "n_coverage_checks_passed": split["coverage"]["n_passed"],
            "n_coverage_checks": split["coverage"]["n_checks"],
        },
        "b2_split_comparison": {
            "note": ("B2 的 test min event EOL 低于 train 与 val 的最小值 —— "
                     "本阶段修的就是这个 lifetime-support mismatch。"
                     "B2 的划分与判定本身不变。"),
            "b2_min_event_eol": {"train": 22336, "val": 22426, "test": 21658},
            "b21_min_event_eol": split["coverage"]["min_event_eol"],
        },
        "seeds": list(gm["seeds"]),
        "methods": list(gm["methods"]),
        "primary_metric": "test_info_trajectory_macro_rmse",
        "frozen_training": dict(gm["frozen_training"]),
        "test_lifetime_bin_counts": dict(gm["test_lifetime_bin_counts"]),
        "per_seed": per_seed,
        "aggregate": {
            "target_only_rmse": _stat(_col("target_only_rmse")),
            "const_mean_info_rmse": _stat(_col("const_mean_info_rmse")),
            "damage_extrapolation_rmse":
                _stat(_col("damage_extrapolation_rmse")),
            "gain_vs_const": _stat(_col("gain_vs_const")),
            "macro_corr": _stat(_col("macro_corr")),
            "psr": _stat(_col("psr")),
            "warning_coverage": _stat(_col("warning_coverage")),
            "warning_miss": _stat(_col("warning_miss")),
            "censored_lb_violation_rate":
                _stat(_col("censored_lb_violation_rate")),
            **{f"{n}_life_rmse": _stat(_col(f"{n}_life_rmse")) for n in bins},
        },
        "lifetime_by_method": by_method_lifetime,
        "catastrophic": {
            "definition": dict(b21["catastrophic"]),
            "n_seeds_no_catastrophic_short": int(
                sum(bool(x["no_catastrophic_short_life"]) for x in per_seed)),
            "short_life_catastrophic_counts": {
                str(x["seed"]): x["n_catastrophic_short_life"]
                for x in per_seed},
            "overall_catastrophic_counts": {
                str(x["seed"]): x["n_catastrophic_overall"]
                for x in per_seed},
            "thresholds_per_seed": {str(x["seed"]): x["catastrophic_threshold"]
                                    for x in per_seed},
        },
        "decision": {
            "conditions": dec["conditions"],
            "n_conditions": len(dec["conditions"]),
            "n_passed": n_pass,
            "all_required": True,
            "mean_paired_gain": dec["mean_paired_gain"],
            "bootstrap": dec["bootstrap"],
            "thresholds_frozen_before_run": True,
            "threshold_source": dec["threshold_source"],
            "recomputed_from_conditions": True,
        },
        "verdict": verdict,
        "verdict_meaning": (PASS_MEANING
                            if verdict == "B21_GENERALIZATION_PASS"
                            else FAIL_MEANING),
        "exit_discipline": EXIT_DISCIPLINE,
        "next_step": (NEXT_STAGE_ON_PASS
                      if verdict == "B21_GENERALIZATION_PASS"
                      else "stop_do_not_run_transfer"),
        "b5_auto_run": False,
    }
    sp = ROOT / cfg["paths"]["summary_json"]
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(out, indent=2, ensure_ascii=False,
                             default=float) + "\n", encoding="utf-8")

    # protocol hash (B2.1 自己的产物, 不进 baseline 契约)
    ph = {"stage": "BASILISK_B21", "label": str(b21["label"]),
          "verdict": verdict,
          "b2_verdict_unchanged": "B2_GENERALIZATION_FAIL",
          "b3x_verdict_unchanged": "B3X_NO_STABILIZING_SIGNAL",
          "b4x_verdict_unchanged_exploratory":
              "B4X_TRANSFER_STABILIZATION_SIGNAL",
          "frozen_before_any_b21_number": True}
    for k, rel in (("b21_protocol", cfg["protocol"]["path"]),
                   ("b21_config", a.config),
                   ("b21_split_manifest", cfg["paths"]["split_json"]),
                   ("b21_gate_metrics", cfg["paths"]["metrics_json"]),
                   ("b21_summary", cfg["paths"]["summary_json"])):
        p = ROOT / rel
        ph[k] = (hashlib.sha256(p.read_bytes()).hexdigest()
                 if p.exists() else "MISSING")
    hp = ROOT / cfg["protocol"]["hash_path"]
    hp.parent.mkdir(parents=True, exist_ok=True)
    hp.write_text(json.dumps(ph, indent=2, ensure_ascii=False) + "\n",
                  encoding="utf-8")

    print(f"\n{'=' * 66}\n[B2.1 §12] 判定 (从 conditions 重算)\n{'=' * 66}")
    for c in dec["conditions"]:
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] 条件 {c['id']} "
              f"{c['name']}: 需 {c['need']}, 实得 {c['got']}")
    print(f"  {n_pass}/{len(dec['conditions'])} 通过  =>  {verdict}")
    print(f"  B2 判定不变: B2_GENERALIZATION_FAIL")
    print(f"  下一步: {out['next_step']}  (b5_auto_run=False)")
    print(f"{'=' * 66}")
    print(f">> 已写 {sp.relative_to(ROOT)} / {hp.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
