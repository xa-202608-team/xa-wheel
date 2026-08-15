#!/usr/bin/env python
"""scripts/basilisk_b6/final_transfer_verdict.py

BASILISK-B6 §15/§16/§18/§19/§20/§21/§24-8 —— PRIMARY 十条门槛判定 + 飞轮线最终迁移结论。

**只有 PRIMARY 档 (n_event_labeled = 5) 能产生正式低标签迁移判定** (§8)。
FT 与 MMD 各自独立判定, 十条**全部**满足才算 B6_PRIMARY_LOW_LABEL_POSITIVE_TRANSFER:
  1 mean paired gain > 0
  2 median paired gain > 0
  3 improve_count >= 4/5
  4 seed 级 paired bootstrap 95% CI 下界 > 0
  5 short/medium/long 中 >= 2/3 个 bin 的 gain >= 0
  6 macro corr 不低于 target_only - 0.02
  7 catastrophic rate 不高于 target_only
  8 warning miss rate 不高于 target_only + 0.05
  9 same-data-same-budget 公平性 PASS
 10 该 source 方法至少打赢 const_mean_info

§16: SECONDARY (n=3/10/21) 只出趋势与 SECONDARY_LOW_LABEL_SIGNAL_AT_N{n} 标签,
标为 NOT_PRIMARY_CONFIRMATORY_EVIDENCE, **不得推翻 PRIMARY 判定**。

§20: B5 负 + B6 primary 负 -> NO_POSITIVE_TRANSFER_SUPPORTED;
     B5 负 + B6 n=5 正 -> NO_GENERAL_POSITIVE_TRANSFER,
                          CONDITIONAL_BENEFIT_UNDER_FAILURE_LABEL_SCARCITY。
**绝对禁止把后者写成"迁移学习总体显著优于 Target-only"。**

阈值全部从 protocol_hash.json 读 (出数字之前冻结的那一份), 并与 live config 交叉
核对; 两边不一致即 B6_INVALID —— 事后降低门槛可被检出。NaN 一律判不通过。

用法:
    python scripts/basilisk_b6/final_transfer_verdict.py
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

NAN = float("nan")
INVALID = "B6_INVALID"

MAIN_ROWS = ("target_only", "source_finetune", "source_mmd_finetune",
             "const_mean_info", "damage_extrapolation")

GATE_BAR_DISCIPLINE = (
    "十条阈值全部从 checkpoints/basilisk_b6/protocol_hash.json 读取 —— "
    "那是在任何 B6 数字之前冻结的一份, 并与 live config 逐项交叉核对。"
    "两边不一致即 B6_INVALID。禁止降低门槛、换 seed、改超参、事后挑子集来翻转结论。"
    "NaN 一律判**不通过** —— 不可评估不等于通过。"
)

PRIMARY_ONLY = (
    "§8: PRIMARY = n_event_labeled 5, 且该档在看到任何 B6 结果之前就已固定。"
    "它是**唯一**允许产生正式低标签迁移判定的档 —— 避免\"挑一个最好看的标签档\"。"
)

PASS_MEANING = (
    "B6_PRIMARY_LOW_LABEL_POSITIVE_TRANSFER 只意味着: 在 B2.1 划分、CORE_ONLY "
    "输入、n_event_labeled = 5 (event 5 + censored 24) 这一个标签预算、五个正式 "
    "seed 下, 该 source 方法的 test information-zone trajectory-macro RMSE "
    "相对 Target-only 稳定更低, 且未在 corr / catastrophic / warning miss / "
    "寿命分 bin 上出现系统性退化。"
    "它**不意味着**迁移学习总体优于 Target-only (B5 已在全数据上给出否定结论), "
    "**不意味着**它优于物理 damage extrapolation 基线, "
    "**不意味着**结论可外推到其它划分、其它输入 schema 或其它标签预算。"
)

FAIL_MEANING = (
    "B6_NO_PRIMARY_LOW_LABEL_POSITIVE_TRANSFER 意味着: 即使把目标域失效标签压到 "
    "5 条完整失效轨迹 (同时保留全部 24 条 censored), source 预训练在预登记的十条"
    "门槛下也没能证明相对 Target-only 的稳定增益。"
    "这是一个正当的负面结论, 不是实验失败。"
)

CASE_B_FORBIDDEN = "迁移学习总体显著优于 Target-only"

CASE_B_DISCIPLINE = (
    "§20 情况 B (B5 负 + B6 n=5 正) 的唯一合法表述是"
    "\"无总体正转移, 仅在失效标签稀缺条件下存在条件性收益\"。"
    f"**绝对禁止**写成\"{CASE_B_FORBIDDEN}\" —— "
    "那会把一个单一标签预算下的条件性结果冒充为全局结论。"
)

DAMAGE_PHRASING = "学习模型未超过基于已知累计损伤结构的物理外推基线。"

RECOMMENDATION_BASIS = (
    "§19: 单一工程推荐, 候选只有 target_only / source_finetune / "
    "source_mmd_finetune / damage_extrapolation。"
    "优先级冻结: (1) primary 档 test info macro RMSE, (2) warning miss, "
    "(3) catastrophic rate, (4) 简洁性 / 可部署性。"
    "const_mean_info / persistence / true_rul_lookup 是标尺或 oracle, 排除在外。"
    "允许出现\"迁移判定为正\"同时\"推荐 damage_extrapolation\" —— "
    "前者问预训练是否帮到了这个模型, 后者问现在该部署什么。"
)

EXIT_DISCIPLINE = (
    "§26: B6 到此停止。不自动跑 B7, 不自动跑 B8, 不新增算法, 不调 MMD, "
    "不改 split, 不改 HI/RUL。下一步只允许 B7 (图表 + 报告) 与 B8 (封装 + Docker)。"
)


def _f(x) -> float:
    if x is None:
        return NAN
    x = float(x)
    return x if np.isfinite(x) else NAN


def _pos(a) -> bool:
    """> 0, 且必须有限。NaN 判不通过。"""
    a = _f(a)
    return bool(np.isfinite(a) and a > 0)


def _le(a, b) -> bool:
    a, b = _f(a), _f(b)
    return bool(np.isfinite(a) and np.isfinite(b) and a <= b)


def _ge(a, b) -> bool:
    a, b = _f(a), _f(b)
    return bool(np.isfinite(a) and np.isfinite(b) and a >= b)


def decide_primary(method: str, S: dict, Lv: dict, Wv: dict, thr: dict,
                   labels: dict) -> dict:
    """§15 十条 —— 对单一 source 方法在 PRIMARY 档判定。任一条不满足即 FAIL。"""
    key = "ft" if method == "source_finetune" else "mmd"
    g = S["gain"][f"gain_{key}"]
    corr_s = _f(S["corr_mean_by_group"][method])
    corr_t = _f(S["corr_mean_by_group"]["target_only"])
    cat_s = _f(S["catastrophic_rate_mean_by_group"][method])
    cat_t = _f(S["catastrophic_rate_mean_by_group"]["target_only"])
    gi = Wv["gate_inputs"]
    miss_s = _f(gi[f"{key}_miss_mean"])
    miss_t = _f(gi["target_only_miss_mean"])
    n_bins = int(Lv["n_bins_gain_nonneg"][key])
    minus_const = _f(S["aggregate_rmse"][f"{key}_minus_const"]["mean"])

    conds = [
        {"id": 1, "name": "mean_gain_positive",
         "detail": f"mean(gain) = {_f(g['mean']):+.6f} > 0",
         "passed": _pos(g["mean"]) if thr["require_mean_positive"] else True},
        {"id": 2, "name": "median_gain_positive",
         "detail": f"median(gain) = {_f(g['median']):+.6f} > 0",
         "passed": _pos(g["median"]) if thr["require_median_positive"] else True},
        {"id": 3, "name": "improve_count",
         "detail": f"{int(g['improve_count'])}/{thr['n_seeds']} seed gain > 0, "
                   f"要求 >= {thr['min_improve_count']}",
         "passed": bool(int(g["improve_count"]) >= int(thr["min_improve_count"]))},
        {"id": 4, "name": "bootstrap_ci_lower_positive",
         "detail": f"seed 级 paired bootstrap CI95 下界 = "
                   f"{_f(g['ci95_lower']):+.6f} > 0",
         "passed": (_pos(g["ci95_lower"])
                    if thr["require_ci_lower_positive"] else True)},
        {"id": 5, "name": "lifetime_bins_gain_nonneg",
         "detail": f"{n_bins}/{thr['n_lifetime_bins']} 个寿命 bin 的 gain >= 0, "
                   f"要求 >= {thr['min_lifetime_bins_gain_nonneg']}",
         "passed": bool(n_bins >= int(thr["min_lifetime_bins_gain_nonneg"]))},
        {"id": 6, "name": "no_corr_degradation",
         "detail": f"macro corr {corr_s:.4f} >= target {corr_t:.4f} - "
                   f"{thr['corr_tol']}",
         "passed": _ge(corr_s, corr_t - float(thr["corr_tol"]))},
        {"id": 7, "name": "catastrophic_not_worse",
         "detail": f"catastrophic rate {cat_s:.6f} <= target {cat_t:.6f}",
         "passed": (_le(cat_s, cat_t)
                    if thr["require_catastrophic_not_worse"] else True)},
        {"id": 8, "name": "warning_miss_not_worse",
         "detail": f"miss rate {miss_s:.6f} <= target {miss_t:.6f} + "
                   f"{thr['warning_miss_tol']}",
         "passed": bool(gi[f"{key}_within_tol"])},
        {"id": 9, "name": "same_data_same_budget_fairness",
         "detail": ("同 cell 内 train/val/test IDs、batch 顺序、初始权重、优化器、"
                    "训练预算、early stopping、checkpoint 规则、删失损失、"
                    f"evaluator 同一 = {bool(S['fairness_pass'])}"),
         "passed": (bool(S["fairness_pass"])
                    if thr["require_fairness_pass"] else True)},
        {"id": 10, "name": "beats_const_mean_info",
         "detail": f"{key} − const = {minus_const:+.6f} < 0 "
                   f"(RMSE 低于常数标尺)",
         "passed": (bool(np.isfinite(minus_const) and minus_const < 0)
                    if thr["require_beat_const_mean_info"] else True)},
    ]
    n_passed = sum(1 for c in conds if c["passed"])
    all_ok = n_passed == len(conds)
    return {
        "method": method,
        "level": int(S["n_event_labeled"]),
        "is_primary": True,
        "conditions": conds,
        "n_conditions": len(conds),
        "n_passed": n_passed,
        "all_required": True,
        "verdict": labels["primary_pass"] if all_ok else labels["primary_fail"],
        "low_label_positive_transfer": bool(all_ok),
        "failed_conditions": [c["id"] for c in conds if not c["passed"]],
        "thresholds_frozen_before_run": True,
        "nan_counts_as_fail": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wheel_basilisk_b6.yaml")
    a = ap.parse_args()

    cfg = load_cfg(a.config)
    b6 = cfg["b6"]
    P = cfg["paths"]

    # --- 阈值只从冻结的 protocol_hash 读, 并与 live config 交叉核对 ---
    proto = json.loads((ROOT / cfg["protocol"]["hash_path"]).read_text("utf-8"))
    thr = dict(proto["gate_thresholds"])
    labels = dict(proto["gate_labels"])
    now = hashlib.sha256(
        (ROOT / cfg["protocol"]["path"]).read_bytes()).hexdigest()
    if now != str(proto["protocol_sha256"]):
        raise SystemExit(f"!! protocol.md 在冻结之后被改动 —— {INVALID}")
    live = {k: b6["gate"][k] for k in thr if k in b6["gate"]}
    for k, v in live.items():
        if type(thr[k]) is bool or type(v) is bool:
            same = bool(thr[k]) == bool(v)
        else:
            same = float(thr[k]) == float(v)
        if not same:
            raise SystemExit(f"!! {INVALID}: gate.{k} config={v} != 冻结值 "
                             f"{thr[k]} —— 禁止降低门槛")
    if dict(b6["gate"]["labels"]) != labels:
        raise SystemExit(f"!! gate.labels 与冻结件不符 —— {INVALID}")

    M = json.loads((ROOT / P["metrics_json"]).read_text("utf-8"))
    S = json.loads((ROOT / P["paired_json"]).read_text("utf-8"))
    L = json.loads((ROOT / P["lifetime_json"]).read_text("utf-8"))
    W = json.loads((ROOT / P["warning_json"]).read_text("utf-8"))
    U = json.loads((ROOT / P["summary_json"]).read_text("utf-8"))
    if bool(M.get("fast")):
        raise SystemExit("!! metrics 来自 --fast 冒烟跑, 不得下正式结论")
    for nm, blk in (("paired", S), ("lifetime", L), ("warning", W),
                    ("summary", U)):
        for k in ("protocol_sha256", "subset_manifest_sha256", "split_sha256"):
            if str(blk[k]) != str(M[k]):
                raise SystemExit(f"!! {nm} 的 {k} 与 metrics 不一致 —— {INVALID}")

    levels = [int(x) for x in M["label_levels"]]
    primary = int(M["primary_level"])
    if primary != int(proto["primary"]["n_event_labeled"]) or primary != 5:
        raise SystemExit(f"!! PRIMARY 必须是 n_event_labeled = 5 —— {INVALID}")
    secondary = [int(x) for x in proto["secondary"]["levels"]]
    if sorted(secondary) != sorted(set(levels) - {primary}):
        raise SystemExit(f"!! SECONDARY 档集合与冻结件不符 —— {INVALID}")

    Sp = S["by_level"][str(primary)]
    Lp = L["by_level"][str(primary)]
    Wp = W["by_level"][str(primary)]
    if not bool(Sp["is_primary"]):
        raise SystemExit(f"!! primary 标记错位 —— {INVALID}")
    if not bool(Sp["checkpoint_validation_only"]):
        raise SystemExit(f"!! primary 档存在非 validation-only 选点 —— {INVALID}")

    dec_ft = decide_primary("source_finetune", Sp, Lp, Wp, thr, labels)
    dec_mmd = decide_primary("source_mmd_finetune", Sp, Lp, Wp, thr, labels)
    ft_ok = dec_ft["low_label_positive_transfer"]
    mmd_ok = dec_mmd["low_label_positive_transfer"]
    primary_positive = bool(ft_ok or mmd_ok)
    primary_verdict = (labels["primary_pass"] if primary_positive
                       else labels["primary_fail"])

    # --- §16 SECONDARY 只出信号, 不得推翻 primary ---
    sig_cfg = b6["secondary_signal"]
    sec_signals = []
    for n in secondary:
        Ss, Ls = S["by_level"][str(n)], L["by_level"][str(n)]
        for meth, key in (("source_finetune", "ft"),
                          ("source_mmd_finetune", "mmd")):
            g = Ss["gain"][f"gain_{key}"]
            # 信号的门槛只用"方向 + 一致性"三条, 与 primary 十条不同, 故不可混用
            strong = bool(_pos(g["mean"]) and _pos(g["median"])
                          and int(g["improve_count"])
                          >= int(thr["min_improve_count"]))
            if strong:
                sec_signals.append({
                    "label": str(sig_cfg["label_template"]).format(n=n),
                    "n_event_labeled": n,
                    "method": meth,
                    "mean_gain": _f(g["mean"]),
                    "median_gain": _f(g["median"]),
                    "improve_count": int(g["improve_count"]),
                    "ci95_lower": _f(g["ci95_lower"]),
                    "n_bins_gain_nonneg": int(Ls["n_bins_gain_nonneg"][key]),
                    "mark_as": str(sig_cfg["mark_as"]),
                    "cannot_override_primary": True,
                    "criteria": "mean>0 且 median>0 且 improve>=4/5 "
                                "(仅三条方向性条件, 不是 primary 的十条门槛)",
                })
    secondary_block = {
        "levels": secondary,
        "role": str(proto["secondary"]["role"]),
        "signals": sec_signals,
        "n_signals": len(sec_signals),
        "cannot_override_primary": True,
        "override_attempted": False,
        "forbid_reason": str(sig_cfg["forbid_reason"]),
        "trend": U["gain_vs_labels"],
    }

    # --- §18 damage 比较 (主表内, 每档都报) ---
    per_level_damage = {}
    for n in levels:
        vd = U["main_tables"][str(n)]["vs_damage"]
        per_level_damage[str(n)] = {
            "target_minus_damage": _f(vd["target_minus_damage"]),
            "ft_minus_damage": _f(vd["ft_minus_damage"]),
            "mmd_minus_damage": _f(vd["mmd_minus_damage"]),
            "rmse_damage": _f(vd["rmse_damage"]),
            "damage_leads_all_learned": bool(vd["damage_leads_all_learned"]),
        }
    damage_leads_primary = bool(
        per_level_damage[str(primary)]["damage_leads_all_learned"])
    damage_leads_everywhere = bool(
        all(v["damage_leads_all_learned"] for v in per_level_damage.values()))
    damage_block = {
        "kept_in_main_table": True,
        "demoted_to_footnote": False,
        "per_level": per_level_damage,
        "definition": "正数 = 该学习方法 RMSE 更高, 即不如物理外推基线",
        "damage_leads_at_primary": damage_leads_primary,
        "damage_leads_at_every_level": damage_leads_everywhere,
        "mandatory_statement": (DAMAGE_PHRASING if damage_leads_primary
                                else None),
        "forbid_hiding_because_theme_is_transfer_learning": True,
    }

    # --- §19 单一工程推荐: 依据 primary 档主指标, 优先级冻结 ---
    erc = b6["engineering_recommendation"]
    excluded = list(erc["excluded_oracle_methods"])
    cands = [g for g in erc["candidates"] if g not in excluded]
    rowsP = U["main_tables"][str(primary)]["rows"]
    scored = []
    for g in cands:
        v = _f(rowsP[g]["info_macro_rmse"])
        if not np.isfinite(v):
            continue
        scored.append({"method": g, "info_macro_rmse": v,
                       "miss_rate": _f(rowsP[g]["miss_rate"]),
                       "catastrophic_rate": _f(rowsP[g]["catastrophic_rate"])})
    if not scored:
        raise SystemExit("!! primary 档无候选可评分 —— 拒绝给推荐")
    # 优先级冻结: RMSE -> miss -> catastrophic (NaN 排到最后, 不当成 0)
    def _k(d: dict):
        def q(x):
            x = _f(x)
            return (1, 0.0) if not np.isfinite(x) else (0, x)
        return (q(d["info_macro_rmse"]), q(d["miss_rate"]),
                q(d["catastrophic_rate"]))
    scored.sort(key=_k)
    rec = str(scored[0]["method"])
    recommendation = {
        "recommendation": rec,
        "single_recommendation": True,
        "basis": "primary_level_formal_test_table",
        "primary_level": primary,
        "primary_metric": str(b6["primary_metric"]),
        "priority_order": list(erc["priority_order"]),
        "priority_frozen": bool(erc["priority_frozen"]),
        "ranking": scored,
        "excluded_oracle_methods": excluded,
        "meaning": RECOMMENDATION_BASIS,
        "separate_from_transfer_verdict": True,
        "damage_still_first": bool(rec == "damage_extrapolation"),
        "note_if_diverges": (
            "迁移判定与工程推荐指向不同对象; 两者回答的问题不同。"
            if primary_positive and rec not in ("source_finetune",
                                                "source_mmd_finetune") else None),
    }

    # --- §20 B5 / B6 关系 -> §21 最终结论 ---
    rel = b6["b5_b6_relationship"]
    b5_negative = str(b6["frozen_facts"]["b5_verdict"]) == "B5_NO_POSITIVE_TRANSFER"
    if not b5_negative:
        raise SystemExit(f"!! B5 冻结结论被改动 —— {INVALID}")
    allowed_final = list(b6["final_conclusion"]["allowed_values"])
    if primary_positive:
        case, case_desc = "B", str(rel["case_B"]["final"])
        final_conclusion = "CONDITIONAL_LOW_LABEL_TRANSFER_ONLY"
    else:
        case, case_desc = "A", str(rel["case_A"]["final"])
        final_conclusion = "NO_POSITIVE_TRANSFER_SUPPORTED"
    if final_conclusion not in allowed_final:
        raise SystemExit(f"!! 最终结论 {final_conclusion} 不在允许值内 —— {INVALID}")

    b5_b6 = {
        "b5_role": str(rel["b5_role"]),
        "b5_verdict": "B5_NO_POSITIVE_TRANSFER",
        "b5_conclusion_permanent": True,
        "b5_overwritten_by_b6": False,
        "b6_role": str(rel["b6_role"]),
        "b6_primary_verdict": primary_verdict,
        "case": case,
        "case_condition": str(rel[f"case_{case}"]["condition"]),
        "combined_statement": case_desc,
        "forbidden_phrasing": CASE_B_FORBIDDEN,
        "forbid_writing_case_B_as_overall_superiority": True,
        "case_b_discipline": CASE_B_DISCIPLINE,
        "b5_full_data_numbers": {
            "target_only": _f(b6["frozen_facts"]
                              ["b5_target_only_mean_info_macro_rmse"]),
            "source_finetune": _f(b6["frozen_facts"]
                                  ["b5_source_finetune_mean_info_macro_rmse"]),
            "source_mmd_finetune":
                _f(b6["frozen_facts"]
                   ["b5_source_mmd_finetune_mean_info_macro_rmse"]),
            "damage_extrapolation":
                _f(b6["frozen_facts"]
                   ["b5_damage_extrapolation_mean_info_macro_rmse"]),
            "gain_ft_mean": _f(b6["frozen_facts"]["b5_gain_ft_mean"]),
            "gain_mmd_mean": _f(b6["frozen_facts"]["b5_gain_mmd_mean"]),
            "gain_ft_improve_count":
                int(b6["frozen_facts"]["b5_gain_ft_improve_count"]),
            "gain_mmd_improve_count":
                int(b6["frozen_facts"]["b5_gain_mmd_improve_count"]),
            "engineering_recommendation":
                str(b6["frozen_facts"]["b5_engineering_recommendation"]),
        },
    }

    er = b6["exit_rules"]
    out = {
        "stage": "BASILISK_B6",
        "label": str(b6["label"]),
        "formal_stage": True,
        "last_stage_allowed_to_produce_core_numbers":
            bool(b6["last_stage_allowed_to_produce_core_numbers"]),
        "question": str(b6["question"]),
        "primary_only": PRIMARY_ONLY,
        "gate_bar_discipline": GATE_BAR_DISCIPLINE,
        "pass_meaning": PASS_MEANING,
        "fail_meaning": FAIL_MEANING,
        "exit_discipline": EXIT_DISCIPLINE,
        "protocol_sha256": str(M["protocol_sha256"]),
        "config_sha256": str(M["config_sha256"]),
        "subset_manifest_sha256": str(M["subset_manifest_sha256"]),
        "split_sha256": str(M["split_sha256"]),
        "input_schema": str(M["input_schema"]),
        "n_features": int(M["n_features"]),
        "formal_seeds": [int(s) for s in M["formal_seeds"]],
        "seed_role": str(M["seed_role"]),
        "label_levels": levels,
        "primary_level": primary,
        "secondary_levels": secondary,
        "primary_train_composition": dict(Sp["train_composition"]),
        "gate_thresholds": thr,
        "gate_labels": labels,
        "primary_decision": {"source_finetune": dec_ft,
                             "source_mmd_finetune": dec_mmd},
        "primary_verdict": primary_verdict,
        "B6_PRIMARY_VERDICT": primary_verdict,
        "primary_positive": {"ft": ft_ok, "mmd": mmd_ok,
                             "any": primary_positive},
        "secondary": secondary_block,
        "secondary_cannot_override_primary": True,
        "primary_main_table": {g: dict(rowsP[g]) for g in MAIN_ROWS},
        "lifetime_bins_primary": {
            "gain_by_bin": Lp["gain_by_bin"],
            "n_bins_gain_nonneg": Lp["n_bins_gain_nonneg"],
            "localized_transfer_benefit": Lp["localized_transfer_benefit"]},
        "warning_primary": Wp["gate_inputs"],
        "damage_baseline": damage_block,
        "engineering_recommendation": recommendation,
        "ENGINEERING_RECOMMENDATION": rec,
        "b5_b6_relationship": b5_b6,
        "FINAL_TRANSFER_CONCLUSION": final_conclusion,
        "final_conclusion_allowed_values": allowed_final,
        "final_conclusion_doc": str(b6["final_conclusion"]["path"]),
        "final_conclusion_must_explain_separately":
            list(b6["final_conclusion"]["must_explain_separately"]),
        "stop_after_b6": bool(er["stop_after_b6"]),
        "forbid_auto_run_b7": bool(er["forbid_auto_run_b7"]),
        "forbid_auto_run_b8": bool(er["forbid_auto_run_b8"]),
        "next_stage_allowed_only": list(er["next_stage_allowed_only"]),
        "b7_auto_run": False,
        "b8_auto_run": False,
    }

    # 从 conditions 重算一次, 与写入的 verdict 对不上就报错 —— 防止手改结论
    for nm, dec in (("ft", dec_ft), ("mmd", dec_mmd)):
        if all(c["passed"] for c in dec["conditions"]) != \
                dec["low_label_positive_transfer"]:
            raise SystemExit(f"!! {nm} 判定与 conditions 不一致 —— 拒绝写出")
    if primary_positive != (ft_ok or mmd_ok):
        raise SystemExit("!! primary 汇总判定自相矛盾 —— 拒绝写出")
    if (final_conclusion == "CONDITIONAL_LOW_LABEL_TRANSFER_ONLY") != \
            primary_positive:
        raise SystemExit("!! 最终结论与 primary 判定不一致 —— 拒绝写出")

    op = ROOT / P["verdict_json"]
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2, ensure_ascii=False,
                             default=float) + "\n", encoding="utf-8")
    print(f">> 已写 {op.relative_to(ROOT)}")

    print(f"\n{'=' * 78}")
    print(f"PRIMARY = n_event_labeled {primary} "
          f"(event {Sp['train_composition']['event']} + censored "
          f"{Sp['train_composition']['censored']} = "
          f"{Sp['train_composition']['total']})")
    print(f"{'=' * 78}")
    for nm, dec in (("FT ", dec_ft), ("MMD", dec_mmd)):
        print(f"[{nm}] {dec['verdict']}  ({dec['n_passed']}/10 条通过)")
        for c in dec["conditions"]:
            print(f"    {'PASS' if c['passed'] else 'FAIL'} {c['id']:>2}. "
                  f"{c['name']}: {c['detail']}")
    print(f"\n>> B6_PRIMARY_VERDICT = {primary_verdict}")
    if sec_signals:
        for s in sec_signals:
            print(f">> SECONDARY 信号 {s['label']} [{s['method']}] "
                  f"mean {s['mean_gain']:+.6f} improve {s['improve_count']}/5 "
                  f"—— 标记为 {s['mark_as']}, 不推翻 primary")
    else:
        print(">> SECONDARY 档无满足方向性条件的信号")
    for n in levels:
        d = per_level_damage[str(n)]
        print(f">> n={n:<2} vs damage: target {d['target_minus_damage']:+.6f} "
              f"ft {d['ft_minus_damage']:+.6f} mmd {d['mmd_minus_damage']:+.6f} "
              f"(damage RMSE {d['rmse_damage']:.6f})")
    if damage_leads_primary:
        print(f">> §18 必须原样写出: {DAMAGE_PHRASING}")
    print(f">> ENGINEERING_RECOMMENDATION = {rec} "
          f"(primary 档 info macro RMSE {scored[0]['info_macro_rmse']:.6f})")
    print(f">> §20 情况 {case}: {case_desc}")
    print(f">> FINAL_TRANSFER_CONCLUSION = {final_conclusion}")
    print(f">> {EXIT_DISCIPLINE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
