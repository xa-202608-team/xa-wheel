#!/usr/bin/env python
"""scripts/basilisk_b6/freeze_protocol.py

BASILISK-B6 §15/§24-2 —— 在**任何 B6 数字产生之前**冻结 protocol 与 config。

写出 checkpoints/basilisk_b6/protocol_hash.json, 内含 protocol.md 与
wheel_basilisk_b6.yaml 的 sha256, 以及从 config 抽出的全部 Gate 阈值、标签档定义、
子集配额与 subset_seed。之后 final_transfer_verdict.py 只能按这份哈希里的阈值判定
—— 出数字后改阈值会让哈希对不上。

同时硬断言:
  * B6 的输出目录里还没有任何 metrics —— 保证"protocol 先于数字" (§24);
  * 正式 seed 与禁用 seed 无交集 (§9);
  * 输入 schema 为 CORE_ONLY / n_features=12 且与 B3X 判定一致 (§4);
  * 划分与 B2.1 逐项一致, train = 45 (event 21 / censored 24) (§5);
  * 标签档 = {3,5,10,21} 且每档都保留全部 24 条 censored (§6);
  * 每档 bin 配额之和等于该档 n_event, 且不超过各 bin 的可用 event 数 (§7);
  * bin edges 等于 B2.1 冻结的 [26846, 37395] (§7);
  * PRIMARY 恰为 n=5 (§8);
  * B5 的正式结论仍是 B5_NO_POSITIVE_TRANSFER (§0/§20);
  * mmd_lambda 与 B2 冻结值一致 (§10);
  * Gate 阈值等于 protocol.md §15 写死的那十条。

用法:
    python scripts/basilisk_b6/freeze_protocol.py --config configs/wheel_basilisk_b6.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.basilisk_b11.calibrate_degradation import load_b11_config  # noqa: E402

# §15 的十条 Gate 阈值。写成模块级常量, 与 protocol.md §15 的表格逐条对应;
# freeze 时与 config 交叉校验, 两边不一致就拒绝冻结 —— 防止"文档写 4/5, 代码
# 跑 3/5"这种只有读代码才能发现的偏差。
GATE_SPEC_FROM_PROTOCOL = {
    "n_conditions": 10,
    "n_seeds": 5,
    "require_mean_positive": True,
    "require_median_positive": True,
    "min_improve_count": 4,
    "require_ci_lower_positive": True,
    "min_lifetime_bins_gain_nonneg": 2,
    "n_lifetime_bins": 3,
    "corr_tol": 0.02,
    "require_catastrophic_not_worse": True,
    "warning_miss_tol": 0.05,
    "require_fairness_pass": True,
    "require_beat_const_mean_info": True,
}

# §6/§7 标签档定义。同样与 protocol.md 的两张表逐条对应。
LABEL_SPEC_FROM_PROTOCOL = {
    "levels": [3, 5, 10, 21],
    "n_censored_train_always": 24,
    "primary": 5,
    "secondary": [3, 10, 21],
    "subset_seed": 20260814,
    "bin_edges": [26846.0, 37395.0],
    "quota": {3: {"short": 1, "medium": 1, "long": 1},
              5: {"short": 2, "medium": 1, "long": 2},
              10: {"short": 3, "medium": 3, "long": 4},
              21: {"short": 7, "medium": 7, "long": 7}},
}

FREEZE_MEANING = (
    "本文件的存在意味着: B6 的判定规则、标签档定义、子集配额与 subset_seed 在看到"
    "任何 B6 数字之前已经固定。final_transfer_verdict.py 必须按此处记录的阈值判定; "
    "若 config 的阈值与此处不符, 说明有人在出数字后改了门槛, 应判 B6_INVALID "
    "而不是采信新阈值。"
    "PRIMARY = n_event_labeled 5 在此固定 —— 它是唯一允许产生正式低标签迁移判定的档, "
    "选它的理由是它位于 3 与 10 之间且在任何结果之前就已确定, "
    "而不是因为它的结果好看。"
    "SECONDARY (3/10/21) 只能用于趋势描述, 不得单独推翻 primary。"
    "B5_NO_POSITIVE_TRANSFER 是终局, 不得因 B6 被覆盖。"
)


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wheel_basilisk_b6.yaml")
    a = ap.parse_args()

    cfg = load_b11_config(a.config)
    b6 = cfg["b6"]
    paths = cfg["paths"]
    proto = ROOT / cfg["protocol"]["path"]
    out = ROOT / cfg["protocol"]["hash_path"]

    if not proto.exists():
        raise SystemExit(f"!! protocol 缺失: {proto} —— 必须先写 protocol")

    # --- 硬断言 1: 冻结必须发生在任何数字之前 (§24) ---
    for key in ("subset_manifest_json", "metrics_json", "paired_json",
                "lifetime_json", "warning_json", "summary_json",
                "verdict_json"):
        p = ROOT / paths[key]
        if p.exists():
            raise SystemExit(
                f"!! {p.relative_to(ROOT)} 已存在 —— protocol 必须在出数字前冻结。"
                " 若要重跑, 请先删除 B6 的输出再重新 freeze。")

    # --- 硬断言 2: 正式 seed 全新 (§9) ---
    seeds = [int(s) for s in b6["seeds"]]
    forbidden = {int(s) for s in b6["forbid_reused_seeds"]}
    overlap = sorted(set(seeds) & forbidden)
    if overlap:
        raise SystemExit(f"!! 正式 seed 与已观察 seed 重叠 {overlap} —— B6_INVALID")
    if len(set(seeds)) != len(seeds):
        raise SystemExit(f"!! 正式 seed 有重复: {seeds}")

    # --- 硬断言 3: 输入 schema 与 B3X 判定一致, 且不得恢复 mission arm (§4) ---
    isch = b6["input_schema"]
    b3x = json.loads((ROOT / paths["b3x_summary_json"]).read_text("utf-8"))
    b3x_verdict = str(b3x["verdict"])
    want = str(isch["rule"][b3x_verdict])
    if str(isch["formal"]).lower() != want.lower():
        raise SystemExit(
            f"!! schema 不一致: B3X={b3x_verdict} 应得 {want}, "
            f"config 写 {isch['formal']} —— B6_INVALID")
    if int(isch["n_features"]) != 12:
        raise SystemExit(f"!! §4 要求 n_features=12, config 写 {isch['n_features']}")
    for flag in ("forbid_mission_features", "forbid_restoring_mission_arm"):
        if not bool(isch[flag]):
            raise SystemExit(f"!! {flag} 必须为 true (§4)")

    # --- 硬断言 4: 划分与 B2.1 逐项一致 (§5) ---
    man = json.loads((ROOT / paths["b21_split_json"]).read_text("utf-8"))
    if str(man["verdict"]) != "B21_SPLIT_VALID":
        raise SystemExit(f"!! B2.1 划分级判定 {man['verdict']} 不是 "
                         f"B21_SPLIT_VALID —— B6_INVALID")
    sp = b6["split"]
    if str(man["split_sha256"]) != str(sp["expected_sha256"]):
        raise SystemExit("!! B2.1 split hash 不符 —— B6_INVALID")
    b21 = json.loads((ROOT / paths["b21_summary_json"]).read_text("utf-8"))
    if str(b21["verdict"]) != str(sp["require_b21_verdict"]):
        raise SystemExit(f"!! B2.1 阶段判定 {b21['verdict']} != 要求值 "
                         f"{sp['require_b21_verdict']} —— B6_INVALID")
    tr = man["splits"]["train"]
    for key, got, want_v in (("n_train", int(tr["n"]), int(sp["n_train"])),
                             ("n_train_event", int(tr["n_event"]),
                              int(sp["n_train_event"])),
                             ("n_train_censored", int(tr["n_censored"]),
                              int(sp["n_train_censored"])),
                             ("n_val", int(man["splits"]["val"]["n"]),
                              int(sp["n_val"])),
                             ("n_test", int(man["splits"]["test"]["n"]),
                              int(sp["n_test"]))):
        if got != want_v:
            raise SystemExit(f"!! §5 {key}: manifest={got} != config={want_v}")
    if not bool(sp["val_test_never_touched"]):
        raise SystemExit("!! val_test_never_touched 必须为 true (§5)")

    # --- 硬断言 5: 标签档定义 (§6) ---
    ls = b6["label_scarcity"]
    levels = [int(x) for x in ls["levels"]]
    if levels != LABEL_SPEC_FROM_PROTOCOL["levels"]:
        raise SystemExit(f"!! §6 标签档必须是 "
                         f"{LABEL_SPEC_FROM_PROTOCOL['levels']}, 得到 {levels}")
    n_cen = int(ls["n_censored_train_always"])
    if n_cen != int(sp["n_train_censored"]):
        raise SystemExit(f"!! §6 每档必须保留全部 {sp['n_train_censored']} 条 "
                         f"censored, config 写 {n_cen}")
    comp = {int(k): v for k, v in ls["train_composition"].items()}
    for n in levels:
        c = comp[n]
        if int(c["event"]) != n or int(c["censored"]) != n_cen \
                or int(c["total"]) != n + n_cen:
            raise SystemExit(f"!! §6 n={n} 的 train_composition 不自洽: {c}")
    for flag in ("always_keep_all_censored_train",
                 "forbid_relabel_known_eol_as_censored",
                 "forbid_using_future_or_eol_info",
                 "forbid_smuggling_as_unlabelled_input",
                 "forbid_defining_scarcity_as_total_n_train"):
        if not bool(ls[flag]):
            raise SystemExit(f"!! {flag} 必须为 true (§6)")

    # --- 硬断言 6: 子集配额与 bin edges (§7) ---
    ss = b6["subset_selection"]
    if int(ss["subset_seed"]) != LABEL_SPEC_FROM_PROTOCOL["subset_seed"]:
        raise SystemExit(f"!! §7 subset_seed 必须是 "
                         f"{LABEL_SPEC_FROM_PROTOCOL['subset_seed']}")
    edges = [float(x) for x in ss["bin_edges"]]
    b21_edges = [float(x) for x in man["lifetime_bins"]["event"]["edges"]]
    if edges != b21_edges:
        raise SystemExit(f"!! §7 bin edges {edges} != B2.1 冻结值 {b21_edges}")
    if edges != LABEL_SPEC_FROM_PROTOCOL["bin_edges"]:
        raise SystemExit(f"!! §7 bin edges 与 protocol.md 不一致: {edges}")
    names = list(ss["bin_names"])
    if names != list(man["lifetime_bins"]["event"]["names"]):
        raise SystemExit(f"!! bin 名与 B2.1 manifest 不一致: {names}")
    if not bool(ss["forbid_recompute_edges_from_b6_test"]):
        raise SystemExit("!! forbid_recompute_edges_from_b6_test 必须为 true (§7)")
    quota = {int(k): {str(b): int(v) for b, v in q.items()}
             for k, q in ss["per_level_bin_quota"].items()}
    if quota != LABEL_SPEC_FROM_PROTOCOL["quota"]:
        raise SystemExit(f"!! §7 bin 配额与 protocol.md 不一致: {quota}")
    # 配额之和必须等于该档 n_event; 且不得超过 B2.1 train 内该 bin 的可用条数。
    avail = {str(k).split("/")[-1]: int(v)
             for k, v in tr["strata"].items() if str(k).startswith("event/")}
    for n in levels:
        q = quota[n]
        if sum(q.values()) != n:
            raise SystemExit(f"!! §7 n={n} 配额之和 {sum(q.values())} != {n}")
        for b in names:
            if q[b] > avail.get(b, 0):
                raise SystemExit(
                    f"!! §7 n={n} 的 {b} 配额 {q[b]} 超过 B2.1 train 可用 "
                    f"{avail.get(b, 0)} 条 —— 配额不可满足, B6_INVALID")
    if not bool(ss["must_be_fixed_before_any_training"]):
        raise SystemExit("!! must_be_fixed_before_any_training 必须为 true (§7)")

    # --- 硬断言 7: PRIMARY 恰为 n=5 (§8) ---
    pri = int(b6["primary"]["n_event_labeled"])
    if pri != LABEL_SPEC_FROM_PROTOCOL["primary"]:
        raise SystemExit(f"!! §8 PRIMARY 必须是 n_event_labeled=5, 得到 {pri}")
    sec = [int(x) for x in b6["secondary"]["levels"]]
    if sec != LABEL_SPEC_FROM_PROTOCOL["secondary"]:
        raise SystemExit(f"!! §8 SECONDARY 必须是 {LABEL_SPEC_FROM_PROTOCOL['secondary']}")
    if not bool(b6["primary"]["only_level_allowed_to_produce_formal_verdict"]):
        raise SystemExit("!! primary 必须是唯一允许出正式判定的档 (§8)")
    if not bool(b6["secondary"]["cannot_override_primary"]):
        raise SystemExit("!! secondary.cannot_override_primary 必须为 true (§16)")

    # --- 硬断言 8: B5 结论仍是终局 (§0/§20) ---
    b5 = json.loads((ROOT / paths["b5_summary_json"]).read_text("utf-8"))
    if str(b5["verdict"]) != "B5_NO_POSITIVE_TRANSFER":
        raise SystemExit(f"!! B5 正式判定 {b5['verdict']} != "
                         f"B5_NO_POSITIVE_TRANSFER —— B5 结论被改动, B6_INVALID")
    if not bool(b6["frozen_facts"]["b5_must_not_be_overwritten"]):
        raise SystemExit("!! b5_must_not_be_overwritten 必须为 true (§0)")

    # --- 硬断言 9: mmd_lambda 冻结 (§10) ---
    lam = float(cfg["transfer"]["mmd_lambda"])
    b2cfg = load_b11_config("configs/wheel_basilisk_b2.yaml")
    lam_b2 = float(b2cfg["transfer"]["mmd_lambda"])
    if abs(lam - lam_b2) > 0:
        raise SystemExit(f"!! mmd_lambda {lam} != B2 冻结值 {lam_b2} —— B6_INVALID")
    for flag in ("forbid_new_transfer_methods", "forbid_wiener_pf",
                 "forbid_rate_model", "forbid_new_mmd_lambda",
                 "forbid_mission_feature_arm"):
        if not bool(b6[flag]):
            raise SystemExit(f"!! {flag} 必须为 true (§10)")

    # --- 硬断言 10: Gate 阈值 == protocol.md §15 ---
    g = b6["gate"]
    got = {
        "n_conditions": int(g["n_conditions"]),
        "n_seeds": len(seeds),
        "require_mean_positive": bool(g["require_mean_positive"]),
        "require_median_positive": bool(g["require_median_positive"]),
        "min_improve_count": int(g["min_improve_count"]),
        "require_ci_lower_positive": bool(g["require_ci_lower_positive"]),
        "min_lifetime_bins_gain_nonneg":
            int(g["min_lifetime_bins_gain_nonneg"]),
        "n_lifetime_bins": int(g["n_lifetime_bins"]),
        "corr_tol": float(g["corr_tol"]),
        "require_catastrophic_not_worse":
            bool(g["require_catastrophic_not_worse"]),
        "warning_miss_tol": float(g["warning_miss_tol"]),
        "require_fairness_pass": bool(g["require_fairness_pass"]),
        "require_beat_const_mean_info": bool(g["require_beat_const_mean_info"]),
    }
    if got != GATE_SPEC_FROM_PROTOCOL:
        diff = {k: (GATE_SPEC_FROM_PROTOCOL[k], got[k])
                for k in got if got[k] != GATE_SPEC_FROM_PROTOCOL[k]}
        raise SystemExit(f"!! Gate 阈值与 protocol.md §15 不一致: {diff}")
    if not bool(g["all_required"]):
        raise SystemExit("!! §15 十条必须全部满足 (all_required)")
    if not bool(g["forbid_lowering_bar"]):
        raise SystemExit("!! forbid_lowering_bar 必须为 true (§15)")

    # --- 硬断言 11: 训练组恰三个, eval-only 恰两个 (§10) ---
    tg = list(b6["trained_groups"])
    if tg != ["target_only", "source_finetune", "source_mmd_finetune"]:
        raise SystemExit(f"!! 训练组不符 §10: {tg}")
    eg = list(b6["eval_only_groups"])
    if sorted(eg) != sorted(["damage_extrapolation", "const_mean_info"]):
        raise SystemExit(f"!! eval-only 组不符 §10: {eg}")

    # --- 硬断言 12: 最终结论只允许两种取值 (§21) ---
    allowed = list(b6["final_conclusion"]["allowed_values"])
    if sorted(allowed) != sorted(["NO_POSITIVE_TRANSFER_SUPPORTED",
                                  "CONDITIONAL_LOW_LABEL_TRANSFER_ONLY"]):
        raise SystemExit(f"!! §21 最终结论取值集合不符: {allowed}")

    # --- 硬断言 13: 出口纪律 (§26) ---
    er = b6["exit_rules"]
    for flag in ("stop_after_b6", "forbid_auto_run_b7", "forbid_auto_run_b8",
                 "forbid_new_algorithms", "forbid_tuning_mmd",
                 "forbid_changing_split", "forbid_changing_hi_rul"):
        if not bool(er[flag]):
            raise SystemExit(f"!! exit_rules.{flag} 必须为 true (§26)")

    rec = {
        "stage": "BASILISK_B6",
        "label": str(b6["label"]),
        "freeze_meaning": FREEZE_MEANING,
        "frozen_before_any_b6_number": True,
        "last_stage_allowed_to_produce_core_numbers":
            bool(b6["last_stage_allowed_to_produce_core_numbers"]),
        "protocol_path": str(cfg["protocol"]["path"]),
        "protocol_sha256": sha256_file(proto),
        "config_path": a.config,
        "config_sha256": sha256_file(ROOT / a.config),
        "question": str(b6["question"]),
        "scarcity_axis": str(b6["scarcity_axis"]),
        "scarcity_axis_is_not": str(b6["scarcity_axis_is_not"]),
        "formal_seeds": seeds,
        "forbidden_seeds": sorted(forbidden),
        "input_schema": {"formal": str(isch["formal"]),
                         "n_features": int(isch["n_features"]),
                         "decided_by_b3x_verdict": b3x_verdict,
                         "forbid_mission_features":
                             bool(isch["forbid_mission_features"]),
                         "forbid_restoring_mission_arm":
                             bool(isch["forbid_restoring_mission_arm"])},
        "split": {"sha256": str(man["split_sha256"]),
                  "b21_split_verdict": str(man["verdict"]),
                  "b21_stage_verdict": str(b21["verdict"]),
                  "n": {k: int(man["splits"][k]["n"])
                        for k in ("train", "val", "test")},
                  "n_train_event": int(tr["n_event"]),
                  "n_train_censored": int(tr["n_censored"]),
                  "val_test_never_touched": True},
        "label_scarcity": {
            "axis": str(ls["axis"]),
            "levels": levels,
            "n_censored_train_always": n_cen,
            "train_composition": {str(n): comp[n] for n in levels},
            "unselected_events": str(ls["unselected_events"]),
        },
        "subset_selection": {
            "subset_seed": int(ss["subset_seed"]),
            "bin_names": names,
            "bin_edges": edges,
            "bin_edges_source": str(ss["bin_source"]),
            "forbid_recompute_edges_from_b6_test": True,
            "per_level_bin_quota": {str(n): quota[n] for n in levels},
            "available_event_per_bin_in_b21_train": avail,
            "method": str(ss["method"]),
            "forbid_change_after_written": bool(ss["forbid_change_after_written"]),
        },
        "primary": {"n_event_labeled": pri,
                    "reason": str(b6["primary"]["reason"]),
                    "only_level_allowed_to_produce_formal_verdict": True},
        "secondary": {"levels": sec,
                      "role": str(b6["secondary"]["role"]),
                      "cannot_override_primary": True},
        "trained_groups": tg,
        "eval_only_groups": eg,
        "gate_thresholds": got,
        "gate_labels": dict(g["labels"]),
        "forbid_lowering_bar": True,
        "bootstrap": {"unit": str(b6["bootstrap"]["unit"]),
                      "n": int(b6["bootstrap"]["n"]),
                      "samples": int(b6["bootstrap"]["samples"]),
                      "seed": int(b6["bootstrap"]["seed"])},
        "mmd_lambda_frozen": lam,
        "training_hyper_frozen": {
            "max_epochs": int(cfg["training"]["max_epochs"]),
            "early_stop_metric": str(cfg["training"]["early_stop_metric"]),
            "early_stop_patience": int(cfg["training"]["early_stop_patience"]),
            "weight_decay": float(cfg["training"]["weight_decay"]),
            "finetune_lr": float(cfg["transfer"]["finetune_lr"]),
            "batch_size": int(cfg["pretrain"]["batch_size"])},
        "b5_frozen_conclusion": {
            "verdict": str(b5["verdict"]),
            "engineering_recommendation":
                str(b5["ENGINEERING_RECOMMENDATION"]),
            "must_not_be_overwritten": True},
        "b5_b6_relationship": dict(b6["b5_b6_relationship"]),
        "final_conclusion_allowed_values": allowed,
        "damage_baseline_in_main_table":
            bool(b6["damage_baseline"]["keep_in_main_table"]),
        "engineering_recommendation_priority":
            list(b6["engineering_recommendation"]["priority_order"]),
        "sensitivity_forbid_trend_line_fit":
            bool(b6["sensitivity"]["forbid_trend_line_fit"]),
        "exit_rules": dict(er),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(f"[freeze] protocol 已冻结 -> {out.relative_to(ROOT)}")
    print(f"[freeze] protocol_sha256 = {rec['protocol_sha256']}")
    print(f"[freeze] config_sha256   = {rec['config_sha256']}")
    print(f"[freeze] 正式 seed = {seeds} (与已观察 seed 无交集)")
    print(f"[freeze] 输入 schema = {rec['input_schema']['formal']} "
          f"n_features={rec['input_schema']['n_features']} (由 {b3x_verdict} 决定)")
    print(f"[freeze] 标签档 = {levels}, 每档保留全部 {n_cen} 条 censored")
    print(f"[freeze] PRIMARY = n_event_labeled {pri}; SECONDARY = {sec} "
          f"(不得推翻 primary)")
    print(f"[freeze] subset_seed = {rec['subset_selection']['subset_seed']}, "
          f"bin edges = {edges} (取自 B2.1, 未重算)")
    print(f"[freeze] bin 可用 event 数 (B2.1 train) = {avail}")
    print(f"[freeze] Gate 十条阈值与 protocol.md §15 一致; "
          f"mmd_lambda = {lam} (冻结)")
    print(f"[freeze] B5 结论 {b5['verdict']} 仍为终局, 不得被 B6 覆盖")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
