#!/usr/bin/env python
"""scripts/basilisk_b5/freeze_protocol.py

BASILISK-B5 §9/§17 —— 在**任何 B5 数字产生之前**冻结 protocol 与 config。

写出 checkpoints/basilisk_b5/protocol_hash.json, 内含 protocol.md 与
wheel_basilisk_b5.yaml 的 sha256, 以及从 config 抽出的全部 Gate 阈值。之后
summarize_b5.py 只能按这份哈希里的阈值判定 —— 出数字后改阈值会让哈希对不上。

同时硬断言:
  * B5 的输出目录里还没有任何 metrics —— 保证"protocol 先于数字";
  * 正式 seed 与禁用 seed 无交集 (§5);
  * 输入 schema 为 CORE_ONLY 且与 B3X 判定一致 (§3);
  * mmd_lambda 与 B2 冻结值一致 (§16);
  * Gate 阈值等于 protocol.md 里写死的那八条 (§9)。

用法:
    python scripts/basilisk_b5/freeze_protocol.py --config configs/wheel_basilisk_b5.yaml
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

# §9 的八条 Gate 阈值。这里写成模块级常量, 与 protocol.md §9 的表格逐条对应;
# freeze 时与 config 交叉校验, 两边不一致就拒绝冻结 —— 防止"文档写 4/5, 代码
# 跑 3/5"这种只有读代码才能发现的偏差。
GATE_SPEC_FROM_PROTOCOL = {
    "min_improve_count": 4,
    "n_seeds": 5,
    "require_mean_positive": True,
    "require_median_positive": True,
    "require_ci_lower_positive": True,
    "corr_tol": 0.02,
    "require_catastrophic_not_worse": True,
    "warning_miss_tol": 0.05,
    "min_lifetime_bins_gain_positive": 2,
    "n_lifetime_bins": 3,
}

FREEZE_MEANING = (
    "本文件的存在意味着: B5 的判定规则在看到任何 B5 数字之前已经固定。"
    "summarize_b5.py 必须按此处记录的阈值判定; 若 config 的阈值与此处不符, "
    "说明有人在出数字后改了门槛, 应判 B5_INVALID 而不是采信新阈值。"
    "PSR / HIGH_VARIANCE_WARNING 是诊断项, 明确不在 Gate 内 (§17) —— "
    "出数字后往 Gate 里加条件, 无论多合理都是事后调门槛。"
)


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wheel_basilisk_b5.yaml")
    a = ap.parse_args()

    cfg = load_b11_config(a.config)
    b5 = cfg["b5"]
    paths = cfg["paths"]
    proto = ROOT / cfg["protocol"]["path"]
    out = ROOT / cfg["protocol"]["hash_path"]

    if not proto.exists():
        raise SystemExit(f"!! protocol 缺失: {proto} —— 必须先写 protocol")

    # --- 硬断言 1: 冻结必须发生在任何数字之前 ---
    for key in ("metrics_json", "gain_json", "lifetime_json", "warning_json",
                "summary_json"):
        p = ROOT / paths[key]
        if p.exists():
            raise SystemExit(
                f"!! {p.relative_to(ROOT)} 已存在 —— protocol 必须在出数字前冻结。"
                " 若要重跑, 请先删除 B5 的输出再重新 freeze。")

    # --- 硬断言 2: 正式 seed 全新 (§5) ---
    seeds = [int(s) for s in b5["seeds"]]
    forbidden = {int(s) for s in b5["forbid_reused_seeds"]}
    overlap = sorted(set(seeds) & forbidden)
    if overlap:
        raise SystemExit(f"!! 正式 seed 与开发 seed 重叠 {overlap} —— B5_INVALID")
    if len(set(seeds)) != len(seeds):
        raise SystemExit(f"!! 正式 seed 有重复: {seeds}")

    # --- 硬断言 3: 输入 schema 与 B3X 判定一致 (§3) ---
    isch = b5["input_schema"]
    b3x = json.loads((ROOT / paths["b3x_summary_json"]).read_text("utf-8"))
    b3x_verdict = str(b3x["verdict"])
    want = str(isch["rule"][b3x_verdict])
    if str(isch["formal"]).lower() != want.lower():
        raise SystemExit(
            f"!! schema 不一致: B3X={b3x_verdict} 应得 {want}, "
            f"config 写 {isch['formal']} —— B5_INVALID")

    # --- 硬断言 4: B2.1 判定与划分哈希 (§2) ---
    # 两个层级的判定必须分别核: manifest 里的 verdict 是**划分级**
    # (B21_SPLIT_VALID, 由 build_b21_split 写), 阶段级判定 B21_GENERALIZATION_PASS
    # 在 summary.json。只核其中一个都会漏掉另一半。
    man = json.loads((ROOT / paths["b21_split_json"]).read_text("utf-8"))
    if str(man["verdict"]) != "B21_SPLIT_VALID":
        raise SystemExit(f"!! B2.1 划分级判定 {man['verdict']} 不是 "
                         f"B21_SPLIT_VALID —— B5_INVALID")
    if str(man["split_sha256"]) != str(b5["split"]["expected_sha256"]):
        raise SystemExit("!! B2.1 split hash 不符 —— B5_INVALID")
    b21 = json.loads((ROOT / paths["b21_summary_json"]).read_text("utf-8"))
    if str(b21["verdict"]) != str(b5["split"]["require_b21_verdict"]):
        raise SystemExit(f"!! B2.1 阶段判定 {b21['verdict']} != 要求值 "
                         f"{b5['split']['require_b21_verdict']} —— B5_INVALID")
    if str(b21["split"]["split_sha256"]) != str(man["split_sha256"]):
        raise SystemExit("!! B2.1 summary 与 manifest 的 split hash 互不一致")

    # --- 硬断言 5: mmd_lambda 冻结 (§16) ---
    lam = float(cfg["transfer"]["mmd_lambda"])
    b2cfg = load_b11_config("configs/wheel_basilisk_b2.yaml")
    lam_b2 = float(b2cfg["transfer"]["mmd_lambda"])
    if abs(lam - lam_b2) > 0:
        raise SystemExit(f"!! mmd_lambda {lam} != B2 冻结值 {lam_b2} —— B5_INVALID")

    # --- 硬断言 6: Gate 阈值 == protocol.md §9 ---
    g = b5["gate"]
    got = {
        "min_improve_count": int(g["min_improve_count"]),
        "n_seeds": len(seeds),
        "require_mean_positive": bool(g["require_mean_positive"]),
        "require_median_positive": bool(g["require_median_positive"]),
        "require_ci_lower_positive": bool(g["require_ci_lower_positive"]),
        "corr_tol": float(g["corr_tol"]),
        "require_catastrophic_not_worse": bool(g["require_catastrophic_not_worse"]),
        "warning_miss_tol": float(g["warning_miss_tol"]),
        "min_lifetime_bins_gain_positive": int(g["min_lifetime_bins_gain_positive"]),
        "n_lifetime_bins": int(g["n_lifetime_bins"]),
    }
    if got != GATE_SPEC_FROM_PROTOCOL:
        diff = {k: (GATE_SPEC_FROM_PROTOCOL[k], got[k])
                for k in got if got[k] != GATE_SPEC_FROM_PROTOCOL[k]}
        raise SystemExit(f"!! Gate 阈值与 protocol.md §9 不一致: {diff}")

    # --- 硬断言 7: 只有三个训练组, 不得新增迁移方法 (§4) ---
    tg = list(b5["trained_groups"])
    if tg != ["target_only", "source_finetune", "source_mmd_finetune"]:
        raise SystemExit(f"!! 训练组不符 §4: {tg}")
    if not bool(b5["forbid_new_transfer_methods"]):
        raise SystemExit("!! forbid_new_transfer_methods 必须为 true")

    rec = {
        "stage": "BASILISK_B5",
        "label": str(b5["label"]),
        "freeze_meaning": FREEZE_MEANING,
        "frozen_before_any_b5_number": True,
        "protocol_path": str(cfg["protocol"]["path"]),
        "protocol_sha256": sha256_file(proto),
        "config_path": a.config,
        "config_sha256": sha256_file(ROOT / a.config),
        "questions": dict(b5["questions"]),
        "formal_seeds": seeds,
        "forbidden_seeds": sorted(forbidden),
        "input_schema": {"formal": str(isch["formal"]),
                         "decided_by_b3x_verdict": b3x_verdict,
                         "forbid_mission_features":
                             bool(isch["forbid_mission_features"])},
        "split": {"sha256": str(man["split_sha256"]),
                  "b21_split_verdict": str(man["verdict"]),
                  "b21_stage_verdict": str(b21["verdict"]),
                  "n": {k: int(man["splits"][k]["n"])
                        for k in ("train", "val", "test")}},
        "lifetime_bins_from_b21": {
            "edges": [float(x) for x in man["lifetime_bins"]["event"]["edges"]],
            "names": list(man["lifetime_bins"]["event"]["names"]),
            "forbid_recompute_from_b5_test": True},
        "trained_groups": tg,
        "eval_only_groups": list(b5["eval_only_groups"]),
        "gate_thresholds": got,
        "gate_labels": dict(g["labels"]),
        "forbid_lowering_bar": bool(g["forbid_lowering_bar"]),
        "bootstrap": {"unit": str(b5["bootstrap"]["unit"]),
                      "n": int(b5["bootstrap"]["n"]),
                      "samples": int(b5["bootstrap"]["samples"]),
                      "seed": int(b5["bootstrap"]["seed"])},
        "mmd_lambda_frozen": lam,
        "training_hyper_frozen": {
            "max_epochs": int(cfg["training"]["max_epochs"]),
            "early_stop_metric": str(cfg["training"]["early_stop_metric"]),
            "early_stop_patience": int(cfg["training"]["early_stop_patience"]),
            "weight_decay": float(cfg["training"]["weight_decay"]),
            "finetune_lr": float(cfg["transfer"]["finetune_lr"]),
            "batch_size": int(cfg["pretrain"]["batch_size"])},
        "psr_diagnostic_not_in_gate": {
            "in_gate": bool(b5["output_stability"]["in_gate"]),
            "high_variance_psr": float(b5["output_stability"]["high_variance_psr"]),
            "label": str(b5["output_stability"]["high_variance_label"]),
            "reason": str(b5["output_stability"]["reason"])},
        "damage_baseline_in_main_table":
            bool(b5["damage_baseline"]["keep_in_main_table"]),
        "engineering_recommendation_separate":
            bool(b5["engineering_recommendation"]["separate_from_transfer_verdict"]),
        "exit_rules": dict(b5["exit_rules"]),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(f"[freeze] protocol 已冻结 -> {out.relative_to(ROOT)}")
    print(f"[freeze] protocol_sha256 = {rec['protocol_sha256']}")
    print(f"[freeze] config_sha256   = {rec['config_sha256']}")
    print(f"[freeze] 正式 seed = {seeds} (与开发 seed 无交集)")
    print(f"[freeze] 输入 schema = {rec['input_schema']['formal']} "
          f"(由 {b3x_verdict} 决定)")
    print(f"[freeze] Gate 八条阈值与 protocol.md §9 一致; mmd_lambda = {lam} (冻结)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
