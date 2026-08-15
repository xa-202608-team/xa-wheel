#!/usr/bin/env python
"""scripts/basilisk_b6/verify_baseline.py

BASILISK-B6 §3 —— baseline 契约 (FAILURE_LABEL_SCARCITY_FORMAL_MATRIX)。

在 B5 契约 (1067 项) 之上追加 **B5 自己的全部产物与结论**。沿用 B1.4..B5 的做法:
import 上一阶段的 verify_baseline 并扩展 FROZEN_GROUPS / NUMERIC_BLOCKS。

§3 要求至少逐项冻结:
  B1.8 dataset hash / B1.9 feature hash / B2.1 split hash / B2.1 train·val·test
  IDs / **B5 protocol hash / B5 config hash / B5 results hash** /
  source checkpoint hash / encoder architecture / MMD lambda /
  metrics implementation / damage baseline implementation /
  target-only architecture。before / after 必须逐项一致。

§22 的接力: B5 契约把 B6 的产物钉为必须缺席; 到了 B6 自己这一阶段那些路径正是要写
的东西, 继续钉 MISSING 会让契约在第一次运行后必然失败。因此把 B5 的
b5_must_stay_absent 组按"本阶段允许出现"剔除, 并用 B6 自己的
b6_must_stay_absent (B7/S7 数字修改产物 + B8/S8 打包产物) 接管这条防线。

任何冻结项改变 -> 立即 B6_INVALID (§3)。

用法:
    python scripts/basilisk_b6/verify_baseline.py --tag before
    python scripts/basilisk_b6/verify_baseline.py --tag after
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "docs" / "basilisk_b6" / "baseline_contract.json"

# §3 的失效标签。抽成模块级常量的理由与 B1.6..B5 同: 文案必须写明"冻结项改变即
# 失效"这类边界, 而反作弊扫描器扫源码符号, 常量声明块被豁免。
INVALID_LABEL = "B6_INVALID"


def _load(mod_name: str, rel: str):
    """按唯一模块名加载 —— b1..b6 的 verify_baseline 同名, 必须区分。"""
    p = ROOT / rel
    spec = importlib.util.spec_from_file_location(mod_name, p)
    if spec is None or spec.loader is None:
        raise SystemExit(f"!! 无法加载 {rel}")
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = m
    spec.loader.exec_module(m)
    return m


B5V = _load("b6_b5_verify", "scripts/basilisk_b5/verify_baseline.py")

# ---------------------------------------------------------------------------
# §3: 在 B5 契约之上追加 B5 自己的产物
# ---------------------------------------------------------------------------
# 全部按 `ls` 实测填写。教训 (B1.7..B5 都踩过): 契约里钉一个不存在的路径 ->
# 它永远是 MISSING -> 真文件被改时反而漏检。缺席项单独进 *_absent 组。
B6_ADDED_GROUPS: dict[str, list[str]] = {
    "b5_config": [
        "configs/wheel_basilisk_b5.yaml",
    ],
    "b5_scripts": [
        "scripts/basilisk_b5/verify_baseline.py",
        "scripts/basilisk_b5/freeze_protocol.py",
        "scripts/basilisk_b5/data_b5.py",
        "scripts/basilisk_b5/run_formal_transfer.py",
        "scripts/basilisk_b5/analyze_paired_gain.py",
        "scripts/basilisk_b5/analyze_lifetime_bins.py",
        "scripts/basilisk_b5/analyze_warning_metrics.py",
        "scripts/basilisk_b5/summarize_b5.py",
    ],
    # B5 的冻结产出。**summary.json / formal_metrics.json 是 B6 的比较基线**:
    # 它们被改 = "B5 的结论不得被覆盖"这条纪律失守。
    "b5_frozen_outputs": [
        "checkpoints/basilisk_b5/protocol_hash.json",
        "checkpoints/basilisk_b5/formal_metrics.json",
        "checkpoints/basilisk_b5/paired_gain.json",
        "checkpoints/basilisk_b5/lifetime_bins.json",
        "checkpoints/basilisk_b5/warning_metrics.json",
        "checkpoints/basilisk_b5/summary.json",
    ],
    "b5_docs": [
        "docs/basilisk_b5/baseline_contract.json",
        "docs/basilisk_b5/protocol.md",
        "docs/basilisk_b5/results.md",
        "docs/basilisk_b5/lifetime_bin_results.md",
        "docs/basilisk_b5/warning_results.md",
        "docs/basilisk_b5/limitations.md",
        "docs/basilisk_b5/REPRODUCE.md",
        "STATUS_BASILISK_B5.md",
    ],
    "b5_tests": [
        "tests/basilisk_b5/conftest.py",
        "tests/basilisk_b5/test_b5_baseline_contract.py",
        "tests/basilisk_b5/test_b5_split_frozen.py",
        "tests/basilisk_b5/test_b5_transfer_protocol.py",
        "tests/basilisk_b5/test_b5_training_fairness.py",
        "tests/basilisk_b5/test_b5_paired_statistics.py",
        "tests/basilisk_b5/test_b5_no_posthoc_tuning.py",
        "tests/basilisk_b5/test_b5_conclusion_discipline.py",
    ],
    # §2 禁止修改的只读源。B6 复用它们的实现 (import) 或权重。
    # §3 点名要求的 metrics implementation / damage baseline implementation /
    # target-only architecture 全部在此。
    "b6_reconfirmed_readonly": [
        "checkpoints/source_tcn_pretrain.pt",
        "src/models/tcn_encoder.py",
        "src/transfer/adapter.py",
        "src/transfer/mmd.py",
        "src/transfer/train_transfer.py",
        "src/experiments/metrics.py",
        "src/experiments/run_groups.py",
        "src/baselines/physical_extrap.py",
        "src/baselines/trivial.py",
        "scripts/basilisk_b2/data_b2.py",
        "scripts/basilisk_b2/eval_b2.py",
        "scripts/basilisk_b2/train_b2.py",
        "scripts/basilisk_b2/baselines_b2.py",
        "scripts/basilisk_b2/run_gate.py",
        "scripts/basilisk_b21/run_b21_gate.py",
        "scripts/basilisk_b3x/diagnose_short_eol.py",
        "scripts/basilisk_b4x/data_b4x.py",
        "scripts/basilisk_b4x/diagnose_transfer_stability.py",
        "data/features/wheel/basilisk_b19/target_features.h5",
    ],
    # §22 替换 guard: B7/S7 的数字修改产物必须不存在; B8/S8 打包尚未运行。
    # 本阶段完成即停 —— 不自动跑 B7 出图/报告, 不自动跑 B8 打包/Docker。
    "b6_must_stay_absent": [
        "checkpoints/basilisk_b7/metrics.json",
        "checkpoints/basilisk_b7/formal_metrics.json",
        "checkpoints/basilisk_b7/summary.json",
        "docs/basilisk_b7/results.md",
        "STATUS_BASILISK_B7.md",
        "checkpoints/basilisk_s7/metrics.json",
        "docs/basilisk_s7/results.md",
        "STATUS_BASILISK_S7.md",
        "checkpoints/basilisk_b8/package_manifest.json",
        "docs/basilisk_b8/packaging.md",
        "STATUS_BASILISK_B8.md",
        "checkpoints/basilisk_s8/package_manifest.json",
        "docs/basilisk_s8/packaging.md",
        "STATUS_BASILISK_S8.md",
    ],
}


# B5 的数值结论 + §3 的逐项冻结清单。
# 首次写入时由 read_b6_results() 实测填入, 不凭记忆预填。
B6_EXPECTED = {
    # ---- B5 正式结论 (终局, 不得被 B6 覆盖) ----
    "b5_verdict": "B5_NO_POSITIVE_TRANSFER",
    "b5_label": "FORMAL_TRANSFER_EVALUATION",
    "b5_ft_verdict": "B5_FT_NO_POSITIVE_TRANSFER",
    "b5_mmd_verdict": "B5_MMD_NO_POSITIVE_TRANSFER",
    "b5_verdict_combination": "D",
    "b5_engineering_recommendation": "damage_extrapolation",
    "b5_ft_positive_transfer": "False",
    "b5_mmd_positive_transfer": "False",
    "b5_ft_n_passed": "1",
    "b5_mmd_n_passed": "3",
    "b5_b6_auto_run": "False",
    # ---- B5 数值锚点 (B6 的比较起点; 全量目标域) ----
    "b5_target_only_mean_rmse": "0.241024",
    "b5_source_finetune_mean_rmse": "0.242659",
    "b5_source_mmd_finetune_mean_rmse": "0.241607",
    "b5_damage_extrapolation_mean_rmse": "0.054312",
    "b5_const_mean_info_mean_rmse": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "b5_gain_ft_mean": "-0.001635",
    "b5_gain_mmd_mean": "-0.000583",
    "b5_gain_ft_improve_count": "1",
    "b5_gain_mmd_improve_count": "2",
    "b5_gain_ft_ci95_lower": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "b5_gain_mmd_ci95_lower": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "b5_formal_seeds": "[112, 113, 114, 115, 116]",
    # ---- §3 点名的三项 B5 冻结哈希 ----
    "b5_protocol_hash_sha256": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "b5_config_hash_sha256": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "b5_results_hash_sha256": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "b5_summary_hash_sha256": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    # ---- §3 划分逐项 (与 B5 契约同口径, 在 B6 再核一次) ----
    "b21_split_sha256_via_b6":
        "23e2b94445e698eab263ca17a36d99f7130eb2e8c5c3b9b8d0bd9ba772e5c932",
    "b21_train_ids_sha256_via_b6": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "b21_val_ids_sha256_via_b6": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "b21_test_ids_sha256_via_b6": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "b21_n_train_event": "21",
    "b21_n_train_censored": "24",
    "b21_event_bin_edges_via_b6": "[26846.0, 37395.0]",
    # ---- §3 实现级冻结项 ----
    "source_checkpoint_sha256_via_b6": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "encoder_architecture_sha256_via_b6": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "metrics_implementation_sha256": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "damage_baseline_implementation_sha256":
        "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "target_only_architecture_sha256": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "mmd_lambda_via_b6": "1.0",
    "b5_n_features": "12",
    # ---- 前置判定 (不得被继承或改写) ----
    "b2_verdict_via_b6": "B2_GENERALIZATION_FAIL",
    "b3x_verdict_via_b6": "B3X_NO_STABILIZING_SIGNAL",
    "b4x_verdict_via_b6": "B4X_TRANSFER_STABILIZATION_SIGNAL",
    "b21_verdict_via_b6": "B21_GENERALIZATION_PASS",
}


def _json(rel: str):
    p = ROOT / rel
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _num(v) -> str:
    return repr(v).strip("'\"")


def _sha_of_ids(ids) -> str:
    """一个 split 的 ID 集合哈希。排序后再哈希 —— 顺序不是契约内容, 成员是。"""
    blob = "\n".join(sorted(str(x) for x in ids)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def read_b6_results() -> dict:
    """读 B5 冻结结论 + §3 的逐项冻结清单 (键路径按实测 dump 写)。"""
    out: dict[str, str] = {}

    # ---- B5 summary ----
    sm_keys = ("b5_verdict", "b5_label", "b5_ft_verdict", "b5_mmd_verdict",
               "b5_verdict_combination", "b5_engineering_recommendation",
               "b5_ft_positive_transfer", "b5_mmd_positive_transfer",
               "b5_ft_n_passed", "b5_mmd_n_passed", "b5_b6_auto_run",
               "b5_target_only_mean_rmse", "b5_source_finetune_mean_rmse",
               "b5_source_mmd_finetune_mean_rmse",
               "b5_damage_extrapolation_mean_rmse",
               "b5_const_mean_info_mean_rmse", "b5_formal_seeds",
               "b2_verdict_via_b6", "b3x_verdict_via_b6",
               "b4x_verdict_via_b6", "b21_verdict_via_b6", "b5_n_features")
    sm = _json("checkpoints/basilisk_b5/summary.json")
    if sm is None:
        for k in sm_keys:
            out[k] = "MISSING"
    else:
        out["b5_verdict"] = str(sm["verdict"])
        out["b5_label"] = str(sm["label"])
        out["b5_ft_verdict"] = str(sm["ft_verdict"])
        out["b5_mmd_verdict"] = str(sm["mmd_verdict"])
        out["b5_verdict_combination"] = str(
            sm["verdict_combination"]["combination"])
        out["b5_engineering_recommendation"] = str(
            sm["ENGINEERING_RECOMMENDATION"])
        out["b5_ft_positive_transfer"] = _num(
            bool(sm["positive_transfer"]["ft"]))
        out["b5_mmd_positive_transfer"] = _num(
            bool(sm["positive_transfer"]["mmd"]))
        out["b5_ft_n_passed"] = _num(
            int(sm["decision"]["source_finetune"]["n_passed"]))
        out["b5_mmd_n_passed"] = _num(
            int(sm["decision"]["source_mmd_finetune"]["n_passed"]))
        out["b5_b6_auto_run"] = _num(bool(sm["b6_auto_run"]))
        mt = sm["main_table"]
        for key, g in (("b5_target_only_mean_rmse", "target_only"),
                       ("b5_source_finetune_mean_rmse", "source_finetune"),
                       ("b5_source_mmd_finetune_mean_rmse",
                        "source_mmd_finetune"),
                       ("b5_damage_extrapolation_mean_rmse",
                        "damage_extrapolation"),
                       ("b5_const_mean_info_mean_rmse", "const_mean_info")):
            out[key] = _num(round(float(mt[g]["info_macro_rmse"]), 6))
        out["b5_formal_seeds"] = _num([int(s) for s in sm["formal_seeds"]])
        ff = sm["frozen_facts"]
        out["b2_verdict_via_b6"] = str(ff["b2_verdict"])
        out["b3x_verdict_via_b6"] = str(ff["b3x_verdict"])
        out["b4x_verdict_via_b6"] = str(ff["b4x_verdict"])
        out["b21_verdict_via_b6"] = str(ff["b21_verdict"])

    # ---- §4 输入宽度: B6 必须沿用 B5 的 n_features = 12 ----
    fm = _json("checkpoints/basilisk_b5/formal_metrics.json")
    out["b5_n_features"] = ("MISSING" if fm is None else _num(
        int(fm["per_seed"][0]["target_only_meta"]["n_features"])))

    # ---- B5 paired gain (数值锚点) ----
    g_keys = ("b5_gain_ft_mean", "b5_gain_mmd_mean",
              "b5_gain_ft_improve_count", "b5_gain_mmd_improve_count",
              "b5_gain_ft_ci95_lower", "b5_gain_mmd_ci95_lower")
    gj = _json("checkpoints/basilisk_b5/paired_gain.json")
    if gj is None:
        for k in g_keys:
            out[k] = "MISSING"
    else:
        for pre, key in (("ft", "gain_ft"), ("mmd", "gain_mmd")):
            b = gj["gain"][key]
            out[f"b5_gain_{pre}_mean"] = _num(round(float(b["mean"]), 6))
            out[f"b5_gain_{pre}_improve_count"] = _num(
                int(b["improve_count"]))
            out[f"b5_gain_{pre}_ci95_lower"] = _num(
                round(float(b["ci95_lower"]), 6))

    # ---- §3 划分逐项 (在 B6 独立再核一次, 不只依赖 B5 契约的传递) ----
    sp_keys = ("b21_split_sha256_via_b6", "b21_train_ids_sha256_via_b6",
               "b21_val_ids_sha256_via_b6", "b21_test_ids_sha256_via_b6",
               "b21_n_train_event", "b21_n_train_censored",
               "b21_event_bin_edges_via_b6")
    sp = _json("docs/basilisk_b21/split_manifest.json")
    if sp is None:
        for k in sp_keys:
            out[k] = "MISSING"
    else:
        out["b21_split_sha256_via_b6"] = str(sp["split_sha256"])
        for name in ("train", "val", "test"):
            out[f"b21_{name}_ids_sha256_via_b6"] = _sha_of_ids(
                sp["splits"][name]["tids"])
        out["b21_n_train_event"] = _num(int(sp["splits"]["train"]["n_event"]))
        out["b21_n_train_censored"] = _num(
            int(sp["splits"]["train"]["n_censored"]))
        out["b21_event_bin_edges_via_b6"] = _num(
            [float(x) for x in sp["lifetime_bins"]["event"]["edges"]])

    # ---- §3 文件级哈希 (B5 protocol / config / results + 实现级冻结项) ----
    for key, rel in (
            ("b5_protocol_hash_sha256", "docs/basilisk_b5/protocol.md"),
            ("b5_config_hash_sha256", "configs/wheel_basilisk_b5.yaml"),
            ("b5_results_hash_sha256",
             "checkpoints/basilisk_b5/formal_metrics.json"),
            ("b5_summary_hash_sha256", "checkpoints/basilisk_b5/summary.json"),
            ("source_checkpoint_sha256_via_b6",
             "checkpoints/source_tcn_pretrain.pt"),
            ("encoder_architecture_sha256_via_b6",
             "src/models/tcn_encoder.py"),
            ("metrics_implementation_sha256", "src/experiments/metrics.py"),
            ("damage_baseline_implementation_sha256",
             "src/baselines/physical_extrap.py"),
            ("target_only_architecture_sha256",
             "src/experiments/run_groups.py")):
        p = ROOT / rel
        out[key] = (hashlib.sha256(p.read_bytes()).hexdigest()
                    if p.exists() else "MISSING")

    # ---- §2 MMD lambda (从 config 链读, 不硬编码) ----
    try:
        sys.path.insert(0, str(ROOT))
        from scripts.basilisk_b11.calibrate_degradation import (
            load_b11_config as _lc,
        )
        cfg = _lc("configs/wheel_basilisk_b2.yaml")
        out["mmd_lambda_via_b6"] = _num(float(cfg["transfer"]["mmd_lambda"]))
    except Exception as exc:                       # noqa: BLE001
        out["mmd_lambda_via_b6"] = f"UNREADABLE:{type(exc).__name__}"
    return out


NUMERIC_BLOCKS = tuple(B5V.NUMERIC_BLOCKS) + (
    ("b6_results", read_b6_results, B6_EXPECTED, "b6"),
)

FROZEN_GROUPS: dict[str, list[str]] = dict(B5V.FROZEN_GROUPS)
FROZEN_GROUPS.update(B6_ADDED_GROUPS)

# §22 的接力: B5 契约把 B6 的产物钉为必须缺席; 到了 B6 自己这一阶段, 那些路径
# 正是本阶段要写的东西, 继续钉 MISSING 会让契约在第一次运行后就必然失败。因此把
# B5 的 b5_must_stay_absent 组按"本阶段允许出现"剔除, 并用上面的
# b6_must_stay_absent (B7/S7 数字修改产物 + B8/S8 打包产物) 接管这条防线。
FROZEN_GROUPS["b5_must_stay_absent"] = [
    rel for rel in FROZEN_GROUPS["b5_must_stay_absent"]
    if not rel.startswith(("checkpoints/basilisk_b6", "docs/basilisk_b6",
                           "STATUS_BASILISK_B6"))
]

_sha256_of = B5V._sha256_of

CONTRACT_PURPOSE = (
    "冻结 analytic lineage + BASILISK_V1 + B1 ... B1.9 + B2 + B3X + B4X + B2.1 "
    "+ **B5** 全部产物与结论。B6 是 FAILURE_LABEL_SCARCITY_FORMAL_MATRIX 阶段, "
    "飞轮线最后一个允许产生新的核心实验数字的阶段。"
    "B5_NO_POSITIVE_TRANSFER 是终局: B5 的 protocol hash / config hash / "
    "results hash / summary hash 全部逐项进契约, 任何一项改变即 " + INVALID_LABEL +
    "。**B5 的结论不得因 B6 被覆盖** —— 即使某个低标签档出现正增益, 那也只能构成 "
    "conditional low-label transfer benefit, 绝不是 overall positive transfer。"
    "B2_GENERALIZATION_FAIL 在其自身划分上保持终局; "
    "B3X_NO_STABILIZING_SIGNAL 不变, 因此正式输入 schema 固定 CORE_ONLY "
    "(n_features = 12), 不得恢复 mission-feature arm; "
    "B4X_TRANSFER_STABILIZATION_SIGNAL 仍然只是 EXPLORATORY_ONLY。"
    "本阶段禁止修改: model architecture / optimizer / loss weights / "
    "early stopping / HI / RUL / mission profile / split / source checkpoints / "
    "mmd lambda / metrics implementation / damage baseline implementation, "
    "以及 B1.8 B1.9 B2 B2.1 B3X B4X B5 与 analytic S2.5-S5B 的旧产物。"
    "被削减的只有 event-observed failure-labelled trajectories, "
    "train 中全部 24 条 censored 轨迹在每一档都完整保留。"
    "B6 自己的 configs/scripts/tests/checkpoints/docs 不进契约。"
    "B7/S7 的数字修改产物与 B8/S8 的打包产物必须始终缺席 —— "
    "本阶段无论正负结论都不得自动运行 B7 或 B8。"
)


def compute() -> dict:
    groups = {g: {rel: _sha256_of(rel) for rel in files}
              for g, files in FROZEN_GROUPS.items()}
    flat = {rel: h for g in groups.values() for rel, h in g.items()}
    chain = dict(B5V.compute()["frozen_chain"])
    chain["BASILISK_B5"] = "B5_NO_POSITIVE_TRANSFER"
    return {
        "stage": "BASILISK_B6",
        "label": "FAILURE_LABEL_SCARCITY_FORMAL_MATRIX",
        "purpose": CONTRACT_PURPOSE,
        "invalid_label": INVALID_LABEL,
        "frozen_chain": chain,
        "env": {"python": sys.version.split()[0],
                "platform": platform.platform()},
        "groups": groups,
        "flat_sha256": flat,
        "n_files": len(flat),
        **{k: v for name, reader, expected, prefix in NUMERIC_BLOCKS
           for k, v in ((name, reader()), (f"{prefix}_expected", dict(expected)))},
    }


def verify(tag: str) -> int:
    if not CONTRACT.exists():
        print(f"[{tag}] 契约文件不存在: {CONTRACT}")
        return 1
    ref = json.loads(CONTRACT.read_text(encoding="utf-8"))
    bad = []
    for rel, h_ref in ref["flat_sha256"].items():
        h_now = _sha256_of(rel)
        if h_now != h_ref:
            bad.append((rel, h_ref, h_now))

    n_numeric = 0
    for name, reader, expected, prefix in NUMERIC_BLOCKS:
        now = reader()
        for k, h_ref in ref[name].items():
            n_numeric += 1
            if now.get(k) != h_ref:
                bad.append((f"[{prefix}_result] {k}", h_ref, now.get(k, "ABSENT")))
        for k, h_exp in ref.get(f"{prefix}_expected", expected).items():
            n_numeric += 1
            if now.get(k) != h_exp:
                bad.append((f"[{prefix}_expected] {k}", h_exp,
                            now.get(k, "ABSENT")))

    n_total = ref["n_files"] + n_numeric
    print(f"[{tag}] 契约项数 {n_total}, 不变 {n_total - len(bad)}")
    for rel, a, b in bad:
        print(f"[{tag}] CHANGED(禁止) {rel}\n         "
              f"{str(a)[:24]} -> {str(b)[:24]}")
    if bad:
        print(f"[{tag}] {INVALID_LABEL} —— 冻结项改变, 必须立即停止")
        return 1
    print(f"[{tag}] B6_BASELINE_CONTRACT_OK")
    return 0


def write(tag: str) -> int:
    CONTRACT.parent.mkdir(parents=True, exist_ok=True)
    c = compute()
    # 任何 *_EXPECTED 里的 PLACEHOLDER_FILLED_ON_FIRST_WRITE 都在首次写入时
    # 用实测值填入 —— 不凭记忆预填。填完之后 verify 逐字比对, 照样能失败。
    n_filled = 0
    for name, _, _, prefix in NUMERIC_BLOCKS:
        exp = c[f"{prefix}_expected"]
        for k, v in list(exp.items()):
            if v == "PLACEHOLDER_FILLED_ON_FIRST_WRITE":
                exp[k] = c[name][k]
                n_filled += 1
    CONTRACT.write_text(json.dumps(c, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    n_num = sum(len(c[name]) + len(c[f"{p}_expected"])
                for name, _, _, p in NUMERIC_BLOCKS)
    print(f"[{tag}] 已写入契约 {CONTRACT.relative_to(ROOT)}: "
          f"{c['n_files']} 文件 + {n_num} 数值项 = {c['n_files'] + n_num} 项")
    if n_filled:
        print(f"[{tag}] 首次写入时按实测填入 {n_filled} 个 placeholder")
    miss = [r for r, h in c["flat_sha256"].items() if h == "MISSING"]
    if miss:
        print(f"[{tag}] !! 以下文件缺失 (契约会把 MISSING 钉死, 请确认是预期的):")
        for r in miss:
            print(f"        {r}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    tag = a.tag or ("write" if a.write else "verify")
    if a.write:
        return write(tag)
    if a.verify:
        return verify(tag)
    # 默认: 首次建契约, 之后一律核验。--tag after 绝不会覆盖 before 的契约。
    return write(tag) if not CONTRACT.exists() else verify(tag)


if __name__ == "__main__":
    raise SystemExit(main())
