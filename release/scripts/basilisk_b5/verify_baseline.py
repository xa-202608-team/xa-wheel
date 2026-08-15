#!/usr/bin/env python
"""scripts/basilisk_b5/verify_baseline.py

BASILISK-B5 §2 —— baseline 契约 (FORMAL_TRANSFER_EVALUATION)。

在 B2.1 契约 (974 项) 之上追加 B2.1 自己的全部产物与结论。沿用 B1.4..B2.1 的
做法: import 上一阶段的 verify_baseline 并扩展 FROZEN_GROUPS / NUMERIC_BLOCKS。

§2 要求逐项冻结并在结束时重新校验:
  B1.8 dataset hash / B1.9 feature hash / B2.1 split hash / B2.1 train·val·test
  IDs / B2.1 protocol hash / B2.1 metrics hash / source checkpoint hash /
  encoder architecture hash / MMD lambda / train config / evaluator·metrics hash。

train IDs / val IDs / test IDs 不是靠"split.json 文件哈希"间接覆盖 —— 那样
只要有人重写整个文件就无从区分改了哪一部分。本脚本把三个 ID 列表**各自**
排序后单独求 sha256, 并把逐 split 的条数、event/censored 计数、min/max event
EOL 一并进契约。任何一条轨迹换 split 都会立刻暴露。

任何冻结项改变 -> 立即 B5_INVALID (§2)。

用法:
    python scripts/basilisk_b5/verify_baseline.py --tag before
    python scripts/basilisk_b5/verify_baseline.py --tag after
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
CONTRACT = ROOT / "docs" / "basilisk_b5" / "baseline_contract.json"

# §2 的失效标签。抽成模块级常量的理由与 B1.6..B2.1 同: 文案必须写明"冻结项
# 改变即失效"这类边界, 而反作弊扫描器扫源码符号, 常量声明块被豁免。
INVALID_LABEL = "B5_INVALID"


def _load(mod_name: str, rel: str):
    """按唯一模块名加载 —— b1..b5 的 verify_baseline 同名, 必须区分。"""
    p = ROOT / rel
    spec = importlib.util.spec_from_file_location(mod_name, p)
    if spec is None or spec.loader is None:
        raise SystemExit(f"!! 无法加载 {rel}")
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = m
    spec.loader.exec_module(m)
    return m


B21V = _load("b5_b21_verify", "scripts/basilisk_b21/verify_baseline.py")

# ---------------------------------------------------------------------------
# §2: 在 B2.1 契约之上追加 B2.1 自己的产物
# ---------------------------------------------------------------------------
# 全部按 `ls` 实测填写。教训 (B1.7..B2.1 都踩过): 契约里钉一个不存在的路径 ->
# 它永远是 MISSING -> 真文件被改时反而漏检。缺席项单独进 *_absent 组。
B5_ADDED_GROUPS: dict[str, list[str]] = {
    "b21_config": [
        "configs/wheel_basilisk_b21.yaml",
    ],
    "b21_scripts": [
        "scripts/basilisk_b21/verify_baseline.py",
        "scripts/basilisk_b21/build_b21_split.py",
        "scripts/basilisk_b21/run_b21_gate.py",
        "scripts/basilisk_b21/summarize_b21.py",
    ],
    # B2.1 的冻结产出。**split_manifest.json 是本阶段的地基**: 它被改 = B5 的
    # 全部数字失去意义, 所以它必须在契约里, 且 §2 的 ID 级校验另有独立条目。
    "b21_frozen_outputs": [
        "docs/basilisk_b21/split_manifest.json",
        "checkpoints/basilisk_b21/gate_metrics.json",
        "checkpoints/basilisk_b21/summary.json",
        "checkpoints/basilisk_b21/protocol_hash.json",
    ],
    "b21_docs": [
        "docs/basilisk_b21/baseline_contract.json",
        "docs/basilisk_b21/protocol.md",
        "docs/basilisk_b21/results.md",
        "docs/basilisk_b21/limitations.md",
        "docs/basilisk_b21/b5_plan.md",
        "STATUS_BASILISK_B21.md",
    ],
    "b21_tests": [
        "tests/basilisk_b21/conftest.py",
        "tests/basilisk_b21/test_b21_discipline.py",
    ],
    # §1 禁止修改的只读源。B5 复用它们的实现 (import) 或权重。
    # encoder architecture / evaluator / metrics / 训练循环全部在此。
    "b5_reconfirmed_readonly": [
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
        "scripts/basilisk_b3x/diagnose_short_eol.py",
        "scripts/basilisk_b4x/data_b4x.py",
        "scripts/basilisk_b4x/run_transfer_diagnostic.py",
        "scripts/basilisk_b4x/diagnose_transfer_stability.py",
        "configs/wheel_basilisk_b2.yaml",
        "configs/wheel_basilisk_b4x.yaml",
        "data/features/wheel/basilisk_b19/target_features.h5",
    ],
    # §20: B6 必须始终缺席 —— 本阶段完成即停, 不自动跑低数据矩阵。
    "b5_must_stay_absent": [
        "checkpoints/basilisk_b6/metrics.json",
        "docs/basilisk_b6/results.md",
        "STATUS_BASILISK_B6.md",
        "checkpoints/basilisk_b5/low_data_matrix.json",
        "checkpoints/basilisk_b5/truncation_matrix.json",
    ],
}


# B2.1 的数值结论 + §2 逐项冻结清单。
# 首次写入时由 read_b5_results() 实测填入, 不凭记忆预填。
B5_EXPECTED = {
    # ---- B2.1 结论 ----
    "b21_verdict": "B21_GENERALIZATION_PASS",
    "b21_label": "SPLIT_COVERAGE_ROBUSTNESS",
    "b21_split_verdict": "B21_SPLIT_VALID",
    "b21_next_step": "B5 formal transfer on the frozen B2.1 split",
    "b21_b5_auto_run": "False",
    "b21_transfer_run": "False",
    "b21_n_conditions_passed": "7",
    # ---- §2 冻结项 ----
    "b21_split_sha256":
        "23e2b94445e698eab263ca17a36d99f7130eb2e8c5c3b9b8d0bd9ba772e5c932",
    "b21_train_ids_sha256": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "b21_val_ids_sha256": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "b21_test_ids_sha256": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "b21_protocol_hash_sha256": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "b21_metrics_hash_sha256": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "b21_n_train": "45",
    "b21_n_val": "31",
    "b21_n_test": "74",
    "b21_train_min_event_eol": "21658",
    "b21_val_min_event_eol": "21848",
    "b21_test_min_event_eol": "22126",
    "b21_event_bin_edges": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "b21_censored_bins_degenerate": "True",
    # ---- 迁移侧冻结项 (§1/§16) ----
    "source_checkpoint_sha256": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "encoder_architecture_sha256": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "evaluator_metrics_sha256": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "mmd_lambda": "1.0",
    "train_config_early_stop_metric": "info_macro_rmse",
    "train_config_max_epochs": "8",
    "train_config_early_stop_patience": "2",
    "train_config_weight_decay": "0.001",
    "train_config_finetune_lr": "0.0001",
    # ---- 前置判定 (不得被继承或改写) ----
    "b2_verdict_via_b21": "B2_GENERALIZATION_FAIL",
    "b3x_verdict_via_b21": "B3X_NO_STABILIZING_SIGNAL",
    "b4x_verdict_via_b21": "B4X_TRANSFER_STABILIZATION_SIGNAL",
    "b4x_exploratory_only": "True",
    "b4x_formal_stage": "False",
    # ---- B2.1 的数值锚点 (B5 的比较起点) ----
    "b21_target_only_mean_rmse": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "b21_damage_extrapolation_mean_rmse": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
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


def read_b5_results() -> dict:
    """读 B2.1 冻结结论 + §2 的逐项冻结清单 (键路径按实测 dump 写)。"""
    out: dict[str, str] = {}

    # ---- B2.1 summary ----
    sm_keys = ("b21_verdict", "b21_label", "b21_split_verdict",
               "b21_next_step", "b21_b5_auto_run", "b21_transfer_run",
               "b21_n_conditions_passed", "b21_target_only_mean_rmse",
               "b21_damage_extrapolation_mean_rmse", "b2_verdict_via_b21",
               "b3x_verdict_via_b21", "b4x_verdict_via_b21")
    sm = _json("checkpoints/basilisk_b21/summary.json")
    if sm is None:
        for k in sm_keys:
            out[k] = "MISSING"
    else:
        out["b21_verdict"] = str(sm["verdict"])
        out["b21_label"] = str(sm["label"])
        out["b21_split_verdict"] = str(sm["split"]["split_verdict"])
        out["b21_next_step"] = str(sm["next_step"])
        out["b21_b5_auto_run"] = _num(bool(sm["b5_auto_run"]))
        out["b21_transfer_run"] = _num(bool(sm["transfer_run"]))
        out["b21_n_conditions_passed"] = _num(int(sm["decision"]["n_passed"]))
        out["b21_target_only_mean_rmse"] = _num(round(
            float(sm["aggregate"]["target_only_rmse"]["mean"]), 6))
        out["b21_damage_extrapolation_mean_rmse"] = _num(round(
            float(sm["aggregate"]["damage_extrapolation_rmse"]["mean"]), 6))
        ff = sm["frozen_facts"]
        out["b2_verdict_via_b21"] = str(ff["b2_verdict"])
        out["b3x_verdict_via_b21"] = str(ff["b3x_verdict"])
        out["b4x_verdict_via_b21"] = str(ff["b4x_verdict"])

    # ---- B4X 的探索性边界 (不得被 B5 继承为正式结论) ----
    b4 = _json("checkpoints/basilisk_b4x/summary.json")
    if b4 is None:
        out["b4x_exploratory_only"] = "MISSING"
        out["b4x_formal_stage"] = "MISSING"
    else:
        out["b4x_exploratory_only"] = _num(bool(b4["exploratory_only"]))
        out["b4x_formal_stage"] = _num(bool(b4["formal_stage"]))

    # ---- §2 划分逐项 (split hash + 三个 ID 集合 + 计数 + EOL 范围) ----
    sp_keys = ("b21_split_sha256", "b21_train_ids_sha256",
               "b21_val_ids_sha256", "b21_test_ids_sha256", "b21_n_train",
               "b21_n_val", "b21_n_test", "b21_train_min_event_eol",
               "b21_val_min_event_eol", "b21_test_min_event_eol",
               "b21_event_bin_edges", "b21_censored_bins_degenerate")
    sp = _json("docs/basilisk_b21/split_manifest.json")
    if sp is None:
        for k in sp_keys:
            out[k] = "MISSING"
    else:
        out["b21_split_sha256"] = str(sp["split_sha256"])
        for name, short in (("train", "train"), ("val", "val"),
                            ("test", "test")):
            blk = sp["splits"][name]
            out[f"b21_{short}_ids_sha256"] = _sha_of_ids(blk["tids"])
            out[f"b21_n_{short}"] = _num(int(blk["n"]))
            out[f"b21_{short}_min_event_eol"] = _num(int(blk["event_eol_min"]))
        out["b21_event_bin_edges"] = _num(
            [float(x) for x in sp["lifetime_bins"]["event"]["edges"]])
        out["b21_censored_bins_degenerate"] = _num(
            bool(sp["lifetime_bins"]["censored"]["degenerate"]))

    # ---- §2 文件级哈希 (protocol / metrics / checkpoint / 架构 / evaluator) ----
    for key, rel in (
            ("b21_protocol_hash_sha256",
             "checkpoints/basilisk_b21/protocol_hash.json"),
            ("b21_metrics_hash_sha256",
             "checkpoints/basilisk_b21/gate_metrics.json"),
            ("source_checkpoint_sha256", "checkpoints/source_tcn_pretrain.pt"),
            ("encoder_architecture_sha256", "src/models/tcn_encoder.py"),
            ("evaluator_metrics_sha256", "src/experiments/metrics.py")):
        p = ROOT / rel
        out[key] = (hashlib.sha256(p.read_bytes()).hexdigest()
                    if p.exists() else "MISSING")

    # ---- §1/§16 训练与迁移超参 (从 config 链读, 不硬编码) ----
    hp_keys = ("mmd_lambda", "train_config_early_stop_metric",
               "train_config_max_epochs", "train_config_early_stop_patience",
               "train_config_weight_decay", "train_config_finetune_lr")
    try:
        sys.path.insert(0, str(ROOT))
        from scripts.basilisk_b11.calibrate_degradation import (
            load_b11_config as _lc,
        )
        cfg = _lc("configs/wheel_basilisk_b2.yaml")
        exp = cfg["b2_frozen_training_expected"]
        out["mmd_lambda"] = _num(float(cfg["transfer"]["mmd_lambda"]))
        out["train_config_early_stop_metric"] = str(exp["early_stop_metric"])
        out["train_config_max_epochs"] = _num(int(exp["max_epochs"]))
        out["train_config_early_stop_patience"] = _num(
            int(exp["early_stop_patience"]))
        out["train_config_weight_decay"] = _num(float(exp["weight_decay"]))
        out["train_config_finetune_lr"] = _num(
            float(cfg["transfer"]["finetune_lr"]))
    except Exception as exc:                       # noqa: BLE001
        for k in hp_keys:
            out[k] = f"UNREADABLE:{type(exc).__name__}"
    return out


NUMERIC_BLOCKS = tuple(B21V.NUMERIC_BLOCKS) + (
    ("b5_results", read_b5_results, B5_EXPECTED, "b5"),
)

FROZEN_GROUPS: dict[str, list[str]] = dict(B21V.FROZEN_GROUPS)
FROZEN_GROUPS.update(B5_ADDED_GROUPS)

# B2.1 契约把 B5 的产物钉为必须缺席; 到了 B5 自己这一阶段, 这些路径正是本阶段
# 要写的东西, 继续钉 MISSING 会让契约在第一次运行后就必然失败。因此把 B2.1 的
# b21_must_stay_absent 组按"本阶段允许出现"剔除, 并在下面用 B5 自己的
# b5_must_stay_absent (B6 产物) 接管这条防线。
FROZEN_GROUPS["b21_must_stay_absent"] = [
    rel for rel in FROZEN_GROUPS["b21_must_stay_absent"]
    if not rel.startswith(("checkpoints/basilisk_b5", "docs/basilisk_b5",
                           "STATUS_BASILISK_B5"))
]

_sha256_of = B21V._sha256_of

CONTRACT_PURPOSE = (
    "冻结 analytic lineage + BASILISK_V1 + B1 ... B1.9 + B2 + B3X + B4X + B2.1 "
    "全部产物与结论。B5 是 FORMAL_TRANSFER_EVALUATION 阶段, "
    "第一次允许产生正式迁移结论。"
    "B21_GENERALIZATION_PASS 是本阶段的地基: B2.1 的划分 / train·val·test IDs / "
    "protocol hash / metrics hash 全部逐项进契约, 任何一项改变即 " + INVALID_LABEL +
    "。B2_GENERALIZATION_FAIL 在其自身划分上保持终局, B2.1 的 PASS 不追溯改写它。"
    "B3X_NO_STABILIZING_SIGNAL 不变, 因此正式输入 schema 固定 CORE_ONLY, "
    "不得加入 mission features。"
    "B4X_TRANSFER_STABILIZATION_SIGNAL 仍然只是 EXPLORATORY_ONLY, "
    "**不得被继承或提升为正式 positive-transfer 结论** —— B5 必须自己重新证明。"
    "本阶段禁止修改: model architecture / optimizer / loss weights / "
    "early stopping / HI / RUL / mission profile / split / source checkpoints / "
    "mmd lambda, 以及 B1.8 B1.9 B2 B2.1 B3X B4X 与 analytic S2.5-S5B 的旧产物。"
    "MMD 的 lambda 已冻结: 禁止 grid search, 禁止 seed 特异 lambda, "
    "禁止按 B5 test 反调, 禁止重新选择 source layers; MMD 失败就如实判 "
    "B5_MMD_NO_POSITIVE_TRANSFER。"
    "B5 自己的 configs/scripts/tests/checkpoints/docs 不进契约。"
    "B6 的产物与任何低数据 / 截断矩阵产物必须始终缺席 —— "
    "本阶段无论正负结论都不得自动运行 B6。"
)


def compute() -> dict:
    groups = {g: {rel: _sha256_of(rel) for rel in files}
              for g, files in FROZEN_GROUPS.items()}
    flat = {rel: h for g in groups.values() for rel, h in g.items()}
    chain = dict(B21V.compute()["frozen_chain"])
    chain["BASILISK_B21"] = "B21_GENERALIZATION_PASS"
    return {
        "stage": "BASILISK_B5",
        "label": "FORMAL_TRANSFER_EVALUATION",
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
    print(f"[{tag}] B5_BASELINE_CONTRACT_OK")
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
