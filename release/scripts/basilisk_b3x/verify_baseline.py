#!/usr/bin/env python
"""scripts/basilisk_b3x/verify_baseline.py

BASILISK-B3X §2 —— baseline 契约 (POST_B2_FAIL_EXPLORATORY)。

在 B2 契约 (743 项 = 319 文件 + 424 数值) 之上追加 B2 自己的产物与结论。
沿用 B1.4..B2 的做法: import 上一阶段的 verify_baseline 并扩展其
FROZEN_GROUPS / NUMERIC_BLOCKS —— 不复制粘贴。

§2 明列必须记录的对象, 逐项落位:
  * B1.8 dataset hash            -> 继承 (b18_results)
  * B1.9 feature hash            -> 继承 (b19_results)
  * B2 split hash                -> b2_results
  * B2 train / val / test IDs    -> b2_results (三个 ID 列表的 sha256)
  * B2 config hash               -> b2_results
  * B2 metrics hash              -> b2_results
  * B2 verdict                   -> b2_results
  * B2 checkpoint hashes         -> b2_frozen_outputs (文件组)
  * short-EOL diagnostic IDs     -> b2_results (19 / 109 / 111)

B3X 会训练模型, 因此:
  * B2 的全部产物进契约 (只读, 必须逐字不变) —— 这是"不得修改 B2 namespace"
    这条纪律的机器化核验;
  * B3X 自己的 checkpoints / docs 不进契约;
  * 正式 B3 标签类产物进 `b3x_must_stay_absent`, 两次都必须 MISSING。

用法:
    python scripts/basilisk_b3x/verify_baseline.py --tag before
    python scripts/basilisk_b3x/verify_baseline.py --tag after
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
CONTRACT = ROOT / "docs" / "basilisk_b3x" / "baseline_contract.json"


def _load(mod_name: str, rel: str):
    """按唯一模块名加载 —— b1..b3x 的 verify_baseline 同名, 必须区分。"""
    p = ROOT / rel
    spec = importlib.util.spec_from_file_location(mod_name, p)
    if spec is None or spec.loader is None:
        raise SystemExit(f"!! 无法加载 {rel}")
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = m
    spec.loader.exec_module(m)
    return m


B2V = _load("b3x_b2_verify", "scripts/basilisk_b2/verify_baseline.py")

# ---------------------------------------------------------------------------
# §2: 在 B2 契约之上追加 B2 自己的产物
# ---------------------------------------------------------------------------
# 全部按 `ls` 实测填写。教训 (B1.7/B1.8/B1.9/B2 都踩过): 契约里钉一个不存在的
# 路径 -> 它永远是 MISSING -> 真文件被改时反而漏检。缺席项单独进 *_absent 组。
B3X_ADDED_GROUPS: dict[str, list[str]] = {
    "b2_config": [
        "configs/wheel_basilisk_b2.yaml",
    ],
    "b2_scripts": [
        "scripts/basilisk_b2/verify_baseline.py",
        "scripts/basilisk_b2/build_split.py",
        "scripts/basilisk_b2/data_b2.py",
        "scripts/basilisk_b2/eval_b2.py",
        "scripts/basilisk_b2/baselines_b2.py",
        "scripts/basilisk_b2/train_b2.py",
        "scripts/basilisk_b2/run_gate.py",
    ],
    # B2 的冻结产出 —— B3X 的前置事实, 必须逐字不变。
    # 这一组就是"不得修改 B2 namespace"的机器化保证。
    "b2_frozen_outputs": [
        "checkpoints/basilisk_b2/split.json",
        "checkpoints/basilisk_b2/metrics.json",
        "checkpoints/basilisk_b2/protocol_hash.json",
    ],
    "b2_docs": [
        "docs/basilisk_b2/baseline_contract.json",
        "docs/basilisk_b2/protocol.md",
        "docs/basilisk_b2/results.md",
        "STATUS_BASILISK_B2.md",
    ],
    "b2_tests": [
        "tests/basilisk_b2/conftest.py",
        "tests/basilisk_b2/test_b2_split_protocol.py",
        "tests/basilisk_b2/test_b2_censor_contract.py",
        "tests/basilisk_b2/test_b2_input_contract.py",
        "tests/basilisk_b2/test_b2_gate_discipline.py",
        "tests/basilisk_b2/test_b2_baseline_contract.py",
        "tests/basilisk_b2/test_b2_evaluator_integrity.py",
    ],
    # §0 禁止修改的只读源。B3X 复用它们的实现 (import), 因此哈希变动
    # 会直接改变 B3X 的训练语义 —— 必须钉死。
    "b3x_reconfirmed_readonly": [
        "src/transfer/train_transfer.py",
        "src/transfer/adapter.py",
        "src/experiments/run_groups.py",
        "src/experiments/metrics.py",
        "src/baselines/physical_extrap.py",
        "src/models/tcn_encoder.py",
        "configs/wheel.yaml",
        "configs/wheel_basilisk_b19.yaml",
        "data/features/wheel/basilisk_b19/target_features.h5",
    ],
    # §9 禁止正式标签。以下两次都必须缺席 —— 它们的存在意味着
    # 探索性结论被提升成了正式 claim。
    "b3x_must_stay_absent": [
        "checkpoints/basilisk_b3/metrics.json",
        "docs/basilisk_b3/results.md",
        "STATUS_BASILISK_B3.md",
        "checkpoints/basilisk_b3x/formal_verdict.json",
    ],
}


def _ids_sha(lst) -> str:
    """ID 列表的顺序敏感 sha256 —— 划分被重排也算改动。"""
    h = hashlib.sha256()
    for t in lst:
        h.update(str(t).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


# B2 的数值结论 —— B3X 的前置事实, 逐项钉死。
# 首次写入时由 read_b2_results() 实测填入, 不凭记忆预填。
B2_EXPECTED = {
    "b2_verdict": "B2_GENERALIZATION_FAIL",
    "b2_n_conditions_passed": "4",
    "b2_n_conditions": "7",
    "b2_split_sha256":
        "079296e5fba206da6ba9dab1078a8eaa6957e37ea97d0782b7a3fa01164113ae",
    "b2_split_seed": "20260810",
    "b2_n_traj": "150",
    "b2_n_event_observed": "71",
    "b2_n_censored": "79",
    "b2_n_train": "44",
    "b2_n_val": "31",
    "b2_n_test": "75",
    "b2_train_ids_sha256": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "b2_val_ids_sha256": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "b2_test_ids_sha256": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "b2_hi_key": "hi_damage_obs",
    "b2_improve_count": "2",
    "b2_mean_paired_gain": "-0.31762785914640695",
    "b2_bootstrap_ci_lower": "-0.574033907616108",
    "b2_const_mean_info_rmse": "0.2886",
    "b2_seeds": "[72, 73, 74, 75, 76]",
    # §2 要求记录的 short-EOL 诊断 ID。写死在契约里, 使
    # "事后换一组更好看的轨迹" 这件事不可能悄悄发生。
    "b2_short_eol_diagnostic_ids": "[19, 109, 111]",
    "b2_collapsed_seeds": "[72, 74, 76]",
    "b2_normal_seeds": "[73, 75]",
    "b2_frozen_max_epochs": "8",
    "b2_frozen_patience": "2",
    "b2_frozen_weight_decay": "0.001",
    "b2_frozen_early_stop_metric": "info_macro_rmse",
    "b2_source_transfer_run": "False",
    "b2_truncation_applied": "False",
}

# B2 逐 seed 的主指标 —— B3X 的配对基准。任何一个变了都说明 B2 被重跑/篡改。
B2_PER_SEED_KEYS = ("b2_rmse_seed72", "b2_rmse_seed73", "b2_rmse_seed74",
                    "b2_rmse_seed75", "b2_rmse_seed76")
for _k in B2_PER_SEED_KEYS:
    B2_EXPECTED[_k] = "PLACEHOLDER_FILLED_ON_FIRST_WRITE"


def _json(rel: str):
    p = ROOT / rel
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _num(v) -> str:
    return repr(v).strip("'\"")


def read_b2_results() -> dict:
    """读 B2 的冻结数值结论 (键路径已实测 dump, 不凭记忆)。"""
    out: dict[str, str] = {}
    keys_from_split = ("b2_split_sha256", "b2_split_seed", "b2_n_traj",
                       "b2_n_event_observed", "b2_n_censored", "b2_n_train",
                       "b2_n_val", "b2_n_test", "b2_train_ids_sha256",
                       "b2_val_ids_sha256", "b2_test_ids_sha256")
    sp = _json("checkpoints/basilisk_b2/split.json")
    if sp is None:
        for k in keys_from_split:
            out[k] = "MISSING"
    else:
        out["b2_split_sha256"] = str(sp["split_sha256"])
        out["b2_split_seed"] = _num(sp["split_seed"])
        out["b2_n_traj"] = _num(sp["n_traj"])
        out["b2_n_event_observed"] = _num(sp["n_event_observed"])
        out["b2_n_censored"] = _num(sp["n_censored"])
        s = sp["splits"]
        out["b2_n_train"] = _num(s["train"]["n"])
        out["b2_n_val"] = _num(s["val"]["n"])
        out["b2_n_test"] = _num(s["test"]["n"])
        out["b2_train_ids_sha256"] = _ids_sha(s["train"]["tids"])
        out["b2_val_ids_sha256"] = _ids_sha(s["val"]["tids"])
        out["b2_test_ids_sha256"] = _ids_sha(s["test"]["tids"])

    keys_from_metrics = (("b2_verdict", "b2_n_conditions_passed",
                          "b2_n_conditions", "b2_hi_key", "b2_improve_count",
                          "b2_mean_paired_gain", "b2_bootstrap_ci_lower",
                          "b2_const_mean_info_rmse", "b2_seeds",
                          "b2_source_transfer_run", "b2_truncation_applied",
                          "b2_frozen_max_epochs", "b2_frozen_patience",
                          "b2_frozen_weight_decay",
                          "b2_frozen_early_stop_metric")
                         + B2_PER_SEED_KEYS)
    mt = _json("checkpoints/basilisk_b2/metrics.json")
    if mt is None:
        for k in keys_from_metrics:
            out[k] = "MISSING"
    else:
        d = mt["decision"]
        out["b2_verdict"] = str(mt["verdict"])
        out["b2_n_conditions_passed"] = _num(d["n_passed"])
        out["b2_n_conditions"] = _num(d["n_conditions"])
        out["b2_hi_key"] = str(mt["hi_key"])
        out["b2_improve_count"] = _num(
            sum(1 for r in mt["per_seed"] if r["paired"]["better_than_const"]))
        out["b2_mean_paired_gain"] = _num(d["mean_paired_gain"])
        out["b2_bootstrap_ci_lower"] = _num(d["bootstrap"]["ci_lower"])
        out["b2_const_mean_info_rmse"] = _num(round(
            float(mt["per_seed"][0]["paired"]["const_mean_info"]), 4))
        out["b2_seeds"] = _num(list(mt["seeds"]))
        out["b2_source_transfer_run"] = _num(d["source_transfer_run"])
        out["b2_truncation_applied"] = _num(d["truncation_applied"])
        ft = mt["frozen_training"]
        out["b2_frozen_max_epochs"] = _num(ft["max_epochs"])
        out["b2_frozen_patience"] = _num(ft["early_stop_patience"])
        out["b2_frozen_weight_decay"] = _num(ft["weight_decay"])
        out["b2_frozen_early_stop_metric"] = str(ft["early_stop_metric"])
        by = {int(r["seed"]): r["paired"]["target_only"]
              for r in mt["per_seed"]}
        for k in B2_PER_SEED_KEYS:
            s = int(k.replace("b2_rmse_seed", ""))
            out[k] = _num(by[s]) if s in by else "MISSING"

    # short-EOL 诊断 ID 与 seed 分组来自 B3X config (声明), 但必须与 B2 metrics
    # 里的事实一致 —— 这里读 config 声明值, 由 tests 交叉核验其一致性。
    cfg_p = ROOT / "configs" / "wheel_basilisk_b3x.yaml"
    if not cfg_p.exists():
        for k in ("b2_short_eol_diagnostic_ids", "b2_collapsed_seeds",
                  "b2_normal_seeds"):
            out[k] = "MISSING"
    else:
        import yaml
        c = yaml.safe_load(cfg_p.read_text(encoding="utf-8"))["b3x"]
        out["b2_short_eol_diagnostic_ids"] = _num(list(c["short_eol_subset"]))
        out["b2_collapsed_seeds"] = _num(list(c["collapsed_seeds"]))
        out["b2_normal_seeds"] = _num(list(c["normal_seeds"]))
    return out


NUMERIC_BLOCKS = tuple(B2V.NUMERIC_BLOCKS) + (
    ("b2_results", read_b2_results, B2_EXPECTED, "b2"),
)

FROZEN_GROUPS: dict[str, list[str]] = dict(B2V.FROZEN_GROUPS)
FROZEN_GROUPS.update(B3X_ADDED_GROUPS)

_sha256_of = B2V._sha256_of

# 契约 purpose 文案。抽成模块级常量的理由与 B1.6..B2 同: 文案必须写明
# "不是正式阶段"这类边界, 而反作弊扫描器会扫源码里的这些符号;
# 常量声明块被扫描器豁免, 函数体内的字符串不会。
CONTRACT_PURPOSE = (
    "冻结 analytic lineage + BASILISK_V1 + B1 ... B1.9 + B2 全部产物与结论。"
    "B3X 是 POST_B2_FAIL_EXPLORATORY 诊断阶段, 不是正式 B3。"
    "它只回答一个问题: Basilisk mission features 是否能缓解 B2 在短寿命轨迹"
    "(tid 19/109/111) 上的失控? "
    "B2 的判定 B2_GENERALIZATION_FAIL 保持终局, 本阶段不覆盖、不重新解释。"
    "B2 的 split / IDs / 架构 / 早停 / max_epochs / weight_decay / loss 权重 / "
    "HI / RUL / 删失契约 / mission profile 一律不得修改; 两臂唯一差异是输入列。"
    "B2 的全部产物进契约 (必须逐字不变); B3X 自己的 checkpoints/docs 不进契约。"
    "正式 B3 标签类产物必须始终缺席。"
)


def compute() -> dict:
    groups = {g: {rel: _sha256_of(rel) for rel in files}
              for g, files in FROZEN_GROUPS.items()}
    flat = {rel: h for g in groups.values() for rel, h in g.items()}
    chain = dict(B2V.compute()["frozen_chain"])
    chain["BASILISK_B2"] = "B2_GENERALIZATION_FAIL"
    return {
        "stage": "BASILISK_B3X",
        "label": "POST_B2_FAIL_EXPLORATORY",
        "purpose": CONTRACT_PURPOSE,
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
        print(f"[{tag}] B3X_BASELINE_CONTRACT_VIOLATION —— 必须立即停止")
        return 1
    print(f"[{tag}] B3X_BASELINE_CONTRACT_OK")
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
