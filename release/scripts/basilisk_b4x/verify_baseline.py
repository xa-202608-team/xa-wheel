#!/usr/bin/env python
"""scripts/basilisk_b4x/verify_baseline.py

BASILISK-B4X §2 —— baseline 契约 (POST_B2_FAIL_EXPLORATORY)。

在 B3X 契约 (838 项) 之上追加 B3X 自己的产物与结论。沿用 B1.4..B3X 的做法:
import 上一阶段的 verify_baseline 并扩展 FROZEN_GROUPS / NUMERIC_BLOCKS。

§2 要求记录的对象在 B3X 契约里已全部落位并继承过来:
  B1.8 dataset hash / B1.9 feature hash / B2 split hash / B2 train·val·test IDs /
  B2 config hash / B2 metrics hash / B2 verdict / B2 checkpoint hashes /
  short-EOL diagnostic IDs (19,109,111)。

B4X 额外钉死的前置事实:
  * B3X verdict = B3X_NO_STABILIZING_SIGNAL  -> b3x_results
  * §10 由该 verdict 推出的 B4X 输入 schema = core_only -> b3x_results
  * B3X 逐 seed 两臂 RMSE -> b3x_results (防止事后重跑 B3X 改数)
  * B3X 全部 checkpoints / docs / scripts / tests -> 文件组 (只读)
  * source checkpoint (§0 禁改) -> b4x_reconfirmed_readonly
  * 正式 B4 / 迁移结论类产物 -> b4x_must_stay_absent, 两次都必须 MISSING

用法:
    python scripts/basilisk_b4x/verify_baseline.py --tag before
    python scripts/basilisk_b4x/verify_baseline.py --tag after
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "docs" / "basilisk_b4x" / "baseline_contract.json"


def _load(mod_name: str, rel: str):
    """按唯一模块名加载 —— b1..b4x 的 verify_baseline 同名, 必须区分。"""
    p = ROOT / rel
    spec = importlib.util.spec_from_file_location(mod_name, p)
    if spec is None or spec.loader is None:
        raise SystemExit(f"!! 无法加载 {rel}")
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = m
    spec.loader.exec_module(m)
    return m


B3XV = _load("b4x_b3x_verify", "scripts/basilisk_b3x/verify_baseline.py")

# ---------------------------------------------------------------------------
# §2: 在 B3X 契约之上追加 B3X 自己的产物
# ---------------------------------------------------------------------------
# 全部按 `ls` 实测填写。教训 (B1.7..B3X 都踩过): 契约里钉一个不存在的路径
# -> 它永远是 MISSING -> 真文件被改时反而漏检。缺席项单独进 *_absent 组。
B4X_ADDED_GROUPS: dict[str, list[str]] = {
    "b3x_config": [
        "configs/wheel_basilisk_b3x.yaml",
    ],
    "b3x_scripts": [
        "scripts/basilisk_b3x/verify_baseline.py",
        "scripts/basilisk_b3x/data_b3x.py",
        "scripts/basilisk_b3x/run_mission_ablation.py",
        "scripts/basilisk_b3x/diagnose_short_eol.py",
        "scripts/basilisk_b3x/summarize_b3x.py",
    ],
    # B3X 的冻结产出 —— B4X 输入 schema 的唯一依据 (§10)。必须逐字不变,
    # 否则"schema 由 B3X verdict 决定"这条规则就可以被事后重写。
    "b3x_frozen_outputs": [
        "checkpoints/basilisk_b3x/mission_ablation_metrics.json",
        "checkpoints/basilisk_b3x/short_eol_diagnostics.json",
        "checkpoints/basilisk_b3x/summary.json",
        "checkpoints/basilisk_b3x/protocol_hash.json",
        "checkpoints/basilisk_b3x/ablation_raw.npz",
    ],
    "b3x_docs": [
        "docs/basilisk_b3x/baseline_contract.json",
        "docs/basilisk_b3x/protocol.md",
        "docs/basilisk_b3x/mission_ablation_results.md",
        "docs/basilisk_b3x/short_eol_diagnostics.md",
        "docs/basilisk_b3x/limitations.md",
        "STATUS_BASILISK_B3X.md",
    ],
    "b3x_tests": [
        "tests/basilisk_b3x/conftest.py",
        "tests/basilisk_b3x/test_b3x_discipline.py",
    ],
    # §0 禁止修改的只读源。B4X 复用它们的实现 (import) 或权重, 哈希变动会
    # 直接改变 B4X 的训练语义 —— 必须钉死。source checkpoint 尤其关键:
    # §0 明列 "禁止修改 source checkpoint"。
    "b4x_reconfirmed_readonly": [
        "checkpoints/source_tcn_pretrain.pt",
        "src/transfer/train_transfer.py",
        "src/transfer/adapter.py",
        "src/transfer/mmd.py",
        "src/models/tcn_encoder.py",
        "src/experiments/metrics.py",
        "scripts/basilisk_b2/data_b2.py",
        "scripts/basilisk_b2/eval_b2.py",
        "scripts/basilisk_b2/train_b2.py",
        "configs/wheel_basilisk_b2.yaml",
        "data/features/wheel/basilisk_b19/target_features.h5",
        "data/features/wheel/schema_v1/source_features.h5",
    ],
    # §17 三个允许标签之外的任何正式化产物必须始终缺席。
    "b4x_must_stay_absent": [
        "checkpoints/basilisk_b4/metrics.json",
        "docs/basilisk_b4/results.md",
        "STATUS_BASILISK_B4.md",
        "checkpoints/basilisk_b4x/formal_verdict.json",
        "checkpoints/basilisk_b4x/positive_transfer_proven.json",
    ],
}


# B3X 的数值结论 —— B4X 的前置事实, 逐项钉死。
# 首次写入时由 read_b3x_results() 实测填入, 不凭记忆预填。
B3X_EXPECTED = {
    "b3x_verdict": "B3X_NO_STABILIZING_SIGNAL",
    "b3x_label": "POST_B2_FAIL_EXPLORATORY",
    "b3x_b4x_input_schema": "core_only",
    "b3x_n_conditions_passed": "3",
    "b3x_n_conditions": "4",
    "b3x_n_collapsed_recovered": "0",
    "b3x_seeds": "[72, 73, 74, 75, 76]",
    "b3x_short_eol_subset": "[19, 109, 111]",
    "b3x_recovered_threshold": "0.2646356195158311",
    "b3x_forbidden_label_used": "False",
    "b3x_mean_gain": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "b3x_median_gain": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "b3x_improve_count": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "b3x_short_eol_mean_gain": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "b3x_normal_eol_mean_gain": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
}

# B3X 逐 seed 两臂主指标 —— 防止事后重跑 B3X 改数后再讲 B4X 故事。
B3X_PER_SEED_KEYS: list[str] = []
for _s in (72, 73, 74, 75, 76):
    for _arm in ("core", "mission"):
        _k = f"b3x_rmse_{_arm}_seed{_s}"
        B3X_PER_SEED_KEYS.append(_k)
        B3X_EXPECTED[_k] = "PLACEHOLDER_FILLED_ON_FIRST_WRITE"


def _json(rel: str):
    p = ROOT / rel
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _num(v) -> str:
    return repr(v).strip("'\"")


def read_b3x_results() -> dict:
    """读 B3X 的冻结数值结论 (键路径按实测 dump 写, 不凭记忆)。"""
    out: dict[str, str] = {}
    sm_keys = ("b3x_verdict", "b3x_label", "b3x_b4x_input_schema",
               "b3x_n_conditions_passed", "b3x_n_conditions",
               "b3x_n_collapsed_recovered", "b3x_seeds",
               "b3x_short_eol_subset", "b3x_recovered_threshold",
               "b3x_forbidden_label_used", "b3x_mean_gain", "b3x_median_gain",
               "b3x_improve_count", "b3x_short_eol_mean_gain",
               "b3x_normal_eol_mean_gain")
    sm = _json("checkpoints/basilisk_b3x/summary.json")
    if sm is None:
        for k in sm_keys:
            out[k] = "MISSING"
    else:
        out["b3x_verdict"] = str(sm["verdict"])
        out["b3x_label"] = str(sm["label_discipline"])
        out["b3x_b4x_input_schema"] = str(sm["b4x_input_schema"])
        d = sm["decision"]
        out["b3x_n_conditions_passed"] = _num(d["n_passed"])
        out["b3x_n_conditions"] = _num(d["n_conditions"])
        ans = sm["answers"]
        g = ans["D_gain_mission"]
        out["b3x_mean_gain"] = _num(g["overall"]["mean"])
        out["b3x_median_gain"] = _num(g["overall"]["median"])
        out["b3x_improve_count"] = _num(g["improve_count"])
        out["b3x_seeds"] = _num(list(sm["seeds"]))
        out["b3x_short_eol_subset"] = _num(list(sm["short_eol_subset"]))
        rec = ans["A_collapsed_seeds_recovered"]
        out["b3x_n_collapsed_recovered"] = _num(rec["n_recovered"])
        out["b3x_recovered_threshold"] = _num(rec["threshold"])
        # 只允许 §9 的两个标签。这一项恒为 False, 若变 True 说明用了禁止标签。
        out["b3x_forbidden_label_used"] = _num(
            str(sm["verdict"]) == str(sm["forbidden_formal_label"]))
        sub = ans["B_where_the_gain_comes_from"]
        out["b3x_short_eol_mean_gain"] = _num(sub["short_eol_gain"]["mean"])
        out["b3x_normal_eol_mean_gain"] = _num(sub["normal_eol_gain"]["mean"])

    ab = _json("checkpoints/basilisk_b3x/mission_ablation_metrics.json")
    if ab is None:
        for k in B3X_PER_SEED_KEYS:
            out[k] = "MISSING"
    else:
        by: dict[tuple[str, int], float] = {}
        for r in ab["per_seed"]:
            s = int(r["seed"])
            by[("core", s)] = r["core_only"]["info_macro_rmse"]
            by[("mission", s)] = r["core_plus_mission"]["info_macro_rmse"]
        for k in B3X_PER_SEED_KEYS:
            arm, _, sd = k.replace("b3x_rmse_", "").partition("_seed")
            key = (arm, int(sd))
            out[k] = _num(by[key]) if key in by else "MISSING"
    return out


NUMERIC_BLOCKS = tuple(B3XV.NUMERIC_BLOCKS) + (
    ("b3x_results", read_b3x_results, B3X_EXPECTED, "b3x"),
)

FROZEN_GROUPS: dict[str, list[str]] = dict(B3XV.FROZEN_GROUPS)
FROZEN_GROUPS.update(B4X_ADDED_GROUPS)

_sha256_of = B3XV._sha256_of

# 契约 purpose 文案。抽成模块级常量的理由与 B1.6..B3X 同: 文案必须写明
# "不是正式阶段"这类边界, 而反作弊扫描器会扫源码里的这些符号;
# 常量声明块被扫描器豁免, 函数体内的字符串不会。
CONTRACT_PURPOSE = (
    "冻结 analytic lineage + BASILISK_V1 + B1 ... B1.9 + B2 + B3X 全部产物与结论。"
    "B4X 是 POST_B2_FAIL_EXPLORATORY 诊断阶段, 不是正式 B4, 不是正式迁移证据。"
    "它只回答一个问题: source_finetune / source_mmd_finetune 究竟是只比不稳定的"
    "target_only 更稳, 还是确实提供了跨域增益? "
    "B2 的判定 B2_GENERALIZATION_FAIL 保持终局, 本阶段不覆盖、不重新解释。"
    "B2 的 split / IDs / 早停 / max_epochs / weight_decay / loss 权重 / HI / RUL / "
    "删失契约 / mission profile / source checkpoint / MMD lambda 一律不得修改。"
    "输入 schema 由 B3X verdict 按 §10 规则唯一确定 (B3X_NO_STABILIZING_SIGNAL "
    "-> core_only), 该规则在 B3X 出数之前已写入 protocol。"
    "catastrophic 阈值只允许来自 target_only 的 validation 轨迹 RMSE 中位数, "
    "禁止用 test 分布选阈值。新 seeds (92,93,94) 与 replay seeds (72,74,76) "
    "必须分开报告, 不得合并成一个 6-seed 平均。"
    "B2 与 B3X 的全部产物进契约 (必须逐字不变); B4X 自己的 checkpoints/docs 不进契约。"
    "正式 B4 标签类产物与任何 positive-transfer 断言类产物必须始终缺席。"
)


def compute() -> dict:
    groups = {g: {rel: _sha256_of(rel) for rel in files}
              for g, files in FROZEN_GROUPS.items()}
    flat = {rel: h for g in groups.values() for rel, h in g.items()}
    chain = dict(B3XV.compute()["frozen_chain"])
    chain["BASILISK_B3X"] = "B3X_NO_STABILIZING_SIGNAL"
    return {
        "stage": "BASILISK_B4X",
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
        print(f"[{tag}] B4X_BASELINE_CONTRACT_VIOLATION —— 必须立即停止")
        return 1
    print(f"[{tag}] B4X_BASELINE_CONTRACT_OK")
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
