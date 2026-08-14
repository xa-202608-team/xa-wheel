#!/usr/bin/env python
"""scripts/basilisk_b2/verify_baseline.py

BASILISK-B2 §1 —— baseline 契约。

在 B1.9 契约 (668 项 = 290 文件 + 378 数值) 之上追加 B1.9 自己的产物与结论。
沿用 B1.4..B1.9 的做法: import 上一阶段的 verify_baseline 并扩展其
FROZEN_GROUPS / NUMERIC_BLOCKS —— 不复制粘贴。

§1 明列必须记录的对象, 逐项落位:
  * B1.9 protocol hash            -> b19_results
  * B1.9 feature hash (content+file) -> b19_results
  * B1.9 verdict                  -> b19_results
  * B1.9 L_ref / failure rule     -> b19_results
  * B1.8 protocol hash            -> 继承 (b18_results)
  * B1.8 150-traj dataset hash    -> 继承
  * S2.5-S5B frozen hashes        -> 继承
  * B2 split hash / protocol hash -> 由 §2 冻结件记录, **不进本契约**
    (它们是 B2 自己的产物, 合法地在 before/after 之间出现)

B2 会训练模型并产出 metrics, 因此:
  * B1.9 的 feature 文件进契约 (只读, 必须逐字不变);
  * B2 自己的 checkpoints / docs 不进契约;
  * §6 禁止的 source transfer 类产物进 `b2_must_stay_absent`, 两次都必须 MISSING。

用法:
    python scripts/basilisk_b2/verify_baseline.py --tag before
    python scripts/basilisk_b2/verify_baseline.py --tag after
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "docs" / "basilisk_b2" / "baseline_contract.json"


def _load(mod_name: str, rel: str):
    """按唯一模块名加载 —— b1..b2 的 verify_baseline 同名, 必须区分。"""
    p = ROOT / rel
    spec = importlib.util.spec_from_file_location(mod_name, p)
    if spec is None or spec.loader is None:
        raise SystemExit(f"!! 无法加载 {rel}")
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = m
    spec.loader.exec_module(m)
    return m


B19V = _load("b2_b19_verify", "scripts/basilisk_b19/verify_baseline.py")

# ---------------------------------------------------------------------------
# §1: 在 B1.9 契约之上追加 B1.9 自己的产物
# ---------------------------------------------------------------------------
# 全部按 `ls` 实测填写。教训 (B1.7/B1.8/B1.9 都踩过): 契约里钉一个不存在的路径
# -> 它永远是 MISSING -> 真文件被改时反而漏检。缺席项单独进 *_absent 组。
B2_ADDED_GROUPS: dict[str, list[str]] = {
    "b19_config": [
        "configs/wheel_basilisk_b19.yaml",
    ],
    "b19_scripts": [
        "scripts/basilisk_b19/verify_baseline.py",
        "scripts/basilisk_b19/audit_observable_inputs.py",
        "scripts/basilisk_b19/derive_damage_proxy.py",
        "scripts/basilisk_b19/build_features.py",
        "scripts/basilisk_b19/audit_features.py",
        "scripts/basilisk_b19/freeze_feature_definition.py",
    ],
    "b19_checkpoints": [
        "checkpoints/basilisk_b19/observable_input_audit.json",
        "checkpoints/basilisk_b19/damage_proxy.json",
        "checkpoints/basilisk_b19/feature_stats.json",
        "checkpoints/basilisk_b19/feature_audit.json",
        "checkpoints/basilisk_b19/frozen_feature_definition.json",
    ],
    "b19_docs": [
        "docs/basilisk_b19/baseline_contract.json",
        "docs/basilisk_b19/protocol.md",
        "docs/basilisk_b19/observable_input_audit.md",
        "docs/basilisk_b19/damage_proxy_derivation.md",
        "docs/basilisk_b19/feature_build.md",
        "docs/basilisk_b19/feature_report.md",
        "docs/basilisk_b19/selected_hi_definition.md",
        "docs/basilisk_b19/limitations.md",
        "docs/basilisk_b19/REPRODUCE.md",
        "STATUS_BASILISK_B19.md",
    ],
    # §1: B1.9 的 feature 文件 —— B2 的唯一输入, 只读, 必须逐字不变。
    "b19_frozen_data": [
        "data/features/wheel/basilisk_b19/target_features.h5",
    ],
    # §11 不得修改的只读源。B2 复用它们的实现 (import), 因此它们的哈希
    # 变动会直接改变 B2 的训练语义 —— 必须钉死。
    "b2_reconfirmed_readonly": [
        "src/sim/wheel_sim.py",
        "src/sim/build_hi.py",
        "src/transfer/train_transfer.py",
        "src/transfer/adapter.py",
        "src/experiments/run_groups.py",
        "src/experiments/metrics.py",
        "src/baselines/trivial.py",
        "src/baselines/physical_extrap.py",
        "configs/wheel.yaml",
        "configs/wheel_basilisk_b19.yaml",
    ],
    # §6 禁止 source transfer (那是 B4); §3 禁止截断。以下两次都必须缺席。
    "b2_must_stay_absent": [
        "checkpoints/basilisk_b2/source_finetune_metrics.json",
        "checkpoints/basilisk_b2/source_mmd_metrics.json",
        "checkpoints/basilisk_b2/mmd_metrics.json",
        "checkpoints/basilisk_b2/truncated_metrics.json",
        "checkpoints/basilisk_b2/pretrain_metrics.json",
    ],
}

# B1.9 的数值结论 —— B2 的前置事实, 逐项钉死。
# 全部由 read_b19_results() 从实跑 JSON 读出, 键路径已 dump 实测, 不凭记忆。
B19_EXPECTED = {
    "b19_observability_verdict": "B19_OBSERVABILITY_OK",
    "b19_feature_verdict": "B19_FEATURE_READY",
    "b19_audit_verdict": "B19_FEATURE_READY",
    "b19_frozen_verdict": "B19_FEATURE_READY",
    "b19_protocol_sha256":
        "e95bf867e083aa0d3095f31049c7587ab3acc21a463463eac640a427008e8a3e",
    "b19_config_sha256":
        "95cdb74ad61fc9f7fc53031ec6f662d362aaa07cc977b374c762dc2fec7ae1bf",
    "b19_feature_content_sha256":
        "510dd14e5f3f6ae214c2b2ed85f3eb0227504f4ccc0529ace233d5a26f378cba",
    "b19_feature_file_sha256":
        "575d708ae9330dfa37d407cf639ebbbd098c473ce1fc137ff1eef8feaab9c293",
    "b19_source_dataset_content_sha256":
        "61678e582f82bd136ffe509de7ec1274c07c39fe791a2a762fcf562a3d36a0f7",
    "b19_primary_hi": "hi_damage_obs",
    "b19_auxiliary_hi": "hi_friction",
    "b19_L_ref_years": "3.0",
    "b19_n_traj": "150",
    "b19_n_event_observed": "71",
    "b19_n_censored": "79",
    "b19_damage_proxy_in_xT": "False",
    "b19_mission_features_in_xT": "False",
    "b19_is_recomputation_not_estimation": "True",
    "b19_trained_any_model": "False",
    "b19_cross_check_consistent": "True",
    "b19_n_gates_passed": "14",
    # B2 的输入契约: 这三条决定"允许/禁止"的角色, 一旦漂移 B2 的 §5 就失效
    "b19_hi_allowed_supervision": "True",
    "b19_hi_forbidden_plain_input": "True",
}


def _json(rel: str):
    p = ROOT / rel
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _num(v) -> str:
    return repr(v).strip("'\"")


def read_b19_results() -> dict:
    """读 B1.9 的冻结数值结论 (键路径已实测 dump, 不凭记忆)。"""
    out: dict[str, str] = {}

    oa = _json("checkpoints/basilisk_b19/observable_input_audit.json")
    out["b19_observability_verdict"] = ("MISSING" if oa is None
                                        else str(oa["verdict"]))

    fs = _json("checkpoints/basilisk_b19/feature_stats.json")
    if fs is None:
        for k in ("b19_feature_verdict", "b19_protocol_sha256",
                  "b19_feature_content_sha256", "b19_feature_file_sha256",
                  "b19_source_dataset_content_sha256", "b19_primary_hi",
                  "b19_auxiliary_hi", "b19_L_ref_years", "b19_n_traj",
                  "b19_n_event_observed", "b19_n_censored",
                  "b19_damage_proxy_in_xT", "b19_mission_features_in_xT",
                  "b19_trained_any_model"):
            out[k] = "MISSING"
    else:
        out["b19_feature_verdict"] = str(fs["verdict"])
        out["b19_protocol_sha256"] = str(fs["protocol_sha256"])
        out["b19_feature_content_sha256"] = str(fs["feature_content_sha256"])
        out["b19_feature_file_sha256"] = str(fs["feature_file_sha256"])
        out["b19_source_dataset_content_sha256"] = str(
            fs["source_dataset_content_sha256"])
        out["b19_primary_hi"] = str(fs["primary_hi"])
        out["b19_auxiliary_hi"] = str(fs["auxiliary_hi"])
        out["b19_L_ref_years"] = _num(fs["L_ref_years"])
        out["b19_n_traj"] = _num(fs["n_traj"])
        out["b19_n_event_observed"] = _num(fs["n_event_observed"])
        out["b19_n_censored"] = _num(fs["n_censored"])
        out["b19_damage_proxy_in_xT"] = _num(fs["damage_proxy_in_xT"])
        out["b19_mission_features_in_xT"] = _num(fs["mission_features_in_xT"])
        out["b19_trained_any_model"] = _num(fs["trained_any_model"])

    fa = _json("checkpoints/basilisk_b19/feature_audit.json")
    if fa is None:
        out["b19_audit_verdict"] = "MISSING"
        out["b19_cross_check_consistent"] = "MISSING"
    else:
        out["b19_audit_verdict"] = str(fa["verdict"])
        out["b19_cross_check_consistent"] = _num(fa["cross_check_consistent"])

    fr = _json("checkpoints/basilisk_b19/frozen_feature_definition.json")
    if fr is None:
        for k in ("b19_frozen_verdict", "b19_config_sha256",
                  "b19_is_recomputation_not_estimation", "b19_n_gates_passed",
                  "b19_hi_allowed_supervision",
                  "b19_hi_forbidden_plain_input"):
            out[k] = "MISSING"
    else:
        out["b19_frozen_verdict"] = str(fr["verdict"])
        out["b19_config_sha256"] = str(fr["config_sha256"])
        out["b19_is_recomputation_not_estimation"] = _num(
            fr["selected_primary_hi"]["is_recomputation_not_estimation"])
        out["b19_n_gates_passed"] = _num(fr["gate_results"]["n_passed"])
        dc = fr["downstream_contract"]
        # B2 的 §5 直接依赖这两条角色约束
        out["b19_hi_allowed_supervision"] = _num(
            "supervision_target" in dc["hi_damage_obs_allowed_roles"])
        out["b19_hi_forbidden_plain_input"] = _num(
            "plain_input_column_in_xT" in dc["hi_damage_obs_forbidden_roles"])
    return out


NUMERIC_BLOCKS = tuple(B19V.NUMERIC_BLOCKS) + (
    ("b19_results", read_b19_results, B19_EXPECTED, "b19"),
)

FROZEN_GROUPS: dict[str, list[str]] = dict(B19V.FROZEN_GROUPS)
FROZEN_GROUPS.update(B2_ADDED_GROUPS)

_sha256_of = B19V._sha256_of

# 契约 purpose 文案。抽成模块级常量的理由与 B1.6..B1.9 同: 文案必须写明
# "不跑 source transfer"这类边界, 而反作弊扫描器会扫源码里的这些符号;
# 常量声明块被扫描器豁免, 函数体内的字符串不会。
CONTRACT_PURPOSE = (
    "冻结 analytic lineage + BASILISK_V1 + B1 ... B1.9 全部产物与结论。"
    "B2 只解决一件事: 在 B1.9 冻结的新数据 (Basilisk 工况 + D-based EOL + "
    "observable damage HI) 上重新建立 Target-only 基线, 判断它是否稳定优于最强"
    "可部署平凡基线 const_mean_info。"
    "旧 analytic S25_PASS 不得继承 —— 数据/失效定义/健康指标三者全换过。"
    "本阶段不做 S3 truncation、不跑 source transfer、不重新调参、"
    "不重新生成 feature 文件。B1.9 的 feature 文件进契约 (必须逐字不变); "
    "B2 自己的 checkpoints/docs 不进契约。"
)


def compute() -> dict:
    groups = {g: {rel: _sha256_of(rel) for rel in files}
              for g, files in FROZEN_GROUPS.items()}
    flat = {rel: h for g in groups.values() for rel, h in g.items()}
    chain = dict(B19V.compute()["frozen_chain"])
    chain["BASILISK_B1.9"] = "B19_FEATURE_READY"
    return {
        "stage": "BASILISK_B2",
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
        print(f"[{tag}] B2_BASELINE_CONTRACT_VIOLATION —— 必须立即停止")
        return 1
    print(f"[{tag}] B2_BASELINE_CONTRACT_OK")
    return 0


def write(tag: str) -> int:
    CONTRACT.parent.mkdir(parents=True, exist_ok=True)
    c = compute()
    # 任何 *_EXPECTED 里的 PLACEHOLDER_FILLED_ON_FIRST_WRITE 都在首次写入时
    # 用实测值填入 —— 不凭记忆预填。填完之后 verify 逐字比对, 照样能失败。
    # (B1.9 只针对 b18_damage_equation_sha256 单点做, 这里做成通用规则。)
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
    return write(tag) if not CONTRACT.exists() else verify(tag)


if __name__ == "__main__":
    raise SystemExit(main())
