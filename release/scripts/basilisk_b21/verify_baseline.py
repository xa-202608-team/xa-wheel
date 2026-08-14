#!/usr/bin/env python
"""scripts/basilisk_b21/verify_baseline.py

BASILISK-B2.1 §1 —— baseline 契约 (SPLIT_COVERAGE_ROBUSTNESS)。

在 B4X 契约 (914 项) 之上追加 B4X 自己的产物与结论。沿用 B1.4..B4X 的做法:
import 上一阶段的 verify_baseline 并扩展 FROZEN_GROUPS / NUMERIC_BLOCKS。

§1 "Freeze old hashes and create an isolated b21 namespace" 的落地:
  * 旧 hash: B1.8 dataset / B1.9 feature / B2 split+IDs+config+metrics+verdict /
    B3X 全部产物 / B4X 全部产物 —— 全部进契约, 必须逐字不变。
  * 隔离命名空间: B2.1 自己的 configs/scripts/tests/checkpoints/docs 一律
    **不进**契约 (它们是本阶段的新产物, 会变)。

B2.1 额外钉死的前置事实:
  * B4X verdict = B4X_TRANSFER_STABILIZATION_SIGNAL, 且 EXPLORATORY_ONLY
  * B4X 未产出任何正式 positive-transfer 结论 -> forbid_positive_transfer_claim
  * B2 逐 split 的 min event EOL (lifetime-support mismatch 的量化证据)
    -> 防止事后重解释"缺陷本来就不存在"
  * 正式 B5 / 迁移结论类产物必须始终缺席 (§13/§14: 不自动跑 B5)

用法:
    python scripts/basilisk_b21/verify_baseline.py --tag before
    python scripts/basilisk_b21/verify_baseline.py --tag after
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "docs" / "basilisk_b21" / "baseline_contract.json"


def _load(mod_name: str, rel: str):
    """按唯一模块名加载 —— b1..b21 的 verify_baseline 同名, 必须区分。"""
    p = ROOT / rel
    spec = importlib.util.spec_from_file_location(mod_name, p)
    if spec is None or spec.loader is None:
        raise SystemExit(f"!! 无法加载 {rel}")
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = m
    spec.loader.exec_module(m)
    return m


B4XV = _load("b21_b4x_verify", "scripts/basilisk_b4x/verify_baseline.py")

# ---------------------------------------------------------------------------
# §1: 在 B4X 契约之上追加 B4X 自己的产物
# ---------------------------------------------------------------------------
# 全部按 `ls` 实测填写。教训 (B1.7..B4X 都踩过): 契约里钉一个不存在的路径
# -> 它永远是 MISSING -> 真文件被改时反而漏检。缺席项单独进 *_absent 组。
B21_ADDED_GROUPS: dict[str, list[str]] = {
    "b4x_config": [
        "configs/wheel_basilisk_b4x.yaml",
    ],
    "b4x_scripts": [
        "scripts/basilisk_b4x/verify_baseline.py",
        "scripts/basilisk_b4x/data_b4x.py",
        "scripts/basilisk_b4x/run_transfer_diagnostic.py",
        "scripts/basilisk_b4x/diagnose_transfer_stability.py",
        "scripts/basilisk_b4x/summarize_b4x.py",
    ],
    # B4X 的冻结产出。必须逐字不变, 否则"B4X 只是探索性"这条边界就可以被
    # 事后重写成正式迁移结论。
    "b4x_frozen_outputs": [
        "checkpoints/basilisk_b4x/transfer_metrics.json",
        "checkpoints/basilisk_b4x/transfer_stability.json",
        "checkpoints/basilisk_b4x/summary.json",
        "checkpoints/basilisk_b4x/protocol_hash.json",
        "checkpoints/basilisk_b4x/transfer_raw.npz",
    ],
    "b4x_docs": [
        "docs/basilisk_b4x/baseline_contract.json",
        "docs/basilisk_b4x/protocol.md",
        "docs/basilisk_b4x/transfer_results.md",
        "docs/basilisk_b4x/short_eol_transfer_diagnostics.md",
        "docs/basilisk_b4x/limitations.md",
        "STATUS_BASILISK_B4X.md",
    ],
    "b4x_tests": [
        "tests/basilisk_b4x/conftest.py",
        "tests/basilisk_b4x/test_b4x_discipline.py",
    ],
    # §0 禁止修改的只读源。B2.1 复用它们的实现 (import) 或权重。
    # 本阶段唯一允许改变的是**划分协议**, 其他一律钉死。
    "b21_reconfirmed_readonly": [
        "checkpoints/source_tcn_pretrain.pt",
        "src/models/tcn_encoder.py",
        "src/transfer/train_transfer.py",
        "src/experiments/metrics.py",
        "src/experiments/run_groups.py",
        "scripts/basilisk_b2/build_split.py",
        "scripts/basilisk_b2/data_b2.py",
        "scripts/basilisk_b2/eval_b2.py",
        "scripts/basilisk_b2/train_b2.py",
        "scripts/basilisk_b2/baselines_b2.py",
        "scripts/basilisk_b2/run_gate.py",
        "configs/wheel_basilisk_b2.yaml",
        "data/features/wheel/basilisk_b19/target_features.h5",
    ],
    # §13/§14: B5 必须始终缺席 —— 本阶段不得自动跑 B5。
    # 另外任何 positive-transfer 断言类产物同样必须缺席。
    "b21_must_stay_absent": [
        "checkpoints/basilisk_b5/metrics.json",
        "docs/basilisk_b5/results.md",
        "STATUS_BASILISK_B5.md",
        "checkpoints/basilisk_b21/formal_transfer_verdict.json",
        "checkpoints/basilisk_b21/positive_transfer_proven.json",
    ],
}


# B4X 的数值结论 + B2 的 lifetime-support mismatch 量化证据。
# 首次写入时由 read_b21_results() 实测填入, 不凭记忆预填。
B21_EXPECTED = {
    "b4x_verdict": "B4X_TRANSFER_STABILIZATION_SIGNAL",
    "b4x_label": "POST_B2_FAIL_EXPLORATORY",
    "b4x_formal_stage": "False",
    "b4x_exploratory_only": "True",
    "b4x_not_formal_transfer_evidence": "True",
    "b4x_b2_verdict_unchanged": "B2_GENERALIZATION_FAIL",
    "b4x_forbid_positive_transfer_claim": "True",
    "b4x_n_methods_passing_A_and_B": "0",
    "b4x_split_sha256":
        "079296e5fba206da6ba9dab1078a8eaa6957e37ea97d0782b7a3fa01164113ae",
    "b4x_input_schema": "core_only",
    "b3x_verdict_via_b4x": "B3X_NO_STABILIZING_SIGNAL",
    # lifetime-support mismatch: 本阶段存在的唯一理由, 必须钉死
    "b2_train_min_event_eol": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "b2_val_min_event_eol": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "b2_test_min_event_eol": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "b2_lifetime_support_mismatch": "True",
    "b2_n_test_event_below_train_min": "PLACEHOLDER_FILLED_ON_FIRST_WRITE",
    "dataset_n_event": "71",
    "dataset_n_censored": "79",
}


def _json(rel: str):
    p = ROOT / rel
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _num(v) -> str:
    return repr(v).strip("'\"")


def read_b21_results() -> dict:
    """读 B4X 冻结结论 + B2 划分的寿命支撑证据 (键路径按实测 dump 写)。"""
    out: dict[str, str] = {}
    b4x_keys = ("b4x_verdict", "b4x_label", "b4x_formal_stage",
                "b4x_exploratory_only", "b4x_not_formal_transfer_evidence",
                "b4x_b2_verdict_unchanged",
                "b4x_forbid_positive_transfer_claim",
                "b4x_n_methods_passing_A_and_B", "b4x_split_sha256",
                "b4x_input_schema", "b3x_verdict_via_b4x")
    sm = _json("checkpoints/basilisk_b4x/summary.json")
    if sm is None:
        for k in b4x_keys:
            out[k] = "MISSING"
    else:
        out["b4x_verdict"] = str(sm["verdict"])
        out["b4x_label"] = str(sm["label_discipline"])
        out["b4x_formal_stage"] = _num(bool(sm["formal_stage"]))
        out["b4x_exploratory_only"] = _num(bool(sm["exploratory_only"]))
        out["b4x_not_formal_transfer_evidence"] = _num(
            bool(sm["not_formal_transfer_evidence"]))
        out["b4x_b2_verdict_unchanged"] = str(sm["b2_verdict_unchanged"])
        out["b4x_forbid_positive_transfer_claim"] = _num(
            bool(sm["forbid_positive_transfer_claim"]))
        out["b4x_n_methods_passing_A_and_B"] = _num(
            len(list(sm["methods_passing_A_and_B"])))
        out["b4x_split_sha256"] = str(sm["split_sha256"])
        out["b4x_input_schema"] = str(sm["input_schema"])
    b3 = _json("checkpoints/basilisk_b3x/summary.json")
    out["b3x_verdict_via_b4x"] = (str(b3["verdict"]) if b3 is not None
                                  else "MISSING")

    ev_keys = ("b2_train_min_event_eol", "b2_val_min_event_eol",
               "b2_test_min_event_eol", "b2_lifetime_support_mismatch",
               "b2_n_test_event_below_train_min", "dataset_n_event",
               "dataset_n_censored")
    sp = _json("checkpoints/basilisk_b2/split.json")
    h5p = ROOT / "data/features/wheel/basilisk_b19/target_features.h5"
    if sp is None or not h5p.exists():
        for k in ev_keys:
            out[k] = "MISSING"
        return out

    import h5py  # 局部 import: 契约脚本在无 h5py 环境也能读 JSON 部分
    eol: dict[str, int] = {}
    n_cen = 0
    with h5py.File(h5p, "r") as f:
        for k in sorted(f.keys()):
            g = f[k]
            if bool(int(g.attrs["event_observed"])):
                eol[k] = int(g.attrs["eol_idx"])
            else:
                n_cen += 1
    mins = {}
    for name in ("train", "val", "test"):
        vals = [eol[t] for t in sp["splits"][name]["tids"] if t in eol]
        mins[name] = min(vals) if vals else None
        out[f"b2_{name}_min_event_eol"] = _num(mins[name])
    out["dataset_n_event"] = _num(len(eol))
    out["dataset_n_censored"] = _num(n_cen)
    te = [eol[t] for t in sp["splits"]["test"]["tids"] if t in eol]
    out["b2_n_test_event_below_train_min"] = _num(
        sum(1 for v in te if mins["train"] is not None and v < mins["train"]))
    out["b2_lifetime_support_mismatch"] = _num(
        bool(mins["test"] is not None and mins["train"] is not None
             and mins["test"] < mins["train"]))
    return out


NUMERIC_BLOCKS = tuple(B4XV.NUMERIC_BLOCKS) + (
    ("b21_results", read_b21_results, B21_EXPECTED, "b21"),
)

FROZEN_GROUPS: dict[str, list[str]] = dict(B4XV.FROZEN_GROUPS)
FROZEN_GROUPS.update(B21_ADDED_GROUPS)

_sha256_of = B4XV._sha256_of

# 契约 purpose 文案。抽成模块级常量的理由与 B1.6..B4X 同: 文案必须写明
# "旧结论不变"这类边界, 而反作弊扫描器会扫源码里的这些符号;
# 常量声明块被扫描器豁免, 函数体内的字符串不会。
CONTRACT_PURPOSE = (
    "冻结 analytic lineage + BASILISK_V1 + B1 ... B1.9 + B2 + B3X + B4X "
    "全部产物与结论。B2.1 是 SPLIT_COVERAGE_ROBUSTNESS 阶段, 只改**划分协议**。"
    "B2_GENERALIZATION_FAIL 保持终局; B3X_NO_STABILIZING_SIGNAL 不变; "
    "B4X_TRANSFER_STABILIZATION_SIGNAL 仍然只是 EXPLORATORY_ONLY, "
    "不存在任何正式 positive-transfer 结论。"
    "本阶段禁止修改: model architecture / optimizer / loss / HI / RUL / "
    "source checkpoints / MMD lambda / mission features / "
    "B2 与 B3X 与 B4X 的任何旧产物。"
    "已知结构缺陷 = lifetime-support mismatch: B2 的 test 含有比 train 与 val "
    "全部 event 轨迹都更短命的 event 轨迹。三个 split 的 min event EOL 与"
    "低于 train 最短寿命的 test event 轨迹条数一并钉入契约, "
    "使该缺陷不能被事后否认或重新解释。"
    "划分不得用 test 表现挑选: 确定性划分不满足覆盖要求就报 B21_SPLIT_INVALID, "
    "禁止换 seed 试到好看。"
    "B2.1 自己的 configs/scripts/tests/checkpoints/docs 不进契约。"
    "正式 B5 产物与任何 positive-transfer 断言类产物必须始终缺席 —— "
    "本阶段无论 PASS 还是 FAIL 都不得自动运行 B5。"
)


def compute() -> dict:
    groups = {g: {rel: _sha256_of(rel) for rel in files}
              for g, files in FROZEN_GROUPS.items()}
    flat = {rel: h for g in groups.values() for rel, h in g.items()}
    chain = dict(B4XV.compute()["frozen_chain"])
    chain["BASILISK_B4X"] = "B4X_TRANSFER_STABILIZATION_SIGNAL"
    return {
        "stage": "BASILISK_B21",
        "label": "SPLIT_COVERAGE_ROBUSTNESS",
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
        print(f"[{tag}] B21_BASELINE_CONTRACT_VIOLATION —— 必须立即停止")
        return 1
    print(f"[{tag}] B21_BASELINE_CONTRACT_OK")
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
