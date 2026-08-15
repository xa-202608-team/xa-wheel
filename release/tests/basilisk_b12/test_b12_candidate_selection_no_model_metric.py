"""tests/basilisk_b12/test_b12_candidate_selection_no_model_metric.py

文件名带 b12_ 前缀: pytest 无 __init__.py 时按 basename 建模块名, 与
tests/basilisk_b1/test_candidate_selection_no_model_metric.py 同名会 collection error。
测试**函数名**仍按 §19 要求为 test_candidate_selection_no_model_metric。

§19 test_candidate_selection_no_model_metric —— Gate 与选择只看物理量/数据分布。

与 test_candidate_selection_no_rul 的分工:
  * no_rul        关注"有没有碰 RUL 训练/评估链路" (模块与标识符层面);
  * no_model_metric 关注"Gate 判据本身是否只由物理量与数据分布构成"
    —— 逐条 Gate 检查其 detail 里的量是否都能在 per_trajectory 的物理字段中找到,
    并断言没有任何 Gate 依赖模型输出。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# per_trajectory 里允许 Gate 引用的物理量/分布量白名单
ALLOWED_QUANTITIES = {
    "failed", "eol_idx", "eol_window", "eol_frac_of_horizon", "n", "n_windows",
    "stat_max", "stat_p95", "threshold",
    "degradation_fraction_at_eol", "command_fraction_at_eol",
    "friction_fraction_at_eol", "eol_segment",
    "healthy_trigger_fraction", "healthy_n_windows",
    "healthy_stat_p95_over_thr",
    "b_init", "b_final", "b_ratio", "b_monotone_frac", "label_monotone",
    "omega_absmax", "Im_absmax", "T_min", "T_max",
    "n_nan", "all_finite", "g_duty", "speed_util_mean", "torque_util_mean",
    "idx",
}

MODEL_METRIC_WORDS = ("rmse", "mae", "phm", "nphm", "loss", "accuracy",
                      "r2_score", "auc", "f1_score", "prediction")


def test_candidate_selection_no_model_metric(audit_rec, sel_rec, cand_rec):
    if audit_rec is None:
        pytest.skip("candidate_audit.json 尚未生成")

    assert audit_rec["used_model_metric"] is False
    assert audit_rec["used_rul_metric"] is False

    # ---- per_trajectory 字段必须全在物理量白名单内 ----
    if cand_rec is not None:
        for name, cd in cand_rec["per_candidate"].items():
            keys = set(cd["per_trajectory"][0].keys())
            extra = keys - ALLOWED_QUANTITIES
            assert not extra, \
                f"{name} per_trajectory 出现非物理量字段 {extra} —— " \
                "Gate 只允许看物理量与数据分布"

    # ---- 逐条 Gate: detail 文本不得含模型指标词 ----
    for name, r in audit_rec["per_candidate"].items():
        assert len(r["gates"]) == 12, f"{name} Gate 数不是 12: {len(r['gates'])}"
        for x in r["gates"]:
            low = (x["name"] + " " + x["detail"]).lower()
            for w in MODEL_METRIC_WORDS:
                assert w not in low, f"{name} Gate '{x['name']}' 含模型指标词 '{w}'"
        # Gate 10 必须显式自证
        g10 = [x for x in r["gates"] if x["name"].startswith("10 ")]
        assert len(g10) == 1 and g10[0]["pass"] is True, \
            f"{name} Gate 10 (无 RUL/模型指标) 未通过"

    # ---- 整份审计 JSON 不得含模型指标键 ----
    raw = json.dumps(audit_rec, ensure_ascii=False).lower()
    for w in ("val_loss", "test_rmse", "rul_rmse", "phm_score", "y_pred"):
        assert w not in raw, f"candidate_audit.json 含 '{w}'"

    if sel_rec is None:
        pytest.skip("selection_result.json 尚未生成")
    assert sel_rec["used_model_metric"] is False
    raw2 = json.dumps(sel_rec, ensure_ascii=False).lower()
    for w in ("val_loss", "test_rmse", "rul_rmse", "phm_score", "y_pred"):
        assert w not in raw2, f"selection_result.json 含 '{w}'"


def test_gate_names_match_protocol(audit_rec):
    """12 条 Gate 必须与 protocol.md 登记的一一对应, 编号不得错位或缺失。"""
    if audit_rec is None:
        pytest.skip("candidate_audit.json 尚未生成")
    md = (ROOT / "docs" / "basilisk_b12" / "protocol.md").read_text(encoding="utf-8")
    for name, r in audit_rec["per_candidate"].items():
        nums = [x["name"].split(" ", 1)[0] for x in r["gates"]]
        assert nums == [str(i) for i in range(1, 13)], \
            f"{name} Gate 编号错位: {nums}"
    # protocol 中确实登记了 12 条
    for i in range(1, 13):
        assert f"| {i} |" in md, f"protocol.md 缺 Gate {i} 的登记行"
