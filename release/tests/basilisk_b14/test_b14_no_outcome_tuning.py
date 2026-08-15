# -*- coding: utf-8 -*-
"""BASILISK-B1.4 §4/§11: 禁止 outcome tuning 与多次 probe 的测试。

核心命名测试:
  - test_q0_max_not_outcome_tuned
  - test_only_q0_upper_bound_changes
  - test_single_probe_only
  - test_no_rul_metric
"""
import io
import json
import os

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# §0 逐字禁止清单里出现的 B1.3 结果数字。判据不是「文件里不许出现」——
# config 被要求逐字记录禁止清单 (含 "18/60") —— 而是更强的两条:
#   (a) token 只能出现在注释行, 绝不能成为任何 YAML 值;
#   (b) q0_prior 整块 (含注释) 不得出现任何 token。
B13_OUTCOME_TOKENS = (
    "0.766667", "0.233333", "0.7666666", "0.2333333",
    "18/60", "46/60", "14/60",
    "early_eol_fraction=0.3", "early_eol=0.3",
)
Q0_MIN_NEW = 0.0017901569929920683
Q0_MAX_NEW = 0.017901569929920685
Q0_OLD = (0.04137931034482758, 0.41379310344827586)


def _p(*parts):
    return os.path.join(ROOT, *parts)


def _read(rel):
    with io.open(_p(*rel.split("/")), encoding="utf-8") as fh:
        return fh.read()


# --------------------------------------------------------------------------
# §21: test_q0_max_not_outcome_tuned
# --------------------------------------------------------------------------
def test_q0_max_not_outcome_tuned(prior_rec, prior_math_rec):
    """q0_max 必须来自 §2 独立依据的除法, 不是为了过 Gate 试出来的数字。"""
    if prior_rec is None:
        pytest.skip("healthy_entry_prior.json 未生成")
    # 1) 只有一个候选 —— 没有 sweep / 二分
    assert prior_rec["n_candidates"] == 1
    assert prior_rec["hardcoded_target"] is False
    # 2) 五个 outcome-usage 开关全 False
    assert prior_rec["used_failure_fraction_for_calibration"] is False
    assert prior_rec["used_early_eol_fraction_for_calibration"] is False
    assert prior_rec["used_rul_metric"] is False
    assert prior_rec["used_model_metric"] is False
    assert prior_rec["all_checks_pass"] is True
    # 3) q0_max 不是 §4 禁止的那几个「试出来」的候选值
    for banned in (0.40, 0.35, 0.30):
        assert abs(Q0_MAX_NEW - banned) > 0.1
    # 4) 独立验证脚本自己也判定 outcome 无污染
    if prior_math_rec is not None:
        assert prior_math_rec["verdict"] == "B14_PRIOR_MATH_VALID"
        v2 = prior_math_rec["groups"]["V2 无 outcome 污染"]
        assert all(it["pass"] for it in v2)
        assert len(v2) >= 7


def test_no_b13_outcome_tokens_as_values(b14_cfg):
    """config 里 outcome token 只能出现在注释行, 且 q0_prior 整块必须干净。"""
    txt = _read("configs/wheel_basilisk_b14.yaml")
    lines = txt.splitlines()
    for i, line in enumerate(lines, 1):
        code = line.split("#", 1)[0]
        for tok in B13_OUTCOME_TOKENS:
            assert tok not in code, "第 %d 行把 B1.3 结果 %r 写成了 YAML 值" % (i, tok)
    # q0_prior 块 (缩进块) 整体不得含 token, 连注释也不行
    block, inside, indent = [], False, None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("q0_prior:"):
            inside = True
            indent = len(line) - len(line.lstrip())
            continue
        if inside:
            if stripped and not stripped.startswith("#"):
                cur = len(line) - len(line.lstrip())
                if cur <= indent:
                    break
            block.append(line)
    blob = "\n".join(block)
    for tok in B13_OUTCOME_TOKENS:
        assert tok not in blob, "q0_prior 块内出现 B1.3 结果 %r" % tok


def test_derivation_doc_outcome_free():
    """推导脚本与推导文档不得出现任何 B1.3 结果数字。"""
    for rel in ("scripts/basilisk_b14/derive_healthy_entry_prior.py",
                "docs/basilisk_b14/healthy_entry_derivation.md"):
        txt = _read(rel)
        for tok in B13_OUTCOME_TOKENS:
            assert tok not in txt, "%s 含 B1.3 结果 %r" % (rel, tok)


# --------------------------------------------------------------------------
# §21: test_only_q0_upper_bound_changes
# --------------------------------------------------------------------------
def test_only_q0_upper_bound_changes(prior_math_rec, protocol_rec):
    """§10: 相对 B1.3 唯一变化 = q0 支撑区间; 其余全部逐位不变。"""
    if prior_math_rec is None:
        pytest.skip("prior_math_validation.json 未生成")
    v3 = prior_math_rec["groups"]["V3 单自由度"]
    assert all(it["pass"] for it in v3)
    names = [it["check"] for it in v3]
    for must in ("F2 threshold_Nm unchanged", "b_fail_f2 unchanged",
                 "omega_ref unchanged", "horizon unchanged 3 years",
                 "CRN paired_trajectory_hash reused", "only q0 support changed"):
        assert must in names, "缺少单自由度检查: %s" % must
    if protocol_rec is not None:
        fv = protocol_rec["frozen_values"]
        # 只有 b0 被重标定; Delta / tau 明确列为不变参数
        assert fv["rescale_params"] == ["b0"]
        assert fv["invariant_params"] == ["Delta", "tau_years"]
        assert fv["q0_min_new"] == Q0_MIN_NEW
        assert fv["q0_max_new"] == Q0_MAX_NEW
        assert fv["q0_old_range"] == list(Q0_OLD)
        assert fv["horizon_years"] == 3.0


# --------------------------------------------------------------------------
# §21: test_single_probe_only
# --------------------------------------------------------------------------
def test_single_probe_only(b14_cfg, protocol_rec, probe_rec):
    """§11: 只允许一次 probe。config/protocol 声明 runs_allowed=1, 磁盘上不得有第二份 probe。"""
    probe_cfg = b14_cfg["probe"]
    assert int(probe_cfg["runs_allowed"]) == 1
    assert int(probe_cfg["n_traj"]) == 60
    if protocol_rec is not None:
        assert protocol_rec["frozen_values"]["runs_allowed"] == 1
        assert protocol_rec["frozen_values"]["probe_n_traj"] == 60
    # 磁盘证据: 不存在 probe_summary_v2 / _run2 / _retry 之类的第二次产物
    ckpt = _p("checkpoints", "basilisk_b14")
    if os.path.isdir(ckpt):
        extra = [f for f in os.listdir(ckpt)
                 if f.startswith("probe_summary") and f != "probe_summary.json"]
        assert extra == [], "出现第二次 probe 产物: %s" % extra
    if probe_rec is not None:
        assert probe_rec["n_traj"] == 60
        assert probe_rec["only_q0_prior_changed"] is True


def test_probe_used_frozen_protocol_hash(probe_rec, protocol_rec):
    """probe 必须引用冻结后的 protocol hash —— 证明 protocol 先冻结再跑。"""
    if probe_rec is None or protocol_rec is None:
        pytest.skip("probe / protocol 产物未生成")
    assert probe_rec["protocol_sha256"] == protocol_rec["protocol_sha256"]
    assert protocol_rec["frozen_before_any_probe_data"] is True


def test_prediction_pre_registered_before_probe(prior_math_rec, protocol_rec, probe_rec):
    """§4 反推禁令的正面证据: 预测在 probe 之前落盘, 且 probe 后不得被改写。"""
    if prior_math_rec is None:
        pytest.skip("prior_math_validation.json 未生成")
    pred = prior_math_rec["pre_registered_prediction"]
    assert pred["written_before_probe"] is True
    assert pred["failure_fraction"] == 0.0
    assert pred["gate1_pass"] is False
    assert pred["verdict"] == "B14_CALIBRATION_FAIL"
    if protocol_rec is not None:
        # protocol 冻结时把同一预测抄了一份, 两处必须一致 (防事后单方面改口)
        assert protocol_rec["pre_registered_prediction"] == pred
    if probe_rec is not None:
        pm = _p("checkpoints", "basilisk_b14", "prior_math_validation.json")
        ps = _p("checkpoints", "basilisk_b14", "probe_summary.json")
        assert os.path.getmtime(pm) < os.path.getmtime(ps), "预测文件在 probe 之后被改写"


# --------------------------------------------------------------------------
# §21: test_no_rul_metric
# --------------------------------------------------------------------------
def test_no_rul_metric():
    """§20: 本阶段禁止任何 RUL / transfer / 模型指标。b14 产物与脚本不得出现这些字段。"""
    banned = ("rmse", "phm_score", "nphm", "mae_days", "transfer_gain",
              "target_only", "source_finetune", "mmd_loss")
    ckpt = _p("checkpoints", "basilisk_b14")
    if os.path.isdir(ckpt):
        for fn in sorted(os.listdir(ckpt)):
            if not fn.endswith(".json"):
                continue
            with io.open(os.path.join(ckpt, fn), encoding="utf-8") as fh:
                low = fh.read().lower()
            for tok in banned:
                assert tok not in low, "%s 出现被禁指标 %r" % (fn, tok)
    # 脚本层面: 不得 import 训练/迁移模块
    sdir = _p("scripts", "basilisk_b14")
    for fn in sorted(os.listdir(sdir)):
        if not fn.endswith(".py"):
            continue
        low = _read("scripts/basilisk_b14/" + fn).lower()
        for tok in ("import torch", "from src.train", "from src.transfer",
                    "src.models", "particle_filter", "wiener"):
            assert tok not in low, "%s 触及被禁模块 %r" % (fn, tok)
