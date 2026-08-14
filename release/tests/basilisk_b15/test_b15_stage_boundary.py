"""tests/basilisk_b15/test_b15_stage_boundary.py

Basilisk-B1.5 —— 停止边界与 config 一致性测试。

钉死三件事:
  1. config 里不得出现任何可调仿真参数 (§6 禁止清单对应的键);
  2. §8/§9 verdict 下不得有 probe / dataset / 训练产物;
  3. B1.5 没有修改任何既有阶段的产物。
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "configs" / "wheel_basilisk_b15.yaml"
CKPT = ROOT / "checkpoints" / "basilisk_b15"

# §6 明令禁止调整的量 —— 这些**不得**作为可写参数出现在 b15 config 里
FORBIDDEN_TUNABLES = ("Delta_range", "tau_years_range", "q0_min", "q0_max",
                      "q0_range", "b0_range", "duration_years", "horizon_years",
                      "Im_rated", "im_rated", "safety_factor", "eta_margin_new",
                      "threshold_new", "T_f_fail_new")


def _cfg() -> dict:
    assert CFG.exists(), f"缺少 {CFG}"
    return yaml.safe_load(io.open(CFG, encoding="utf-8").read())


def test_config_has_no_tunable_simulation_params():
    """B1.5 是纯审计阶段: config 不得含任何可调仿真参数。

    `forbidden:` 块是 §6 要求**逐字记录**的禁令清单 —— 记录禁令本身不构成
    使用该参数, 故扫描时跳过该块。跳过范围是精确的: 只跳 `forbidden:` 下的
    列表项, 一旦回到零缩进的新键就恢复扫描。
    """
    raw = io.open(CFG, encoding="utf-8").read()
    in_forbidden = False
    n_scanned = 0
    for line in raw.splitlines():
        stripped = line.rstrip()
        if stripped.startswith("forbidden:"):
            in_forbidden = True
            continue
        if in_forbidden:
            # 列表项或空行仍在块内; 任何零缩进非空内容意味着块结束
            if stripped and not stripped.startswith((" ", "\t", "-")):
                in_forbidden = False
            else:
                continue
        code = line.split("#", 1)[0]           # 剥注释: 记录禁令不算使用
        n_scanned += 1
        for tok in FORBIDDEN_TUNABLES:
            assert tok not in code, \
                f"config 出现禁止的可调参数 {tok!r}: {line.strip()!r}"
    assert n_scanned > 40, "扫描行数过少 —— 跳过逻辑可能吞掉了整个文件"


def test_config_records_forbidden_list_verbatim():
    c = _cfg()
    f = c["forbidden"]
    assert len(f) == 8
    for s in ("用 B13/B14 failure fraction 推阈值", "用 RUL RMSE",
              "用 target_only/transfer", "调 Delta", "调 q0", "调 horizon",
              "调 Im_rated", "新造任意 safety factor"):
        assert s in f, f"§6 禁止项缺失: {s}"


def test_config_readonly_values_match_frozen():
    """config 登记的审计对象数值必须与 B1.2 冻结产物逐位一致。"""
    c = _cfg()
    rv = c["audit_target"]["readonly_values"]
    b12 = json.loads(io.open(
        ROOT / "checkpoints" / "basilisk_b12" / "candidates.json",
        encoding="utf-8").read())
    d = b12["per_candidate"]["FRICTION_TORQUE_P95"]["threshold_derivation"]
    assert rv["torque_util_ref_p95"] == float(d["torque_util_ref_p95"])
    assert rv["eta_margin"] == float(d["eta_margin"])
    assert rv["u_max_Nm"] == float(d["u_max_Nm"])
    assert rv["T_f_fail_Nm"] == float(
        b12["per_candidate"]["FRICTION_TORQUE_P95"]["threshold"])


def test_config_confidence_labels_are_exactly_four():
    c = _cfg()
    labels = c["audit_target"]["confidence_labels"]
    assert labels == ["DIRECT", "DERIVED", "ASSUMED", "UNAVAILABLE"]


def test_config_tolerance_not_relaxed():
    c = _cfg()
    assert c["q0_accounting"]["tolerance_rel"] == 1e-12
    assert c["q0_accounting"]["must_not_modify_trajectories"] is True


def test_config_declares_no_mutable_groups():
    c = _cfg()
    sb = c["stop_boundary"]
    assert sb["mutable_groups"] == []
    for k in ("no_rul", "no_transfer", "no_s6", "no_new_failure_dataset",
              "no_second_probe", "no_threshold_modification"):
        assert sb[k] is True


def test_verdict_domain_is_the_three_declared():
    c = _cfg()
    assert set(c["verdicts"]) == {
        "B15_THRESHOLD_PROVENANCE_SUFFICIENT",
        "B15_THRESHOLD_PROVENANCE_INSUFFICIENT",
        "B15_FAILURE_SEMANTICS_MISMATCH",
    }
    # 实际 verdict 必须在域内
    r = json.loads(io.open(CKPT / "failure_threshold_audit.json",
                           encoding="utf-8").read())
    assert r["verdict"] in c["verdicts"]


def test_probe_gate_wired_to_sufficient_only():
    c = _cfg()
    p = c["probe"]
    assert p["allowed_only_if_verdict"] == "B15_THRESHOLD_PROVENANCE_SUFFICIENT"
    assert p["max_runs"] == 1
    assert p["n_traj"] == 60
    assert p["must_not_modify_threshold_after_probe"] is True
    r = json.loads(io.open(CKPT / "failure_threshold_audit.json",
                           encoding="utf-8").read())
    # verdict 非 SUFFICIENT ⇒ probe_allowed 必须 False
    assert r["probe_allowed"] == (
        r["verdict"] == p["allowed_only_if_verdict"])


def test_no_b15_simulation_or_feature_artifacts():
    """§8: 不生成新 failure dataset; 本阶段也不构建特征。"""
    for d in (ROOT / "data" / "sim" / "wheel_basilisk_b15",
              ROOT / "data" / "features" / "wheel" / "basilisk_b15"):
        if d.exists():
            assert not list(d.rglob("*.h5")), f"{d} 下出现 h5 —— 越界"


def test_b15_not_in_requirements_or_docker():
    """Basilisk 不得写入 requirements.txt / Dockerfile / docker-compose.yml。"""
    for rel in ("requirements.txt", "Dockerfile", "docker-compose.yml"):
        p = ROOT / rel
        if p.exists():
            txt = io.open(p, encoding="utf-8", errors="replace").read()
            assert "basilisk_b15" not in txt.lower()


def test_checkpoint_files_are_exactly_the_two_audits():
    """本阶段只应产出两份审计 JSON, 没有 frozen_* / probe_* 产物。"""
    names = sorted(p.name for p in CKPT.glob("*.json"))
    assert names == ["failure_threshold_audit.json", "q0_accounting_audit.json"]
    for bad in ("frozen_calibration.json", "frozen_threshold.json",
                "probe_summary.json", "probe_audit.json"):
        assert not (CKPT / bad).exists(), f"越界产物: {bad}"
