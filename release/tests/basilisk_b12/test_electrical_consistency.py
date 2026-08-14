"""tests/basilisk_b12/test_electrical_consistency.py

§19 —— 电气 provenance 与电流分解的三项测试:
  * test_electrical_provenance_recorded
  * test_current_limit_not_tuned_from_failure_fraction
  * test_current_decomposition
"""
from __future__ import annotations

import numpy as np
import pytest

CONF = {"DIRECT", "DERIVED", "ASSUMED", "UNAVAILABLE"}


def test_electrical_provenance_recorded(prov_rec, b12_cfg):
    """§3: 每个必需参数都必须有 source / source_type / value / unit / confidence。"""
    assert prov_rec is not None, "缺 electrical_provenance.json (§20 步骤 2 未跑)"
    tbl = prov_rec["param_table"]["required_params"]
    required = list(b12_cfg["electrical"]["required_params"])
    assert set(tbl.keys()) == set(required), \
        f"参数表与 config required_params 不一致: {set(tbl) ^ set(required)}"
    for name, e in tbl.items():
        assert e["confidence"] in CONF, f"{name} confidence 非法: {e['confidence']}"
        for f in ("source", "source_type", "value", "unit", "confidence"):
            assert f in e, f"{name} 缺字段 {f}"
        if e["confidence"] == "UNAVAILABLE":
            # UNAVAILABLE 必须明确写 None, 不许填一个猜的数字冒充 (§5)
            assert e["value"] is None, \
                f"{name} 标 UNAVAILABLE 却有数值 {e['value']} —— §5 禁止猜值"
        else:
            assert e["value"] is not None, f"{name} 非 UNAVAILABLE 却无值"
            st = e["source_type"]
            assert st in b12_cfg["electrical"]["allowed_source_types"], \
                f"{name} source_type={st} 不在白名单 (§3: 禁论坛/博客/二手网页)"

    # §4 verdict 必须是三态之一, 且与 f1_eligible 自洽
    d = prov_rec["decision"]
    assert d["verdict"] in {"ELECTRICALLY_CONSISTENT", "ELECTRICALLY_INCONSISTENT",
                            "PROVENANCE_INSUFFICIENT"}
    assert d["f1_eligible"] == (d["verdict"] == "ELECTRICALLY_CONSISTENT")
    if d["verdict"] != "ELECTRICALLY_CONSISTENT":
        assert d["current_based_failure_status"] == "PROVENANCE_UNSUPPORTED", \
            "§5: provenance 不足时 current_based_failure 必须标 PROVENANCE_UNSUPPORTED"

    # §4 明令: 不得通过修改参数来改变判断
    assert prov_rec["params_modified_by_this_script"] is False
    assert prov_rec["used_rul_metric"] is False
    assert prov_rec["used_model_metric"] is False


def test_current_limit_not_tuned_from_failure_fraction(prov_rec, cand_rec, b12_cfg):
    """§2/§5: Im_rated 不得为了通过 Gate 而手工填, 也不得由 failure fraction 反推。

    证据链:
      1. provenance 记录声明未修改任何参数;
      2. config 里的 Im_rated_A 与 B1.1 / wheel.yaml 完全一致 (未因 B1.2 改动);
      3. F1 的 threshold_derivation 明确标 PROVENANCE_UNSUPPORTED 且 eligible=false,
         即没有"编一个能过 Gate 的电流值"这条路。
    """
    import yaml
    assert prov_rec["params_modified_by_this_script"] is False

    b12_im = float(b12_cfg["sim"]["failure"]["Im_rated_A"])
    base = yaml.safe_load(
        (__import__("pathlib").Path("configs/wheel.yaml")).read_text(encoding="utf-8"))
    assert float(base["sim"]["failure"]["Im_rated_A"]) == b12_im, \
        "B1.2 改动了 Im_rated_A —— §2 明令禁止为通过 Gate 手工填电流值"

    if cand_rec is None:
        pytest.skip("candidates.json 尚未生成")
    f1 = cand_rec["per_candidate"]["CURRENT_WINDOW_P95"]
    assert f1["eligible"] is False, \
        "electrical verdict 非 CONSISTENT 时 F1 必须 ineligible (§5)"
    assert f1["threshold_confidence"] == "PROVENANCE_UNSUPPORTED"
    assert float(f1["threshold"]) == b12_im, "F1 阈值被改成了非 in-use 值"


def test_current_decomposition(b12_mech, mech_rec):
    """§6: I_cmd + I_coulomb + I_viscous 必须能重构仿真器的无噪 I_m。

    构造一段解析可验的输入直接校验分解代数, 再核对审计记录里的残差量级
    (残差 = 量测噪声 + 直流偏置, 不是分解误差)。
    """
    Kt, Tc = 0.025, 1.25e-3
    om = np.array([100.0, -100.0, 50.0, -50.0])
    tcmd = np.array([0.01, -0.02, 0.005, 0.0])
    b = np.array([1e-5, 2e-5, 3e-5, 4e-5])
    im_clean = (tcmd + Tc * np.sign(om) + b * om) / Kt

    class G(dict):
        def __getitem__(self, k):
            return dict.__getitem__(self, k)

    g = G(omega_cmd=om, T_cmd=tcmd, b_true=b, I_m=im_clean)
    # decompose 用 g["x"][:] 取值, ndarray 支持切片
    d = b12_mech.decompose(g, Kt, Tc)
    assert np.allclose(d["i_cmd"], tcmd / Kt)
    assert np.allclose(d["i_coul"], Tc * np.sign(om) / Kt)
    assert np.allclose(d["i_visc"], b * om / Kt)
    assert np.allclose(d["i_total"], im_clean), "三项之和未能重构 I_m"
    assert np.allclose(d["recon_resid"], 0.0, atol=1e-12), \
        "无噪输入下重构残差应为 0"

    # 退化占比定义: 与力矩域恒等 (三项同除 Kt)
    eps = 1e-12
    fr = b12_mech.frac_series(d, eps)
    fr_torque = (np.abs(b * om)
                 / (np.abs(tcmd) + np.abs(Tc * np.sign(om)) + np.abs(b * om) + eps))
    assert np.allclose(fr, fr_torque), \
        "Gate 11 的电流域与力矩域定义应恒等 (protocol §5 的立论基础)"

    if mech_rec is None:
        pytest.skip("failure_mechanism_audit.json 尚未生成")
    assert mech_rec["degradation_fraction_threshold"] == 0.50
    assert mech_rec["threshold_pre_registered"] is True
    assert mech_rec["used_rul_metric"] is False
    assert mech_rec["used_model_metric"] is False
    # 残差应远小于电流量级 (否则说明分解用错了 omega 路)
    assert mech_rec["recon_resid_rms"]["p95"] < 0.5, \
        "重构残差过大, 可能误用了含噪 omega 而非 omega_cmd"
