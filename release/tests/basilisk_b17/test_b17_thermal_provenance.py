"""tests/basilisk_b17/test_b17_thermal_provenance.py

§4 / §5 / §6 / §7 / §17 —— 温度阈值 provenance、operating-vs-storage 语义、
无来源参数禁令、analog 相似性门槛。
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

USABLE = ("DIRECT", "DERIVED_FROM_DIRECT", "LITERATURE_ANALOG")


def test_temperature_limit_direct(b17_temperature):
    """§17 —— 温度记录必须是本阶段**重新抓取**的 DIRECT 证据, 带 content hash。

    注意本测试断言的是"抓取是真实且可追溯的", 不是"它可以当 EOL 用" ——
    后者由 test_operating_not_storage_temperature 单独把关。
    """
    probes = b17_temperature["network"]["probes"]
    assert b17_temperature["network"]["n_reachable"] >= 1, "一级来源全部不可达"
    ok = [p for p in probes if p["reachable"]]
    for p in ok:
        assert p["http_code"] == "200"
        assert p["content_sha256"] not in ("", "MISSING")
        assert len(p["content_sha256"]) == 64
        assert (ROOT / p["local_cache"]).exists(), "本地缓存缺失, 无法复核"
    f = b17_temperature["findings"]
    assert len(f) >= 1, "未抓到任何温度行 —— 若为 0, 先怀疑探测器而非来源"
    for x in f:
        assert x["unit"] == "degC"
        assert x["max_degC"] is not None
        assert len(x["line_sha256"]) == 64
        assert x["source_url"].startswith("https://nanoavionics.com"), (
            "温度证据必须来自一级官方来源")


def test_temperature_refetched_not_copied_from_b16(b17_temperature):
    """§2 —— 不得从 B1.6 提取字符串直接用; 必须重新抓取并可复现。"""
    cc = b17_temperature["crosscheck_with_b16"]
    assert cc["b16_line_reproduced_by_b17_refetch"] is True, (
        "B1.7 重抓未能复现 B1.6 记录的那一行 —— 需查清是页面变了还是记录失真")
    # 证据必须挂在本阶段的 cache 上, 而不是 b16 的 json 上
    for x in b17_temperature["findings"]:
        assert x["local_cache"].startswith("checkpoints/basilisk_b17/"), (
            "provenance 载体必须是 B1.7 自己的 cache")


def test_operating_not_storage_temperature(b17_temperature, b17_config):
    """§17 —— operating 与 storage 必须区分; 只有 operating max 可作 EOL。

    第三种情形同样受约束: 措辞既无 operating 也无 storage 时记 unlabeled,
    **不得**被计入可用阈值。
    """
    sem = b17_config["temperature_semantics"]
    adm = set(sem["eol_admissible_kinds"])
    assert adm == {"operating"}, "只有 operating 允许作运行失效阈值"
    assert "storage" in set(sem["eol_inadmissible_kinds"])
    assert "unlabeled" in set(sem["eol_inadmissible_kinds"]), (
        "unlabeled 必须被明确列为不可用 —— 否则会被默默当成 operating")

    for x in b17_temperature["findings"]:
        assert x["semantics"] in (
            "operating", "storage", "unlabeled", "ambiguous_both_words")

    usable = [x for x in b17_temperature["findings"]
              if x["semantics"] in adm]
    assert b17_temperature["usable_for_eol_count"] == len(usable)

    # 若存在 unlabeled, 必须有显式披露文字 (§4 要求)
    n_un = b17_temperature["semantics_breakdown"]["unlabeled"]
    if n_un > 0:
        assert b17_temperature["unlabeled_disclosure"], (
            "存在 unlabeled 温度记录却无披露说明")
        assert "不得默认当作 operating" in \
            b17_temperature["unlabeled_disclosure"]

    # 条件 A 必须与"可用阈值数量"一致 —— 不允许有阈值却报 A=True
    assert b17_temperature[
        "gate_condition_A_direct_operating_temperature_max"] == bool(usable)

    # 若 A 不成立, T_max_direct 必须为 None (不得留一个"备用数字")
    if not usable:
        assert b17_temperature["T_max_direct_degC"] is None
        assert b17_temperature["T_max_direct_provenance"] == "UNAVAILABLE"


def test_no_unsourced_thermal_parameter(b17_registry):
    """§17 (拼写: test_no_unsourced_thermal_parameter) ——
    每个热参数都带 provenance 级别; 且未被赋予任何来源外数值。
    """
    levels = {"DIRECT", "DERIVED_FROM_DIRECT", "LITERATURE_ANALOG",
              "ASSUMED", "UNAVAILABLE"}
    pr = b17_registry["parameter_registry"]
    assert pr, "参数 registry 为空"
    for k, v in pr.items():
        assert "provenance" in v, f"{k} 缺 provenance 级别标注"
        assert v["provenance"] in levels, f"{k} 级别非法: {v['provenance']}"
        # §10: 值只能来自 registry。未通过 Gate 时不得有任何参数被赋值。
        if v["provenance"] in ("UNAVAILABLE", "ASSUMED"):
            assert v.get("value_used_in_model") is None, (
                f"{k} 无可用来源却已被赋值: {v.get('value_used_in_model')}")
    assert b17_registry["thermal_params_hardcoded_outside_registry"] is False
    assert b17_registry["model_memory_used_to_fill_parameters"] is False


def test_direct_claims_carry_verbatim_evidence(b17_registry):
    """判 DIRECT 的参数必须有逐字来源行 —— 不能只写个级别。"""
    for k, v in b17_registry["parameter_registry"].items():
        if v["provenance"] == "DIRECT":
            assert v["tier1_verbatim_hits"], f"{k} 判 DIRECT 但无逐字证据"


def test_power_point_not_overclaimed_as_loss_model(b17_registry):
    """功耗工作点 ≠ loss model。若判 DIRECT 必须带限定并标 is_full_loss_model。"""
    v = b17_registry["parameter_registry"]["efficiency_or_loss_model"]
    if v["provenance"] == "DIRECT":
        assert v.get("caveat"), "功耗点判 DIRECT 却无限定说明 (会被读成损耗模型已有依据)"
        assert v.get("is_full_loss_model") is False
        # 且不得被 B 条件当作 efficiency-derived loss 的依据
        assert b17_registry["gate"]["B_forms"][
            "efficiency_derived_loss"] is False


def test_analog_similarity_gate(b17_registry, b17_config):
    """§17 / §7 —— LITERATURE_ANALOG 必须过 0.5<=ratio<=2.0; 否则不得使用。"""
    an = b17_registry["analog_similarity"]
    lo = b17_config["analog_similarity"]["ratio_min"]
    hi = b17_config["analog_similarity"]["ratio_max"]
    assert an["ratio_bounds"] == [lo, hi]

    if not an["analog_source_available"]:
        # 无 analog 来源 => 必须判不可迁移, 且不得计算比值 (无从计算)
        assert an["analog_not_transferable"] is True
        assert an["ratios_computed"] is False
        assert an["reason"], "需说明为何无 analog 来源"
    else:
        assert an["ratios_computed"] is True
        for key in ("mass_ratio", "torque_ratio", "power_ratio"):
            r = an[key]
            in_band = lo <= r <= hi
            if not in_band:
                assert an["analog_not_transferable"] is True, (
                    f"{key}={r} 越界却未判不可迁移")

    # 不可迁移时, 任何 LITERATURE_ANALOG 参数都不得进入 C 条件
    if an["analog_not_transferable"]:
        for k, lvl in b17_registry["gate"]["C_sources"].items():
            if lvl == "LITERATURE_ANALOG":
                assert b17_registry["gate"][
                    "C_thermal_inertia_or_time_constant"] is False, (
                    f"{k} 为不可迁移的 analog, 却让 C 条件通过")


def test_existing_T_not_auto_direct(b17_registry):
    """§17 / §8 —— 既有 `T` 遥测不得因既已存在而自动标 DIRECT。"""
    e = b17_registry["existing_temperature_signal_audit"]
    assert e["auto_labelled_DIRECT"] is False
    assert e["provenance_level"] != "DIRECT", (
        "项目内部经验模型不得记 DIRECT")
    assert e["role"] == "implementation_reference"
    assert e["may_serve_as_provenance_proof"] is False
    # 四个 §8 问题必须都有回答
    for q in ("q1_is_real_thermal_state_or_synthetic",
              "q2_depends_on_ground_truth_degradation_params",
              "q3_contains_wear_or_friction_heating",
              "q4_has_independent_R_C_provenance"):
        assert q in e, f"§8 问题未回答: {q}"
    # 无独立 R/C provenance 是本阶段的关键事实
    assert e["q4_has_independent_R_C_provenance"] is False
    assert e["has_energy_balance"] is False, (
        "若 wheel_sim 真有能量平衡, 本阶段结论需重审")
    # 且它的硬编码参数确实不在 config 里 (审计声称如此, 这里独立复核)
    src = (ROOT / "src/sim/wheel_sim.py").read_text(encoding="utf-8")
    cfg = (ROOT / "configs/wheel.yaml").read_text(encoding="utf-8")
    assert "T_base" in src and "T_amp" in src
    assert "T_base" not in cfg and "T_amp" not in cfg, (
        "T_base/T_amp 已进入 config —— 审计结论需更新")


def test_gate_decision_consistent(b17_registry):
    """Gate 的四条件与最终 verdict 必须自洽 —— 不允许条件不满足却报 PASS。"""
    g = b17_registry["gate"]
    expect = (g["A_direct_operating_temperature_max"]
              and g["B_heat_generation_mapping"]
              and g["C_thermal_inertia_or_time_constant"]
              and g["D_not_all_assumed"])
    assert g["all_pass"] == expect
    assert b17_registry["verdict"] == (
        "B17_THERMAL_PROVENANCE_PASS" if expect
        else "B17_THERMAL_PROVENANCE_INSUFFICIENT")
