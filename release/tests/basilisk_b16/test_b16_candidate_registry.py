"""tests/basilisk_b16/test_b16_candidate_registry.py —— §15: 候选来自运行时枚举。"""
from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCR = ROOT / "scripts" / "basilisk_b16"

# 已知轮型名 —— 只用于**证明脚本里没有它们**, 不用于生成候选。
KNOWN_WHEEL_NAMES = ("HR12", "HR14", "HR16", "RWP015", "RW0", "RW4",
                     "Honeywell", "NanoAvionics", "BCT", "Blue Canyon")


def test_candidates_enumerated_from_runtime(b16_registry, b16_inventory):
    """候选必须来自 Basilisk 运行时反射, 且注册表与清单一一对应。"""
    assert b16_registry["candidates_enumerated_from_runtime"] is True
    inv_names = {w["model_name"] for w in b16_inventory["wheels"]}
    reg_names = {c["wheel_model"] for c in b16_registry["candidates"]}
    assert inv_names == reg_names, "注册表候选与运行时清单不一致"
    assert len(reg_names) == b16_registry["n_candidates"] >= 1
    # 清单必须记录反射来源文件与其哈希 (可复现)
    src = ROOT / b16_inventory["basilisk_source_file"] \
        if not Path(b16_inventory["basilisk_source_file"]).is_absolute() \
        else Path(b16_inventory["basilisk_source_file"])
    assert len(b16_inventory["basilisk_source_sha256"]) == 64
    if src.exists():
        assert hashlib.sha256(src.read_bytes()).hexdigest() == \
            b16_inventory["basilisk_source_sha256"]


def test_no_hardcoded_candidate_names():
    """枚举 / 注册 / 打分脚本里不得出现任何轮型名字面量。

    §6: "候选数量由实际审计决定; 不要预先硬编码候选名字。" 若脚本里写了
    `if name == "Honeywell_HR16"`, 那候选集就是我挑的, 不是环境给的。
    """
    for fname in ("enumerate_wheel_models.py", "build_candidate_registry.py",
                  "score_provenance.py"):
        txt = io.open(SCR / fname, encoding="utf-8").read()
        # 只检查代码行 —— 注释与 docstring 里解释陷阱时可以提型号名
        code = []
        in_doc = False
        for line in txt.splitlines():
            s = line.strip()
            if s.startswith('"""') or s.startswith("'''"):
                # 单行 docstring 不切换状态
                if len(s) < 6 or not s.endswith(s[:3]):
                    in_doc = not in_doc
                continue
            if in_doc or s.startswith("#"):
                continue
            code.append(line.split("#")[0])
        blob = "\n".join(code)
        for nm in KNOWN_WHEEL_NAMES:
            assert nm not in blob, \
                f"{fname} 代码行里出现硬编码轮型名 {nm!r}"


def test_registry_records_undecidable_state(b16_registry):
    """不可判定必须有独立状态, 不得被压成 True/False。"""
    lbl = b16_registry["undecidable_label"]
    assert lbl == "UNDECIDABLE_IN_B16"
    vals = {c["direct_overspeed_limit"]["usable_as_degradation_EOL"]
            for c in b16_registry["candidates"] if c.get("basilisk_supported")}
    # 允许 True / False / UNDECIDABLE 三态, 不允许出现别的东西
    assert vals <= {True, False, lbl}, vals


def test_direct_source_required(b16_registry, b16_sources):
    """任何 DIRECT 级限值必须有可指认来源; 自述 estimate 的器件参数不得记 DIRECT。

    关键区分 (§5 / protocol §3): "自述 estimate" 只污染**该 preset 源码自带的
    器件参数** (omega/u/H/摩擦标称值), 不污染从厂商官网**逐字读到**的规格 ——
    后者的 provenance 来自官网, 与源码注释无关。混为一谈会把真实的 DIRECT 依据
    错误降级。
    """
    assert b16_sources["model_memory_used_to_fill_parameters"] is False
    for c in b16_registry["candidates"]:
        if not c.get("basilisk_supported"):
            continue
        assert c["datasheet_source"], f"{c['wheel_model']} 无来源记录"
        est = c["self_declared_estimate_markers"]
        lvl = c["device_param_provenance"]
        assert lvl == ("ASSUMED" if est else "DIRECT"), \
            f"{c['wheel_model']} 器件参数级别 {lvl} 与自述标记 {est} 不符"
        # 源码自带的器件参数必须跟随该级别
        for key in ("direct_speed_limit", "direct_torque_limit",
                    "direct_momentum_limit"):
            assert c[key]["provenance"] == lvl, \
                f"{c['wheel_model']}.{key} 级别未跟随 device_param_provenance"
        # 任何记为 DIRECT 的失效相关限值都必须附逐字来源行
        for key in ("direct_friction_limit", "direct_thermal_limit",
                    "direct_current_limit"):
            if c[key]["provenance"] == "DIRECT":
                assert c[key]["lines"], \
                    f"{c['wheel_model']}.{key} 记为 DIRECT 却无逐字来源行"


def test_assumed_device_params_cannot_yield_eligible_limit(b16_registry,
                                                           b16_scores):
    """自述 estimate 的 preset, 其速度/动量限不得成为可用 EOL (§9)。"""
    est_models = {c["wheel_model"] for c in b16_registry["candidates"]
                  if c.get("basilisk_supported")
                  and c["self_declared_estimate_markers"]}
    for m in est_models:
        row = next(r for r in b16_scores["rows"] if r["wheel_model"] == m)
        for u in row["usable_limits"]:
            assert u["provenance"] != "ASSUMED", \
                f"{m} 的 ASSUMED 阈值进了可用集"


def test_forbidden_sources_not_used(b16_sources):
    """§4: 博客 / CSDN / 知乎 / 问答 / 聚合站不得作为来源。"""
    assert b16_sources["forbidden_sources_encountered"] == [] or \
        all(not p.get("used_as_provenance", False)
            for p in b16_sources["forbidden_sources_encountered"])
    for p in b16_sources["probes"]:
        u = p["url"].lower()
        for bad in ("csdn", "zhihu", "blog", "bbs", "baidu", "cnblogs"):
            assert bad not in u, f"探测了禁止来源 {p['url']}"


def test_network_state_recorded_honestly(b16_sources):
    """网络不可达必须记 NETWORK_UNAVAILABLE, 不得用模型记忆补参数。"""
    assert b16_sources["network_state"] in (
        "NETWORK_UNAVAILABLE", "NETWORK_PARTIAL", "NETWORK_OK")
    n_ok = b16_sources["n_reachable"]
    n_all = b16_sources["n_probes"]
    assert 0 <= n_ok <= n_all
    if n_ok == 0:
        assert b16_sources["network_state"] == "NETWORK_UNAVAILABLE"
    elif n_ok < n_all:
        assert b16_sources["network_state"] == "NETWORK_PARTIAL"
    # 不可达探测必须留下 code 000 之类的实证, 不能只写一句"没连上"
    for p in b16_sources["probes"]:
        assert "http_code" in p
