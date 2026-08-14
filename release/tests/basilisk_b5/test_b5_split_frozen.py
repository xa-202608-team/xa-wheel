"""tests/basilisk_b5/test_b5_split_frozen.py —— §2/§3/§11 划分与输入 schema 冻结。

本阶段的地基是 B2.1 的冻结划分。B5 **不重划、不重分层、不改 bin 边界**。
输入 schema 固定 CORE_ONLY (§3), 因为 B3X_NO_STABILIZING_SIGNAL 已经否掉了
mission 特征这条路。
"""
from __future__ import annotations

import json

from conftest import ROOT, code_only

B21_SPLIT_SHA = "23e2b94445e698eab263ca17a36d99f7130eb2e8c5c3b9b8d0bd9ba772e5c932"


def test_b5_b21_split_exact(b5_data_mod, b21_split_ro):
    """test_b5_b21_split_exact (§21) —— 脚本常量、manifest、B2.1 三方一致。"""
    assert b5_data_mod.B5_EXPECTED_SPLIT_SHA256 == B21_SPLIT_SHA
    assert str(b21_split_ro["split_sha256"]) == B21_SPLIT_SHA
    assert b5_data_mod.SPLIT_MISMATCH_LABEL == "B5_INVALID"


def test_b5_config_pins_same_split(b5_config):
    s = b5_config["b5"]["split"]
    assert str(s["expected_sha256"]) == B21_SPLIT_SHA
    assert str(s["require_b21_verdict"]) == "B21_GENERALIZATION_PASS"
    assert bool(s["forbid_restratify"]) is True


def test_b5_metrics_used_the_frozen_split(b5_metrics):
    assert str(b5_metrics["split_sha256"]) == B21_SPLIT_SHA


def test_b5_same_target_data(b5_metrics, b21_split_ro):
    """test_b5_same_target_data (§21) —— 三组用同一份 train/val/test IDs。"""
    n = b5_metrics["split_n"]
    for k in ("train", "val", "test"):
        assert int(n[k]) == len(b21_split_ro["splits"][k]["tids"]), \
            f"{k} 规模与 B2.1 不一致"
    assert int(n["train"]) == 45 and int(n["val"]) == 31 and int(n["test"]) == 74


def test_b5_input_schema_is_core_only(b5_metrics, b5_config, b3x_summary_ro):
    """§3: 因 B3X_NO_STABILIZING_SIGNAL, 正式输入固定 CORE_ONLY, 无 mission 特征。"""
    assert str(b5_metrics["input_schema"]) == "core_only"
    sc = b5_config["b5"]["input_schema"]
    assert str(sc["formal"]) == "CORE_ONLY"
    # schema 由 B3X 的判定决定, 不是本阶段临时挑的
    assert str(sc["rule"]["B3X_NO_STABILIZING_SIGNAL"]) == "core_only"
    # 泄漏列必须被显式列为禁止输入
    forb = set(sc["forbidden_input_cols"])
    assert {"hi_damage_obs", "D_true", "rul", "eol_idx"} <= forb
    assert bool(sc["forbid_mission_features"]) is True
    assert bool(sc["shared_across_groups"]) is True
    assert str(b5_metrics["input_schema_decided_by"]) != ""
    assert str(b3x_summary_ro["verdict"]) == "B3X_NO_STABILIZING_SIGNAL"


def test_b5_no_leakage_features_in_input(b5_data_mod):
    """输入端绝不能出现 hi_damage_obs / D_true / 未来任务统计 / EOL 信息。"""
    src = code_only("scripts/basilisk_b5/data_b5.py")
    for bad in ("D_true", "future_mission", "eol_lookup", "true_rul_lookup"):
        assert bad not in src, f"data_b5 出现疑似泄漏字段 {bad}"


def test_b5_bins_from_b21(b5_lifetime, b21_split_ro):
    """test_b5_bins_from_b21 (§21/§11) —— bin 边界来自 B2.1, 不按 B5 test 重算。"""
    b = b5_lifetime["bins"]
    exp = [float(x) for x in b21_split_ro["lifetime_bins"]["event"]["edges"]]
    assert [float(x) for x in b["edges"]] == exp
    assert list(b["names"]) == ["short", "medium", "long"]
    assert str(b["source"]).startswith("b21")
    assert bool(b["recomputed_on_b5_test"]) is False


def test_b5_lifetime_bin_source_is_single_definition():
    """bin 定义只允许有一处 —— B5 从 B2.1 import, 不复制一份。"""
    src = code_only("scripts/basilisk_b5/analyze_lifetime_bins.py")
    assert "lifetime_bin_map" in src, "未从 B2.1 复用 bin 映射"
    assert "np.digitize" not in src, "B5 不得自行 digitize 重算 bin"


def test_b5_split_ids_hash_matches_contract(b5_contract, b21_split_ro):
    import hashlib
    num = {}
    for k, v in b5_contract.items():
        if k.endswith("_results") and isinstance(v, dict):
            num.update(v)

    def _h(ids):
        return hashlib.sha256(
            "\n".join(sorted(str(x) for x in ids)).encode("utf-8")).hexdigest()

    for k in ("train", "val", "test"):
        assert _h(b21_split_ro["splits"][k]["tids"]) == \
            num[f"b21_{k}_ids_sha256"], \
            f"{k} ID 集合哈希与契约不符 —— B5_INVALID"


def test_b5_lifetime_coverage_complete(b21_split_ro):
    """B2.1 的立论前提: train/val/test 的寿命覆盖都完整 (最短寿命都进得去)。"""
    sp = b21_split_ro["splits"]
    mins = {k: float(sp[k]["event_eol_min"]) for k in ("train", "val", "test")}
    # B2.1 的核心修复: train 的最短事件寿命不高于 test 的 —— 否则短寿命不可学
    assert mins["train"] <= mins["test"], f"train 未覆盖 test 的短寿命端: {mins}"
    assert max(mins.values()) - min(mins.values()) < 5000, \
        f"三个划分的最短事件寿命差异过大, 覆盖不匹配: {mins}"
    for k in ("train", "val", "test"):
        assert set(sp[k]["event_bins_present"]) == {"short", "medium", "long"}
