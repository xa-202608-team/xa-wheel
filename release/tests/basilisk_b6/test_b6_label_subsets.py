"""tests/basilisk_b6/test_b6_label_subsets.py —— §5/§6/§7 划分与标签子集纪律。

- 划分完全复用 B2.1, 哈希逐字符相同, val/test 永不改动 (§5)
- 稀缺轴是 n_event_labeled, **不是 n_train** —— 每档始终保留全部 24 条 censored (§6)
- 未被选中的 event 轨迹必须**完全移除**: 不得改标 censored, 不得以 unlabelled
  形式塞回, 不得使用其 EOL / 未来信息 (§6)
- bin 边界取自 B2.1 冻结 manifest, **不得从 B6 test 重算** (§7)
- 子集 manifest 在任何训练之前生成并冻结, 哈希可复算 (§7)
"""
from __future__ import annotations

import hashlib
import json

from conftest import ROOT, code_nospace, code_only

B6_SCRIPTS = ("scripts/basilisk_b6/build_label_subsets.py",
              "scripts/basilisk_b6/audit_label_subsets.py",
              "scripts/basilisk_b6/data_b6.py",
              "scripts/basilisk_b6/run_formal_matrix.py")

FROZEN_EDGES = [26846.0, 37395.0]
SPLIT_SHA = "23e2b94445e698eab263ca17a36d99f7130eb2e8c5c3b9b8d0bd9ba772e5c932"


def test_b6_split_exactly_b21(b6_manifest, b21_split_ro, b6_config):
    """test_b6_split_exactly_b21 (§23/§5) —— 划分完全复用 B2.1, 不重划。"""
    assert str(b6_manifest["split_sha256"]) == str(b21_split_ro["split_sha256"])
    assert str(b6_manifest["split_sha256"]) == SPLIT_SHA
    assert str(b6_manifest["split_source"]) == "docs/basilisk_b21/split_manifest.json"
    sp = b6_config["b6"]["split"]
    assert str(sp["expected_sha256"]) == SPLIT_SHA
    assert bool(sp["forbid_restratify"]) is True
    assert str(sp["require_b21_verdict"]) == "B21_GENERALIZATION_PASS"
    assert (int(sp["n_train"]), int(sp["n_train_event"]),
            int(sp["n_train_censored"]), int(sp["n_val"]),
            int(sp["n_test"])) == (45, 21, 24, 31, 74)
    tr = b21_split_ro["splits"]["train"]
    assert int(tr["n"]) == 45
    assert int(b21_split_ro["splits"]["val"]["n"]) == 31
    assert int(b21_split_ro["splits"]["test"]["n"]) == 74


def test_b6_val_test_never_touched(b6_manifest, b21_split_ro):
    """§5: val / test 永远不动 —— manifest 不得携带任何 val/test 改动。"""
    assert bool(b6_manifest["val_test_never_touched"]) is True
    blob = json.dumps(b6_manifest, ensure_ascii=False)
    for name in ("val", "test"):
        ids = list(b21_split_ro["splits"][name]["tids"])
        # manifest 只应记录 train 侧清单; val/test 的 tid 清单不该被复制改写
        assert f'"{name}_tids"' not in blob, f"manifest 里出现 {name}_tids —— 越界"
        assert len(ids) == (31 if name == "val" else 74)


def test_b6_label_subset_manifest_frozen(b6_manifest, b6_protocol_hash):
    """test_b6_label_subset_manifest_frozen (§23/§7) —— 哈希可复算且已冻结。"""
    core = {
        "levels": b6_manifest["levels"],
        "subset_seed": b6_manifest["subset_seed"],
        "bin_names": b6_manifest["bin_names"],
        "bin_edges": b6_manifest["bin_edges"],
        "per_level_bin_quota": b6_manifest["per_level_bin_quota"],
    }
    got = hashlib.sha256(json.dumps(core, sort_keys=True,
                                    ensure_ascii=False).encode("utf-8")).hexdigest()
    assert got == str(b6_manifest["subset_manifest_sha256"]), \
        "manifest 内容与其自身哈希不符 —— 被改动过"
    assert str(b6_manifest["subset_manifest_sha256"]) == \
        "f7fe1ababe7b23db5d0ba41a7c644505267f684400a22aebb5144bf6ce041b1d"
    assert bool(b6_manifest["forbid_change_after_written"]) is True
    assert bool(b6_manifest["generated_before_any_training"]) is True
    assert str(b6_manifest["protocol_sha256"]) == \
        str(b6_protocol_hash["protocol_sha256"])


def test_b6_subset_seed_and_edges_frozen(b6_manifest, b6_protocol_hash,
                                         b21_split_ro):
    """§7: subset_seed 与 bin 边界都取自冻结件, 不得从 B6 test 重算。"""
    ss = b6_protocol_hash["subset_selection"]
    assert int(b6_manifest["subset_seed"]) == 20260814 == int(ss["subset_seed"])
    assert [float(x) for x in b6_manifest["bin_edges"]] == FROZEN_EDGES
    assert [float(x) for x in ss["bin_edges"]] == FROZEN_EDGES
    assert [float(x) for x in
            b21_split_ro["lifetime_bins"]["event"]["edges"]] == FROZEN_EDGES
    assert bool(b6_manifest["forbid_recompute_edges_from_b6_test"]) is True
    assert list(b6_manifest["bin_names"]) == ["short", "medium", "long"]


def _lv(mf: dict, n: int) -> dict:
    return mf["levels"][str(int(n))]


def test_b6_n3_bin_coverage(b6_manifest):
    """test_b6_n3_bin_coverage (§23/§7) —— n=3 每 bin 各 1 条。"""
    L = _lv(b6_manifest, 3)
    assert int(L["n_event_labeled"]) == 3
    assert dict(L["event_bin_counts"]) == {"short": 1, "medium": 1, "long": 1}
    assert len(L["event_tids"]) == 3
    assert int(L["n_train_total"]) == 27


def test_b6_n5_bin_coverage(b6_manifest):
    """test_b6_n5_bin_coverage (§23/§7) —— n=5 (PRIMARY) 2/1/2。"""
    L = _lv(b6_manifest, 5)
    assert int(L["n_event_labeled"]) == 5
    assert str(L["role"]) == "PRIMARY"
    assert dict(L["event_bin_counts"]) == {"short": 2, "medium": 1, "long": 2}
    assert len(L["event_tids"]) == 5
    assert int(L["n_train_total"]) == 29


def test_b6_n10_bin_coverage(b6_manifest):
    """test_b6_n10_bin_coverage (§23/§7) —— n=10 为 3/3/4。"""
    L = _lv(b6_manifest, 10)
    assert int(L["n_event_labeled"]) == 10
    assert dict(L["event_bin_counts"]) == {"short": 3, "medium": 3, "long": 4}
    assert len(L["event_tids"]) == 10
    assert int(L["n_train_total"]) == 34


def test_b6_n21_is_all_train_events(b6_manifest, b21_split_ro):
    """n=21 必须正好是 B2.1 train 的全部 event 轨迹, 且无移除。"""
    L = _lv(b6_manifest, 21)
    assert int(L["n_event_labeled"]) == 21
    assert dict(L["event_bin_counts"]) == {"short": 7, "medium": 7, "long": 7}
    assert list(L["removed_event_tids"]) == []
    assert int(L["n_train_total"]) == 45 == int(
        b21_split_ro["splits"]["train"]["n"])


def test_b6_all_censored_train_retained(b6_manifest):
    """test_b6_all_censored_train_retained (§23/§6) —— 每档全部 24 条 censored。"""
    assert int(b6_manifest["n_censored_train_always"]) == 24
    assert bool(b6_manifest["always_keep_all_censored_train"]) is True
    base = list(b6_manifest["censored_train_tids"])
    assert len(base) == 24 == len(set(base))
    for n in (3, 5, 10, 21):
        L = _lv(b6_manifest, n)
        assert list(L["censored_tids"]) == base, f"n={n} 的 censored 集合被改动"
        assert int(L["n_censored"]) == 24
        assert int(L["n_train_total"]) == int(L["n_event_labeled"]) + 24
        assert set(base) <= set(L["train_tids"])


def test_b6_scarcity_axis_is_labels_not_total(b6_manifest, b6_config):
    """§6: 稀缺轴是失效标签数量, 不是 target 轨迹总数。"""
    assert str(b6_manifest["scarcity_axis"]) == \
        "event_observed_failure_labelled_trajectories"
    assert str(b6_manifest["scarcity_axis_is_not"]) == "total_target_trajectories"
    ls = b6_config["b6"]["label_scarcity"]
    assert str(ls["axis"]) == "n_event_labeled"
    assert bool(ls["forbid_defining_scarcity_as_total_n_train"]) is True
    assert [int(x) for x in ls["levels"]] == [3, 5, 10, 21]
    # train 总数必须随 event 数变化, 而 censored 恒为 24
    tc = {int(k): v for k, v in ls["train_composition"].items()}
    for n, exp in ((3, 27), (5, 29), (10, 34), (21, 45)):
        assert int(tc[n]["censored"]) == 24
        assert int(tc[n]["event"]) == n
        assert int(tc[n]["total"]) == exp


def test_b6_unused_events_not_relabelled_censored(b6_manifest, b21_split_ro):
    """test_b6_unused_events_not_relabelled_censored (§23/§6)。

    未被选中的 event 轨迹必须完全从该档 train 移除 —— 既不在 train_tids 里,
    也没有被塞进 censored_tids (那是伪造删失: 我们已知其真 EOL)。
    """
    base_cen = set(b6_manifest["censored_train_tids"])
    all_ev = set(_lv(b6_manifest, 21)["event_tids"])
    assert len(all_ev) == 21
    assert not (all_ev & base_cen), "event 与 censored 集合有交叠"
    for n in (3, 5, 10, 21):
        L = _lv(b6_manifest, n)
        sel, rem = set(L["event_tids"]), set(L["removed_event_tids"])
        assert sel | rem == all_ev, f"n={n} 的 选中+移除 != 全部 event"
        assert not (sel & rem), f"n={n} 同一条既选中又移除"
        assert len(rem) == 21 - n
        assert not (rem & set(L["train_tids"])), \
            f"n={n} 移除的 event 仍在 train 里"
        assert not (rem & set(L["censored_tids"])), \
            f"n={n} 移除的 event 被改标成 censored —— 伪造删失"
        # 被移除轨迹的 EOL 不得出现在该档记录里 (那等于用了未来信息)
        assert not (rem & set(L.get("event_eol", {}).keys())), \
            f"n={n} 记录了被移除轨迹的 EOL"
    assert bool(b6_manifest["forbid_relabel_known_eol_as_censored"]) is True
    assert bool(b6_manifest["forbid_smuggling_as_unlabelled_input"]) is True


def test_b6_levels_are_nested(b6_manifest):
    """§7: 各档嵌套 (n=3 ⊂ n=5 ⊂ n=10 ⊂ n=21), 否则趋势会被"换了一组轨迹"混淆。"""
    seq = [3, 5, 10, 21]
    for lo, hi in zip(seq, seq[1:]):
        a = set(_lv(b6_manifest, lo)["event_tids"])
        b = set(_lv(b6_manifest, hi)["event_tids"])
        assert a <= b, f"n={lo} 不是 n={hi} 的子集"
    checks = {(int(c["lower"]), int(c["upper"])): c
              for c in b6_manifest["nesting_check"]}
    for lo, hi in zip(seq, seq[1:]):
        assert bool(checks[(lo, hi)]["subset"]) is True


def test_b6_quota_within_available(b6_manifest, b6_protocol_hash):
    """每档每 bin 的配额不得超过该 bin 在 B2.1 train 中的可用 event 数。"""
    avail = {k: int(v) for k, v in b6_manifest["available_event_per_bin"].items()}
    assert avail == {"short": 7, "medium": 7, "long": 7}
    for n, q in b6_manifest["per_level_bin_quota"].items():
        assert sum(int(v) for v in q.values()) == int(n), f"n={n} 配额和不等于 n"
        for b, v in q.items():
            assert int(v) <= avail[b], f"n={n} 的 {b} 配额超出可用数"
    assert b6_manifest["per_level_bin_quota"] == \
        b6_protocol_hash["subset_selection"]["per_level_bin_quota"]


def test_b6_no_recompute_bins_from_b6_test():
    """§7 禁令: 不得在 B6 脚本里重算寿命 bin 边界。"""
    for rel in B6_SCRIPTS:
        src = code_nospace(rel)
        for bad in ("np.quantile", "np.percentile", "quantile("):
            assert bad.replace(" ", "") not in src, \
                f"{rel} 出现重算分位数的写法: {bad}"
        assert "lifetime_bin_map(" not in src or "test" not in rel


def test_b6_no_global_seed_in_b6_scripts():
    """禁止全局 np.random.seed() —— 只允许局部 default_rng / torch generator。"""
    for rel in B6_SCRIPTS:
        src = code_nospace(rel)
        assert "np.random.seed(" not in src, f"{rel} 使用了全局 np.random.seed"
        assert "random.seed(" not in src.replace("np.random.seed(", ""), \
            f"{rel} 使用了全局 random.seed"


def test_b6_manifest_matches_audit_doc(b6_manifest):
    """§24-4 的人工确认文档必须与 manifest 逐条一致 (不是另写一份数字)。"""
    p = ROOT / "docs" / "basilisk_b6" / "label_subset_manifest.md"
    assert p.exists(), "缺 §25 要求的 label_subset_manifest.md"
    txt = p.read_text(encoding="utf-8")
    assert str(b6_manifest["subset_manifest_sha256"]) in txt
    for n in (3, 5, 10, 21):
        for t in _lv(b6_manifest, n)["event_tids"]:
            assert t in txt, f"文档缺 n={n} 的 {t}"


def test_b6_forbid_mission_feature_arm(b6_config):
    """§4: 输入 schema 沿用 B5 的 CORE_ONLY, 不得恢复 mission-feature arm。"""
    isch = b6_config["b6"]["input_schema"]
    assert str(isch["formal"]) == "CORE_ONLY"
    assert int(isch["n_features"]) == 12
    assert bool(isch["forbid_mission_features"]) is True
    assert bool(isch["forbid_restoring_mission_arm"]) is True
    assert bool(b6_config["b6"]["forbid_mission_feature_arm"]) is True
    for rel in B6_SCRIPTS:
        src = code_only(rel)
        assert "core_plus_mission" not in src, f"{rel} 出现 mission arm"
