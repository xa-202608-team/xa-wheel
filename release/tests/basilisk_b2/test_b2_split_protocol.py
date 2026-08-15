"""tests/basilisk_b2/test_b2_split_protocol.py —— §2 划分协议不变量。

划分是所有后续数字的地基: 一旦轨迹级泄漏或分层失效, 主指标就没有意义。
本文件把 §2 的四条要求逐条钉死, 并且**独立重算**划分而非只读 JSON。
"""
from __future__ import annotations

import numpy as np


def test_split_is_trajectory_level_and_disjoint(b2_split):
    """train/val/test 两两不相交且并集覆盖全部 150 条 —— 轨迹级无泄漏。"""
    s = b2_split["splits"]
    tr, va, te = (set(s[k]["tids"]) for k in ("train", "val", "test"))
    assert tr & va == set(), tr & va
    assert tr & te == set(), tr & te
    assert va & te == set(), va & te
    assert len(tr | va | te) == b2_split["n_traj"] == 150
    assert b2_split["level"] == "trajectory"


def test_split_ratios_match_protocol(b2_split, b2_config):
    """比例 30/20/50, 且实际条数与比例一致 (允许取整偏差 <= 1 条)。"""
    sc = b2_config["b2"]["split"]
    assert (sc["train"], sc["val"], sc["test"]) == (0.30, 0.20, 0.50)
    n = b2_split["n_traj"]
    for k, r in (("train", 0.30), ("val", 0.20), ("test", 0.50)):
        got = b2_split["splits"][k]["n"]
        assert abs(got - r * n) <= 1.0, (k, got, r * n)


def test_split_stratifies_event_and_censored(b2_split):
    """event/censored 在三个 split 里都按比例出现。

    这是 §2 的分层要求: 若某个 split 里 event 比例严重偏离全局, 主指标
    (只在 event 轨迹上可算) 的样本量就会失衡。容差 0.12 是绝对比例差。
    """
    assert b2_split["n_event_observed"] == 71
    assert b2_split["n_censored"] == 79
    glob = 71.0 / 150.0
    for k in ("train", "val", "test"):
        s = b2_split["splits"][k]
        assert s["n_event"] > 0 and s["n_censored"] > 0, k
        assert abs(s["event_frac"] - glob) < 0.12, (k, s["event_frac"], glob)


def test_split_is_deterministic_given_seed(b2_split_mod, b2_config, b2_split):
    """重算两次结果完全相同, 且与冻结的 split.json 逐字一致。"""
    a = b2_split_mod.build_split(b2_config)
    b = b2_split_mod.build_split(b2_config)
    assert a["split_sha256"] == b["split_sha256"]
    assert a["split_sha256"] == b2_split["split_sha256"]
    for k in ("train", "val", "test"):
        assert a["splits"][k]["tids"] == b2_split["splits"][k]["tids"], k


def test_split_seed_is_frozen_value(b2_split, b2_config_raw):
    """split_seed = 20260810 写在 b2 config 自己的文件里, 不是继承来的默认值。"""
    assert b2_split["split_seed"] == 20260810
    assert b2_config_raw["b2"]["split"]["split_seed"] == 20260810


def test_different_seed_changes_split(b2_split_mod, b2_config):
    """哨兵: 换 split_seed 必须真的换出不同划分。

    若不成立, 说明划分根本没用到 seed —— 那么"确定性"就是假的确定性
    (任何 seed 都给同一结果), 上面的确定性测试也就失去意义。
    """
    import copy
    c2 = copy.deepcopy(b2_config)
    c2["b2"]["split"]["split_seed"] = 20260811
    other = b2_split_mod.build_split(c2)
    base = b2_split_mod.build_split(b2_config)
    assert other["split_sha256"] != base["split_sha256"]


def test_censored_trajectories_never_get_eol(b2_split_mod, b2_config):
    """§4: 删失轨迹不得被赋予 EOL —— 分层键读的是 event_observed 属性。"""
    tids, ev, key = b2_split_mod.read_traj_meta(
        b2_split_mod.ROOT / b2_config["transfer"]["target_feature_path"])
    assert len(tids) == 150
    assert int(ev.sum()) == 71
    assert int((~ev).sum()) == 79
    # censored 的分层键是观测长度 (全部相同的 52596), 不是任何"推断的 EOL"
    assert np.all(np.isfinite(key))
    assert len(np.unique(key[~ev])) == 1, "censored 键应为统一的观测长度"
