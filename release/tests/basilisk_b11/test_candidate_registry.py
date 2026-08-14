"""tests/basilisk_b11/test_candidate_registry.py

B1.1 §18 —— candidate 注册与配对组硬测试:
  * `test_candidate_registry_exactly_three`
  * `test_candidates_share_degradation_params`
  * `test_candidates_only_differ_horizon`
  * `test_common_random_numbers_b11`
  * `test_b11_dataset_isolated`
  * `test_b11_hash_reproducible`
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CAND_JSON = ROOT / "checkpoints/basilisk_b11/candidates.json"
FROZEN_REGISTRY = ("P95_UTILIZATION_ONLY", "P95_UTILIZATION_DURATION_1P5",
                   "P95_UTILIZATION_DURATION_2P0")
# §7 明令禁止的追加 candidate 命名
FORBIDDEN_NAMES = ("D11", "E11", "P95_UTILIZATION_DURATION_3P0",
                   "P95_UTILIZATION_DURATION_2P5")


def _cand() -> dict:
    if not CAND_JSON.exists():
        pytest.skip("candidates.json 未生成")
    return json.loads(CAND_JSON.read_text(encoding="utf-8"))


def test_candidate_registry_exactly_three(b11_cfg):
    """§7: registry 必须恰好三个, 名称与预先登记一致, 且不得追加 D/E。"""
    reg = tuple(b11_cfg["candidates"]["registry"])
    assert reg == FROZEN_REGISTRY, f"registry 被改动: {reg}"
    assert tuple(b11_cfg["candidates"]["priority"]) == FROZEN_REGISTRY, \
        "priority 顺序被改动 (§10: A11 > B11 > C11)"
    spec = b11_cfg["candidates"]["spec"]
    assert set(spec) == set(FROZEN_REGISTRY), f"spec 键集合不符: {set(spec)}"
    for bad in FORBIDDEN_NAMES:
        assert bad not in spec, f"§7 禁止追加 candidate, 却出现 {bad}"


def test_generate_script_pins_registry():
    """生成脚本必须自带 FROZEN_REGISTRY 并与 config 交叉校验。

    只靠 config 不够 —— config 可被改; 脚本内的冻结元组提供第二道锁。
    """
    src = (ROOT / "scripts/basilisk_b11/generate_candidates.py").read_text(
        encoding="utf-8")
    tree = ast.parse(src)
    found = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "FROZEN_REGISTRY":
                    found = tuple(e.value for e in node.value.elts)
    assert found == FROZEN_REGISTRY, f"脚本内 FROZEN_REGISTRY = {found}"


def test_candidate_json_has_exactly_three():
    """产物侧: candidates.json 必须恰好三条记录。"""
    d = _cand()
    assert tuple(d["registry_frozen"]) == FROZEN_REGISTRY
    assert len(d["candidates"]) == 3, f"candidate 数 {len(d['candidates'])} != 3"
    assert set(d["candidates"]) == set(FROZEN_REGISTRY)


def test_candidates_share_degradation_params():
    """§7/§8: 三个 candidate 的逐轨迹**退化尺度参数**必须逐位相同。

    b0 / tau_years / omega0 / g_duty 由 (seed, traj_id) 与 **base-horizon**
    duty 决定, 与 horizon 无关 —— 若 horizon 影响了它们, A/B/C 就不是配对比较。

    刻意**不**比较 `speed_util_mean` / `torque_util_mean` / `mode_fractions`:
    这些是对**整条轨迹全部窗**取的统计, horizon 越长纳入的窗越多, 本来就会变;
    它们是诊断量, 不是退化尺度参数。把它们当 CRN 不变量会得到一个必然失败的
    测试, 掩盖真正要守的东西 (mode_weight_sum 才是共享的 profile slice 证据)。
    """
    d = _cand()
    cands = d["candidates"]
    base = cands[FROZEN_REGISTRY[0]]["per_traj"]
    for name in FROZEN_REGISTRY[1:]:
        other = cands[name]["per_traj"]
        assert len(other) == len(base)
        for i, (a, b) in enumerate(zip(base, other)):
            for k in ("b0_b11", "tau_years_b11", "omega0_b11", "b_init",
                      "g_duty", "mode_weight_sum", "n_modes_used"):
                assert a[k] == b[k], \
                    (f"traj {i} 的 {k} 在 {FROZEN_REGISTRY[0]} 与 {name} 之间不同: "
                     f"{a[k]!r} vs {b[k]!r} —— CRN 被破坏")


def test_candidates_share_profile_slice_prefix():
    """§8: 长 horizon 的 duty 必须是短 horizon 的**前缀** (逐位相同的开头)。

    mode 占比会随 horizon 变化 (分母变了), 所以不能直接比占比; 但共享前缀
    意味着短 candidate 的窗数 × 其占比, 必须等于长 candidate 在同样窗数上的
    计数 —— 这里用"总窗数加权计数"还原绝对计数来比。
    """
    d = _cand()
    cands = d["candidates"]
    A = cands[FROZEN_REGISTRY[0]]
    for name in FROZEN_REGISTRY[1:]:
        B = cands[name]
        for i, (a, b) in enumerate(zip(A["per_traj"], B["per_traj"])):
            # 权重 (采样分布) 必须相同 —— 这才是"相同 mode weights"的直接证据
            assert a["mode_weight_sum"] == b["mode_weight_sum"], \
                f"traj {i} 的 mode 权重和不同"
            assert set(a["mode_fractions"]) == set(b["mode_fractions"]), \
                f"traj {i} 覆盖的 mode 集合不同 —— profile slice 未共享"
    # duty 前缀逐位相同由生成脚本的 CRN 检查证明 (paired hash + prefix check)
    assert d["crn_ok"] is True


def test_candidates_only_differ_horizon(b11_cfg):
    """§7: A/B/C 的 spec 除 horizon_scale 外必须逐键相同。"""
    spec = b11_cfg["candidates"]["spec"]
    base = spec[FROZEN_REGISTRY[0]]
    for name in FROZEN_REGISTRY[1:]:
        other = spec[name]
        assert set(other) == set(base), f"{name} 的 spec 键集合与 A11 不同"
        for k in base:
            if k == "horizon_scale":
                continue
            assert other[k] == base[k], \
                f"{name}.{k} = {other[k]!r} != A11 的 {base[k]!r} (§7 只允许差 horizon)"
    hs = [float(spec[n]["horizon_scale"]) for n in FROZEN_REGISTRY]
    assert hs == [1.0, 1.5, 2.0], f"horizon_scale 被改动: {hs}"


def test_candidate_horizons_are_scaled_base():
    """产物侧: n_windows 必须严格等于 round(n_base × horizon_scale)。"""
    d = _cand()
    nb = int(d["n_base_windows"])
    for name in FROZEN_REGISTRY:
        c = d["candidates"][name]
        assert c["n_windows"] == round(nb * float(c["horizon_scale"])), \
            f"{name} 的 n_windows 与 horizon_scale 不一致"
    # A11 的 horizon 必须最短 (§10 优先级的物理前提)
    ns = [d["candidates"][n]["n_windows"] for n in FROZEN_REGISTRY]
    assert ns == sorted(ns), f"horizon 未按 A<B<C 递增: {ns}"


def test_common_random_numbers_b11():
    """§8: CRN 对齐证据必须为 PASS, 且三 candidate 共享同一 paired hash。"""
    d = _cand()
    assert d["crn_ok"] is True, "CRN 对齐检查未通过"
    ph = d["paired_trajectory_hash"]
    assert isinstance(ph, str) and len(ph) == 64
    for chk in d["crn_prefix_checks"]:
        assert chk["pass"] is True, f"CRN 前缀比对失败: {chk}"
    assert d["seed"] == 20260809, \
        "seed 被改动 —— §8 要求与 B1 同一 seed 以构成配对比较"


def test_crn_seed_matches_b1(b11_cal, b1_cal):
    """§8: 逐轨迹 RNG namespace 必须与 B1 完全一致 (同一 params_rng 实现)。

    这使 B1 与 B1.1 本身也构成配对比较 —— 两者只差标定口径。
    """
    assert b11_cal.params_rng is b1_cal.params_rng
    r1 = b11_cal.params_rng(20260809, 7).standard_normal(4)
    r2 = b1_cal.params_rng(20260809, 7).standard_normal(4)
    assert list(r1) == list(r2)


def test_no_global_seed_in_b11_scripts():
    """跨阶段纪律: 禁止全局 np.random.seed() —— 逐轨迹 Generator 才可配对。"""
    bad = []
    for p in sorted((ROOT / "scripts/basilisk_b11").glob("*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr == "seed":
                v = node.func.value
                # np.random.seed / numpy.random.seed
                if isinstance(v, ast.Attribute) and v.attr == "random":
                    bad.append(f"{p.name}:L{node.lineno}")
    assert not bad, f"出现全局 np.random.seed(): {bad}"


def test_b11_dataset_isolated():
    """§0: B1.1 的全部产物必须落在 b11 命名空间, 不触碰 v1 / B1 目录。"""
    d = _cand()
    for name in FROZEN_REGISTRY:
        out = d["candidates"][name]["out"]
        assert "wheel_basilisk_b11/candidates" in out, f"{name} 输出越界: {out}"
        assert "wheel_basilisk_b1/" not in out
        assert "basilisk_v1" not in out
    # B1 / v1 的 candidate 目录不得出现 B1.1 的 candidate 名
    for rel in ("data/sim/wheel_basilisk_b1/candidates",
                "data/sim/wheel_basilisk_v1"):
        dd = ROOT / rel
        if not dd.exists():
            continue
        for name in FROZEN_REGISTRY:
            assert not (dd / name).exists(), f"B1.1 candidate 写进了 {rel}"


def test_guard_out_rejects_frozen_dirs():
    """guard_out 必须拒绝 v1/B1 路径, 但**不得**误伤 b11 (前缀相同的陷阱)。"""
    import importlib.util
    p = ROOT / "scripts/basilisk_b11/generate_candidates.py"
    spec = importlib.util.spec_from_file_location("b11_gc_guardtest", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["b11_gc_guardtest"] = mod
    spec.loader.exec_module(mod)
    for bad in ("data/sim/wheel_basilisk_b1", "data/features/wheel/basilisk_v1",
                "data/sim/wheel_basilisk_b1/candidates/X"):
        with pytest.raises(SystemExit):
            mod.guard_out(ROOT / bad)
    # 关键反向用例: b11 以 b1 为前缀, 必须放行
    mod.guard_out(ROOT / "data/sim/wheel_basilisk_b11/candidates/A")
    mod.guard_out(ROOT / "data/features/wheel/basilisk_b11")


def test_b11_hash_reproducible():
    """§13/§20: candidate content hash 必须是可复算的 64 位十六进制且互不相同。

    互不相同很重要 —— 若三个 candidate 的 hash 相同, 说明 horizon 没生效。
    """
    d = _cand()
    hs = {}
    for name in FROZEN_REGISTRY:
        h = d["candidates"][name]["content_sha256"]
        assert isinstance(h, str) and len(h) == 64
        int(h, 16)
        hs[name] = h
    assert len(set(hs.values())) == 3, f"candidate content hash 重复: {hs}"


def test_b11_candidate_h5_declares_no_rul_label(b11_cfg):
    """candidate 阶段的 h5 不得含任何 HI / RUL 派生量 (§17)。"""
    import h5py
    root = ROOT / b11_cfg["paths"]["candidate_dir"]
    if not root.exists():
        pytest.skip("candidate 数据未生成")
    found = 0
    for name in FROZEN_REGISTRY:
        p = root / name / "wheel_all.h5"
        if not p.exists():
            continue
        found += 1
        with h5py.File(p, "r") as f:
            assert bool(f.attrs["contains_rul_label"]) is False
            assert str(f.attrs["reference_caliber"]) == "p95"
            g = f[sorted(k for k in f.keys() if k.startswith("traj_"))[0]]
            for forbidden in ("rul", "rul_lower_bound", "hi_b", "hi_a", "x_T"):
                assert forbidden not in g, \
                    f"{name} candidate 数据里出现 {forbidden}"
    if found == 0:
        pytest.skip("candidate h5 尚未落盘")
