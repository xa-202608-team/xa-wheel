"""tests/basilisk_b1/test_b1_dataset_isolation.py

B1 §0/§12/§13 —— 命名空间硬隔离 + CRN 对齐 + 特征无泄漏 + hash 可复现。

覆盖 §16 点名的:
  * test_b1_dataset_isolated        —— B1 产物绝不落进 v1 / 正式命名空间
  * test_common_random_numbers      —— A/B/C 只差 horizon (CRN 对齐)
  * test_b1_feature_no_truth_leakage —— target_features.h5 无任何真值
  * test_b1_hash_reproducible       —— content hash 口径可复现
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "basilisk_b1"))

import calibrate_degradation as cal                          # noqa: E402
from src.sim.build_hi import XT_COLS                          # noqa: E402
from src.sim.basilisk_bridge import DUTY_KEYS                 # noqa: E402

CFG = cal.load_b1_config("configs/wheel_basilisk_b1.yaml")
# B1 不得写入的命名空间: analytic 正式产物 + 已冻结的 basilisk_v1
FORBIDDEN_NS = ("data/simulated/wheel/sim_v1",
                "data/features/wheel/schema_v1",
                "data/sim/wheel_basilisk_v1",
                "data/features/wheel/basilisk_v1",
                "docs/basilisk_v1",
                "checkpoints/basilisk_v1")
FEATURE_H5 = ROOT / CFG["paths"]["feature_h5"]
CAND_DIR = ROOT / CFG["paths"]["candidate_dir"]
TRUTH_ATTRS = {"Kt", "Tc", "b0", "omega0", "b_slope", "J", "seed_traj",
               "tau_years", "Delta", "T_base", "T_amp",
               "_b1_g_duty", "_b1_speed_util_mean"}


# --------------------------------------------------------------------------
# §0 命名空间隔离
# --------------------------------------------------------------------------
def test_b1_dataset_isolated():
    """所有 B1 写路径都必须带 basilisk_b1, 且不落进 v1 / 正式命名空间。"""
    write_keys = ("sim_dir", "candidate_dir", "feature_h5", "ckpt_dir", "doc_dir")
    for k in write_keys:
        rel = CFG["paths"][k]
        assert "basilisk_b1" in rel, f"paths.{k}={rel} 未落在 B1 隔离命名空间"
        for ns in FORBIDDEN_NS:
            assert ns not in rel, f"paths.{k}={rel} 落进禁写命名空间 {ns}"
    # profile 库是**只读复用** v1 冻结产物, 这是允许的 (且必须是 v1 的那一份)
    assert CFG["paths"]["profile_h5"] == "data/mission_profile/basilisk_v1/profiles.h5"

    for rel in ("data/sim/wheel_basilisk_b1", "data/features/wheel/basilisk_b1",
                "docs/basilisk_b1", "checkpoints/basilisk_b1"):
        d = ROOT / rel
        if not d.exists():
            continue
        for p in d.rglob("*"):
            rp = p.relative_to(ROOT).as_posix()
            for ns in FORBIDDEN_NS:
                assert ns not in rp, f"{rp} 落进禁写命名空间 {ns}"


def test_b1_writers_guard_forbidden_namespaces():
    """每个写盘脚本都必须实现禁写命名空间保护 (静态检查)。"""
    for rel in ("scripts/basilisk_b1/generate_candidates.py",
                "scripts/basilisk_b1/generate_dataset.py",
                "scripts/basilisk_b1/build_features.py"):
        src = (ROOT / rel).read_text(encoding="utf-8")
        assert ("guard_out" in src) or ("FORBIDDEN_OUT" in src), \
            f"{rel} 未实现禁写保护 (§0)"
        assert "basilisk_v1" in src, f"{rel} 的保护未显式列出 basilisk_v1"


def test_b1_never_writes_into_v1_paths():
    """v1 路径字面量只允许出现在**禁写清单**里, 不得作为输出路径使用。

    用 AST 精确判定: 收集每个模块里 `FORBIDDEN_OUT` 赋值语句内部的字符串常量,
    任何出现在别处的 v1 路径字面量都是违规 (可能被拼进输出路径)。
    """
    for rel in ("scripts/basilisk_b1/generate_candidates.py",
                "scripts/basilisk_b1/generate_dataset.py",
                "scripts/basilisk_b1/build_features.py"):
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
        allowed = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and "FORBIDDEN" in t.id
                    for t in node.targets):
                for c in ast.walk(node.value):
                    if isinstance(c, ast.Constant) and isinstance(c.value, str):
                        allowed.add(id(c))
        bad = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant)
                    and isinstance(node.value, str)) or id(node) in allowed:
                continue
            v = node.value
            if not any(t in v for t in ("basilisk_v1", "schema_v1", "sim_v1")):
                continue
            # 只关心"看起来就是一条路径"的字面量: 无空白、无换行、无 markdown 符号。
            # docstring 与报告文本里提到 v1 是说明性文字, 不构成写入风险。
            if any(ch in v for ch in " \n\t|`*") or "/" not in v:
                continue
            bad.append(f"L{node.lineno}: {v!r}")
        assert not bad, (f"{rel} 在禁写清单之外出现 v1 路径字面量 "
                         f"(可能被拼成输出路径):\n" + "\n".join(bad))


def test_basilisk_not_in_requirements_or_docker():
    """Basilisk 不得写入 requirements.txt / Dockerfile / docker-compose.yml。"""
    for rel in ("requirements.txt", "Dockerfile", "docker-compose.yml"):
        p = ROOT / rel
        if not p.exists():
            continue
        assert "basilisk" not in p.read_text(encoding="utf-8").lower(), \
            f"{rel} 引入了 Basilisk 依赖"


# --------------------------------------------------------------------------
# §7 common random numbers
# --------------------------------------------------------------------------
def test_common_random_numbers():
    """A/B/C 的同一 trajectory 必须共享退化尺度与 duty 前缀, 唯一差异是长度。

    这是 §7 的核心不变量: 若被破坏, "A/B/C 的差异来自 horizon" 这个结论就不成立,
    candidate 比较也就失去意义。
    """
    import h5py
    reg = list(CFG["candidates"]["registry"])
    paths = {c: CAND_DIR / c / "wheel_all.h5" for c in reg}
    if not all(p.exists() for p in paths.values()):
        pytest.skip("candidate 数据未全部生成")

    fs = {c: h5py.File(p, "r") for c, p in paths.items()}
    try:
        base = reg[0]
        keys = sorted(k for k in fs[base].keys() if k.startswith("traj_"))
        n_check = min(5, len(keys))
        for k in keys[:n_check]:
            ga = fs[base][k]
            na = ga["omega_cmd"].shape[0]
            for c in reg[1:]:
                gb = fs[c][k]
                # (a) 退化尺度参数逐位相同
                for at in ("b0", "tau_years", "omega0", "Kt", "Tc", "J",
                           "Delta", "seed_traj", "_b1_g_duty"):
                    assert float(ga.attrs[at]) == float(gb.attrs[at]), \
                        f"{k}.{at} 在 {base} 与 {c} 不同 -> CRN 破坏"
                # (b) duty 前缀逐位相同
                assert gb["omega_cmd"].shape[0] > na, f"{c} 的 horizon 未变长"
                np.testing.assert_array_equal(
                    ga["omega_cmd"][:], gb["omega_cmd"][:na],
                    err_msg=f"{k} omega_cmd 前缀在 {base} 与 {c} 不同 -> CRN 破坏")
                # (c) 长度比必须等于 horizon_scale 比
                hs_a = float(fs[base].attrs["horizon_scale"])
                hs_b = float(fs[c].attrs["horizon_scale"])
                assert abs(gb["omega_cmd"].shape[0] / na - hs_b / hs_a) < 1e-6
    finally:
        for f in fs.values():
            f.close()


def test_paired_trajectory_hash_recorded():
    """paired_trajectory_hash 必须被记录, 且三 candidate 共享同一值。"""
    p = ROOT / "checkpoints/basilisk_b1/candidates.json"
    if not p.exists():
        pytest.skip("candidates.json 未生成")
    d = json.loads(p.read_text(encoding="utf-8"))
    assert isinstance(d["paired_trajectory_hash"], str)
    assert len(d["paired_trajectory_hash"]) == 64
    assert d["crn_ok"] is True, "CRN 对齐检查未通过"


def test_params_rng_independent_of_horizon():
    """物理参数 RNG 只依赖 (seed, traj_id), 与 horizon / 循环历史无关。"""
    from src.sim.wheel_sim import sample_params
    seed = int(CFG["seed"])
    for i in (0, 7, 41):
        a = sample_params(cal.params_rng(seed, i), CFG["sim"])
        b = sample_params(cal.params_rng(seed, i), CFG["sim"])
        assert a == b, f"traj {i} 的参数采样不可复现"
    # 不同 traj_id 必须给出不同参数 (否则 60 条是同一条)
    p0 = sample_params(cal.params_rng(seed, 0), CFG["sim"])
    p1 = sample_params(cal.params_rng(seed, 1), CFG["sim"])
    assert p0 != p1


def test_no_global_seed_in_b1_scripts():
    """禁止全局 np.random.seed() —— 会让 CRN 与可复现性同时失效。

    用 AST 查**真实调用**而非子串: 报告文本里写 "无任何全局 np.random.seed()"
    是合法的自我声明, 不该被判违规。
    """
    bad = []
    for p in sorted((ROOT / "scripts/basilisk_b1").glob("*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            # np.random.seed(...) / numpy.random.seed(...) / random.seed(...)
            if isinstance(f, ast.Attribute) and f.attr == "seed":
                owner = f.value
                if isinstance(owner, ast.Attribute) and owner.attr == "random":
                    bad.append(f"{p.name}:L{node.lineno} 全局 np.random.seed()")
                elif isinstance(owner, ast.Name) and owner.id == "random":
                    bad.append(f"{p.name}:L{node.lineno} 全局 random.seed()")
    assert not bad, "出现全局 seed 调用:\n" + "\n".join(bad)


def test_b1_rng_is_hash_derived():
    """逐轨迹 RNG 必须由 (seed, traj_id) 的 sha256 派生, 与循环历史无关。"""
    src = (ROOT / "scripts/basilisk_b1/calibrate_degradation.py").read_text(
        encoding="utf-8")
    assert "basilisk_b1_params|" in src, "params RNG 未采用 hash 派生口径"
    assert "default_rng" in src


# --------------------------------------------------------------------------
# §12 特征无真值泄漏
# --------------------------------------------------------------------------
def test_b1_feature_no_truth_leakage():
    """B1 target_features.h5 不得含任何仿真真值 attrs 或额外 dataset。"""
    import h5py
    if not FEATURE_H5.exists():
        pytest.skip("B1 target_features.h5 未生成")
    allowed = {"x_T", "hi_b", "hi_a", "b_hat", "b_true", "rul",
               "rul_lower_bound", "label_fail", "mission_features"}
    with h5py.File(FEATURE_H5, "r") as f:
        assert f.attrs["lineage"] == "basilisk_b1"
        assert bool(f.attrs["mission_features_in_xT"]) is False
        for k in (k for k in f.keys() if k.startswith("traj_")):
            g = f[k]
            bad_a = TRUTH_ATTRS & set(g.attrs.keys())
            assert not bad_a, f"{k} 泄漏真值 attrs {bad_a}"
            bad_d = set(g.keys()) - allowed
            assert not bad_d, f"{k} 含未预期 dataset {bad_d}"
            assert g["x_T"].shape[1] == len(XT_COLS) == 10


def test_b1_feature_script_feeds_only_telemetry():
    """静态检查: B1 build_features 只喂遥测列, 且不读输入仿真组的 attrs。

    与 v1 同款检查 —— 输入组变量名是 g / fin, 输出侧 gg / mg / fout 才允许写 attrs。
    """
    src = (ROOT / "scripts/basilisk_b1/build_features.py").read_text(encoding="utf-8")
    assert 'TELEMETRY_COLS + ["b_true", "label_fail"]' in src, \
        "df 构造被改动, 需重新核查真值泄漏"
    tree = ast.parse(src)
    forbidden_owners = {"g", "fin"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "attrs" \
                and isinstance(node.value, ast.Name) \
                and node.value.id in forbidden_owners:
            raise AssertionError(
                f"build_features.py L{node.lineno} 读取了输入仿真组的 attrs "
                f"({node.value.id}.attrs) —— 可能触及 Kt/Tc/b0/omega0 真值")


def test_b1_uses_frozen_build_features():
    """§12: 必须直接调用 src.sim.build_hi.build_features, 不得复制 HI 数学。"""
    src = (ROOT / "scripts/basilisk_b1/build_features.py").read_text(encoding="utf-8")
    assert "from src.sim.build_hi import" in src and "build_features" in src
    import inspect
    from src.sim.build_hi import build_features
    assert list(inspect.signature(build_features).parameters) == ["df", "sim_cfg"]


def test_b1_core_xt_columns_preserved(b1_build_features):
    """§12: x_T 核心 8 列语义与位置一字不改。"""
    assert tuple(XT_COLS[:8]) == b1_build_features.CORE_XT_COLS == (
        "I_m", "omega", "T", "T_cmd", "sigma_Im", "b_hat", "dT", "omega_err")


def test_b1_mission_features_auxiliary_only():
    """§12: Basilisk 工况统计只在 mission_features/ 辅助组, 不进 x_T。"""
    import h5py
    if not FEATURE_H5.exists():
        pytest.skip("B1 target_features.h5 未生成")
    with h5py.File(FEATURE_H5, "r") as f:
        k = sorted(x for x in f.keys() if x.startswith("traj_"))[0]
        g = f[k]
        assert "mission_features" in g
        for key in DUTY_KEYS:
            assert key in g["mission_features"], f"缺 {key}"
            assert g["mission_features"][key].shape[0] == g["x_T"].shape[0]


def test_b1_censored_rul_is_nan_not_fabricated():
    """右删失轨迹的 rul 必须是 NaN —— 绝不伪造 EOL (跨阶段铁律)。"""
    import h5py
    if not FEATURE_H5.exists():
        pytest.skip("B1 target_features.h5 未生成")
    n_obs = n_cens = 0
    with h5py.File(FEATURE_H5, "r") as f:
        for k in (k for k in f.keys() if k.startswith("traj_")):
            g = f[k]
            if int(g.attrs["event_observed"]):
                n_obs += 1
                assert np.isfinite(g["rul"][:]).all(), f"{k} 观测轨迹 rul 含 NaN"
            else:
                n_cens += 1
                assert np.isnan(g["rul"][:]).all(), \
                    f"{k} 删失轨迹被伪造了 EOL (rul 非全 NaN)"
                assert np.isfinite(g["rul_lower_bound"][:]).all(), \
                    f"{k} 删失轨迹缺 rul_lower_bound"
    assert n_obs > 0 and n_cens > 0, \
        f"数据集应同时含观测({n_obs})与删失({n_cens})轨迹"


# --------------------------------------------------------------------------
# §13 hash 可复现
# --------------------------------------------------------------------------
def test_b1_hash_reproducible(b1_build_features):
    """content hash 必须可用同一口径重算出来 (不依赖 HDF5 容器元数据)。

    HDF5 文件本身含创建时间戳, file hash 天然不可复现; 这正是全链一律用
    **数值 content hash** 作为版本身份的原因。
    """
    bf = b1_build_features
    stats = ROOT / "checkpoints/basilisk_b1/feature_stats.json"
    if not (FEATURE_H5.exists() and stats.exists()):
        pytest.skip("B1 特征未生成")
    d = json.loads(stats.read_text(encoding="utf-8"))
    got = bf.feature_content_hash(FEATURE_H5)
    assert got == d["feature_content_sha256"], \
        f"content hash 不可复现: {got[:16]} != {d['feature_content_sha256'][:16]}"
    # 连算两次必须一致 (排序遍历 + 固定 dtype)
    assert bf.feature_content_hash(FEATURE_H5) == got


def test_b1_dataset_hash_recorded_in_feature_file():
    """特征文件必须记录上游数据集的 content hash (谱系可追溯)。"""
    import h5py
    if not FEATURE_H5.exists():
        pytest.skip("B1 target_features.h5 未生成")
    with h5py.File(FEATURE_H5, "r") as f:
        for at in ("source_dataset", "source_dataset_content_sha256",
                   "calibration_protocol_hash", "candidate"):
            assert at in f.attrs, f"特征文件缺谱系 attr {at}"
            assert str(f.attrs[at]) != "ABSENT", f"{at} 未被记录"


def test_calibration_protocol_hash_stable():
    """calibration_protocol_hash 对同一 payload 必须稳定 (json sort_keys)。"""
    payload = {"b": 2, "a": [1, 2, 3], "c": {"y": 1, "x": 2}}
    h1 = cal.calibration_protocol_hash(payload)
    h2 = cal.calibration_protocol_hash({"c": {"x": 2, "y": 1}, "a": [1, 2, 3], "b": 2})
    assert h1 == h2 and len(h1) == 64
