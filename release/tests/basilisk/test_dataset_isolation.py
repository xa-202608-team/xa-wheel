"""tests/basilisk/test_dataset_isolation.py

§18 —— 硬隔离与"旧结果零变化"测试。这一组是 §0 冻结契约的自动化守卫:
  - 旧 analytic lineage 的 30 个冻结产物 byte 级不变
  - analytic 代码路径逐位保持原行为 (§2)
  - basilisk_v1 产物完全不落在正式命名空间里 (§1)
  - basilisk 特征文件无真值泄漏 (§16)
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.basilisk_config import load_basilisk_config     # noqa: E402
from src.sim.build_hi import TELEMETRY_COLS, XT_COLS           # noqa: E402

# --------------------------------------------------------------------------
# 共享生命周期治理注册表 —— 全库唯一来源。
# B7 授权改写 docs/results.md 作为最终交付表达面, 因此该文件的"哈希永久不变"
# 约束退役, 改由语义审计 (scripts/basilisk_b7/audit_report_numbers.py) 看守。
# 算法产物 (数据/特征/split/权重/metrics/verdict) 仍然逐字节冻结, 不受影响。
# 退役理由逐条记录: docs/basilisk_b7/stale_guard_retirement.md
# --------------------------------------------------------------------------
import importlib.util as _ilu   # noqa: E402

_lr_spec = _ilu.spec_from_file_location(
    "lifecycle_registry", ROOT / "tests" / "lifecycle_registry.py")
lifecycle = _ilu.module_from_spec(_lr_spec)
_lr_spec.loader.exec_module(lifecycle)

CONTRACT = ROOT / "docs/basilisk_v1/baseline_contract.json"
CFG = load_basilisk_config("configs/wheel_basilisk.yaml")
FEATURE_H5 = ROOT / CFG["paths"]["feature_h5"]
SIM_DIR = ROOT / CFG["paths"]["sim_dir"]

# 正式 (analytic) lineage 的命名空间 —— basilisk_v1 的任何产物都不得落在这里
OFFICIAL_NAMESPACES = ("data/simulated/wheel/sim_v1",
                       "data/features/wheel/schema_v1")


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _contract() -> dict:
    if not CONTRACT.exists():
        pytest.skip("baseline_contract.json 未生成")
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# §0 旧结果 hash 零变化
# --------------------------------------------------------------------------
def test_old_results_hash_unchanged():
    """契约中除 allowed_minimal_edit 组外, 全部文件 hash 必须与冻结值一致。

    MISSING 本身也是契约的一部分: 一个当时不存在的文件后来出现, 同样是违约
    (说明有流程往正式命名空间里写了东西)。
    """
    c = _contract()
    mutable = c.get("mutable_group", "allowed_minimal_edit")
    violations = []
    _n_hashed = _n_exempt = 0
    for group, files in c["groups"].items():
        if group == mutable:
            continue
        for rel, expect in files.items():
            p = ROOT / rel
            if expect == "MISSING":
                if p.exists():
                    violations.append(f"{rel}: 冻结时不存在, 现已出现")
                continue
            if lifecycle.hash_exempt(rel):
                _n_exempt += 1
                continue
            if not p.exists():
                violations.append(f"{rel}: 冻结时存在, 现已丢失")
                continue
            got = _sha256(p)
            _n_hashed += 1
            if got != expect:
                violations.append(f"{rel}: {expect[:12]} -> {got[:12]}")
    assert not violations, "BASELINE_CONTRACT_VIOLATION:\n" + "\n".join(violations)
    # 防掏空 (B7 生命周期治理): 豁免只允许命中封闭白名单, 且实检项须占绝大多数。
    lifecycle.assert_registry_is_not_widened()
    assert _n_exempt <= 3, f"哈希豁免项过多 ({_n_exempt}) —— 疑似滥用"
    assert _n_hashed >= 20, f"逐字节实检项过少 ({_n_hashed}) —— 契约可能被掏空"



def test_contract_covers_all_frozen_lineages():
    """契约必须覆盖 S2.5/S3/S4/S5/S5B 全部关键产物 (防止契约被悄悄缩小)。"""
    c = _contract()
    allrel = {r for files in c["groups"].values() for r in files}
    must = ["configs/wheel.yaml",
            "data/features/wheel/schema_v1/target_features.h5",
            "checkpoints/s3_gate_metrics.json",
            "checkpoints/s4_metrics.json",
            "checkpoints/s5_rate_metrics.json",
            "checkpoints/s5b_wiener_pf_metrics.json",
            "docs/diagnostics_s25_results.md",
            "docs/diagnostics_s3_results.md",
            "docs/diagnostics_s4_results.md",
            "docs/diagnostics_s5b_results.md",
            "docs/results.md",
            "src/sim/build_hi.py"]
    missing = [m for m in must if m not in allrel]
    assert not missing, f"契约漏掉冻结产物: {missing}"


# --------------------------------------------------------------------------
# §2 analytic 路径逐位不变
# --------------------------------------------------------------------------
def test_old_analytic_path_unchanged():
    """simulate(duty=None) 必须逐位复现冻结的 analytic wheel_all.h5。

    这是 §2 "source=analytic 必须逐位保持现有行为" 的直接检验: 用同一
    seed 重跑仿真, 与磁盘上冻结的正式数据集逐点比对 6 个遥测列。
    """
    import h5py
    import pandas as pd
    from src.utils.config import load_config
    from src.sim.wheel_sim import simulate, sample_params

    frozen = ROOT / "data/simulated/wheel/sim_v1/seed_42/wheel_all.h5"
    if not frozen.exists():
        pytest.skip("冻结 analytic 数据集不存在")
    cfg = load_config("configs/wheel.yaml")
    seed = int(cfg["seed"])
    rng = np.random.default_rng(seed)
    n_check = 5
    cols = ["omega", "omega_cmd", "I_m", "T", "T_cmd", "b_true"]
    with h5py.File(frozen, "r") as f:
        keys = sorted(k for k in f.keys() if k.startswith("traj_"))
        for i in range(min(n_check, len(keys))):
            params = sample_params(rng, cfg["sim"])
            traj_rng = np.random.default_rng(params["seed_traj"])
            df, eol, failed = simulate(params, cfg["sim"], traj_rng)   # duty=None
            g = f[keys[i]]
            for c in cols:
                ref = g[c][:]
                assert len(df) == len(ref), \
                    f"{keys[i]}/{c} 长度 {len(df)} != 冻结 {len(ref)}"
                assert np.array_equal(df[c].values.astype(ref.dtype), ref), \
                    f"{keys[i]}/{c} 与冻结 analytic 结果不一致 -> §2 违约"


def test_analytic_duty_none_is_default():
    """simulate 的 duty 参数必须是可选的、默认 None (新增接口而非改旧行为)。"""
    import inspect
    from src.sim.wheel_sim import simulate
    sig = inspect.signature(simulate)
    assert "duty" in sig.parameters
    assert sig.parameters["duty"].default is None
    # 旧的三个位置参数顺序不得改变
    names = list(sig.parameters)
    assert names[:3] == ["params", "sim_cfg", "traj_rng"], names


# --------------------------------------------------------------------------
# §1 命名空间硬隔离
# --------------------------------------------------------------------------
def test_basilisk_dataset_isolated():
    """所有 basilisk_v1 产物路径都必须带 basilisk_v1, 且不在正式命名空间内。"""
    for key in ("profile_h5", "duty_stats", "provenance", "sim_dir",
                "feature_h5", "ckpt_dir", "doc_dir"):
        rel = CFG["paths"][key]
        assert "basilisk_v1" in rel, f"paths.{key}={rel} 未落在隔离命名空间"
        for ns in OFFICIAL_NAMESPACES:
            assert ns not in rel, f"paths.{key}={rel} 落进正式命名空间 {ns}"
    if SIM_DIR.exists():
        for p in SIM_DIR.rglob("*"):
            rp = p.relative_to(ROOT).as_posix()
            for ns in OFFICIAL_NAMESPACES:
                assert ns not in rp, f"{rp} 落进正式命名空间"


def test_basilisk_scripts_never_write_official_namespace():
    """basilisk 脚本中不得出现指向正式产物的写路径 (静态检查)。

    允许**读**冻结产物 (verify_baseline / 本测试自身需要), 因此只检查
    这些字符串是否出现在 open(...,'w') / h5py.File(...,'w') 语境里 ——
    实现上更简单可靠的做法: 每个写盘脚本都必须带 _guard_out 或 forbidden 检查。
    """
    writers = [ROOT / "scripts/basilisk/generate_wheel_dataset.py",
               ROOT / "scripts/basilisk/build_features.py"]
    for p in writers:
        assert p.exists(), f"缺 {p.name}"
        src = p.read_text(encoding="utf-8")
        assert ("_guard_out" in src) or ("forbidden" in src), \
            f"{p.name} 未实现正式命名空间写保护 (§1)"
        assert any(ns in src for ns in OFFICIAL_NAMESPACES), \
            f"{p.name} 的保护未显式列出正式命名空间"


# --------------------------------------------------------------------------
# §16 无真值泄漏
# --------------------------------------------------------------------------
def test_build_hi_no_truth_leakage_basilisk():
    """basilisk_v1 特征文件不得含任何仿真真值 (attrs 或额外 dataset)。"""
    import h5py
    if not FEATURE_H5.exists():
        pytest.skip("basilisk_v1 target_features.h5 未生成")
    truth_attrs = {"Kt", "Tc", "b0", "omega0", "b_slope", "J", "seed_traj"}
    allowed_ds = {"x_T", "hi_b", "hi_a", "b_hat", "b_true", "rul",
                  "rul_lower_bound", "label_fail", "mission_features"}
    with h5py.File(FEATURE_H5, "r") as f:
        assert f.attrs["lineage"] == "basilisk_v1"
        assert bool(f.attrs["mission_features_in_xT"]) is False
        for key in f.keys():
            g = f[key]
            bad_a = truth_attrs & set(g.attrs.keys())
            assert not bad_a, f"{key} 泄漏真值 attrs {bad_a}"
            bad_d = set(g.keys()) - allowed_ds
            assert not bad_d, f"{key} 含未预期 dataset {bad_d}"
            # x_T 必须与遥测派生的列数一致, 且不含工况统计列
            assert g["x_T"].shape[1] == len(XT_COLS)


def test_build_features_signature_has_no_params():
    """§16 接口级保证: build_hi.build_features 不接受 params (拿不到真值)。"""
    import inspect
    from src.sim.build_hi import build_features
    sig = inspect.signature(build_features)
    assert list(sig.parameters) == ["df", "sim_cfg"], list(sig.parameters)


def test_basilisk_feature_script_feeds_only_telemetry():
    """静态检查: build_features.py 传给 build_features 的 df 只含遥测列。

    并且在计算段不得**读取输入仿真组的 attrs** —— 那里存着 Kt/Tc/b0/omega0 真值。
    注意不能简单查子串 "g.attrs": 写输出用的 gg.attrs / mg.attrs 都会命中。
    这里用 AST 精确找 `<name>.attrs` 的读取, 只允许输入组 mission_features
    (mg_in.attrs, 里面只有 mode weights 与采样点数, 不含真值)。
    """
    import ast
    src = (ROOT / "scripts/basilisk/build_features.py").read_text(encoding="utf-8")
    assert 'TELEMETRY_COLS + ["b_true", "label_fail"]' in src, \
        "特征脚本的 df 构造被改动, 需重新核查真值泄漏"
    tree = ast.parse(src)
    # 输入侧变量名: 仿真 h5 的组变量是 g, 文件是 fin
    forbidden_owners = {"g", "fin"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "attrs" \
                and isinstance(node.value, ast.Name) \
                and node.value.id in forbidden_owners:
            raise AssertionError(
                f"特征脚本第 {node.lineno} 行读取了输入仿真组的 attrs "
                f"({node.value.id}.attrs) —— 可能触及 Kt/Tc/b0/omega0 真值")


def test_mission_features_are_auxiliary_only():
    """§17: 工况统计只在 mission_features/ 组, 绝不进 x_T。"""
    import h5py
    from src.sim.basilisk_bridge import DUTY_KEYS
    if not FEATURE_H5.exists():
        pytest.skip("basilisk_v1 target_features.h5 未生成")
    with h5py.File(FEATURE_H5, "r") as f:
        key = sorted(f.keys())[0]
        g = f[key]
        assert "mission_features" in g, "缺 mission_features 辅助组"
        mf = g["mission_features"]
        for k in DUTY_KEYS:
            assert k in mf, f"mission_features 缺 {k}"
            # 长度必须与 x_T 行数一致 (逐窗对齐), 但不作为特征列
            assert mf[k].shape[0] == g["x_T"].shape[0]
        assert g["x_T"].shape[1] == len(XT_COLS) == 10


def test_core_xt_columns_preserved():
    """§17: x_T 前 8 列语义与位置必须与正式 lineage 完全一致。"""
    assert tuple(XT_COLS[:8]) == ("I_m", "omega", "T", "T_cmd", "sigma_Im",
                                  "b_hat", "dT", "omega_err"), XT_COLS


def test_label_logic_matches_frozen_pipeline():
    """本脚本复刻的 derive_labels 必须与冻结 build_hi.main 的标签逐位一致。

    build_hi.py 被合同冻结不可改, 标签派生逻辑无法抽公共函数, 只能在
    scripts/basilisk/build_features.py 里复刻。此测试用**冻结的正式特征文件**
    做黄金参考, 防止两份逻辑漂移。
    """
    import h5py
    sys.path.insert(0, str(ROOT / "scripts" / "basilisk"))
    from build_features import derive_labels

    frozen_feat = ROOT / "data/features/wheel/schema_v1/target_features.h5"
    if not frozen_feat.exists():
        pytest.skip("冻结特征文件不存在")
    cap_ratio = float(CFG["source"]["rul_cap_ratio"])
    with h5py.File(frozen_feat, "r") as f:
        keys = sorted(f.keys())
        n_obs = n_cens = 0
        for key in keys:
            g = f[key]
            lf = g["label_fail"][:]
            rul, rul_lb, eol, observed = derive_labels(lf, cap_ratio)
            np.testing.assert_array_equal(
                rul.astype(np.float32), g["rul"][:],
                err_msg=f"{key} rul 与冻结流程不一致")
            np.testing.assert_array_equal(
                rul_lb.astype(np.float32), g["rul_lower_bound"][:],
                err_msg=f"{key} rul_lower_bound 与冻结流程不一致")
            assert eol == int(g.attrs["eol_idx"])
            assert int(observed) == int(g.attrs["event_observed"])
            n_obs += int(observed)
            n_cens += int(not observed)
        assert n_obs > 0 and n_cens > 0, \
            f"冻结集应同时含观测({n_obs})与删失({n_cens})轨迹, 才算真正验证两条分支"


def test_basilisk_not_in_requirements_or_docker():
    """§23: Basilisk 不得写入 requirements.txt / Dockerfile / docker-compose.yml。"""
    for rel in ("requirements.txt", "Dockerfile", "docker-compose.yml"):
        p = ROOT / rel
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8").lower()
        assert "basilisk" not in txt, f"{rel} 引入了 Basilisk 依赖 (违反 §23)"
