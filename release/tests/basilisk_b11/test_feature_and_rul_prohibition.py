"""tests/basilisk_b11/test_feature_and_rul_prohibition.py

B1.1 §18 —— 特征与 RUL 禁令组硬测试:
  * `test_b11_feature_no_truth_leakage` : AST 静态检查 + (若有产物) 运行时扫描
  * `test_b11_no_rul_in_stage`          : §17 本阶段不得运行任何 RUL

FAIL 路径下 (`B11_CALIBRATION_FAIL`) 特征文件不应存在, 因此运行时部分会 skip;
但**静态部分照常执行** —— 无泄漏合约是对代码的约束, 与产物是否生成无关。
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BF = "scripts/basilisk_b11/build_features.py"
FEATURE_H5 = ROOT / "data/features/wheel/basilisk_b11/target_features.h5"
B11_SCRIPTS = sorted((ROOT / "scripts/basilisk_b11").glob("*.py"))

# 训练 / 迁移 / 实验模块 —— §17 禁止在 B1.1 任何脚本中出现
FORBIDDEN_MODULES = ("torch", "src.models", "src.transfer", "src.experiments",
                     "src.baselines", "pytorch_lightning")
# 禁止调用的训练入口
FORBIDDEN_ENTRYPOINTS = ("target_only", "source_finetune", "source_mmd",
                         "source_mmd_finetune", "train_transfer", "pretrain",
                         "wiener_pf", "rate_model")


def _tree(rel: str):
    p = ROOT / rel
    assert p.exists(), f"缺 {rel}"
    src = p.read_text(encoding="utf-8")
    return ast.parse(src), src


# --------------------------------------------------------------------------
# §15/§16 no-truth-leakage —— 静态
# --------------------------------------------------------------------------

def test_b11_feature_no_truth_leakage_static():
    """build_features 不得从源 h5 的 attrs 读取任何真值。

    用 AST 检查 `g.attrs[...]` / `g.attrs.get(...)` 形式的下标读取:
    源轨迹组的 attrs 里有 b0 / tau_years / omega0 / _b1_g_duty 等真值,
    读到任意一个就是泄漏。注意**不能用子串匹配** —— docstring 里必须能提到
    这些名字来解释为什么不读它们。
    """
    tree, _ = _tree(BF)
    bad = []
    for node in ast.walk(tree):
        # 匹配 <expr>.attrs[<key>] 的读取
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) \
                and node.value.attr == "attrs":
            owner = node.value.value
            # 允许写入自己的输出组 (gg.attrs[...] = ...) —— 由父节点是否 Assign 决定,
            # 这里保守地只允许 owner 名为 gg / fout / mg
            name = owner.id if isinstance(owner, ast.Name) else None
            if name not in ("gg", "fout", "mg"):
                key = node.slice
                k = key.value if isinstance(key, ast.Constant) else "<dyn>"
                bad.append(f"L{node.lineno}: {name}.attrs[{k!r}]")
    assert not bad, "build_features 读取了源组 attrs (可能泄漏真值):\n" + "\n".join(bad)


def test_b11_build_features_calls_frozen_pipeline():
    """§15: 必须直接调用 src.sim.build_hi.build_features, 不得复制 HI 数学。"""
    tree, src = _tree(BF)
    imported = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "src.sim.build_hi":
            names = {a.name for a in node.names}
            assert {"build_features", "TELEMETRY_COLS", "XT_COLS"} <= names, \
                f"未从冻结管线引入必要符号: {names}"
            imported = True
    assert imported, "未 import src.sim.build_hi —— HI 数学疑似被复制"
    # 不得自行定义同名函数覆盖冻结实现
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            assert node.name != "build_features", \
                "本地重定义了 build_features —— §15 要求复用冻结实现"


def test_b11_build_features_signature_has_no_params():
    """§15 的结构性保证: 冻结的 build_features 签名不含 params。"""
    from src.sim.build_hi import build_features
    import inspect
    sig = list(inspect.signature(build_features).parameters)
    assert "params" not in sig, f"build_features 签名出现 params: {sig}"
    assert sig[:2] == ["df", "sim_cfg"], f"签名被改动: {sig}"


def test_b11_feature_core_columns_unchanged(b11_build_features):
    """§16(5): x_T 核心 8 列语义与位置不变。"""
    from src.sim.build_hi import XT_COLS
    bf = b11_build_features
    assert bf.CORE_XT_COLS == ("I_m", "omega", "T", "T_cmd", "sigma_Im",
                               "b_hat", "dT", "omega_err")
    assert tuple(XT_COLS[:8]) == bf.CORE_XT_COLS, \
        f"核心 8 列被改动: {XT_COLS[:8]}"
    assert len(XT_COLS) == 10


def test_b11_derive_labels_censored_not_faked(b11_build_features):
    """右删失轨迹的 rul 必须是 NaN, 只给 rul_lower_bound —— 不伪造 EOL。"""
    import numpy as np
    bf = b11_build_features
    n = 1000
    # (a) event-observed
    lf = np.zeros(n, dtype=np.int8)
    lf[600:] = 1
    rul, lb, eol, obs = bf.derive_labels(lf, 0.5)
    assert obs is True and eol == 600
    assert np.all(rul[600:] == 0.0), "post-EOL 的 rul 未置 0"
    assert np.all(np.isfinite(rul))
    assert rul[599] == 1.0
    # (b) censored
    lf0 = np.zeros(n, dtype=np.int8)
    rul0, lb0, eol0, obs0 = bf.derive_labels(lf0, 0.5)
    assert obs0 is False and eol0 == n - 1
    assert np.all(np.isnan(rul0)), "删失轨迹的 rul 被伪造成数值 —— 严禁"
    assert np.all(np.isfinite(lb0)) and lb0.max() > 0, "删失轨迹缺 rul_lower_bound"


def test_b11_mission_features_not_in_xT():
    """§15: mission_features 只作 auxiliary, 不得进 x_T。"""
    tree, src = _tree(BF)
    assert '"mission_features_in_xT"' in src or "'mission_features_in_xT'" in src
    # 断言写入的是 False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant) \
                    and t.slice.value == "mission_features_in_xT":
                assert isinstance(node.value, ast.Constant) \
                    and node.value.value is False, \
                    f"L{node.lineno}: mission_features_in_xT 未被置 False"


def test_b11_feature_content_hash_skips_mission_features(b11_build_features):
    """content hash 必须跳过 auxiliary 组, 且 NaN 处理确定 (不悄悄转 0)。"""
    src = (ROOT / BF).read_text(encoding="utf-8")
    assert 'if name == "mission_features"' in src, "content hash 未跳过 auxiliary 组"
    assert "nan=-1.0e300" in src, \
        "NaN 未映射到不可能出现的哨兵值 —— nan_to_num 默认会把 NaN 变 0.0, " \
        "那会让删失轨迹与 rul=0 撞 hash"


# --------------------------------------------------------------------------
# §15/§16 no-truth-leakage —— 运行时 (需产物)
# --------------------------------------------------------------------------

def test_b11_feature_no_truth_leakage_runtime():
    """若特征文件存在: 逐组扫描 dataset 名与 attrs, 不得有真值。"""
    if not FEATURE_H5.exists():
        pytest.skip("B1.1 特征文件不存在 (FAIL 路径下按 §16 不应生成)")
    import h5py
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location("b11_bf_leak", ROOT / BF)
    mod = module_from_spec(spec)
    sys.modules["b11_bf_leak"] = mod
    spec.loader.exec_module(mod)
    allowed = {"x_T", "hi_b", "hi_a", "b_hat", "b_true", "rul",
               "rul_lower_bound", "label_fail", "mission_features"}
    leaked = []
    with h5py.File(FEATURE_H5, "r") as f:
        assert bool(f.attrs["mission_features_in_xT"]) is False
        for key in (k for k in f.keys() if k.startswith("traj_")):
            for name in f[key].keys():
                if name not in allowed:
                    leaked.append(f"{key}/{name}")
            for at in f[key].attrs:
                if at in mod.TRUTH_ATTRS:
                    leaked.append(f"{key}.{at}")
    assert not leaked, f"特征文件出现真值泄漏: {leaked[:10]}"


def test_b11_feature_gate_recorded():
    """若 feature_stats.json 存在: 必须记录 6 条 Gate 与 verdict。"""
    p = ROOT / "checkpoints/basilisk_b11/feature_stats.json"
    if not p.exists():
        pytest.skip("feature_stats.json 不存在 (FAIL 路径)")
    d = json.loads(p.read_text(encoding="utf-8"))
    assert len(d["feature_gates"]) == 6
    assert d["verdict"] in ("B11_FEATURE_READY", "B11_FEATURE_INVALID")
    assert d["mission_features_in_xT"] is False
    assert d["trained_any_model"] is False and d["ran_rul_model"] is False


# --------------------------------------------------------------------------
# §17 RUL 禁令
# --------------------------------------------------------------------------

@pytest.mark.parametrize("path", [p.name for p in B11_SCRIPTS])
def test_b11_no_rul_in_stage(path):
    """§17: B1.1 的任何脚本都不得 import 模型/迁移模块或调用训练入口。"""
    rel = f"scripts/basilisk_b11/{path}"
    tree, src = _tree(rel)
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for al in node.names:
                if any(al.name == m or al.name.startswith(m + ".")
                       for m in FORBIDDEN_MODULES):
                    bad.append(f"L{node.lineno}: import {al.name}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            if any(node.module == m or node.module.startswith(m + ".")
                   for m in FORBIDDEN_MODULES):
                bad.append(f"L{node.lineno}: from {node.module}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in FORBIDDEN_ENTRYPOINTS:
            bad.append(f"L{node.lineno}: 调用 {node.func.id}()")
    assert not bad, f"{rel} 违反 §17 RUL 禁令:\n" + "\n".join(bad)


def test_b11_artifacts_declare_no_rul():
    """所有 B1.1 产物 JSON 必须显式声明未跑 RUL / 未用模型指标。"""
    checked = 0
    for rel in ("checkpoints/basilisk_b11/protocol_hash.json",
                "checkpoints/basilisk_b11/candidates.json",
                "checkpoints/basilisk_b11/candidate_audit.json",
                "checkpoints/basilisk_b11/dataset_stats.json",
                "checkpoints/basilisk_b11/feature_stats.json"):
        p = ROOT / rel
        if not p.exists():
            continue
        checked += 1
        d = json.loads(p.read_text(encoding="utf-8"))
        flags = {k: v for k, v in d.items()
                 if k in ("ran_rul_model", "used_model_metric", "used_rul_metric",
                          "trained_any_model")}
        assert flags, f"{rel} 未声明任何 RUL 禁令标志"
        for k, v in flags.items():
            assert v is False, f"{rel} 的 {k} = {v}"
    assert checked >= 3, f"只检查到 {checked} 个产物"


def test_no_checkpoint_or_metric_files_produced():
    """§17: B1.1 目录下不得出现任何模型权重 / 训练指标文件。"""
    d = ROOT / "checkpoints/basilisk_b11"
    if not d.exists():
        pytest.skip("目录不存在")
    bad = [p.name for p in d.iterdir()
           if p.suffix in (".pt", ".pth", ".ckpt", ".onnx")]
    assert not bad, f"出现模型权重文件: {bad}"


def test_b11_not_written_into_requirements_or_docker():
    """跨阶段纪律: Basilisk 不得写入 requirements.txt / Dockerfile / compose。"""
    for rel in ("requirements.txt", "Dockerfile", "docker-compose.yml"):
        p = ROOT / rel
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8").lower()
        assert "basilisk" not in txt, f"{rel} 出现 basilisk 依赖"
