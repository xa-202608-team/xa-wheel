"""tests/basilisk_b12/test_b12_dataset_isolation.py

§19 —— 命名空间隔离与真值边界:
  * test_b12_dataset_isolated
  * test_truth_never_enters_xt
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# §0 硬隔离: B1.2 绝不可写入的 lineage
PROTECTED = (
    "data/sim/wheel_basilisk_v1", "data/sim/wheel_basilisk_b1",
    "data/sim/wheel_basilisk_b11",
    "data/features/wheel/basilisk_v1", "data/features/wheel/basilisk_b1",
    "data/features/wheel/basilisk_b11",
    "checkpoints/basilisk_v1", "checkpoints/basilisk_b1",
    "checkpoints/basilisk_b11",
)

TRUTH_FIELDS = ("b_true", "Kt_true", "Tc_true", "b0_true")

# 写操作方法名 (只读方法如 read_text / open(...,'r') 不在内)
WRITE_CALLS = frozenset({
    "write_text", "write_bytes", "mkdir", "unlink", "rmtree", "rmdir",
    "remove", "makedirs", "dump", "to_hdf", "to_csv", "savez", "save",
    "create_dataset", "copyfile", "move", "rename", "replace",
})
WRITE_MODES = frozenset({"w", "a", "x", "wb", "ab", "xb", "w+", "a+", "r+"})


def _is_protected(s: str) -> bool:
    """路径段边界比对: b12 不得被 b1 前缀误伤。"""
    v = s.replace("\\", "/")
    return any(v == bad or v.startswith(bad + "/") for bad in PROTECTED)


def _call_name(node: ast.Call) -> str:
    f = node.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return ""


def _call_path_literals(node: ast.Call) -> list[str]:
    """调用的**全部**字符串字面量 —— 含 func 侧, 以覆盖 Path('...').mkdir()。"""
    return [c.value for c in ast.walk(node)
            if isinstance(c, ast.Constant) and isinstance(c.value, str)]


def _is_write_call(node: ast.Call, fname: str) -> bool:
    if fname in WRITE_CALLS:
        return True
    if fname in {"open", "File"}:
        modes = [c.value for a in node.args[1:] for c in ast.walk(a)
                 if isinstance(c, ast.Constant) and isinstance(c.value, str)]
        modes += [c.value for kw in node.keywords if kw.arg == "mode"
                  for c in ast.walk(kw.value)
                  if isinstance(c, ast.Constant) and isinstance(c.value, str)]
        return any(m in WRITE_MODES for m in modes)
    return False


def test_b12_dataset_isolated(b12_cfg, b12_gen):
    """§0: B1.2 的所有输出路径都必须落在 b12 命名空间内, 且 guard 真的会拒绝越界。"""
    for key in ("sim_dir", "candidate_dir", "feature_h5", "ckpt_dir", "doc_dir"):
        p = str(b12_cfg["paths"][key]).replace("\\", "/")
        assert "b12" in p, f"paths.{key} = {p} 不在 b12 命名空间"
        for bad in PROTECTED:
            assert not (p == bad or p.startswith(bad + "/")), \
                f"paths.{key} 指向受保护 lineage {bad}"

    # guard_out 必须拒绝受保护路径…
    for bad in PROTECTED:
        with pytest.raises(SystemExit):
            b12_gen.guard_out(ROOT / bad)
        with pytest.raises(SystemExit):
            b12_gen.guard_out(ROOT / bad / "candidates" / "X")
    # …但不得误伤 b12 自己 (前缀比对必须按路径段边界:
    # wheel_basilisk_b12 与 wheel_basilisk_b1 共享前缀)
    b12_gen.guard_out(ROOT / "data/sim/wheel_basilisk_b12")
    b12_gen.guard_out(ROOT / "data/sim/wheel_basilisk_b12/candidates/F2")
    b12_gen.guard_out(ROOT / "checkpoints/basilisk_b12")
    b12_gen.guard_out(ROOT / "data/features/wheel/basilisk_b12")


def test_b12_scripts_never_write_protected_paths():
    """静态检查: b12 脚本不得把受保护 lineage 路径喂给**写操作**。

    口径说明 (为什么不是"源码里不得出现该字符串"):
    `verify_baseline.py` 按 §1 的职责就是**逐个读取并 hash** v1/B1/B1.1 的全部冻结
    产物, `audit_failure_mechanisms.py` 按 §6 必须读 B1.1 的 candidate 轨迹。
    "出现即报警"会把这些合法只读引用全判成违规, 只能靠不断扩白名单来压 ——
    白名单越长, 测试越接近失效。

    因此本测试检查**写入动作**: 受保护路径字面量不得作为参数流入
    `open(..., 'w'/'a'/'x')` / `h5py.File(..., 'w'/'a')` / `write_text` /
    `write_bytes` / `mkdir` / `unlink` / `rmtree` / `json.dump` / `to_hdf` 等。
    只读的 `read_text` / `sha256_of` / `h5py.File(..., 'r')` 不报警。

    前缀比对按**路径段边界** —— `checkpoints/basilisk_b12/...` 与受保护的
    `checkpoints/basilisk_b1` 共享字符串前缀, 裸 startswith 会误伤 b12 自己。
    """
    for p in sorted((ROOT / "scripts" / "basilisk_b12").glob("*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call):
                continue
            fname, strs = _call_name(n), _call_path_literals(n)
            if not _is_write_call(n, fname):
                continue
            for s in strs:
                assert not _is_protected(s), \
                    f"{p.name}:{n.lineno} 写操作 {fname}() 指向受保护路径 {s}"

    # 反向自检: 匹配器必须真能抓到已知违规写法, 否则本测试是"永真"的
    probe = ast.parse(
        "open('checkpoints/basilisk_b1/x.json', 'w')\n"
        "Path('data/sim/wheel_basilisk_b11/y').mkdir()\n"
        "h5py.File('data/features/wheel/basilisk_v1/z.h5', 'w')\n")
    hits = 0
    for n in ast.walk(probe):
        if isinstance(n, ast.Call) and _is_write_call(n, _call_name(n)):
            hits += sum(1 for s in _call_path_literals(n) if _is_protected(s))
    assert hits >= 3, f"写操作匹配器失效 (仅命中 {hits}/3 个已知违规样例)"


def test_truth_never_enters_xt(b12_cfg):
    """§8/§17: b_true / Kt_true / Tc_true / b0_true 绝对禁止进入 x_T。

    两层:
      1. 核心 x_T 列定义 (src/sim/build_hi.XT_COLS) 不含任何真值字段;
      2. b12 特征脚本 (若已存在) 的源码里不得把真值字段塞进特征矩阵。
    """
    from src.sim.build_hi import XT_COLS
    for f in TRUTH_FIELDS:
        assert f not in XT_COLS, f"x_T 定义里出现真值字段 {f}"
    # b_true 是最容易漏的那个: 显式再查一遍任何以 _true 结尾的列
    leaked = [c for c in XT_COLS if c.endswith("_true")]
    assert not leaked, f"x_T 含真值字段 {leaked}"

    bf = ROOT / "scripts" / "basilisk_b12" / "build_features.py"
    if not bf.exists():
        pytest.skip("build_features.py 尚未创建 (§16 PASS 后才生成)")
        return
    tree = ast.parse(bf.read_text(encoding="utf-8"))
    # 找出所有 CORE_XT_COLS / 特征列常量的字符串成员
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            names = {t.id for t in n.targets if isinstance(t, ast.Name)}
            if any("XT" in x or "FEATURE" in x for x in names):
                strs = [s.value for s in ast.walk(n)
                        if isinstance(s, ast.Constant) and isinstance(s.value, str)]
                for f in TRUTH_FIELDS:
                    assert f not in strs, \
                        f"build_features.py 的 {names} 含真值字段 {f}"
                assert not [s for s in strs if s.endswith("_true")], \
                    f"build_features.py 的 {names} 含 *_true 字段"


def test_old_b11_unchanged(b12_verify):
    """§19 test_old_b11_unchanged: B1.1 的冻结数字必须一字不差。

    这是 B1.2 不得回改上游的硬约束 —— 若 B1.1 的 b0_scale / 参考口径 / candidate
    hash / verdict 被动过, 本阶段所有"配对比较"的前提立即失效。
    """
    exp = b12_verify.B11_EXPECTED
    assert exp["b11_verdict"] == "B11_CALIBRATION_FAIL"
    assert exp["b11_selected_candidate"] == "None"
    assert exp["b11_b0_scale"] == "5.906442914788147"
    assert exp["b11_speed_util_ref_p95"] == "0.4041898280945392"
    assert exp["b11_torque_util_ref_p95"] == "0.284426851489331"
    assert exp["b11_paired_trajectory_hash"] == \
        "ff1de559b2746dcb394c4de6d70bda4d96ca594d1b953f58a4fa390366f41cf6"
    assert exp["b11_protocol_sha256"] == \
        "4a6a991edb9497262c5fb720938258ae41ebbb3685ca4eff06bc4880aefc99ff"

    # 与 B1.1 实际产物逐项比对
    import json
    ck = ROOT / "checkpoints" / "basilisk_b11"
    cands = json.loads((ck / "candidates.json").read_text(encoding="utf-8"))
    assert repr(cands["b0_calibration"]["b0_scale"]) == exp["b11_b0_scale"]
    assert repr(cands["reference_duty_p95"]["speed_util"]) == \
        exp["b11_speed_util_ref_p95"]
    assert repr(cands["reference_duty_p95"]["torque_util"]) == \
        exp["b11_torque_util_ref_p95"]
    assert cands["paired_trajectory_hash"] == exp["b11_paired_trajectory_hash"]
    assert cands["protocol_sha256"] == exp["b11_protocol_sha256"]
    for tag, key in (("A", "b11_candidate_A_content_sha256"),
                     ("B", "b11_candidate_B_content_sha256"),
                     ("C", "b11_candidate_C_content_sha256")):
        assert exp[key] in json.dumps(cands), \
            f"B1.1 candidate {tag} 的 content hash 已变化"

    # B1.1 是 FAIL 阶段: 不应存在 frozen_calibration.json
    assert not (ck / "frozen_calibration.json").exists(), \
        "B1.1 是 CALIBRATION_FAIL, 不应有 frozen_calibration.json"
