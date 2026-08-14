"""tests/basilisk_b17/test_b17_no_outcome_tuning.py

§17 —— `test_protocol_frozen_before_feasibility` 与 `test_no_outcome_tuning`。

反作弊扫描器沿用 B1.6 的三重豁免设计 (docstring/注释剥离、`NAME = (...)`
常量声明块豁免、否定键剥离), 并保留**负向自检**: 若不能证明扫描器抓得住
真实违规, 那么"没抓到违规"这一结论毫无信息量。
"""
from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "basilisk_b17"

# 禁止作为热模型/判决依据的量 —— B1.7 源码里绝不读取它们。
FORBIDDEN_SYMBOLS = (
    "failure_fraction", "rul_rmse", "RMSE", "nPHM", "PHM", "MAE",
    "target_only", "expected_RUL", "censored_fraction",
)
# 否定键: 这些键名出现时表达的是"没有用某物", 不构成使用。
NEGATION_KEY = re.compile(
    r'"(?:(?:read_any|used|no)_[a-z_]*|[a-z_]*(?:not_used|not_taken|forbidden|'
    r'generated|run|defined|tuned)[a-z_]*)"\s*:')


def _code_lines(path: Path) -> list[tuple[int, str]]:
    """返回 (行号, 代码文本): 剥离 docstring / 注释, 豁免常量声明块,
    并把否定键从行内**剥掉**(而不是跳过整行 —— 否则同一行上的真实读取会漏检)。
    """
    src = io.open(path, encoding="utf-8").read()
    # 剥 docstring (三引号块)
    src = re.sub(r'(?s)""".*?"""', '""', src)
    src = re.sub(r"(?s)'''.*?'''", "''", src)
    out: list[tuple[int, str]] = []
    depth = 0            # 常量声明块 `NAME = (` 的括号深度
    in_const = False
    for i, raw in enumerate(src.split("\n"), start=1):
        code = raw.split("#", 1)[0]
        if not in_const and re.match(r"^[A-Z_][A-Z0-9_]*\s*[:=]", code):
            # 常量声明: 允许其中出现禁用词 (那是"声明禁止清单", 不是使用)
            in_const = True
            depth = code.count("(") + code.count("[") + code.count("{") \
                - code.count(")") - code.count("]") - code.count("}")
            if depth <= 0:
                in_const = False
            continue
        if in_const:
            depth += (code.count("(") + code.count("[") + code.count("{")
                      - code.count(")") - code.count("]") - code.count("}"))
            if depth <= 0:
                in_const = False
            continue
        out.append((i, NEGATION_KEY.sub("", code)))
    return out


def _scan(paths) -> list[str]:
    bad = []
    for p in paths:
        for ln, code in _code_lines(p):
            for sym in FORBIDDEN_SYMBOLS:
                if sym in code:
                    bad.append(f"{p.name}:{ln}: {sym} -> {code.strip()[:80]}")
    return bad


def test_no_outcome_tuning():
    """§17 —— B1.7 源码不得读取 failure fraction / RUL / RMSE 等结果量。

    这些量若进入本阶段, 就意味着热模型或判决可能被"结果好不好"反向驱动。
    """
    scripts = sorted(SCRIPTS.glob("*.py"))
    assert len(scripts) >= 6, f"脚本数异常: {len(scripts)}"
    bad = _scan(scripts)
    assert bad == [], "本阶段源码出现结果量读取 (禁止):\n" + "\n".join(bad)


def test_scanner_actually_scans_code():
    """确认豁免规则没有把整份源码都豁免掉 (否则上一条测试是空转)。"""
    total = sum(len(_code_lines(p)) for p in sorted(SCRIPTS.glob("*.py")))
    assert total > 200, f"可扫描代码行仅 {total} 行 —— 豁免规则过宽"


def test_scanner_would_catch_a_real_violation(tmp_path):
    """负向自检: 造一行真实违规, 扫描器必须抓到。

    没有这条, 「没抓到违规」无法区分「确实干净」与「扫描器瞎了」。
    """
    p = tmp_path / "planted.py"
    p.write_text(
        "def pick():\n"
        "    r = load()\n"
        "    return r[\"rul_rmse\"] < 1.0\n", encoding="utf-8")
    assert _scan([p]), "扫描器未能抓到植入的真实违规"


def test_scanner_not_fooled_by_same_line_negation(tmp_path):
    """否定键剥离必须只剥键名, 同一行上的真实读取仍要被抓到。"""
    p = tmp_path / "mixed.py"
    p.write_text(
        'def f(r):\n'
        '    return {"rul_generated": False, "x": r["rul_rmse"]}\n',
        encoding="utf-8")
    assert _scan([p]), "同一行的真实读取被否定键掩盖"


def test_no_thermal_param_hardcoded_in_config():
    """§10 —— config 里不得出现可调热参数数值 (R_th / C_th / tau / eta / Kt)。

    只查键名: 若这些键出现在 config, 就意味着热参数可以绕过 provenance
    registry 被手工设定。
    """
    cfg_txt = io.open(ROOT / "configs/wheel_basilisk_b17.yaml",
                      encoding="utf-8").read()
    # 剥注释后再查 —— 注释里提到参数名是说明, 不是设定
    code = "\n".join(ln.split("#", 1)[0] for ln in cfg_txt.split("\n"))
    forbidden_keys = ("R_th:", "C_th:", "tau_th:", "efficiency:", "eta:",
                      "Kt:", "winding_resistance:", "thermal_resistance:",
                      "thermal_capacitance:")
    hit = [k for k in forbidden_keys if k in code]
    assert hit == [], f"config 里出现可手工设定的热参数键: {hit}"


def test_protocol_frozen_before_feasibility(b17_protocol_hash, b17_config):
    """§17 —— 协议必须在任何审计/可行性产物之前冻结, 且此后未被改动。"""
    ph = b17_protocol_hash
    assert ph["frozen_before_all_absent_at_freeze_time"] is True, (
        "冻结时刻已存在审计产物 —— 判据可能是看过数据后才定的")
    # 协议内容未被改动
    p = ROOT / ph["protocol_path"]
    assert p.exists()
    now = hashlib.sha256(p.read_bytes()).hexdigest()
    assert now == ph["protocol_sha256"], (
        "protocol.md 在冻结后被改动 —— 结论作废")
    # config 声明的"必须后于协议"的产物清单与冻结记录一致
    assert set(ph["frozen_before"]) == set(
        b17_config["protocol"]["must_freeze_before"])


def test_all_products_carry_protocol_hash(b17_protocol_hash):
    """每个审计产物都必须带上冻结时的 protocol hash —— 便于溯源作废。"""
    h = b17_protocol_hash["protocol_sha256"]
    for rel in ("checkpoints/basilisk_b17/temperature_provenance.json",
                "checkpoints/basilisk_b17/thermal_parameter_registry.json",
                "checkpoints/basilisk_b17/mission_power_audit.json",
                "checkpoints/basilisk_b17/thermal_feasibility.json",
                "checkpoints/basilisk_b17/b17_verdict.json"):
        p = ROOT / rel
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        assert d.get("protocol_sha256") == h, f"{rel} 的 protocol hash 不匹配"


def test_protocol_states_sole_admissible_reason():
    """§2 —— 协议必须写明候选轮的唯一准入理由, 并逐条否认其它理由。"""
    txt = io.open(ROOT / "docs/basilisk_b17/protocol.md",
                  encoding="utf-8").read()
    assert "唯一理由" in txt or "唯一准入理由" in txt
    assert "failure fraction" in txt, "未逐条否认 failure fraction 作为理由"
    assert "THERMAL_COMPONENT_LIMIT" in txt
    # 必须保留 operating/storage 的区分规则与 unlabeled 处理
    assert "operating" in txt and "storage" in txt and "unlabeled" in txt
    # 必须保留"不得偏向 T1 后回去凑数字"
    assert "不得偏向 T1" in txt


def test_forbidden_fallbacks_declared(b17_verdict):
    """§15 —— 必须逐条声明未采取的回退路线。"""
    ff = b17_verdict["forbidden_fallbacks_not_taken"]
    assert len(ff) >= 5, f"回退路线否认清单过短: {ff}"
    joined = " ".join(ff)
    assert "典型" in joined, "未否认「取典型值试算」这条最可能的捷径"
    assert "T_base" in joined or "wheel_sim" in joined, (
        "未否认「复用既有 T 的经验参数」")
