"""tests/basilisk_b16/test_b16_no_outcome_selection.py —— §15: 不得用结果指标选轮。

这是本阶段最核心的反作弊检查。B1.6 之前的多轮失败, 部分正是因为"先看哪个配置
失效得好, 再回头解释"。所以此处从两侧钉死:

1. **源码侧**: 选择链上的脚本里不得出现 RUL / RMSE / failure fraction 等符号。
2. **产物侧**: 裁决产物里必须显式记录"没读过这些量", 且不存在寿命数据文件。
"""
from __future__ import annotations

import io
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCR = ROOT / "scripts" / "basilisk_b16"
CK = ROOT / "checkpoints" / "basilisk_b16"

SELECTION_CHAIN = ("enumerate_wheel_models.py", "audit_basilisk_wheel_fields.py",
                   "audit_datasheet_sources.py", "build_candidate_registry.py",
                   "score_provenance.py", "freeze_wheel_choice.py")

# 结果类指标 —— 出现在**赋值/读取**位置即为违规
OUTCOME_TOKENS = ("rmse", "nphm", "phm_score", "mae", "expected_rul",
                  "failure_fraction", "censored_fraction", "n_failed",
                  "transfer_gain")


def _code_lines(path: Path) -> list[tuple[int, str]]:
    """剥掉注释与 docstring, 并跳过"禁止清单"常量块。

    两处必须排除, 否则扫描器会咬自己:
    - 注释 / docstring: 反作弊说明本身要提到这些词。
    - `FORBIDDEN_*` / `OUTCOME_*` 常量的字面量列表: 那是**声明禁止**, 不是使用。
      判据是"从常量赋值行起, 直到括号配平为止"。
    - 反作弊**否证声明**, 形如 `"read_any_failure_fraction": False` 或
      `"tiebreak_forbidden_not_used": [...]`: 那是在产物里写下"没读过/没用过",
      语义与"读了"正好相反。做法是把这个 **key 本身**从行里抹掉再继续扫描
      (而不是跳过整行) —— 同一行上真出现取值仍然会被抓到。
    """
    # 只认"否证语义"的 key 名: 含 not_used / forbidden, 或以 read_any_ / used_ 开头
    negation_key = re.compile(
        r'"(?:(?:read_any|used)_[a-z_]*|[a-z_]*(?:not_used|forbidden)[a-z_]*)"'
        r'\s*:')
    out, in_doc, in_decl, depth = [], False, False, 0
    for i, line in enumerate(io.open(path, encoding="utf-8").read().splitlines(),
                             1):
        s = line.strip()
        if s.startswith('"""') or s.startswith("'''"):
            if len(s) < 6 or not s.endswith(s[:3]):
                in_doc = not in_doc
            continue
        if in_doc or s.startswith("#"):
            continue
        code = line.split("#")[0]
        if not in_decl and re.match(r"^[A-Z_]+\s*(:.*)?=\s*[\(\[\{]", code):
            in_decl, depth = True, 0
        if in_decl:
            depth += sum(code.count(c) for c in "([{")
            depth -= sum(code.count(c) for c in ")]}")
            if depth <= 0:
                in_decl = False
            continue
        out.append((i, negation_key.sub("", code)))
    return out


def test_no_failure_fraction_used():
    """选择链源码里不得读取 failure fraction 类量。"""
    for f in SELECTION_CHAIN:
        for i, line in _code_lines(SCR / f):
            low = line.lower()
            for tok in ("failure_fraction", "censored_fraction", "n_failed"):
                assert tok not in low, \
                    f"{f}:{i} 读取了 {tok}: {line.strip()}"


def test_no_rul_metric_used():
    """选择链源码里不得读取 RUL / RMSE / PHM 等结果指标。"""
    for f in SELECTION_CHAIN:
        for i, line in _code_lines(SCR / f):
            low = line.lower()
            for tok in OUTCOME_TOKENS:
                assert tok not in low, \
                    f"{f}:{i} 读取了 {tok}: {line.strip()}"


def test_scanner_actually_scans_code():
    """自检: 排除逻辑不能把整份文件都吃掉, 否则上面两条测试是空转。"""
    for f in SELECTION_CHAIN:
        n = len(_code_lines(SCR / f))
        assert n > 20, f"{f} 只扫到 {n} 行代码 —— 排除逻辑过宽"


def test_forbidden_list_declaration_is_present():
    """自检: 打分脚本必须**声明**禁止清单 (只是不得使用其中的量)。"""
    txt = io.open(SCR / "score_provenance.py", encoding="utf-8").read()
    assert "FORBIDDEN_AS_BASIS" in txt
    assert "failure_fraction" in txt, "禁止清单被删了 —— 声明本身要保留"


def test_scanner_would_catch_a_real_violation(tmp_path):
    """负向自检: 造一行真实的违规取值, 扫描器必须抓到。

    没有这条, 上面两条"没抓到违规"的测试无法区分"确实干净"与"扫描器瞎了"。
    """
    p = tmp_path / "fake_selection.py"
    p.write_text(
        'FORBIDDEN_AS_BASIS = (\n    "failure_fraction",\n)\n'
        '"""docstring 里提到 rmse 不算。"""\n'
        '# 注释里提到 failure_fraction 也不算\n'
        'x = {"read_any_failure_fraction": False}\n'
        'y = {"tiebreak_forbidden_not_used": ["expected_RUL"]}\n'
        'best = max(rows, key=lambda r: r["rul_rmse"])\n',
        encoding="utf-8")
    lines = _code_lines(p)
    hits = [(i, l) for i, l in lines if "rul_rmse" in l.lower()]
    assert hits, "扫描器漏掉了真实的结果指标取值"
    # 确认三类豁免都生效
    assert not any("FORBIDDEN_AS_BASIS" in l for _, l in lines)
    assert not any("read_any_failure_fraction" in l for _, l in lines)
    assert not any("tiebreak_forbidden_not_used" in l for _, l in lines)


def test_no_lifetime_or_result_file_read():
    """选择链不得打开寿命数据 / 实验结果文件。"""
    banned_paths = ("data/sim/wheel_basilisk_b1", "results.md", "experiments/",
                    "rul_", "docs/results")
    for f in SELECTION_CHAIN:
        for i, line in _code_lines(SCR / f):
            low = line.lower()
            if "open(" not in low and "read_text" not in low \
                    and "h5py" not in low:
                continue
            for b in banned_paths:
                assert b not in low, f"{f}:{i} 读了禁止文件 {b}"


def test_verdict_records_no_outcome_basis(b16_scores, b16_verdict):
    """产物侧: 必须显式记录未使用任何结果类依据。"""
    ac = b16_scores["anti_cheat"]
    assert ac["read_any_failure_fraction"] is False
    assert ac["read_any_rul_metric"] is False
    assert ac["read_any_transfer_result"] is False
    assert ac["used_data_availability_as_tiebreak"] is False
    assert b16_verdict["lifetime_data_generated"] is False
    assert b16_verdict["rul_run"] is False
    assert b16_verdict["transfer_run"] is False
    assert b16_verdict["s6_run"] is False


def test_no_new_lifetime_dataset_on_disk():
    """§13 / §18: B1.6 不得留下任何寿命数据集。"""
    assert not (ROOT / "data" / "sim" / "wheel_basilisk_b16").exists()
    assert not (CK / "frozen_wheel_model.json").exists()


def test_scores_inputs_are_audit_only():
    """打分脚本的输入只能是 §3/§4 审计产物。"""
    txt = io.open(SCR / "score_provenance.py", encoding="utf-8").read()
    reads = set(re.findall(r'CK / "([^"]+)"', txt))
    assert reads <= {"candidate_registry.json", "field_audit.json",
                     "provenance_scores.json"}, reads
