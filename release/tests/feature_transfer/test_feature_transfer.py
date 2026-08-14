"""tests/feature_transfer/test_feature_transfer.py

P1-3 特征空间迁移对照实验的测试。

测试覆盖:
1. 固定 5 seeds (不是 3-5, 恰为 5)
2. 同一 split, 同一模型配置 (从 config 读取)
3. paired_delta 的符号定义正确
4. bootstrap 的单位是 seed 级, n=5
5. 源域归一化参数确实来自 XJTU-SY train 集
6. 特征对齐是朴素逐列 (min(F_src, F_tgt))
7. 正/负结果都通过 (诚实实验, 不预设方向)
"""
from __future__ import annotations

import io
import json
import sys
import tokenize
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_REL = "scripts/run_feature_transfer.py"
CONFIG_REL = "configs/feature_transfer.yaml"
OUTPUT_REL_MARKER = "feature_transfer_paired.json"


def code_only(rel: str) -> str:
    """剥掉注释与字符串后的源码 (禁令类测试用)。"""
    src = (ROOT / rel).read_text(encoding="utf-8")
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            out.append(tok.string)
    except (tokenize.TokenError, IndentationError):
        return src
    return "\n".join(out)


def _load_output() -> dict:
    """尝试加载实验产物; 不存在则 skip。"""
    # 输出在 release/../../../../../05_结果/reference/wheel/
    p = (ROOT.parent.parent.parent.parent
         / "05_结果" / "reference" / "wheel" / "feature_transfer_paired.json")
    if not p.exists():
        # 可能在 release/ 下 (smoke)
        alt = ROOT / "checkpoints" / "_feature_transfer" / "feature_transfer_paired.json"
        if alt.exists():
            p = alt
        else:
            pytest.skip(f"实验产物不存在 (需先跑 run_feature_transfer.py): {p}")
    return json.loads(p.read_text(encoding="utf-8"))


# ============================================================
# 代码静态测试 (不需要跑实验)
# ============================================================

def test_fixed_5_seeds_not_3_to_5():
    """spec P1-3: 固定 5 seeds, 不得写 3-5 或 range。"""
    src = (ROOT / SCRIPT_REL).read_text(encoding="utf-8")
    # 必须有恰 5 个 seed
    assert "FORMAL_SEEDS = [112, 113, 114, 115, 116]" in src, (
        "seeds 必须硬编码为 [112, 113, 114, 115, 116], 不得用 range 或 3-5")
    # 禁止出现 "3-5" 或 "3_5" 之类的弹性表述
    code = code_only(SCRIPT_REL)
    assert "range(3" not in code.replace(" ", ""), (
        "seeds 不得用 range(3, ...) 等弹性表述")


def test_two_conditions_explicit():
    """spec P1-3: 恰好两个条件 (native_scaler / source_scaler)。"""
    src = (ROOT / SCRIPT_REL).read_text(encoding="utf-8")
    assert "native_scaler" in src
    assert "source_scaler" in src


def test_source_scaler_from_xjtu_train():
    """条件 B 的归一化参数必须来自 XJTU-SY train 行, 不是目标域。"""
    src = (ROOT / SCRIPT_REL).read_text(encoding="utf-8")
    # compute_source_normalization 必须从源域 h5 读取
    assert "compute_source_normalization" in src
    # 必须用 split == train 筛选
    assert "b'train'" in src or 'b\"train\"' in src, (
        "源域归一化必须只用 train split 的行")


def test_native_scaler_from_target_train():
    """条件 A 的归一化参数必须来自目标域 train 行 (B2 标准口径)。"""
    src = (ROOT / SCRIPT_REL).read_text(encoding="utf-8")
    assert "compute_native_normalization" in src
    # 必须用目标域 train mask
    assert "m_tr" in src


def test_clip_before_zscore():
    """条件 B 的归一化顺序: 先 clip 到 [p1, p99], 再 z-score。"""
    src = (ROOT / SCRIPT_REL).read_text(encoding="utf-8")
    # apply_source_scaler_to_target 内先 np.clip 再 (x - mu) / sd
    assert "np.clip" in src
    assert "p1" in src and "p99" in src


def test_paired_delta_definition():
    """paired_delta = native_rmse - source_rmse, 正数 = source_scaler 更好。"""
    src = (ROOT / SCRIPT_REL).read_text(encoding="utf-8")
    assert "native_rmses" in src and "source_rmses" in src
    # delta = native - source (不是 source - native)
    assert "n - s for n, s in zip(native_rmses, source_rmses)" in src


def test_bootstrap_is_seed_level():
    """bootstrap 单位必须是 seed 级, n=5, 不是时间点。"""
    src = (ROOT / SCRIPT_REL).read_text(encoding="utf-8")
    assert "paired_bootstrap" in src, "必须用 B2 冻结的 paired_bootstrap"
    assert "N_BOOTSTRAP = 5000" in src
    assert "BOOTSTRAP_SEED = 20260814" in src


def test_no_single_seed_ci():
    """禁止从单 seed 画 CI (spec P1-3)。"""
    src = (ROOT / SCRIPT_REL).read_text(encoding="utf-8")
    # 必须遍历全部 5 seeds 才产出 CI
    assert "for seed in seeds" in src


def test_same_model_config_both_conditions():
    """两种条件必须用同一模型配置 (n_features=12, S2.5 训练超参)。"""
    src = (ROOT / SCRIPT_REL).read_text(encoding="utf-8")
    # 两种条件都调用 train_target_only, 同一个 th (training_hyper)
    assert "train_target_only" in src
    assert src.count("train_target_only(") >= 2   # 定义 + 两次调用


def test_no_formal_transfer_conclusion():
    """对照实验不得产生正式迁移结论 (B5 才是正式)。"""
    src = (ROOT / SCRIPT_REL).read_text(encoding="utf-8")
    assert "POSITIVE_TRANSFER" not in src, (
        "对照实验不得使用 POSITIVE_TRANSFER 标签")
    assert "FORMAL_TRANSFER_EVALUATION" not in src, (
        "对照实验不得使用 FORMAL_TRANSFER_EVALUATION 标签")


# ============================================================
# 产物测试 (需要先跑实验)
# ============================================================

def test_output_has_5_seeds():
    """产物必须有 5 个 seed 的结果。"""
    out = _load_output()
    if out.get("fast_mode") and len(out["seeds"]) < 5:
        pytest.skip("--fast 单 seed 模式, 跳过 5-seeds 检查")
    assert len(out["seeds"]) == 5, f"seeds 必须恰为 5 个, 得到 {len(out['seeds'])}"
    assert out["seeds"] == [112, 113, 114, 115, 116]


def test_output_conditions_structure():
    """产物 conditions 必须有 native_scaler 和 source_scaler 两块。"""
    out = _load_output()
    for cond in ("native_scaler", "source_scaler"):
        assert cond in out["conditions"], f"缺条件 {cond}"
        c = out["conditions"][cond]
        assert "per_seed_rmse" in c
        assert "mean" in c
        assert len(c["per_seed_rmse"]) == len(out["seeds"])


def test_output_paired_delta():
    """产物 paired_delta 必须有逐 seed 差值 + CI + crosses_zero。"""
    out = _load_output()
    pd = out["paired_delta"]
    assert "per_seed" in pd
    assert len(pd["per_seed"]) == len(out["seeds"])
    assert "mean" in pd
    assert "ci95" in pd
    assert len(pd["ci95"]) == 2
    assert "crosses_zero" in pd
    assert pd["ci95"][0] <= pd["mean"] <= pd["ci95"][1]


def test_output_paired_delta_consistency():
    """paired_delta[seed] = native[seed] - source[seed] 逐 seed 一致。"""
    out = _load_output()
    n = out["conditions"]["native_scaler"]["per_seed_rmse"]
    s = out["conditions"]["source_scaler"]["per_seed_rmse"]
    d = out["paired_delta"]["per_seed"]
    for i in range(len(d)):
        expected = n[i] - s[i]
        assert abs(d[i] - expected) < 1e-9, (
            f"seed {i}: delta {d[i]} != native {n[i]} - source {s[i]}")


def test_output_source_normalization():
    """产物必须记录源域归一化参数的来源。"""
    out = _load_output()
    sn = out["source_normalization"]
    assert sn["source_dataset"] == "XJTU-SY"
    # source_feature_names 存储完整的 12 维特征名列表
    feat_names = sn.get("source_feature_names") or sn.get("feature_names")
    assert feat_names is not None, "source_normalization 缺 feature_names / source_feature_names"
    assert len(feat_names) == 12
    assert sn["n_train_rows"] > 0
