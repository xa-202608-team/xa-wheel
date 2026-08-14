"""特征工程自检 — 验证 extract_features 公式对已知信号的正确性。

对应 plan P1 feature-researcher 角色: 公式落地后须验证实现无误。
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.preprocess.wheel_features import extract_features, FEATURE_NAMES  # noqa: E402

FS = 25600


def test_feature_count_and_shape():
    assert len(FEATURE_NAMES) == 12
    rng = np.random.default_rng(0)
    feats = extract_features(rng.standard_normal(8192), FS)
    assert feats.shape == (12,)


def test_rms_of_pure_sine():
    """纯正弦 x = A·sin(2πft) 的 RMS = A/√2。"""
    A = 2.0
    t = np.arange(8192) / FS
    x = A * np.sin(2 * np.pi * 100 * t)
    rms = extract_features(x, FS)[FEATURE_NAMES.index("rms")]
    assert abs(rms - A / np.sqrt(2)) < 0.02


def test_kurtosis_sensitive_to_impact():
    """含周期冲击的信号峭度应显著高于纯噪声 (退化冲击敏感)。"""
    rng = np.random.default_rng(1)
    base = rng.standard_normal(8192) * 0.5
    with_imp = base.copy()
    for ix in range(0, 8192, 800):
        with_imp[ix] += 10.0
    k_idx = FEATURE_NAMES.index("kurtosis")
    assert extract_features(with_imp, FS)[k_idx] > extract_features(base, FS)[k_idx]


def test_all_finite():
    rng = np.random.default_rng(2)
    feats = extract_features(rng.standard_normal(4096), FS)
    assert np.all(np.isfinite(feats))
