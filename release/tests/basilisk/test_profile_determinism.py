"""tests/basilisk/test_profile_determinism.py

§9/§18 —— 工况库确定性。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.basilisk_config import load_basilisk_config      # noqa: E402
from src.sim.basilisk_profile import (                          # noqa: E402
    MODES, profile_rng, generate_run,
)

CFG = load_basilisk_config("configs/wheel_basilisk.yaml")
PROFILE_H5 = ROOT / CFG["paths"]["profile_h5"]
PROV_JSON = ROOT / CFG["paths"]["provenance"]

bsk = pytest.importorskip("Basilisk", reason="Basilisk 未安装")


def test_profile_rng_deterministic_and_distinct():
    """逐 (mode, run) RNG 必须可复现, 且不同 (mode, run) 必须不同流。"""
    a = profile_rng(20260808, "cruise", 0).random(8)
    b = profile_rng(20260808, "cruise", 0).random(8)
    assert np.array_equal(a, b), "同 (seed, mode, run) 两次抽样不一致"
    c = profile_rng(20260808, "cruise", 1).random(8)
    d = profile_rng(20260808, "nadir", 0).random(8)
    e = profile_rng(20260809, "cruise", 0).random(8)
    for name, other in [("run", c), ("mode", d), ("seed", e)]:
        assert not np.allclose(a, other), f"不同 {name} 却得到同一随机流"


def test_basilisk_profile_deterministic():
    """§9: 同 seed 重跑一条 run, 数值内容逐位一致。

    只重跑一条短 run (非全库) 以控制测试时长; 全库两次一致性由
    docs/basilisk_v1/profile_report.md 记录的实际两次运行 hash 证明。
    """
    cfg = json.loads(json.dumps(CFG))          # 深拷贝, 不污染模块级 CFG
    cfg["sim"]["profile"]["run_duration_s"] = 600.0
    d1 = generate_run("imaging", 0, cfg, 20260808)
    d2 = generate_run("imaging", 0, cfg, 20260808)
    for k in ("time_s", "wheel_speed_rad_s", "motor_torque_Nm",
              "sigma_BN", "omega_BN_B"):
        assert np.array_equal(d1[k], d2[k]), f"字段 {k} 两次生成不一致"


def test_profile_hash_matches_provenance():
    """已落盘的 profiles.h5 数值内容 hash 必须与 provenance 记录一致。"""
    if not PROFILE_H5.exists() or not PROV_JSON.exists():
        pytest.skip("profiles.h5 / provenance.json 未生成")
    sys.path.insert(0, str(ROOT / "scripts" / "basilisk"))
    from generate_profiles import HASH_FIELDS
    import h5py
    runs = {}
    with h5py.File(PROFILE_H5, "r") as f:
        for mode in MODES:
            for rk in sorted(f[mode].keys()):
                key = f"{mode}/{rk}"
                runs[key] = {fld: np.asarray(f[f"{key}/{fld}"][:], dtype=np.float64)
                             for fld in HASH_FIELDS}
    h = hashlib.sha256()
    for key in sorted(runs.keys()):
        h.update(key.encode())
        for fld in HASH_FIELDS:
            h.update(fld.encode())
            a = np.ascontiguousarray(runs[key][fld], dtype=np.float64)
            h.update(str(a.shape).encode())
            h.update(a.tobytes())
    prov = json.loads(PROV_JSON.read_text(encoding="utf-8"))
    assert h.hexdigest() == prov["profiles_sha256"], \
        "profiles.h5 内容 hash 与 provenance 不符 (文件被改动过)"
    with h5py.File(PROFILE_H5, "r") as f:
        assert f.attrs["content_sha256"] == prov["profiles_sha256"]
