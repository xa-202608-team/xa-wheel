# 临时脚本: 采集 S3 协议所需 hash 与截断审计事实 (只读)
import hashlib
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import load_config
from src.sim.build_hi import XT_COLS
from src.transfer.train_transfer import split_trajectories_from_cfg

cfg = load_config(str(ROOT / "configs" / "wheel.yaml"))
tc = cfg["transfer"]
h5p = ROOT / tc["target_feature_path"]
GATE_SEEDS = [62, 63, 64]
TRUNC = 0.55
N_SEL = 5


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


print("== file hashes ==")
for rel in ["configs/wheel.yaml", "src/transfer/train_transfer.py",
            "src/experiments/run_groups.py", "src/baselines/trivial.py"]:
    print(f"{rel:44s} {sha_file(ROOT / rel)[:16]}")

print(f"target_h5 {sha_file(h5p)}")
print(f"target_h5 bytes {h5p.stat().st_size}")

print("== xt schema ==")
import json
print(list(XT_COLS))
print(hashlib.sha256(json.dumps(list(XT_COLS)).encode()).hexdigest())

print("== splits for gate seeds ==")
with h5py.File(h5p, "r") as f:
    keys = sorted(f.keys())
    obs = [k for k in keys
           if bool(f[k].attrs.get("event_observed",
                                  np.asarray(f[k].get("label_fail", [])).any()))]
n_obs = len(obs)
print(f"n_observed={n_obs}")

lines = []
for s in GATE_SEEDS:
    tr, va, te, _ = split_trajectories_from_cfg(h5p, n_obs, tc, s, observed_only=True)
    tr, va, te = list(map(int, tr)), list(map(int, va)), list(map(int, te))
    print(f"seed {s}: tr={tr} va={va} te={te}")
    lines.append(f"{s}|tr={tr}|va={va}|te={te}")
print("split_hash", hashlib.sha256("\n".join(lines).encode()).hexdigest())
