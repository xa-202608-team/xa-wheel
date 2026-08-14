#!/usr/bin/env python
"""scripts/basilisk_b3x/data_b3x.py

BASILISK-B3X §3/§4/§6 —— 两臂共用的目标域 loader。

`POST_B2_FAIL_EXPLORATORY` / `NOT_FORMAL_EVIDENCE`

与 B2 的唯一差别是**输入列**: B 臂在 B1.9 冻结的 10 列之后追加 6 列
mission features。删失契约 / HI / rul_scale / z-score 口径 / 划分
全部 import 复用 data_b2, 不复制。

§6 要求两臂在同一 seed 下除输入列外完全相同。两处必须显式处理, 否则
"只改了输入"这句话是假的:

  1. **batch order**。data_b2.prepare_b2 建的是 `DataLoader(shuffle=True)`,
     没有传 generator, 于是取样顺序来自**全局 torch RNG**, 而全局 RNG 已经
     被模型初始化消耗过。两臂输入维度不同 (10 vs 16) -> adapter 参数量不同
     -> 初始化抽走的随机数个数不同 -> **batch 顺序会静默错开**。
     因此本模块给 train loader 传显式 `torch.Generator`, 由 seed 独立播种,
     使两臂的 batch 顺序逐 batch 相同。

  2. **初始化 RNG**。两臂在 build 之前用同一个 seed 把全局 RNG 置到同一状态,
     消耗顺序也相同。但 realized 权重**不可能逐值相同**: Adapter 先建且
     参数量随输入维度变化, 之后 encoder 拿到的随机数就整体平移了。
     这是"改变输入维度"这件事的固有后果, 任何输入消融都躲不开。
     诚实口径是: **RNG 种子与消耗顺序相同, 而非权重逐值相同**。
     不写成"初始化完全一致"。

§3 派生: `zero_crossing_rate` 在 h5 里**不存在**, 只有 `zero_crossing_count`。
    rate = count / n_per_window, n_per_window = round(sample_period_s / dt_s)。
    该常数从 config 读并与 sim 配置交叉核验, 不硬编码在代码里。
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.basilisk_b2.data_b2 import (  # noqa: E402
    FORBIDDEN_PLAIN_INPUT_COLS,
    assert_no_hi_in_input,
    load_b2_target,
)
from src.baselines.physical_extrap import caliber_eps  # noqa: E402
from src.transfer.train_transfer import (  # noqa: E402
    CensoredTargetSeqDataset,
    TargetSeqDataset,
)
from src.utils import set_seed  # noqa: E402

# h5 的 mission_features 组里真实存在的列。`zero_crossing_rate` 不在其中,
# 必须派生 —— 直接按 rate 去取会 KeyError。
MISSION_GROUP = "mission_features"
DERIVED_RATE_COL = "zero_crossing_rate"
DERIVED_RATE_FROM = "zero_crossing_count"

# §3 明令禁止进入输入的 mission 类列。抽成模块级常量的理由与前几阶段同:
# 反作弊扫描器扫源码符号, 常量声明块被豁免, 函数体内字符串不会。
# `mode_id` 在此列: 它是 mission profile 的调度标签本身, 不是遥测派生量。
FORBIDDEN_MISSION_COLS = (
    "mode_id",
    "future_mode_schedule",
    "lifetime_mode_fraction",
    "future_mission_information",
    "eol_idx",
    "D_true",
    "b_true",
)


def resolve_n_per_window(cfg: dict) -> int:
    """n_per_window = round(sim.sample_period_s / sim.profile.dt_s)。

    config 里声明的 b3x.n_per_window 必须与 sim 配置算出的值一致,
    不一致直接 abort —— 否则 zero_crossing_rate 的量纲是错的。
    """
    sim = cfg["sim"]
    derived = int(round(float(sim["sample_period_s"])
                        / float(sim["profile"]["dt_s"])))
    declared = int(cfg["b3x"]["n_per_window"])
    if derived != declared:
        raise SystemExit(
            f"!! B3X_CONFIG_INCOMPATIBLE: b3x.n_per_window={declared} != "
            f"round(sim.sample_period_s/sim.profile.dt_s)={derived}")
    if derived <= 0:
        raise SystemExit(f"!! n_per_window 必须为正, 得到 {derived}")
    return derived


def arm_mission_cols(cfg: dict, arm: str) -> list[str]:
    """取某一臂的 mission 列并做 §3 禁止项检查。"""
    arms = cfg["b3x"]["arms"]
    if arm not in arms:
        raise ValueError(f"未知臂 {arm!r}, 只允许 {sorted(arms)}")
    cols = list(arms[arm].get("mission_features") or [])
    bad = [c for c in cols
           if c in FORBIDDEN_MISSION_COLS or c in FORBIDDEN_PLAIN_INPUT_COLS]
    if bad:
        raise AssertionError(f"臂 {arm} 含禁止输入列 {bad} —— 违反 §3")
    dup = sorted({c for c in cols if cols.count(c) > 1})
    if dup:
        raise AssertionError(f"臂 {arm} 有重复列 {dup}")
    return cols


def mission_matrix(h5_path: Path, mission_cols: list[str],
                   n_per_window: int, tid_names: list[str] | None = None,
                   n_rows_limit: dict[str, int] | None = None) -> np.ndarray:
    """按 tid 顺序拼出 mission 特征矩阵 (逐点, 与 x_T 同行数同顺序)。

    n_rows_limit: {tid_name: n} —— 只取该轨迹前 n 行。**仅供 prefix-safety
    测试使用**: 截断后重算的前缀必须与全长的前缀逐值相等。
    """
    if not mission_cols:
        with h5py.File(h5_path, "r") as f:
            names = tid_names or sorted(f.keys())
            n = sum(int(f[t]["x_T"].shape[0]) for t in names)
        return np.zeros((n, 0), dtype=np.float32)

    blocks: list[np.ndarray] = []
    with h5py.File(h5_path, "r") as f:
        names = tid_names or sorted(f.keys())
        for t in names:
            g = f[t]
            n_full = int(g["x_T"].shape[0])
            n = min(int((n_rows_limit or {}).get(t, n_full)), n_full)
            mg = g[MISSION_GROUP]
            cols = []
            for c in mission_cols:
                if c == DERIVED_RATE_COL:
                    # §3: rate 必须派生。除以常数是**逐点**运算, 不引入
                    # 任何跨行依赖, 因此 prefix-safe 天然成立。
                    raw = np.asarray(mg[DERIVED_RATE_FROM][:n], np.float64)
                    cols.append((raw / float(n_per_window)).astype(np.float32))
                    continue
                if c not in mg:
                    raise KeyError(
                        f"mission 列 {c!r} 不在 h5 的 {MISSION_GROUP} 组中; "
                        f"可用: {sorted(mg.keys())}")
                a = np.asarray(mg[c][:n], np.float32)
                if a.shape[0] != n:
                    raise ValueError(
                        f"{t}/{c} 行数 {a.shape[0]} != x_T 行数 {n}")
                cols.append(a)
            blocks.append(np.stack(cols, axis=1))
    return np.concatenate(blocks, axis=0)


def b3x_input_cols(cfg: dict, arm: str) -> list[str]:
    """该臂的完整输入列名 = B1.9 冻结 10 列 + mission 列。"""
    h5 = ROOT / cfg["transfer"]["target_feature_path"]
    return list(assert_no_hi_in_input(h5)) + arm_mission_cols(cfg, arm)


def prepare_b3x(cfg: dict, split: dict, seed: int, arm: str,
                with_test: bool = False, verbose: bool = True) -> dict:
    """按 B3X 契约装载数据。除输入列外, 与 prepare_b2 逐项同口径。

    split 逐字复用 B2 的 checkpoints/basilisk_b2/split.json;
    加载前校验 split_sha256, 不符即 abort (§4 禁止重新 stratify)。
    """
    b3 = cfg["b3x"]
    tc, mc = cfg["transfer"], cfg["model"]
    L = int(mc["input_len_L"])
    K = int(cfg.get("pretrain", {}).get("seq_block_K", 8))
    stride = int(tc.get("target_stride", 50))
    hi_key = str(tc["target_hi_key"])
    h5 = ROOT / tc["target_feature_path"]

    want = str(b3["split"]["expected_sha256"])
    got = str(split["split_sha256"])
    if got != want:
        raise SystemExit(
            f"!! B3X_SPLIT_MISMATCH: split_sha256={got[:16]}… != "
            f"契约钉死的 B2 划分 {want[:16]}… —— §4 禁止重新划分")

    n_per_window = resolve_n_per_window(cfg)
    mission_cols = arm_mission_cols(cfg, arm)

    set_seed(seed, cfg["reproducibility"]["deterministic"],
             cfg["reproducibility"]["cudnn_benchmark"])

    core_cols = assert_no_hi_in_input(h5)
    xT, hiT, rulT, lbT, evT, tidT, n_traj, tids = load_b2_target(h5, hi_key)
    xM = mission_matrix(h5, mission_cols, n_per_window, tid_names=tids)
    if xM.shape[0] != xT.shape[0]:
        raise ValueError(f"mission 行数 {xM.shape[0]} != x_T 行数 {xT.shape[0]}")
    xT = np.concatenate([xT, xM], axis=1) if xM.shape[1] else xT
    cols = list(core_cols) + list(mission_cols)

    name2i = {t: i for i, t in enumerate(tids)}
    tr = np.array([name2i[t] for t in split["splits"]["train"]["tids"]])
    va = np.array([name2i[t] for t in split["splits"]["val"]["tids"]])
    te = np.array([name2i[t] for t in split["splits"]["test"]["tids"]])

    m_tr = np.isin(tidT, tr)
    # rul_scale / z-score 口径与 B2 逐字相同: 只用 train 行, censored 的
    # rul 是 NaN 由 nanmax 跳过。新增的 mission 列同样只用 train 统计量,
    # 不引入 val/test 泄漏。
    rul_scale = max(float(np.nanmax(rulT[m_tr])), 1.0)
    rulT = rulT / rul_scale
    lbT = lbT / rul_scale
    mu, sd = xT[m_tr].mean(axis=0), xT[m_tr].std(axis=0)
    xT = (xT - mu) / (sd + 1e-6)

    bs = int(cfg["pretrain"]["batch_size"])
    eta = float(cfg["b2"]["censor_hinge_eta"])

    def _ds_train():
        m = np.isin(tidT, tr)
        r = rulT[m].copy()
        cen = ~evT[m]
        r[cen] = lbT[m][cen]
        return CensoredTargetSeqDataset(
            xT[m], hiT[m], r, tidT[m], L, K, stride=stride,
            event_observed=evT[m], rul_lower_bound=lbT[m])

    def _ds_eval(ids):
        m = np.isin(tidT, ids)
        return TargetSeqDataset(xT[m], hiT[m], rulT[m], tidT[m], L, 1,
                                stride=stride)

    # §6 batch order: 显式 generator, 与模型初始化的全局 RNG 解耦。
    # 两臂拿到同一个 batch 序列, 尽管两者的 adapter 参数量不同。
    gtr = torch.Generator()
    gtr.manual_seed(int(seed) * 1000 + 7)

    out = {
        "ltr": DataLoader(_ds_train(), batch_size=bs, shuffle=True,
                          generator=gtr),
        "lva": DataLoader(_ds_eval(va), batch_size=bs, shuffle=False),
        "tr": tr, "va": va, "te": te,
        "n_traj": n_traj, "rul_scale": rul_scale,
        "n_target": xT.shape[1], "cap_eps": caliber_eps(cfg),
        "censor_eta": eta,
        "xt_cols": cols,
        "core_cols": list(core_cols),
        "mission_cols": list(mission_cols),
        "n_per_window": n_per_window,
        "arm": arm,
        "hi_key": hi_key,
        "tids": tids,
        "ev_by_traj": {tids[i]: bool(evT[tidT == i][0]) for i in range(n_traj)},
        "device": ("cuda" if (cfg["pretrain"]["device"] == "cuda"
                              and torch.cuda.is_available()) else "cpu"),
        "batch_order_generator_seed": int(seed) * 1000 + 7,
    }
    if with_test:
        out["lte"] = DataLoader(_ds_eval(te), batch_size=bs, shuffle=False)

    if verbose:
        n_ev_tr = int(sum(out["ev_by_traj"][tids[i]] for i in tr))
        print(f">> B3X[{arm}] 数据: train {len(tr)} (event {n_ev_tr}) / "
              f"val {len(va)} / test {len(te)}   with_test={with_test}")
        print(f"   n_target={out['n_target']} "
              f"(core {len(core_cols)} + mission {len(mission_cols)})  "
              f"rul_scale={rul_scale:.1f}")
        if mission_cols:
            print(f"   mission: {mission_cols}  n_per_window={n_per_window}")
    return out
