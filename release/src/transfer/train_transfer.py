"""transfer/train_transfer.py

双迁移路径:
  - run()             : 旧观测层迁移 (adapter + 共享 encoder, 飞轮 + ablation)
  - run_hi_layer(args): 新 HI 动力学层迁移 (路径 A, docs/开发推进计划/hi_layer_refactor_design.md §4.2)

旧路径三阶段 (plan §四 Phase 5):
  S1 复用源域预训练编码器 (冻结)
  S2 目标域 x_T 经 adapter g_φ → 冻结 E_θ → 训 adapter + 头 (防小样本冲掉源域知识)
  S3 MMD (按 HI 健康阶段分箱, 非按时间) + 小样本微调

新 HI 层路径三阶段 (设计文档 §4.2):
  S1 复用源域 HISeqEncoder 预训练权重 (encoder 输入 = [HI, ΔHI])
  S2 冻结 encoder, 训 target HI_head + RUL_head (输入 = 目标域 hi_array 派生的 x_HI 窗)
  S3 解冻 + MMD 按 HI bin 对齐 z_S / z_T (两者都由共享 HISeqEncoder 编码 x_HI 得到)

目标域按完整轨迹划分 train15 / val20 / test65 (禁止时间窗打散)。

用法:
  python -m src.transfer.train_transfer --config configs/wheel.yaml --smoke
  python -m src.transfer.train_transfer --config configs/phased_array.yaml --hi-layer
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from itertools import cycle
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils import load_config, set_seed                          # noqa: E402
from src.transfer.adapter import TransferModel                       # noqa: E402
from src.transfer.mmd import mmd_by_hi_bins, reset_global_memory_bank # noqa: E402
from src.data.preprocess.source_io import load_source_features       # noqa: E402

# HI 层路径依赖 (Agent 1/2 接口, 集成时验证)
from src.transfer.adapter import HIDynamicsModel                     # noqa: E402  (Agent 1, §3.1)
from src.data.preprocess.source_io import make_hi_windows            # noqa: E402  (Agent 2, §3.2)

CKPT_DIR = ROOT / "checkpoints"


# ---------------------------------------------------------------- 数据
class TargetSeqDataset(Dataset):
    """目标域: 同一 traj 连续 K 窗块, 带 hi_b / rul (支撑 mono/smooth)。

    额外暴露 `sample_tids` (与 self.samples 同序的轨迹 id 数组): 供评估侧做**宏平均**
    (先算每条轨迹的指标再对轨迹平均, S2' 改动 3)。只读附加属性, 不改 __getitem__ 返回。
    """

    def __init__(self, x_T, hi_b, rul, traj_ids, L, K, stride=1):
        self.samples = []
        self.sample_tids = []
        df = pd.DataFrame({"tid": traj_ids})
        for tid, g in df.groupby("tid"):
            idxs = g.index.to_numpy()
            T = len(idxs)
            if T < L:
                continue
            starts = list(range(0, T - L + 1, stride))
            wf = [x_T[idxs[s:s + L]] for s in starts]
            wh = [float(hi_b[idxs[s + L - 1]]) for s in starts]
            wr = [float(rul[idxs[s + L - 1]]) for s in starts]
            n = len(starts)
            for s2 in range(0, n - K + 1):
                self.samples.append((
                    np.stack(wf[s2:s2 + K]).astype(np.float32),
                    np.array(wh[s2:s2 + K], dtype=np.float32),
                    np.array(wr[s2:s2 + K], dtype=np.float32),
                ))
                self.sample_tids.append(int(tid))
        self.sample_tids = np.asarray(self.sample_tids, dtype=np.int64)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        f, h, r = self.samples[i]
        return torch.from_numpy(f), torch.from_numpy(h), torch.from_numpy(r)


class SourceWindowDataset(Dataset):
    """源域: 单窗 (L, 12) 带 hi (供 S3 MMD 的 z_S)。"""

    def __init__(self, features, hi, bearing_ids, t_idx, L, stride=1):
        self.samples = []
        df = pd.DataFrame({"bid": bearing_ids})
        for _, g in df.groupby("bid"):
            g = g.assign(t=np.asarray(t_idx)[g.index]).sort_values("t")
            idxs = g.index.to_numpy()
            T = len(idxs)
            if T < L:
                continue
            for s in range(0, T - L + 1, stride):
                self.samples.append((features[idxs[s:s + L]].astype(np.float32),
                                     float(hi[idxs[s + L - 1]])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        f, h = self.samples[i]
        return torch.from_numpy(f), torch.tensor(h, dtype=torch.float32)


def _stratify_quota(strata_sizes, ratios, targets):
    """把每层样本按比例分配到三个划分, 同时满足层内合计与全局配额 (最大余数法)。

    这是一个小型运输问题: 单元 (层 s, 划分 j) 的理想份额 = size_s·ratio_j, 先取下整,
    余量按分数部分降序贪心补 1, 且只补"该层还有余额 且 该划分还缺额"的单元。
    两边余量总和相等, 故只要两边都有余量就必存在可补单元 -> 必然精确收敛。
    返回 quota[s][j] (整数矩阵)。
    """
    S, J = len(strata_sizes), len(ratios)
    ideal = [[strata_sizes[s] * ratios[j] for j in range(J)] for s in range(S)]
    quota = [[int(np.floor(ideal[s][j])) for j in range(J)] for s in range(S)]
    room_s = [strata_sizes[s] - sum(quota[s]) for s in range(S)]          # 层内余额
    need_j = [targets[j] - sum(quota[s][j] for s in range(S)) for j in range(J)]  # 划分缺额
    cells = sorted(((ideal[s][j] - quota[s][j], s, j) for s in range(S) for j in range(J)),
                   key=lambda c: (-c[0], c[1], c[2]))
    for _, s, j in cells:
        if room_s[s] > 0 and need_j[j] > 0:
            quota[s][j] += 1
            room_s[s] -= 1
            need_j[j] -= 1
    while any(n > 0 for n in need_j) and any(r > 0 for r in room_s):
        j = next(j for j in range(J) if need_j[j] > 0)
        s = next(s for s in range(S) if room_s[s] > 0)
        quota[s][j] += 1
        room_s[s] -= 1
        need_j[j] -= 1
    return quota


def split_trajectories(n_traj, ratios, seed, eol=None, n_strata=1):
    """确定性轨迹级划分 (完整轨迹, 不打散窗口)。返回 (train, val, test) 轨迹 id。

    eol=None 或 n_strata<=1: 纯随机划分 (S1.5 及之前的行为)。

    **按 EOL 分层 (S2' 改动 2)**: 传入 eol (逐轨迹 EOL index, 与 tid 同序) 且
    n_strata>1 时, 先按 EOL 分位数把轨迹等频分成 n_strata 层, 层内按 seed 打乱,
    再按比例分配到 train/val/test, 使三者 EOL 分布一致。

    动机 (S1.5 实测): EOL 跨度 2121~52595 而 RUL 截顶在 18408, 每条轨迹的零难度
    样本占 65%~95.6% 且占比由 EOL 决定。随机划分下 val 与 test 的难度不可比 ——
    实测方法排序反相关 (B: val 0.0928/test 0.1373 最好, C: val 0.0853/test 0.1525),
    test/val 相差 1.5~1.8 倍, 早停信号因此失效。
    """
    if n_traj < 3 or len(ratios) != 3 or not np.isclose(sum(ratios), 1.0):
        raise ValueError("轨迹数或划分比例无效")
    rng = np.random.default_rng(seed)
    n_tr = int(round(ratios[0] * n_traj))
    n_va = int(round(ratios[1] * n_traj))
    n_te = n_traj - n_tr - n_va
    if min(n_tr, n_va, n_te) < 1:
        raise ValueError("轨迹划分产生空集合")

    if eol is None or int(n_strata) <= 1:
        perm = rng.permutation(n_traj)
        return perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]

    eol = np.asarray(eol, dtype=float)
    if len(eol) != n_traj:
        raise ValueError(f"eol 长度 {len(eol)} 与 n_traj {n_traj} 不一致")
    n_strata = int(min(int(n_strata), n_traj))
    # 按 EOL 排序后等频切成 n_strata 段 (分位数分层; 并列由 stable 排序确定性打破)
    order = np.argsort(eol, kind="stable")
    strata = [np.asarray(s) for s in np.array_split(order, n_strata)]
    strata = [s for s in strata if len(s)]
    quota = _stratify_quota([len(s) for s in strata], list(ratios), [n_tr, n_va, n_te])

    out = [[], [], []]
    for s_idx, members in enumerate(strata):
        shuffled = rng.permutation(members)          # 层内按 seed 打乱
        pos = 0
        for j in range(3):
            k = quota[s_idx][j]
            out[j].extend(int(t) for t in shuffled[pos:pos + k])
            pos += k
    tr, va, te = (np.asarray(sorted(o), dtype=int) for o in out)
    if min(len(tr), len(va), len(te)) < 1:
        raise ValueError("分层划分产生空集合; 减小 n_strata 或增加轨迹数")
    return tr, va, te


def target_eol_by_tid(h5_path, observed_only=True) -> np.ndarray:
    """逐轨迹 EOL index, 顺序与 load_target 返回的 tid 一致 (供分层划分)。

    必须与 load_target 的遍历/跳过/remap 逻辑严格同序: 都按 sorted(keys) 遍历,
    observed_only=True 时跳过未观测轨迹, 保留顺序即新 tid 0..n-1。
    """
    eols = []
    with h5py.File(h5_path, "r") as f:
        for k in sorted(f.keys()):
            g = f[k]
            observed = bool(g.attrs.get("event_observed",
                                        np.asarray(g.get("label_fail", [])).any()))
            if observed_only and not observed:
                continue
            eols.append(int(g.attrs.get("eol_idx", len(g["rul"]) - 1)))
    return np.asarray(eols, dtype=np.int64)


def split_trajectories_from_cfg(h5_path, n_traj, tcfg, seed, observed_only=True):
    """按 config transfer.split 决定是否分层, 返回 (tr, va, te, eol_by_tid_or_None)。

    tcfg = cfg["transfer"]。stratify_by_eol=false 时回退纯随机 (eol 返回 None)。
    """
    sc = tcfg["split"]
    ratios = [sc["train"], sc["val"], sc["test"]]
    if not bool(sc.get("stratify_by_eol", False)):
        return (*split_trajectories(n_traj, ratios, seed), None)
    eol = target_eol_by_tid(h5_path, observed_only=observed_only)
    n_strata = int(sc.get("n_eol_strata", 4))
    return (*split_trajectories(n_traj, ratios, seed, eol=eol, n_strata=n_strata), eol)


def describe_eol_split(eol, tr, va, te, quantiles=(5, 25, 50, 75, 95)):
    """打印三个划分的 EOL 分位数 (核对分层是否让难度可比)。eol=None 时打印提示。"""
    print("  -- 三划分 EOL 分位数 --")
    if eol is None:
        print("     (未启用 EOL 分层: transfer.split.stratify_by_eol=false)")
        return
    eol = np.asarray(eol, dtype=float)
    head = "  ".join(f"p{q}" for q in quantiles)
    print(f"     {'划分':<6} {'n':>3}  {head}   mean")
    for name, ids in (("train", tr), ("val", va), ("test", te)):
        v = eol[np.asarray(ids, dtype=int)]
        qs = "  ".join(f"{np.percentile(v, q):>7.0f}" for q in quantiles)
        print(f"     {name:<6} {len(v):>3}  {qs}  {v.mean():>8.0f}")



def load_target(h5_path, has_nodes=False, observed_only=True,
                truncate_hi=None, truncate_tids=None, split_role="full",
                cap_ratio=None, censor_out=None):
    """目标域加载。has_nodes=True (相控阵): x_global + x_nodes mean-pool → concat;
    否则 (飞轮) 读 x_T。HI 列: hi_array(相控阵) / hi_b(飞轮)。

    ---- S3 train-only 截断契约 (协议 docs/diagnostics_s3_protocol.md §6) ----
    truncate_hi   : float | None。None = 不截断 (S2.5 及之前行为, 逐行数值等价)。
                    非 None 时, 对 truncate_tids 里的每条轨迹只保留**首次 HI > truncate_hi
                    之前**的可观测段 (切片 [0, c), c = 第一个 HI>thr 的下标) ——
                    保留段 max(HI) <= truncate_hi 严格成立, 其后的真实 EOL 完全不可见。
    truncate_tids : 需要截断的轨迹 id 集合 (remap 后的 0..n-1 空间)。必须显式给出。
    split_role    : "train" | "val" | "test" | "full"。**只有 "train" 允许截断**;
                    对 val/test 传入 truncate_hi 直接抛错 (协议 §6 绝对禁止)。
    cap_ratio     : RUL 截顶比例 (config source.rul_cap_ratio)。截断轨迹的
                    rul_lower_bound 需要它, 与 src/sim/build_hi.py 未观测分支同式。
    censor_out    : 可选 dict 出参。给出时填入
                      "event_observed"   (n_rows,) bool  截断轨迹全 False
                      "rul_lower_bound"  (n_rows,) float 归一前 (原始采样点单位)
                      "rul_observed"     (n_rows,) float 截断轨迹置 NaN (真值不可见)
                      "audit"            逐轨迹审计行 (list[dict], 供 §9 打印)
                    返回值仍是 5 元组 —— 所有既有调用方 (diag_*/run_groups/基线) 不受影响。

    rul_lower_bound 的定义 (协议 §7, **绝不使用未来 EOL 位置**):
        lb(t) = min( t_obs_end - t , cap_ratio * n_full )
      t_obs_end = 保留段最后一个下标 = c - 1, 即"已观测窗口右端"。
      语义: 到 t_obs_end 时刻器件仍未失效 ⇒ 真实 RUL(t) >= t_obs_end - t。
      与 src/sim/build_hi.py 未观测轨迹分支 min(n-1-t, cap_ratio*n) 数学同式,
      仅把"整条轨迹右端 n-1"换成"截断后的观测右端 c-1"。
    """
    do_trunc = truncate_hi is not None
    if do_trunc:
        if str(split_role) != "train":
            raise ValueError(f"截断只允许作用于 train (协议 §6); 收到 split_role={split_role!r}")
        if truncate_tids is None:
            raise ValueError("truncate_hi 非 None 时必须显式给出 truncate_tids")
        if cap_ratio is None:
            raise ValueError("truncate_hi 非 None 时必须给出 cap_ratio (rul_lower_bound 需要)")
    trunc_set = set(int(t) for t in (truncate_tids or []))
    thr = float(truncate_hi) if do_trunc else None

    xT_list, hi_list, rul_list, tid_list = [], [], [], []
    ev_list, lb_list, ro_list, audit = [], [], [], []
    hi_key = "hi_array" if has_nodes else "hi_b"
    with h5py.File(h5_path, "r") as f:
        keys = sorted(f.keys())
        kept = 0
        for k in keys:
            g = f[k]
            observed = bool(g.attrs.get("event_observed",
                                        np.asarray(g.get("label_fail", [])).any()))
            if observed_only and not observed:
                continue
            tid = kept                       # remap 后的 tid (与旧 remap 结果逐条相同)
            kept += 1
            if has_nodes:
                xg = g["x_global"][:].astype(np.float32)
                xn = g["x_nodes"][:].mean(axis=1).astype(np.float32)  # mean-pool 16 子阵节点
                xT = np.concatenate([xg, xn], axis=1)
            else:
                xT = g["x_T"][:]
            hi = g[hi_key][:]
            rul = g["rul"][:]
            lf = np.asarray(g["label_fail"][:]) if "label_fail" in g \
                else np.zeros(len(hi), dtype=np.int64)
            n_full = len(hi)
            ev = np.ones(n_full, dtype=bool)
            lb = np.asarray(rul, dtype=float).copy()      # 未截断: 下界即真值
            ro = np.asarray(rul, dtype=float).copy()

            if do_trunc and tid in trunc_set:
                m = np.asarray(hi, dtype=float) > thr
                if not m.any():
                    raise ValueError(f"轨迹 {k} (tid={tid}) 的 HI 从未超过 {thr}, 无法截断")
                c = int(np.argmax(m))                     # 首次 HI>thr 的下标
                if c < 1:
                    raise ValueError(f"轨迹 {k} (tid={tid}) 首点即超阈 {thr}, 截断后为空")
                t_obs_end = c - 1
                xT, hi = xT[:c], hi[:c]
                lf = lf[:c]
                if int(np.asarray(lf).sum()) != 0:
                    raise ValueError(f"轨迹 {k} 截断段仍含 label_fail=1, 截断无效")
                ev = np.zeros(c, dtype=bool)              # 全部右删失
                lb = np.minimum((t_obs_end - np.arange(c)).astype(float),
                                float(cap_ratio) * float(n_full))
                ro = np.full(c, np.nan, dtype=float)      # 真实 RUL 不可见
                rul = lb                                  # 训练侧只允许看到下界
                audit.append({
                    "tid": int(tid), "key": str(k), "original_n": int(n_full),
                    "observed_n": int(c),
                    "retained_fraction": float(c) / float(n_full),
                    "max_hi": float(np.max(hi)), "truncate_hi": thr,
                    "event_observed": 0, "failure_label_count": 0,
                    "t_obs_end": int(t_obs_end),
                    "rul_lower_bound_max": float(lb.max()),
                    "rul_lower_bound_min": float(lb.min()),
                    "truncated": True,
                })
            else:
                audit.append({
                    "tid": int(tid), "key": str(k), "original_n": int(n_full),
                    "observed_n": int(n_full), "retained_fraction": 1.0,
                    "max_hi": float(np.max(hi)), "truncate_hi": thr,
                    "event_observed": int(bool(observed)),
                    "failure_label_count": int(np.asarray(lf).sum()),
                    "t_obs_end": int(n_full - 1),
                    "rul_lower_bound_max": float(np.nanmax(lb)) if len(lb) else float("nan"),
                    "rul_lower_bound_min": float(np.nanmin(lb)) if len(lb) else float("nan"),
                    "truncated": False,
                })

            xT_list.append(xT)
            hi_list.append(hi)
            rul_list.append(rul)
            tid_list.append(np.full(len(xT), tid))
            ev_list.append(ev)
            lb_list.append(lb)
            ro_list.append(ro)
    if not xT_list:
        raise ValueError("目标域没有已观测失效轨迹")
    tids = np.concatenate(tid_list)
    remap = {old: new for new, old in enumerate(sorted(set(tids)))}
    tids = np.asarray([remap[int(i)] for i in tids], dtype=np.int64)
    if censor_out is not None:
        censor_out["event_observed"] = np.concatenate(ev_list)
        censor_out["rul_lower_bound"] = np.concatenate(lb_list)
        censor_out["rul_observed"] = np.concatenate(ro_list)
        censor_out["audit"] = audit
    return np.concatenate(xT_list), np.concatenate(hi_list), np.concatenate(rul_list), tids, len(remap)


def select_train_trajectories(seed, train_ids, n_select, rule="sha256_seed_tid"):
    """S3 确定性选取 n_select 条 target train 轨迹 (协议 §10)。

    **不使用任何 val/test 表现**: 稳定序 = sha256("{seed}|{tid}") 的十六进制升序,
    取前 n_select 条后按 tid 升序返回。同一 seed 下三个方法必然拿到同一组轨迹。

    返回 (selected_ids: list[int], selection_hash: str)。
    """
    if str(rule) != "sha256_seed_tid":
        raise ValueError(f"未知 s3_selection_rule={rule!r}")
    ids = [int(t) for t in train_ids]
    n = int(n_select)
    if n > len(ids):
        raise ValueError(f"要选 {n} 条但 train 划分只有 {len(ids)} 条")
    keyed = sorted(ids, key=lambda t: hashlib.sha256(f"{int(seed)}|{t}".encode()).hexdigest())
    sel = sorted(int(t) for t in keyed[:n])
    # pool 按 tid 升序序列化: 指纹严格只依赖 {rule, seed, id 集合, selected},
    # 与 split 函数返回的容器顺序无关 (否则同一集合换个顺序会给出不同 hash)。
    payload = f"rule={rule}|seed={int(seed)}|pool={sorted(ids)}|selected={sel}"
    return sel, hashlib.sha256(payload.encode()).hexdigest()


class CensoredTargetSeqDataset(TargetSeqDataset):
    """S3 目标域**训练**集: TargetSeqDataset + 逐窗 (event_observed, rul_lower_bound)。

    与父类唯一差别: __getitem__ 返回 5 元组 (x, hi, rul_label, ev, lb)。
    父类仍返回 3 元组, 因此 val/test 评估 loader、eval_test、collect_pred、
    scripts/diag_* 全部不受影响 (只有 S3 train loader 用本类)。

    rul_label 对删失窗是 rul_lower_bound (真实 RUL 不可见); ev=False 使其被排除在
    Huber 之外, 只经 hinge 参与损失 (协议 §8)。NaN 在此已被上游替换, 不会污染梯度。
    """

    def __init__(self, x_T, hi_b, rul, traj_ids, L, K, stride=1,
                 event_observed=None, rul_lower_bound=None):
        n = len(np.asarray(rul))
        ev_all = (np.ones(n, dtype=bool) if event_observed is None
                  else np.asarray(event_observed, dtype=bool))
        lb_all = (np.asarray(rul, dtype=np.float32) if rul_lower_bound is None
                  else np.asarray(rul_lower_bound, dtype=np.float32))
        # 用与父类完全相同的窗口/块枚举顺序, 把 (ev, lb) 对齐到每个窗末
        super().__init__(x_T, hi_b, rul, traj_ids, L, K, stride=stride)
        ev_s, lb_s = [], []
        df = pd.DataFrame({"tid": traj_ids})
        for tid, g in df.groupby("tid"):
            idxs = g.index.to_numpy()
            T = len(idxs)
            if T < L:
                continue
            starts = list(range(0, T - L + 1, stride))
            we = [bool(ev_all[idxs[s + L - 1]]) for s in starts]
            wl = [float(lb_all[idxs[s + L - 1]]) for s in starts]
            nw = len(starts)
            for s2 in range(0, nw - K + 1):
                ev_s.append(np.array(we[s2:s2 + K], dtype=np.float32))
                lb_s.append(np.array(wl[s2:s2 + K], dtype=np.float32))
        if len(ev_s) != len(self.samples):
            raise AssertionError(f"删失标注窗数 {len(ev_s)} != 样本窗数 {len(self.samples)}")
        self.ev = ev_s
        self.lb = lb_s

    def __getitem__(self, i):
        f, h, r = self.samples[i]
        return (torch.from_numpy(f), torch.from_numpy(h), torch.from_numpy(r),
                torch.from_numpy(self.ev[i]), torch.from_numpy(self.lb[i]))


# ============================================================
# S5: 速率监督 (改动 2)
# ============================================================

def hi_local_slope(hi, cfg_rate: dict, scale: float = 1.0) -> np.ndarray:
    """HI 序列 -> 逐点局部退化速率标签 mu_true (S5 改动 2 的 L_rate 标签)。

    这是 L_rate 的**监督信号**, 必须在长基线上稳健估计: 窗内 (L=64, 32 小时) 的
    ΔHI≈0 且被量测噪声主导, 短窗斜率信噪比 <1, 用它当标签等于在拟合噪声。
    窗长 rate.slope_window (>> L) 由 config 唯一给出。

    scale: 把"ΔHI / 样本" 换算到 "ΔHI / 归一 RUL 单位"。传 rul_scale 后 mu 与 RUL
    同尺度, 使 L_rate 与 L_huber 量级天然可比 (rate.scale_by_rul_scale)。

    estimator:
      "lsq"       长窗因果最小二乘斜率 (复用 build_hi.causal_lsq_slope, 同一份实现)
      "theil_sen" 逐窗中位数斜率 (对 HI 的阶跃/尖峰更稳健, 但 O(W^2) ->
                  按 theil_sen_subsample 在窗内等距子采样后再算所有配对斜率中位数)

    返回**非负**速率: HI 单调非减 (cummax), 真实速率不可能为负; 负值只可能来自
    数值噪声, clip 到 0 而不是保留 —— 保留会让 softplus 输出的 mu 永远无法匹配。
    """
    from src.sim.build_hi import causal_lsq_slope

    hi = np.asarray(hi, dtype=np.float64)
    W = int(cfg_rate["slope_window"])
    mp = int(cfg_rate.get("slope_min_periods", 8))
    est = str(cfg_rate.get("slope_estimator", "lsq"))
    if est == "lsq":
        s = causal_lsq_slope(hi, W, mp)
    elif est == "theil_sen":
        s = _theil_sen_causal(hi, W, mp, int(cfg_rate.get("theil_sen_subsample", 64)))
    else:
        raise ValueError(f"未知 rate.slope_estimator={est!r} (应为 lsq | theil_sen)")
    return np.clip(s * float(scale), 0.0, None)


def _theil_sen_causal(v, win: int, min_periods: int, n_sub: int) -> np.ndarray:
    """因果 Theil-Sen 斜率 (窗内等距子采样后取所有配对斜率中位数)。

    完整 Theil-Sen 在 W=2000 上是 2e6 配对 × 5e4 时刻, 不可接受。窗内等距取 n_sub
    个点再做全配对 (n_sub=64 -> 2016 配对), 保留"中位数抗离群"的核心性质。
    """
    v = np.asarray(v, dtype=np.float64)
    n = len(v)
    W = max(2, int(win))
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - W + 1)
        seg = v[lo:i + 1]
        if len(seg) < max(2, int(min_periods)):
            continue
        k = min(int(n_sub), len(seg))
        pos = np.linspace(0, len(seg) - 1, k).astype(np.int64)
        t = (lo + pos).astype(np.float64)
        y = seg[pos]
        ii, jj = np.triu_indices(k, 1)
        dt = t[jj] - t[ii]
        ok = dt > 0
        if not ok.any():
            continue
        out[i] = float(np.median((y[jj][ok] - y[ii][ok]) / dt[ok]))
    return out


class RateTargetSeqDataset(TargetSeqDataset):
    """S5 目标域数据集: TargetSeqDataset + 逐窗末 (mu_true, event_observed, lb)。

    与 CensoredTargetSeqDataset 的关系: 那个类给 (ev, lb) 支撑 S3 删失 Huber;
    本类**在其之上再加 mu_true**, 因为 S5 的主损失项 L_rate 需要窗末的局部速率标签。
    刻意不改那两个类 —— S3/S4 的 loader 行为必须逐位不变 (冻结结果不可动)。

    返回 6 元组 (x, hi, rul, ev, lb, mu_true)。窗末对齐规则与父类**逐字相同**
    (同一 groupby/starts/块枚举顺序), 由 __init__ 末尾的长度断言钉死。

    注: hi 字段仍是**观测 HI** (hi_b, 自校准量), 首达换算用的就是它 —— 不是模型
    预测的 hi_pred (见 TransferModel.forward_rate 的 docstring)。
    """

    def __init__(self, x_T, hi_b, rul, traj_ids, L, K, stride=1,
                 event_observed=None, rul_lower_bound=None, mu_true=None):
        n = len(np.asarray(rul))
        ev_all = (np.ones(n, dtype=bool) if event_observed is None
                  else np.asarray(event_observed, dtype=bool))
        lb_all = (np.asarray(rul, dtype=np.float32) if rul_lower_bound is None
                  else np.asarray(rul_lower_bound, dtype=np.float32))
        if mu_true is None:
            raise ValueError("RateTargetSeqDataset 必须给出 mu_true (L_rate 的标签)")
        mu_all = np.asarray(mu_true, dtype=np.float32)
        if len(mu_all) != n:
            raise ValueError(f"mu_true 长度 {len(mu_all)} != 样本行数 {n}")
        super().__init__(x_T, hi_b, rul, traj_ids, L, K, stride=stride)
        ev_s, lb_s, mu_s = [], [], []
        df = pd.DataFrame({"tid": traj_ids})
        for tid, g in df.groupby("tid"):
            idxs = g.index.to_numpy()
            T = len(idxs)
            if T < L:
                continue
            starts = list(range(0, T - L + 1, stride))
            we = [bool(ev_all[idxs[s + L - 1]]) for s in starts]
            wl = [float(lb_all[idxs[s + L - 1]]) for s in starts]
            wm = [float(mu_all[idxs[s + L - 1]]) for s in starts]
            nw = len(starts)
            for s2 in range(0, nw - K + 1):
                ev_s.append(np.array(we[s2:s2 + K], dtype=np.float32))
                lb_s.append(np.array(wl[s2:s2 + K], dtype=np.float32))
                mu_s.append(np.array(wm[s2:s2 + K], dtype=np.float32))
        if not (len(ev_s) == len(lb_s) == len(mu_s) == len(self.samples)):
            raise AssertionError(
                f"S5 标注窗数 ({len(ev_s)}, {len(mu_s)}) != 样本窗数 {len(self.samples)}")
        self.ev, self.lb, self.mu = ev_s, lb_s, mu_s

    def __getitem__(self, i):
        f, h, r = self.samples[i]
        return (torch.from_numpy(f), torch.from_numpy(h), torch.from_numpy(r),
                torch.from_numpy(self.ev[i]), torch.from_numpy(self.lb[i]),
                torch.from_numpy(self.mu[i]))



def load_source(h5_path, id_field="bearing_id", val_device_ids=None, train_only=True):
    """源域加载 (v1 扁平 / v2 分组自动适配; 复用 source_io 统一 reader)。

    返回扁平 4 元组 (features, hi, device_ids, t_index), 供 SourceWindowDataset
    构造 MMD 对齐用的源域 z_S + HI 分箱。
    """
    sd = load_source_features(h5_path, id_field=id_field, val_device_ids=val_device_ids)
    # 特征 z-score 归一 (source train 集 mean/std, 与 pretrain 一致) — encoder 加载预训练权重
    # 期望归一输入, 否则 S3 MMD 的 z_S 失准 (诊断: 未归一导致 source 组变差/爆炸)
    split = np.array(sd.split)
    tr = split == "train"
    if tr.any():
        fm = sd.features[tr].mean(axis=0)
        fs = sd.features[tr].std(axis=0) + 1e-6
        feats = ((sd.features - fm) / fs).astype(np.float32)
    else:
        feats = sd.features
    keep = tr if train_only else np.ones_like(tr, dtype=bool)
    return feats[keep], sd.hi[keep], sd.device_id_array[keep], sd.t_index[keep]


# ============================================================ HI 动力学层 (路径 A)
class HIWindowDataset(Dataset):
    """HI 序列窗 + (hi_end, rul_end) 标签 (设计文档 §3.2, HI 动力学层迁移)。

    源 / 目标域共用; 按 id (源=device_id / 目标=traj_id) groupby 后组内滑窗,
    严格不跨个体。features 字段不再使用, 只用 hi + rul + id。

    每窗输出:
      x_HI   : (L, 2) float32  channel 0 = HI, channel 1 = ΔHI (前向差分首位补 0)
      hi_end : float32         窗末 HI 值 (供 MMD 按 HI bin 分箱)
      rul_end: float32         窗末 RUL 值 (监督标签)
    """

    def __init__(self, hi: np.ndarray, rul: np.ndarray,
                 ids: np.ndarray, L: int, stride: int = 1):
        self.samples: list[tuple[np.ndarray, float, float]] = []
        df = pd.DataFrame({"id": np.asarray(ids, dtype=object)})
        for _, g in df.groupby("id"):
            idxs = g.index.to_numpy()
            T = len(idxs)
            if T < L:
                continue
            for s in range(0, T - L + 1, stride):
                seg = np.asarray(hi[idxs[s:s + L]], dtype=np.float32)
                # ΔHI = 前向差分首位补 0, 保证 x_HI[..., 1] 与 seg 同长且不引入边界信息
                dhi = np.concatenate(
                    [np.zeros(1, dtype=np.float32), np.diff(seg)]).astype(np.float32)
                x_HI = np.stack([seg, dhi], axis=-1)               # (L, 2)
                hi_end = float(seg[-1])
                rul_end = float(rul[idxs[s + L - 1]])
                self.samples.append((x_HI, hi_end, rul_end))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        x, h, r = self.samples[i]
        return (torch.from_numpy(x),
                torch.tensor(h, dtype=torch.float32),
                torch.tensor(r, dtype=torch.float32))


def _load_target_hi(h5_path, observed_only=True):
    """读取目标域 hi / rul / traj_id (路径 A 默认不读 x_global/x_nodes)。

    字段兼容: 相控阵=hi_array, 飞轮=hi_b。
    返回 (hi, rul, traj_ids, n_traj) — 供 HIWindowDataset 构造 x_HI_T 窗。
    """
    hi_list, rul_list, tid_list = [], [], []
    with h5py.File(h5_path, "r") as f:
        keys = sorted(f.keys())
        for ti, k in enumerate(keys):
            g = f[k]
            observed = bool(g.attrs.get("event_observed",
                                        np.asarray(g.get("label_fail", [])).any()))
            if observed_only and not observed:
                continue
            hi_key = "hi_array" if "hi_array" in g else "hi_b"
            hi_seg = np.asarray(g[hi_key][:]).astype(np.float32)
            rul_seg = np.asarray(g["rul"][:]).astype(np.float32)
            hi_list.append(hi_seg)
            rul_list.append(rul_seg)
            tid_list.append(np.full(hi_seg.shape[0], ti, dtype=np.int64))
    if not hi_list:
        raise ValueError("目标域没有已观测失效轨迹")
    tids = np.concatenate(tid_list)
    remap = {old: new for new, old in enumerate(sorted(set(tids)))}
    tids = np.asarray([remap[int(i)] for i in tids], dtype=np.int64)
    return np.concatenate(hi_list), np.concatenate(rul_list), tids, len(remap)


# ---------------------------------------------------------------- 评估
@torch.no_grad()
def evaluate_T(model, loader, device, huber, mse, lam):
    model.eval()
    tl = n = 0
    nm_v = nm_t = 0
    preds, labels = [], []
    for x, h, r in loader:
        x, h, r = x.to(device), h.to(device), r.to(device)
        B, Kk = x.size(0), x.size(1)
        hi_p, rul_p, _ = model(x.reshape(B * Kk, x.size(2), x.size(3)))
        hi_p = hi_p.view(B, Kk)
        rul_p = rul_p.view(B, Kk)
        Lr = huber(rul_p, r)
        Lh = mse(hi_p, h)
        d = hi_p[:, 1:] - hi_p[:, :-1]
        mono = torch.relu(-d).mean() if d.numel() else hi_p.new_zeros(())
        smooth = (d * d).mean() if d.numel() else hi_p.new_zeros(())
        loss = Lr + lam[0] * Lh + lam[1] * mono + lam[2] * smooth
        tl += loss.item() * B
        n += B
        nm_v += int((d < -1e-6).sum())
        nm_t += d.numel()
        preds.append(rul_p.cpu().numpy())
        labels.append(r.cpu().numpy())
    rp = np.concatenate(preds) if preds else np.array([0.0])
    rl = np.concatenate(labels) if labels else np.array([0.0])
    return {"loss": tl / max(n, 1),
            "rul_rmse": float(np.sqrt(np.mean((rp - rl) ** 2))),
            "mono_viol": nm_v / max(nm_t, 1)}


# ---------------------------------------------------------------- 训练
def run(args):
    cfg = load_config(args.config)
    set_seed(cfg["seed"], cfg["reproducibility"]["deterministic"],
             cfg["reproducibility"]["cudnn_benchmark"])
    L = int(cfg["model"]["input_len_L"])
    K = int(cfg.get("pretrain", {}).get("seq_block_K", 8))
    tcfg = cfg["transfer"]
    mc = cfg["model"]
    bins = [tuple(b) for b in tcfg["hi_bins"]]
    _lc = cfg["loss"]
    lam = (float(_lc.get("lambda_hi", _lc.get("beta_hi", 1.0))),
           float(_lc.get("lambda_mono", _lc.get("mu_mono", 0.1))),
           float(_lc.get("lambda_smooth", _lc.get("nu_smooth", 0.1))))
    mmd_lambda = float(tcfg["mmd_lambda"])

    has_nodes = bool(tcfg.get("target_has_nodes", False))
    target_h5 = ROOT / tcfg.get("target_feature_path", "data/features/wheel/schema_v1/target_features.h5")
    source_h5 = ROOT / cfg["pretrain"].get("source_feature_path", "data/features/wheel/schema_v1/source_features.h5")
    id_field = cfg["pretrain"].get("source_id_field", "bearing_id")
    if not target_h5.exists():
        print(f"!! 缺 {target_h5}; 先运行目标域特征工程 (build_hi.py / build_array_hi.py)"); sys.exit(1)
    if not source_h5.exists():
        print(f"!! 缺 {source_h5}; 先运行源域特征工程 (wheel_features.py / mosfet_features.py --synthetic)"); sys.exit(1)

    xT, hiT, rulT, tidT, n_traj = load_target(target_h5, has_nodes, observed_only=True)
    tr_ids, va_ids, te_ids, eol_by_tid = split_trajectories_from_cfg(
        target_h5, n_traj, tcfg, cfg["seed"], observed_only=True)
    print(f">> 轨迹划分 train {len(tr_ids)} / val {len(va_ids)} / test {len(te_ids)}")
    describe_eol_split(eol_by_tid, tr_ids, va_ids, te_ids)
    rul_scale = max(float(np.nanmax(rulT[np.isin(tidT, tr_ids)])), 1.0)
    rulT = rulT / rul_scale

    # 目标域 xT z-score 归一 (train 集) — encoder 预训练权重期望归一输入, adapter 处理域差异
    _tr_mask = np.isin(tidT, tr_ids)
    _fm = xT[_tr_mask].mean(axis=0)
    _fs = xT[_tr_mask].std(axis=0) + 1e-6
    xT = (xT - _fm) / _fs

    def mask(ids):
        return np.isin(tidT, ids)

    tstride = int(tcfg.get("target_stride", 50))     # 目标域序列长, 用大 stride 控窗数
    if args.smoke:
        tstride = max(tstride, 1000)
    ds_tr = TargetSeqDataset(xT[mask(tr_ids)], hiT[mask(tr_ids)], rulT[mask(tr_ids)],
                             tidT[mask(tr_ids)], L, K, stride=tstride)
    ds_va = TargetSeqDataset(xT[mask(va_ids)], hiT[mask(va_ids)], rulT[mask(va_ids)],
                             tidT[mask(va_ids)], L, 1, stride=tstride)
    ds_te = TargetSeqDataset(xT[mask(te_ids)], hiT[mask(te_ids)], rulT[mask(te_ids)],
                             tidT[mask(te_ids)], L, 1, stride=tstride)
    if args.smoke:
        ds_tr = Subset(ds_tr, list(range(min(48, len(ds_tr)))))
        ds_va = Subset(ds_va, list(range(min(24, len(ds_va)))))
    bs = 16 if args.smoke else int(cfg["pretrain"]["batch_size"])
    loader_T_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True)
    loader_T_va = DataLoader(ds_va, batch_size=bs, shuffle=False)
    loader_T_te = DataLoader(ds_te, batch_size=bs, shuffle=False)

    featsS, hiS, bidS, tidxS = load_source(
        source_h5, id_field,
        (cfg.get("source", {}).get("split", {}).get("val_device_ids") or
         cfg.get("source", {}).get("split", {}).get("val_bearing_ids") or []),
        train_only=True)
    ds_S = SourceWindowDataset(featsS, hiS, bidS, tidxS, L,
                               stride=max(1, len(featsS) // 2000))
    if args.smoke:
        ds_S = Subset(ds_S, list(range(min(64, len(ds_S)))))
    loader_S = DataLoader(ds_S, batch_size=bs, shuffle=True)

    device = "cuda" if (cfg["pretrain"]["device"] == "cuda" and torch.cuda.is_available()) else "cpu"
    model = TransferModel(
        encoder_type=cfg["model"]["encoder"], n_features=featsS.shape[1], n_target=xT.shape[1],
        input_len=L, channels=mc["tcn"]["channels"], kernel_size=mc["tcn"]["kernel_size"],
        num_blocks=mc["tcn"]["num_blocks"], dropout=mc["tcn"]["dropout"],
        latent_dim=mc["latent_dim"], adapter_hidden=tcfg["adapter_hidden"],
    ).to(device)

    # ---- S1: 加载源域预训练编码器 ----
    comp = Path(args.config).stem
    suffix = "" if comp == "wheel" else f"_{comp}"      # 飞轮保持原名, 相控阵带后缀
    ckpt = args.ckpt or str(CKPT_DIR / f"source{suffix}_{cfg['model']['encoder']}_pretrain.pt")
    if args.smoke and not Path(ckpt).exists():
        ckpt = str(CKPT_DIR / f"source{suffix}_{cfg['model']['encoder']}_smoke.pt")
    if Path(ckpt).exists():
        miss, unexp = model.load_pretrained(ckpt, device)
        print(f">> S1 加载源域编码器: {ckpt}  (missing={len(miss)} keys, adapter 随机初始化)")
    elif args.smoke:
        print(">> [SMOKE] 无 source checkpoint，允许随机初始化")
    else:
        raise FileNotFoundError(f"正式迁移缺少 source checkpoint: {ckpt}")

    huber = nn.HuberLoss(delta=float(cfg["loss"]["huber_delta"]))
    mse = nn.MSELoss()
    e2 = 2 if args.smoke else int(tcfg.get("epochs_s2", 20))
    e3 = 2 if args.smoke else int(tcfg.get("epochs_s3", 20))

    # ---- S2: 训 adapter + 头 (encoder 冻结) ----
    model.freeze_encoder(True)
    opt2 = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                            lr=float(tcfg["finetune_lr"]))
    print(f">> S2 adapter+头训练 (encoder 冻结, {e2} epochs)")
    for ep in range(1, e2 + 1):
        model.train()
        tl, n = 0.0, 0
        for x, h, r in loader_T_tr:
            x, h, r = x.to(device), h.to(device), r.to(device)
            B, Kk = x.size(0), x.size(1)
            hi_p, rul_p, _ = model(x.reshape(B * Kk, x.size(2), x.size(3)))
            hi_p = hi_p.view(B, Kk)
            rul_p = rul_p.view(B, Kk)
            Lr = huber(rul_p, r)
            Lh = mse(hi_p, h)
            d = hi_p[:, 1:] - hi_p[:, :-1]
            loss = Lr + lam[0] * Lh + lam[1] * torch.relu(-d).mean() + lam[2] * (d * d).mean()
            opt2.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt2.step()
            tl += loss.item() * B
            n += B
        vm = evaluate_T(model, loader_T_va, device, huber, mse, lam)
        print(f"  S2 ep{ep:02d} loss={tl / n:.4f} val_rul_rmse={vm['rul_rmse']:.4f} "
              f"mono_viol={vm['mono_viol']:.4f}")

    # ---- S3: MMD (按 HI 分箱) + 微调 ----
    reset_global_memory_bank()
    model.freeze_encoder(False)
    opt3 = torch.optim.Adam(model.parameters(), lr=float(tcfg["finetune_lr"]))
    print(f">> S3 MMD(按HI分箱)+微调 ({e3} epochs)")
    for ep in range(1, e3 + 1):
        model.train()
        tl, n = 0.0, 0
        src_iter = cycle(loader_S)
        for x, h, r in loader_T_tr:
            xs, hs = next(src_iter)
            x, h, r = x.to(device), h.to(device), r.to(device)
            xs, hs = xs.to(device), hs.to(device)
            B, Kk = x.size(0), x.size(1)
            hi_p, rul_p, zT = model(x.reshape(B * Kk, x.size(2), x.size(3)))
            hi_p = hi_p.view(B, Kk)
            rul_p = rul_p.view(B, Kk)
            Lr = huber(rul_p, r)
            Lh = mse(hi_p, h)
            d = hi_p[:, 1:] - hi_p[:, :-1]
            Lmono = torch.relu(-d).mean()
            Ls = (d * d).mean()
            zS = model.encoder(xs)                                       # 源域 latent (无 adapter)
            mmd = mmd_by_hi_bins(zS, hs, zT, h.reshape(-1), bins)        # 按 HI 健康阶段对齐
            loss = Lr + lam[0] * Lh + lam[1] * Lmono + lam[2] * Ls + mmd_lambda * mmd
            opt3.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt3.step()
            tl += loss.item() * B
            n += B
        vm = evaluate_T(model, loader_T_va, device, huber, mse, lam)
        print(f"  S3 ep{ep:02d} loss={tl / n:.4f} val_rul_rmse={vm['rul_rmse']:.4f} "
              f"mono_viol={vm['mono_viol']:.4f}")

    te_m = evaluate_T(model, loader_T_te, device, huber, mse, lam)
    CKPT_DIR.mkdir(exist_ok=True)
    tag = "smoke" if args.smoke else "transfer"
    ckpt_out = CKPT_DIR / f"transfer{suffix}_{cfg['model']['encoder']}_{tag}.pt"
    torch.save({"model": model.state_dict(), "L": L, "K": K, "bins": bins,
                "n_target": xT.shape[1],
                "split_train_val_test": (len(tr_ids), len(va_ids), len(te_ids))}, ckpt_out)
    metrics = {
        "stage": "3-stage-transfer", "encoder": cfg["model"]["encoder"], "smoke": bool(args.smoke),
        "L": L, "K": K, "rul_scale_train": rul_scale,
        "val_rul_rmse": vm["rul_rmse"], "val_mono_viol": vm["mono_viol"],
        "test_rul_rmse": te_m["rul_rmse"],
        "split_train_val_test": [len(tr_ids), len(va_ids), len(te_ids)],
        "mmd_aligned_by": "hi_health_bins (非时间)",
    }
    metrics_path = CKPT_DIR / f"transfer_metrics{suffix}.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f">> transfer checkpoint: {ckpt_out}")
    print(f">> 划分: train {len(tr_ids)} / val {len(va_ids)} / test {len(te_ids)} 轨迹 (完整轨迹级)")


# ============================================================ HI 动力学层路径
@torch.no_grad()
def _eval_hi_layer(model, loader, device, huber, mse, lam):
    """HI 层模型评估: 输入 (x_HI, hi_end, rul_end)。返回 val loss + RUL RMSE。"""
    model.eval()
    tl = n = 0
    preds, labels = [], []
    for x_HI, hi_end, rul_end in loader:
        x_HI = x_HI.to(device)
        hi_end = hi_end.to(device)
        rul_end = rul_end.to(device)
        hi_p, rul_p, _ = model(x_HI)
        Lr = huber(rul_p, rul_end)
        Lh = mse(hi_p, hi_end)
        loss = Lr + lam[0] * Lh
        tl += loss.item() * x_HI.size(0)
        n += x_HI.size(0)
        preds.append(rul_p.cpu().numpy())
        labels.append(rul_end.cpu().numpy())
    rp = np.concatenate(preds) if preds else np.array([0.0])
    rl = np.concatenate(labels) if labels else np.array([0.0])
    return {"loss": tl / max(n, 1),
            "rul_rmse": float(np.sqrt(np.mean((rp - rl) ** 2)))}


def run_hi_layer(args):
    """HI 动力学层迁移训练 (路径 A, 设计文档 §4.2)。

    关键差异 (vs 旧 run()):
      - load_source 后只用 hi/device_id/t_index (弃用 features); 由 make_hi_windows 构造 x_HI_S
      - load_target 后只用 hi_array/rul/traj_id (默认不读 x_global/x_nodes)
      - encoder 输入 (B, L, 2) 固定, 不再依赖 n_features/n_target
      - 无 adapter, MMD 对齐的 z_S / z_T 都来自共享 HISeqEncoder 对 x_HI 的编码

    三阶段:
      S1 加载源域 HISeqEncoder 预训练权重 (`source_*_{encoder}_hilayer_pretrain.pt`, Agent 2 产出)
      S2 冻结 encoder, 训 target HI_head + RUL_head (输入 = 目标域 hi 派生的 x_HI 窗)
      S3 解冻 + MMD 按 HI bin 对齐 z_S / z_T (共享 encoder 编码源 / 目标 x_HI)
    """
    cfg = load_config(args.config)
    if getattr(args, "seed", None) is not None:
        cfg["seed"] = args.seed              # override (多种子消融)
    set_seed(cfg["seed"], cfg["reproducibility"]["deterministic"],
             cfg["reproducibility"]["cudnn_benchmark"])
    L = int(cfg["model"]["input_len_L"])
    tcfg = cfg["transfer"]
    mc = cfg["model"]
    bins = [tuple(b) for b in tcfg["hi_bins"]]
    _lc = cfg["loss"]
    lam = (float(_lc.get("lambda_hi", _lc.get("beta_hi", 1.0))),
           float(_lc.get("lambda_mono", _lc.get("mu_mono", 0.1))),
           float(_lc.get("lambda_smooth", _lc.get("nu_smooth", 0.1))))
    mmd_lambda = float(tcfg["mmd_lambda"])

    hi_cfg = tcfg.get("hi_layer", {}) or {}
    hi_stride_target = int(hi_cfg.get("hi_window_stride", 50))

    target_h5 = ROOT / tcfg.get("target_feature_path",
                                 "data/features/wheel/schema_v1/target_features.h5")
    source_h5 = ROOT / cfg["pretrain"].get(
        "source_feature_path", "data/features/wheel/schema_v1/source_features.h5")
    id_field = cfg["pretrain"].get("source_id_field", "bearing_id")
    if not target_h5.exists():
        print(f"!! 缺 {target_h5}; 先运行目标域特征工程 (build_hi.py / build_array_hi.py)")
        sys.exit(1)
    if not source_h5.exists():
        print(f"!! 缺 {source_h5}; 先运行源域特征工程 (wheel_features.py / mosfet_features.py)")
        sys.exit(1)

    # ---- 目标域: 只读 hi / rul / traj_id (路径 A 默认不用 x_global/x_nodes) ----
    hiT, rulT, tidT, n_traj = _load_target_hi(target_h5, observed_only=True)
    tr_ids, va_ids, te_ids, eol_by_tid = split_trajectories_from_cfg(
        target_h5, n_traj, tcfg, cfg["seed"], observed_only=True)
    print(f">> 轨迹划分 train {len(tr_ids)} / val {len(va_ids)} / test {len(te_ids)}")
    describe_eol_split(eol_by_tid, tr_ids, va_ids, te_ids)
    rul_scale = max(float(np.nanmax(rulT[np.isin(tidT, tr_ids)])), 1.0)
    rulT = rulT / rul_scale

    def mask(ids):
        return np.isin(tidT, ids)

    tstride = int(tcfg.get("target_stride", hi_stride_target))      # 序列长, 大 stride 控窗数
    if args.smoke:
        tstride = max(tstride, 1000)
    ds_tr = HIWindowDataset(hiT[mask(tr_ids)], rulT[mask(tr_ids)],
                            tidT[mask(tr_ids)], L, stride=tstride)
    ds_va = HIWindowDataset(hiT[mask(va_ids)], rulT[mask(va_ids)],
                            tidT[mask(va_ids)], L, stride=tstride)
    ds_te = HIWindowDataset(hiT[mask(te_ids)], rulT[mask(te_ids)],
                            tidT[mask(te_ids)], L, stride=tstride)
    if args.smoke:
        ds_tr = Subset(ds_tr, list(range(min(48, len(ds_tr)))))
        ds_va = Subset(ds_va, list(range(min(24, len(ds_va)))))
    bs = 16 if args.smoke else int(cfg["pretrain"]["batch_size"])
    loader_T_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True)
    loader_T_va = DataLoader(ds_va, batch_size=bs, shuffle=False)
    loader_T_te = DataLoader(ds_te, batch_size=bs, shuffle=False)

    # ---- 源域: 只用 hi / device_id / t_index (弃用 features) ----
    # load_source 仍调 (复用 v1/v2 reader); 但下面 make_hi_windows 只吃 hi/device_id/t_index
    _, hiS, idS, tidxS = load_source(
        source_h5, id_field,
        (cfg.get("source", {}).get("split", {}).get("val_device_ids") or
         cfg.get("source", {}).get("split", {}).get("val_bearing_ids") or []),
        train_only=True)
    x_HI_S, hi_end_S = make_hi_windows(
        hiS, idS, tidxS, L, stride=max(1, len(hiS) // 2000))
    if args.smoke:
        cap = min(64, x_HI_S.shape[0])
        x_HI_S = x_HI_S[:cap]
        hi_end_S = hi_end_S[:cap]
    ds_S = TensorDataset(torch.from_numpy(x_HI_S).float(),
                         torch.from_numpy(hi_end_S).float())
    loader_S = DataLoader(ds_S, batch_size=bs, shuffle=True)

    device = "cuda" if (cfg["pretrain"]["device"] == "cuda"
                        and torch.cuda.is_available()) else "cpu"
    model = HIDynamicsModel(
        encoder_type=cfg["model"]["encoder"], input_len=L,
        channels=mc["tcn"]["channels"], kernel_size=mc["tcn"]["kernel_size"],
        num_blocks=mc["tcn"]["num_blocks"], dropout=mc["tcn"]["dropout"],
        latent_dim=mc["latent_dim"],
    ).to(device)

    # ---- S1: 加载源域 HISeqEncoder 预训练权重 (target_only 跳过) ----
    comp = Path(args.config).stem
    suffix = "" if comp == "wheel" else f"_{comp}"                  # 飞轮保持原名, 相控阵带后缀
    if getattr(args, "target_only", False):
        print(">> [target_only] 跳过 S1 源 ckpt 加载, encoder 随机初始化 + 后续全训 (迁移增益基线)")
    else:
        ckpt = (args.ckpt or
                str(CKPT_DIR / f"source{suffix}_{cfg['model']['encoder']}_hilayer_pretrain.pt"))
        if args.smoke and not Path(ckpt).exists():
            # fallback HI 层 smoke ckpt
            ckpt_smoke = str(CKPT_DIR / f"source{suffix}_{cfg['model']['encoder']}_hilayer_smoke.pt")
            if Path(ckpt_smoke).exists():
                ckpt = ckpt_smoke
        if Path(ckpt).exists():
            miss, unexp = model.load_pretrained(ckpt, device)
            print(f">> S1 加载源 HISeqEncoder: {ckpt}  (missing={len(miss)} keys)")
        elif args.smoke:
            print(">> [HI-layer SMOKE] 无 source checkpoint，允许随机初始化")
        else:
            raise FileNotFoundError(f"正式 HI 层迁移缺少 source checkpoint: {ckpt}")

    huber = nn.HuberLoss(delta=float(cfg["loss"]["huber_delta"]))
    mse = nn.MSELoss()
    e2 = 2 if args.smoke else int(tcfg.get("epochs_s2", 20))
    e3 = 0 if getattr(args, "target_only", False) else (2 if args.smoke else int(tcfg.get("epochs_s3", 20)))

    # ---- S2: 训 HI_head + RUL_head (target_only 解冻全训; 否则 encoder 冻结) ----
    model.freeze_encoder(not getattr(args, "target_only", False))
    opt2 = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                            lr=float(tcfg["finetune_lr"]))
    print(f">> S2 HI/RUL 头训练 (encoder 冻结, {e2} epochs)")
    for ep in range(1, e2 + 1):
        model.train()
        tl, n = 0.0, 0
        for x_HI, hi_end, rul_end in loader_T_tr:
            x_HI = x_HI.to(device)
            hi_end = hi_end.to(device)
            rul_end = rul_end.to(device)
            hi_p, rul_p, _ = model(x_HI)
            Lr = huber(rul_p, rul_end)
            Lh = mse(hi_p, hi_end)
            loss = Lr + lam[0] * Lh
            opt2.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt2.step()
            tl += loss.item() * x_HI.size(0)
            n += x_HI.size(0)
        vm = _eval_hi_layer(model, loader_T_va, device, huber, mse, lam)
        print(f"  S2 ep{ep:02d} loss={tl / n:.4f} val_rul_rmse={vm['rul_rmse']:.4f}")

    # ---- S3: 解冻 + MMD 按 HI bin 对齐 z_S / z_T (共享 HISeqEncoder) ----
    reset_global_memory_bank()
    model.freeze_encoder(False)
    opt3 = torch.optim.Adam(model.parameters(), lr=float(tcfg["finetune_lr"]))
    print(f">> S3 MMD(按HI分箱, HI 动力学层)+微调 ({e3} epochs)")
    for ep in range(1, e3 + 1):
        model.train()
        tl, n = 0.0, 0
        src_iter = cycle(loader_S)
        for x_HI, hi_end, rul_end in loader_T_tr:
            xs_HI, xs_hi_end = next(src_iter)
            x_HI = x_HI.to(device)
            hi_end = hi_end.to(device)
            rul_end = rul_end.to(device)
            xs_HI = xs_HI.to(device)
            xs_hi_end = xs_hi_end.to(device)
            hi_p, rul_p, zT = model(x_HI)
            Lr = huber(rul_p, rul_end)
            Lh = mse(hi_p, hi_end)
            # z_S = 共享 HISeqEncoder 编码源 x_HI (可学, encoder 解冻)
            zS = model.encoder(xs_HI)
            mmd = mmd_by_hi_bins(zS, xs_hi_end, zT, hi_end, bins)
            loss = Lr + lam[0] * Lh + mmd_lambda * mmd
            opt3.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt3.step()
            tl += loss.item() * x_HI.size(0)
            n += x_HI.size(0)
        vm = _eval_hi_layer(model, loader_T_va, device, huber, mse, lam)
        print(f"  S3 ep{ep:02d} loss={tl / n:.4f} val_rul_rmse={vm['rul_rmse']:.4f}")

    # test 评估 (泛化, 对比旧观测层 run_groups 的 test rmse)
    te_m = _eval_hi_layer(model, loader_T_te, device, huber, mse, lam)
    print(f">> TEST rul_rmse={te_m['rul_rmse']:.4f} (test {len(te_ids)} 轨迹)")

    CKPT_DIR.mkdir(exist_ok=True)
    tag = "smoke" if args.smoke else "transfer"
    ckpt_out = CKPT_DIR / f"transfer{suffix}_{cfg['model']['encoder']}_hilayer_{tag}.pt"
    torch.save({"model": model.state_dict(), "L": L, "bins": bins,
                "transfer_mode": "hi_dynamics",
                "split_train_val_test": (len(tr_ids), len(va_ids), len(te_ids))},
               ckpt_out)
    metrics = {
        "stage": "hi-layer-transfer", "transfer_mode": "hi_dynamics",
        "encoder": cfg["model"]["encoder"], "smoke": bool(args.smoke),
        "L": L, "rul_scale_train": rul_scale,
        "val_rul_rmse": vm["rul_rmse"], "test_rul_rmse": te_m["rul_rmse"],
        "split_train_val_test": [len(tr_ids), len(va_ids), len(te_ids)],
        "mmd_aligned_by": "hi_health_bins on HISeqEncoder latent (HI dynamics)",
    }
    metrics_path = CKPT_DIR / f"transfer_metrics{suffix}_hilayer.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f">> HI 层 transfer checkpoint: {ckpt_out}")
    print(f">> 划分: train {len(tr_ids)} / val {len(va_ids)} / test {len(te_ids)} 轨迹 (完整轨迹级)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wheel.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--hi-layer", action="store_true",
                    help="走 HI 动力学层迁移 (HIDynamicsModel); 默认走旧观测层 run()。"
                         "也可经 config transfer.mode=hi_dynamics 触发。")
    ap.add_argument("--target-only", action="store_true",
                    help="HI 层 target_only 基线: 随机初始化 + 全训 (不加载源 ckpt, 不 MMD), 迁移增益对比用")
    ap.add_argument("--seed", type=int, default=None,
                    help="覆盖 config seed (多种子消融; 默认用 config seed)")
    args = ap.parse_args()
    # config transfer.mode == 'hi_dynamics' 时自动切到 HI 层路径 (除非用户显式 --no-hi-layer)
    cfg = load_config(args.config)
    tcfg = cfg.get("transfer", {}) or {}
    if args.hi_layer or tcfg.get("mode") == "hi_dynamics":
        run_hi_layer(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
