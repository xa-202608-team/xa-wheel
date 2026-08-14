"""scripts/run_s5_rate.py — S5 速率预测 + 物理首达换算 Gate

S5 改动 (指令原文, 不在此重新论证根因):
  1. 预测目标 RUL -> 退化速率 mu, RUL 由物理首达换算 (1-HI_t)/mu 得出;
     首达换算独立成模块 src/models/rate_head.py (Wiener+PF 基线必须复用同一模块)。
  2. 损失 = L_rate(mu 对 HI 长基线局部斜率) + w_rul·L_huber(信息区) + w_nll·NLL。
  3. x_T 补长基线趋势特征 Tf_ratio / Tf_slope (第 9/10 维)。
  4. 保留原直接回归头作消融组 direct_rul。

**数据契约完全复用 run_s3_gate.prepare_s3**: 同一 split 函数、同一 5 条 train 轨迹
选择规则、同一截断阈值、同一 rul_scale、同一 batch order 种子 —— 只把 train loader
从 CensoredTargetSeqDataset 换成 RateTargetSeqDataset (多带一个 mu_true 标签)。
这样"S5 vs S3"的差异被限定在"建模方式", 不含任何数据/划分/预算的偷偷变动。

禁止清单 (指令): 不动 mmd_lambda / encoder 结构 / transfer.split / 仿真器 /
mission profile。本脚本一行都不碰这些。

完成判据 (3 seeds, 信息区口径):
  (a) pred_std / true_std >= 0.5
  (b) info 口径 macro RMSE < 0.2618  (const_mean_info)
  (c) coverage_before_eol > 0.5

用法:
  python scripts/run_s5_rate.py --config configs/wheel.yaml
  python scripts/run_s5_rate.py --config configs/wheel.yaml --groups rate_mmd
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import load_config, set_seed                                  # noqa: E402
from src.sim.build_hi import XT_COLS                                         # noqa: E402
from src.transfer.train_transfer import hi_local_slope                       # noqa: E402
from src.experiments.run_groups import (                                     # noqa: E402
    _build_model, eval_test, _train_with_early_stop, training_hyper,
    rate_cfg, CKPT_DIR)
from src.models.rate_head import first_passage_rul                           # noqa: E402
from scripts.run_s3_gate import prepare_s3, s3_cfg, config_hash              # noqa: E402
from scripts.diag_generalization import shape_stats                          # noqa: E402

OUT_JSON = ROOT / "checkpoints" / "s5_rate_metrics.json"
S3_JSON = ROOT / "checkpoints" / "s3_gate_metrics.json"
S4_JSON = ROOT / "checkpoints" / "s4_metrics.json"

# 判据阈值 (指令给定, 不从 config 猜)
CRIT_PSR = 0.5
CRIT_RMSE = 0.2618          # const_mean_info (S4 表 1 冻结值)
CRIT_COV = 0.5

# 组: rate_* 走速率头; direct_rul_* 走原直接回归头 (消融, 改动 4)
GROUPS = ("rate_target_only", "rate_finetune", "rate_mmd", "direct_rul_target_only")
_MODE = {
    "rate_target_only":       ("target_only", True),
    "rate_finetune":          ("source_finetune", True),
    "rate_mmd":               ("source_mmd_finetune", True),
    "direct_rul_target_only": ("target_only", False),
}


# ============================== 数据: S3 契约 + mu_true ==============================

class RateWrap(torch.utils.data.Dataset):
    """把 S3 的 `CensoredTargetSeqDataset` **原样包一层**, 只追加 mu_true。

    为什么不重建数据集: prepare_s3 已经完成划分/截断/删失/归一/z-score, 重建一次
    等于给 S5 一个偷偷改动数据的口子。本类直接持有 S3 的 dataset 对象, x/hi/rul/ev/lb
    全部是**同一份内存**, 逐位相同不需要论证; S5 唯一新增的是第 6 项 mu_true。

    窗末对齐用**交叉校验**钉死, 不只靠长度断言: 复算出的窗末行索引处的 hi 必须与
    base.samples[i][1] (S3 自己算的窗末 hi) 逐元素相等。若枚举顺序有任何偏移, 这里
    会立即失败, 而不是静默把速率标签错配到别的时刻上。
    """

    def __init__(self, base, hi_rows, tid_rows, mu_rows, L: int, K: int, stride: int):
        import pandas as pd
        self.base = base
        self.sample_tids = base.sample_tids
        mu_s, end_idx = [], []
        df = pd.DataFrame({"tid": np.asarray(tid_rows)})
        for _tid, g in df.groupby("tid"):
            idxs = g.index.to_numpy()
            T = len(idxs)
            if T < L:
                continue
            starts = list(range(0, T - L + 1, stride))
            we = [int(idxs[s + L - 1]) for s in starts]
            for s2 in range(0, len(starts) - K + 1):
                blk = we[s2:s2 + K]
                mu_s.append(np.asarray(mu_rows, dtype=np.float32)[blk])
                end_idx.append(blk)
        if len(mu_s) != len(base):
            raise AssertionError(f"S5 速率标注窗数 {len(mu_s)} != S3 样本窗数 {len(base)}")
        hi_rows = np.asarray(hi_rows, dtype=np.float32)
        for i in range(len(mu_s)):
            got = base.samples[i][1]
            want = hi_rows[end_idx[i]]
            if not np.allclose(got, want, atol=1e-6):
                raise AssertionError(f"窗末对齐失败 @block {i}: {got} != {want}")
        self.mu = mu_s
        self.n_checked = len(mu_s)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        f, h, r, ev, lb = self.base[i]
        return f, h, r, ev, lb, torch.from_numpy(self.mu[i])


def prepare_s5(cfg: dict, seed: int, verbose: bool = True) -> dict:
    """prepare_s3 的输出 + 速率标签 mu_true + S5 train loader (RateWrap)。

    mu_true 只在**被选中的 5 条截断 train 轨迹**上算, 且只用它们**截断后可见的**
    HI 段 —— 与训练可见数据严格一致, 不看截断点之后任何一个采样点。
    (prepare_s3 返回的 hi_train 已经是截断后的段。)

    速率单位: 乘 rul_scale 换算到"归一 RUL 单位"下的 ΔHI 速率, 与归一 RUL 同尺度,
    使 L_rate 与 L_huber 量级天然可比 (rate.scale_by_rul_scale)。
    """
    d = prepare_s3(cfg, seed, verbose=verbose)
    rc = rate_cfg(cfg)
    L = int(cfg["model"]["input_len_L"])
    K = int(cfg.get("pretrain", {}).get("seq_block_K", 8))
    stride = int(cfg["transfer"].get("target_stride", 50))
    scale = float(d["rul_scale"]) if rc["scale_by_rul_scale"] else 1.0

    hi_tr = np.asarray(d["hi_train"], dtype=float)
    tid_tr = np.asarray(d["tid_train"], dtype=np.int64)
    # **逐轨迹**算斜率: 跨轨迹拼接后做 rolling 会让窗口跨过轨迹边界 (前一条的尾巴
    # 污染后一条的头部)。这类静默错位正是必须拒绝的。
    mu_true = np.zeros(len(hi_tr), dtype=float)
    per_traj = {}
    for t in np.unique(tid_tr):
        m = tid_tr == t
        mu_true[m] = hi_local_slope(hi_tr[m], rc, scale)
        per_traj[int(t)] = {
            "n": int(m.sum()), "mu_mean": float(mu_true[m].mean()),
            "mu_max": float(mu_true[m].max()),
            "hi_span": float(hi_tr[m].max() - hi_tr[m].min()),
        }

    ds_tr = RateWrap(d["ds_tr"], hi_tr, tid_tr, mu_true, L, K, stride)
    bs = d["bs"]
    shuffle_seed = d["shuffle_seed"]

    def mk_train_loader():
        g = torch.Generator()
        g.manual_seed(shuffle_seed)
        return DataLoader(ds_tr, batch_size=bs, shuffle=True, generator=g)

    # batch order 指纹: 与 S3 同一 shuffle_seed 且 dataset 长度相同 -> 顺序应完全一致
    g0 = torch.Generator()
    g0.manual_seed(shuffle_seed)
    order = torch.randperm(len(ds_tr), generator=g0).tolist()
    d["s5_batch_order_hash"] = hashlib.sha256(json.dumps(order).encode()).hexdigest()
    d["mk_train_loader_s5"] = mk_train_loader
    d["ds_tr_s5"] = ds_tr
    d["mu_true"] = mu_true
    d["mu_per_traj"] = per_traj
    d["rc"] = rc
    if verbose:
        nz = float((mu_true > 0).mean())
        print(f">> S5 速率标签: n={len(mu_true)}  >0 占比 {nz:.4f}  "
              f"mean={mu_true.mean():.4e}  max={mu_true.max():.4e}  "
              f"(estimator={rc['slope_estimator']} W={rc['slope_window']} scale={scale:.1f})")
        print(f">> S5 窗块数 {len(ds_tr)} (= S3 {len(d['ds_tr'])})  窗末对齐校验 "
              f"{ds_tr.n_checked}/{len(ds_tr)} 通过  batch_order 同 S3 = "
              f"{d['s5_batch_order_hash'] == d['batch_order_hash']}")
    return d


# ============================== 训练 ==============================

def train_group(group: str, cfg: dict, data: dict, seed: int) -> dict:
    """训练一个 S5 组。预算旋钮 (max_epochs/patience/bs/optimizer/wd/早停) 与 S3 逐项相同。"""
    mode, use_rate = _MODE[group]
    s3 = s3_cfg(cfg)
    th = training_hyper(cfg)
    tcfg = cfg.get("training", {})
    tc = cfg["transfer"]
    rc = data["rc"] if use_rate else None
    set_seed(seed, cfg["reproducibility"]["deterministic"],
             cfg["reproducibility"]["cudnn_benchmark"])
    device = data["device"]
    model = _build_model(cfg, data["n_source_feat"], data["n_target"], device,
                         rate_head=use_rate)
    pretrained = False
    if mode != "target_only":
        ckpt = CKPT_DIR / f"source_{cfg['model']['encoder']}_pretrain.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"缺少源域 checkpoint: {ckpt}")
        model.load_pretrained(str(ckpt), device)
        pretrained = True

    huber = nn.HuberLoss(delta=float(cfg["loss"]["huber_delta"]))
    mse = nn.MSELoss()
    lam = (float(cfg["loss"]["lambda_hi"]), float(cfg["loss"]["lambda_mono"]),
           float(cfg["loss"]["lambda_smooth"]))
    kw = dict(post_eol_weight=float(tcfg.get("post_eol_weight", 1.0)),
              huber_delta=float(cfg["loss"]["huber_delta"]),
              capped_weight=float(tcfg.get("capped_weight", 1.0)),
              cap_eps=data["cap_eps"],
              early_stop_metric=th["early_stop_metric"],
              early_stop_patience=th["early_stop_patience"],
              censor_eta=s3["eta"], rc=rc)
    wd, e = th["weight_decay"], th["max_epochs"]
    lr = float(tc["finetune_lr"])
    # rate 组用 S5 loader (6 元组), direct_rul 组用 S3 loader (5 元组) —— 后者
    # 与 S3 逐位同一份数据管线, 消融才有参照意义。
    mk_loader = data["mk_train_loader_s5"] if use_rate else data["mk_train_loader"]
    budget: dict = {}
    stages = []
    tag = f"{group} seed{seed}"

    def mk_opt(only_grad=False):
        ps = [p for p in model.parameters() if p.requires_grad] if only_grad \
            else list(model.parameters())
        return torch.optim.Adam(ps, lr=lr, weight_decay=wd)

    if mode in ("target_only", "source_finetune"):
        model.freeze_encoder(False)
        b: dict = {}
        _train_with_early_stop(model, mk_loader(), data["lva"], mk_opt(), device,
                               huber, mse, lam, e, tag, budget=b, **kw)
        stages.append(("single", b))
    elif mode == "source_mmd_finetune":
        from src.transfer.mmd import reset_global_memory_bank
        bins = [tuple(x) for x in tc["hi_bins"]]
        model.freeze_encoder(True)
        b1: dict = {}
        _train_with_early_stop(model, mk_loader(), data["lva"], mk_opt(only_grad=True),
                               device, huber, mse, lam, e, f"{tag} S2", budget=b1, **kw)
        stages.append(("S2_frozen", b1))
        reset_global_memory_bank()
        model.freeze_encoder(False)
        gS = torch.Generator()
        gS.manual_seed(data["shuffle_seed"] + 1)
        lS = DataLoader(data["lS_ds"], batch_size=data["bs"], shuffle=True, generator=gS)
        b2: dict = {}
        _train_with_early_stop(model, mk_loader(), data["lva"], mk_opt(), device,
                               huber, mse, lam, e, f"{tag} S3", use_mmd=True,
                               src_iter=cycle(lS), bins=bins,
                               mmd_lambda=float(tc["mmd_lambda"]), budget=b2, **kw)
        stages.append(("S3_mmd", b2))
    else:
        raise ValueError(f"未知 mode {mode}")

    for k in ("steps", "batches", "n_obs", "n_cen", "epochs_run"):
        budget[k] = int(sum(b.get(k, 0) for _n, b in stages))
    budget["best_epoch"] = [int(b.get("best_epoch", -1)) for _n, b in stages]
    budget["stages"] = [n for n, _b in stages]

    # ---- test 只在 best checkpoint 确定之后调用一次 ----
    m = eval_test(model, data["lte"], device, cap_eps=data["cap_eps"],
                  cfg=cfg, rc=rc)
    p, t, ti, hv, mu_v, sg_v = collect_pred_rate(model, data["lte"], device, rc)
    ss = shape_stats(p, t, ti, data["cap_eps"])
    c = m["calibers"]
    out = {
        "group": group, "mode": mode, "use_rate": bool(use_rate),
        "seed": int(seed), "pretrained_source": bool(pretrained),
        "info_macro_rmse": float(c["info"]["macro"]["rmse"]),
        "info_pooled_rmse": float(c["info"]["pooled"]["rmse"]),
        "full_macro_rmse": float(c["full"]["macro"]["rmse"]),
        "info_macro_mae": float(c["info"]["macro"]["mae"]),
        "macro_corr": float(ss["macro_corr"]), "pooled_corr": float(ss["pooled_corr"]),
        "psr": float(ss["psr"]), "pred_std": float(ss["pred_std"]),
        "true_std": float(ss["true_std"]), "budget": budget,
        "stage_metrics": m["stage_metrics"],
        "prognostic_horizon": {k: v for k, v in m["prognostic_horizon"].items()
                               if k != "per_trajectory"},
        "alpha_lambda": {k: v for k, v in m["alpha_lambda"].items()
                         if k != "per_trajectory"},
        "warning": {k: v for k, v in m["warning"].items() if k != "per_trajectory"},
        "convergence": {k: v for k, v in m["convergence"].items()
                        if k != "per_trajectory"},
        "training": {k: th[k] for k in ("early_stop_metric", "max_epochs",
                                        "early_stop_patience", "weight_decay")},
    }
    if rc is not None:
        out["interval"] = m["interval"]
        out["rate_stats"] = m["rate_stats"]
    return out


@torch.no_grad()
def collect_pred_rate(model, loader, device, rc):
    """test loader 上收集 (pred, true, tids, hi_obs, mu, sigma)。

    rc=None 时走直接回归头 (与 scripts/diag_generalization.collect_pred 同式);
    rc 给出时走速率头 + 首达换算, 换算用**观测 HI**。
    """
    model.eval()
    P, T, TI, H, MU, SG = [], [], [], [], [], []
    ds = loader.dataset
    st = np.asarray(getattr(ds, "sample_tids", np.zeros(len(ds))), dtype=np.int64)
    i = 0
    for batch in loader:
        x, h, r = batch[0], batch[1], batch[2]
        x = x.to(device)
        B, Kk = x.size(0), x.size(1)
        xf = x.reshape(B * Kk, x.size(2), x.size(3))
        hv = h.reshape(-1).numpy()
        if rc is None:
            _, rul_p, _ = model(xf)
            P.append(rul_p.cpu().numpy().reshape(-1))
            MU.append(np.full(B * Kk, np.nan))
            SG.append(np.full(B * Kk, np.nan))
        else:
            _hp, mu_p, sg_p, _z = model.forward_rate(xf)
            mu_v = mu_p.cpu().numpy().reshape(-1)
            sg_v = sg_p.cpu().numpy().reshape(-1)
            P.append(np.asarray(first_passage_rul(hv, mu_v, rc["hi_fail"],
                                                  rc["eps_mu"], rc["rul_clip"]),
                                dtype=float))
            MU.append(mu_v)
            SG.append(sg_v)
        T.append(r.reshape(-1).numpy())
        TI.append(np.repeat(st[i:i + B], Kk))
        H.append(hv)
        i += B
    return tuple(np.concatenate(v) for v in (P, T, TI, H, MU, SG))


# ============================== 判据 ==============================

def judge(rows: dict) -> tuple:
    """三条完成判据 (指令给定, 逐组逐 seed 判)。返回 (verdict, 表)。

    判据在**信息区口径**下, 3 seeds 平均。"同时满足"= 三条都过。
    """
    table = []
    ok_any = False
    for g in rows:
        if not rows[g]:
            continue
        psr = np.array([r["psr"] for r in rows[g]], dtype=float)
        rmse = np.array([r["info_macro_rmse"] for r in rows[g]], dtype=float)
        cov = np.array([r["warning"]["coverage_before_eol"] for r in rows[g]],
                       dtype=float)
        c = {
            f"(a) pred_std/true_std >= {CRIT_PSR}":
                (float(np.mean(psr)) >= CRIT_PSR, f"{np.mean(psr):.4f}"),
            f"(b) info macro RMSE < {CRIT_RMSE}":
                (float(np.mean(rmse)) < CRIT_RMSE, f"{np.mean(rmse):.4f}"),
            f"(c) coverage_before_eol > {CRIT_COV}":
                (float(np.mean(cov)) > CRIT_COV, f"{np.mean(cov):.4f}"),
        }
        passed = all(v[0] for v in c.values())
        ok_any = ok_any or passed
        table.append((g, c, passed))
    return ("S5_RATE_MODEL_PASS" if ok_any else "S5_RATE_MODEL_FAIL"), table


def frozen_hashes() -> dict:
    """S3/S4 冻结结果的 sha256 (证明 S5 未改动它们)。"""
    out = {}
    for name, p in (("s3_gate_metrics.json", S3_JSON), ("s4_metrics.json", S4_JSON),
                    ("s4_stage_diagnostics.json",
                     ROOT / "checkpoints" / "s4_stage_diagnostics.json")):
        out[name] = (hashlib.sha256(p.read_bytes()).hexdigest() if p.exists()
                     else "MISSING")
    return out


# ============================== 主流程 ==============================

def run(cfg: dict, cfg_path: Path, groups: list) -> dict:
    s3 = s3_cfg(cfg)
    rc = rate_cfg(cfg)
    th = training_hyper(cfg)
    seeds = s3["seeds"]
    print("=" * 92)
    print("S5 退化速率预测 + 物理首达换算 Gate")
    print("=" * 92)
    print(f">> config sha256 = {config_hash(cfg_path)}")
    print(f">> x_T = {len(XT_COLS)} 维 {XT_COLS}")
    print(f">> rate: eps_mu={rc['eps_mu']} rul_clip={rc['rul_clip']} "
          f"w_rate={rc['w_rate']} w_rul={rc['w_rul']} w_nll={rc['w_nll']} "
          f"capped_weight={rc['capped_weight']}")
    print(f">> slope: {rc['slope_estimator']} W={rc['slope_window']} "
          f"scale_by_rul_scale={rc['scale_by_rul_scale']}")
    print(f">> 冻结 (未改): truncate_hi={s3['truncate_hi']} n_train={s3['n_train']} "
          f"eta={s3['eta']} seeds={seeds} mmd_lambda={cfg['transfer']['mmd_lambda']}")
    print(f">> 预算: max_epochs={th['max_epochs']} patience={th['early_stop_patience']} "
          f"wd={th['weight_decay']} early_stop={th['early_stop_metric']}")
    hb = frozen_hashes()
    print(f">> S3/S4 冻结结果 sha256 (运行前): "
          + " ".join(f"{k}={v[:16]}" for k, v in hb.items()))

    rows = {g: [] for g in groups}
    for seed in seeds:
        print(f"\n{'=' * 92}\n== seed {seed}\n{'=' * 92}")
        data = prepare_s5(cfg, seed)
        for g in groups:
            r = train_group(g, cfg, data, seed)
            rows[g].append(r)
            iv = r.get("interval", {})
            print(f"  [{g:<22}] info_macro={r['info_macro_rmse']:.4f} "
                  f"psr={r['psr']:.4f} corr={r['macro_corr']:+.4f} "
                  f"cov_pre_eol={r['warning']['coverage_before_eol']:.4f} "
                  f"iv90={iv.get('coverage_info', float('nan')):.4f}")

    verdict, table = judge(rows)
    ha = frozen_hashes()
    unchanged = all(hb[k] == ha[k] for k in hb)

    print(f"\n{'=' * 92}\n== 判据 (3 seeds 平均, 信息区口径)\n{'=' * 92}")
    for g, c, passed in table:
        print(f"\n-- {g}  {'PASS' if passed else 'FAIL'}")
        for k, (ok, val) in c.items():
            print(f"   [{'x' if ok else ' '}] {k:<42} = {val}")
    print(f"\n>> S3/S4 冻结结果 = {'UNCHANGED' if unchanged else 'CHANGED (违规!)'}")
    print(f">> VERDICT: {verdict}")

    res = {
        "config_sha256": config_hash(cfg_path),
        "xt_cols": list(XT_COLS), "n_features_target": len(XT_COLS),
        "rate_config": rc, "seeds": seeds, "groups": groups,
        "criteria": {"psr_min": CRIT_PSR, "info_macro_rmse_max": CRIT_RMSE,
                     "coverage_before_eol_min": CRIT_COV},
        "frozen_before": hb, "frozen_after": ha,
        "frozen_unchanged": bool(unchanged),
        "verdict": verdict,
        "judge_table": [{"group": g, "passed": bool(p),
                         "checks": {k: {"ok": bool(v[0]), "value": v[1]}
                                    for k, v in c.items()}}
                        for g, c, p in table],
        "per_seed": rows,
        "aggregate": {g: _agg(rows[g]) for g in groups if rows[g]},
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=_jdefault)
    print(f">> 写出 {OUT_JSON}")
    return res


def _jdefault(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"不可序列化: {type(o)}")


def _agg(rs: list) -> dict:
    """跨 seed 聚合: 标量取 mean/std, 分阶段表逐 bin 取 mean。"""
    def ms(key, sub=None):
        v = [(r[key][sub] if sub else r[key]) for r in rs]
        v = np.array([float(x) for x in v], dtype=float)
        return {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v, ddof=1))
                if len(v) > 1 else 0.0, "per_seed": v.tolist()}
    out = {
        "n_seeds": len(rs),
        "info_macro_rmse": ms("info_macro_rmse"),
        "info_pooled_rmse": ms("info_pooled_rmse"),
        "full_macro_rmse": ms("full_macro_rmse"),
        "psr": ms("psr"), "macro_corr": ms("macro_corr"),
        "pred_std": ms("pred_std"), "true_std": ms("true_std"),
        "macro_ph": ms("prognostic_horizon", "macro_ph"),
        "coverage_before_eol": ms("warning", "coverage_before_eol"),
        "miss_rate_before_eol": ms("warning", "miss_rate_before_eol"),
        "conditional_lead_before_eol": ms("warning", "conditional_lead_before_eol"),
        "n_warned_before_eol": ms("warning", "n_warned_before_eol"),
        "macro_convergence": ms("convergence", "macro_convergence"),
        "macro_late_convergence": ms("convergence", "macro_late_convergence"),
    }
    if "interval" in rs[0]:
        out["interval_coverage_info"] = ms("interval", "coverage_info")
        out["interval_mean_width_info"] = ms("interval", "mean_width_info")
        out["mu_mean"] = ms("rate_stats", "mu_mean")
        out["frac_mu_at_floor"] = ms("rate_stats", "frac_mu_at_floor")
    # alpha-lambda: 逐 lambda 取 accuracy 均值 (结构见 metrics.alpha_lambda_accuracy)
    out["alpha_lambda"] = {}
    for lam in rs[0]["alpha_lambda"].get("by_lambda", {}):
        v = np.array([float(r["alpha_lambda"]["by_lambda"][lam]["accuracy"])
                      for r in rs], dtype=float)
        el = np.array([int(r["alpha_lambda"]["by_lambda"][lam]["eligible_count"])
                       for r in rs], dtype=int)
        out["alpha_lambda"][lam] = {"accuracy_mean": float(np.nanmean(v)),
                                    "per_seed": v.tolist(),
                                    "eligible_count": el.tolist()}
    # 分阶段: 逐 bin 取各量均值
    nb = len(rs[0]["stage_metrics"]["bins"])
    out["stage_bins"] = []
    for i in range(nb):
        bs = [r["stage_metrics"]["bins"][i] for r in rs]
        rec = {"bin": bs[0]["bin"]}
        for k in ("n_points", "n_trajectories", "macro_rmse", "pooled_rmse",
                  "macro_corr", "pred_std_true_std_ratio", "pred_mean", "pred_std",
                  "true_mean", "true_std"):
            v = np.array([float(b[k]) for b in bs], dtype=float)
            rec[k] = float(np.nanmean(v))
        out["stage_bins"].append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wheel.yaml")
    ap.add_argument("--groups", nargs="*", default=list(GROUPS))
    args = ap.parse_args()
    bad = [g for g in args.groups if g not in GROUPS]
    if bad:
        raise SystemExit(f"未知组 {bad}; 可选 {GROUPS}")
    p = ROOT / args.config
    run(load_config(str(p)), p, list(args.groups))


if __name__ == "__main__":
    main()
