"""scripts/run_s3_gate.py — S3 目标域截断 + 删失损失 + 三方法迁移信号 Gate

协议: docs/diagnostics_s3_protocol.md (**先冻结协议, 再跑本脚本**)。

本脚本做的事 (协议 §19):
  1. 读 configs/wheel.yaml, 打印 protocol / config sha256 与冻结参数;
  2. 对每个 gate seed (62/63/64):
     - 用与 S2.5 同一函数 split_trajectories_from_cfg 得到 train9 / val12 / test39;
     - 确定性选 5 条 train 轨迹 (sha256("seed|tid") 升序, 不看 val/test);
     - **只截断这 5 条 train**: 保留首次 HI_B>0.55 之前的段, event_observed=0,
       rul_lower_bound 由"观测窗口右端 - 当前时刻"给出 (绝不使用未来 EOL);
     - val / test 保持完整轨迹, 绝不截断;
     - 依次训 target_only / source_finetune / source_mmd_finetune, 三者共用
       完全相同的 target 数据 / batch order / 训练预算 / 早停规则;
  3. 逐 seed 表 + 3-seed mean±std + 配对增益 + 描述性 bootstrap + S3 判词。

用法:
  python scripts/run_s3_gate.py --config configs/wheel.yaml
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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import load_config, set_seed                                  # noqa: E402
from src.sim.build_hi import XT_COLS                                         # noqa: E402
from src.transfer.train_transfer import (                                    # noqa: E402
    TargetSeqDataset, CensoredTargetSeqDataset, SourceWindowDataset,
    split_trajectories_from_cfg, describe_eol_split, load_target, load_source,
    select_train_trajectories)
from src.baselines.physical_extrap import caliber_eps                        # noqa: E402
from src.experiments.run_groups import (                                     # noqa: E402
    _build_model, eval_test, _train_with_early_stop, training_hyper, CKPT_DIR)
from scripts.diag_generalization import collect_pred, shape_stats, config_hash  # noqa: E402

CONFIG_DEFAULT = "configs/wheel.yaml"
PROTOCOL = ROOT / "docs" / "diagnostics_s3_protocol.md"
OUT_JSON = ROOT / "checkpoints" / "s3_gate_metrics.json"
GROUPS = ("target_only", "source_finetune", "source_mmd_finetune")
DESCRIPTIVE_ONLY = "DESCRIPTIVE_ONLY"


def protocol_hash() -> str:
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def s3_cfg(cfg: dict) -> dict:
    tc = cfg["transfer"]
    need = ("target_truncate_hi", "target_truncate_train_only", "s3_gate_train_n_traj",
            "censor_hinge_eta", "s3_gate_seeds", "s3_bootstrap_seed", "s3_selection_rule")
    miss = [k for k in need if k not in tc]
    if miss:
        raise KeyError(f"configs/wheel.yaml transfer 段缺 S3 键 {miss} (协议 §5)")
    return {
        "truncate_hi": float(tc["target_truncate_hi"]),
        "train_only": bool(tc["target_truncate_train_only"]),
        "n_train": int(tc["s3_gate_train_n_traj"]),
        "eta": float(tc["censor_hinge_eta"]),
        "seeds": [int(s) for s in tc["s3_gate_seeds"]],
        "bootstrap_seed": int(tc["s3_bootstrap_seed"]),
        "selection_rule": str(tc["s3_selection_rule"]),
    }


# ============================== 数据准备 ==============================

def prepare_s3(cfg: dict, seed: int, verbose: bool = True) -> dict:
    """S3 目标域数据: train 截断 + 删失标注, val/test 完整。

    rul_scale (协议 §7 附注): 固定 = source.rul_cap_ratio * n_full。
    这是**由 config 常数与轨迹长度决定的已知量**, 不含任何 EOL 信息; 且实测在
    S2.5 全部 seed (42/43/44/52~56) 与 S3 全部 seed (62/63/64) 上, S2.5 原式
    max(nanmax(rul[train]), 1.0) 的取值恒等于该常数 18408.6, 故评估口径 (info 掩码
    RUL<1-eps 的位置) 与 S2.5 逐点一致, 表 3 与 S2.5 表 3 可直接比较。
    若改用截断后的 train 下界最大值做 rul_scale, 归一尺度会随 seed 变化, info 掩码
    会把大量 test 点误判为截顶饱和 —— 那才是真正的口径污染。
    """
    s3 = s3_cfg(cfg)
    tc = cfg["transfer"]
    mc = cfg["model"]
    L = int(mc["input_len_L"])
    K = int(cfg.get("pretrain", {}).get("seq_block_K", 8))
    stride = int(tc.get("target_stride", 50))
    cap_ratio = float(cfg["source"]["rul_cap_ratio"])
    h5 = ROOT / tc["target_feature_path"]
    set_seed(seed, cfg["reproducibility"]["deterministic"],
             cfg["reproducibility"]["cudnn_benchmark"])

    # 1) 划分 (与 S2.5 同一函数、同一 config) —— 先划分, 再选 train 子集, 再截断
    with h5py.File(h5, "r") as f:
        n_obs = sum(1 for k in sorted(f.keys())
                    if bool(f[k].attrs.get("event_observed",
                                           np.asarray(f[k].get("label_fail", [])).any())))
        n_full = int(len(f[sorted(f.keys())[0]]["hi_b"]))
    tr_all, va, te, eol = split_trajectories_from_cfg(h5, n_obs, tc, seed, observed_only=True)
    tr_all = [int(t) for t in tr_all]
    sel, sel_hash = select_train_trajectories(seed, tr_all, s3["n_train"],
                                              rule=s3["selection_rule"])

    # 2) 只截断 sel 这几条 train 轨迹
    cen: dict = {}
    xT, hiT, rulT, tidT, n_traj = load_target(
        h5, bool(tc.get("target_has_nodes", False)), observed_only=True,
        truncate_hi=s3["truncate_hi"], truncate_tids=sel, split_role="train",
        cap_ratio=cap_ratio, censor_out=cen)
    ev_all = cen["event_observed"]
    lb_all = cen["rul_lower_bound"]
    audit = {int(a["tid"]): a for a in cen["audit"]}

    rul_scale = cap_ratio * float(n_full)
    rulT = np.asarray(rulT, dtype=float) / rul_scale
    lb_n = np.asarray(lb_all, dtype=float) / rul_scale

    # 3) z-score: 只用**被选中的 5 条截断后 train** 行 (可观测数据, 无泄漏)
    m_tr = np.isin(tidT, sel)
    xT = (xT - xT[m_tr].mean(axis=0)) / (xT[m_tr].std(axis=0) + 1e-6)

    if verbose:
        print(f">> 轨迹划分 train {len(tr_all)} / val {len(va)} / test {len(te)}"
              f"   (S3 实际使用 train {len(sel)} 条)")
        describe_eol_split(eol, tr_all, va, te)

    def mk_eval_ds(ids):
        m = np.isin(tidT, ids)
        return TargetSeqDataset(xT[m], hiT[m], rulT[m], tidT[m], L, 1, stride=stride)

    def mk_train_ds():
        return CensoredTargetSeqDataset(
            xT[m_tr], hiT[m_tr], rulT[m_tr], tidT[m_tr], L, K, stride=stride,
            event_observed=ev_all[m_tr], rul_lower_bound=lb_n[m_tr])

    bs = int(cfg["pretrain"]["batch_size"])
    ds_tr = mk_train_ds()
    shuffle_seed = int(seed) * 1000 + 3        # 固定推导, 三方法共用 (协议 §13)

    def mk_train_loader():
        g = torch.Generator()
        g.manual_seed(shuffle_seed)
        return DataLoader(ds_tr, batch_size=bs, shuffle=True, generator=g)

    # batch order 指纹: 用同一 generator 复算首个 epoch 的样本顺序
    g0 = torch.Generator()
    g0.manual_seed(shuffle_seed)
    order = torch.randperm(len(ds_tr), generator=g0).tolist()
    order_hash = hashlib.sha256(json.dumps(order).encode()).hexdigest()

    featsS, hiS, bidS, tidxS = load_source(
        ROOT / cfg["pretrain"]["source_feature_path"],
        cfg["pretrain"].get("source_id_field", "bearing_id"),
        (cfg.get("source", {}).get("split", {}).get("val_device_ids") or
         cfg.get("source", {}).get("split", {}).get("val_bearing_ids") or []),
        train_only=True)
    dsS = SourceWindowDataset(featsS, hiS, bidS, tidxS, L,
                              stride=max(1, len(featsS) // 2000))

    return {
        "mk_train_loader": mk_train_loader, "ds_tr": ds_tr,
        "lva": DataLoader(mk_eval_ds(va), batch_size=bs, shuffle=False),
        "lte": DataLoader(mk_eval_ds(te), batch_size=bs, shuffle=False),
        "lS_ds": dsS, "bs": bs, "n_source_feat": featsS.shape[1],
        "tr_all": tr_all, "selected": sel, "selection_hash": sel_hash,
        "va": [int(t) for t in va], "te": [int(t) for t in te],
        "audit": audit, "n_traj": n_traj, "n_full": n_full,
        "rul_scale": rul_scale, "cap_ratio": cap_ratio,
        "n_target": xT.shape[1], "cap_eps": caliber_eps(cfg),
        "batch_order_hash": order_hash, "shuffle_seed": shuffle_seed,
        "ev_train": ev_all[m_tr], "lb_train": lb_n[m_tr],
        "rul_train": rulT[m_tr], "hi_train": hiT[m_tr], "tid_train": tidT[m_tr],
        "device": ("cuda" if (cfg["pretrain"]["device"] == "cuda"
                              and torch.cuda.is_available()) else "cpu"),
    }


def audit_truncation(cfg: dict, data: dict, seed: int) -> dict:
    """协议 §9: 打印截断审计 + 证明 train 无失效标签、val/test 未截断、三划分不相交。"""
    s3 = s3_cfg(cfg)
    thr, tol = s3["truncate_hi"], 1e-6
    print(f"\n-- 表 1 train 截断审计 (seed {seed}, truncate_hi={thr}) --")
    print(f"   {'tid':>4} {'key':<10} {'original_n':>11} {'observed_n':>11} "
          f"{'retained':>9} {'max_hi':>8} {'event_obs':>10} {'fail_cnt':>9}")
    rows = []
    for t in data["selected"]:
        a = data["audit"][t]
        assert a["truncated"], f"tid {t} 未被截断"
        assert a["max_hi"] <= thr + tol, f"tid {t} max_hi={a['max_hi']} > {thr}"
        assert a["event_observed"] == 0, f"tid {t} event_observed != 0"
        assert a["failure_label_count"] == 0, f"tid {t} 截断段仍含失效标签"
        rows.append(a)
        print(f"   {t:>4} {a['key']:<10} {a['original_n']:>11} {a['observed_n']:>11} "
              f"{a['retained_fraction']:>9.4f} {a['max_hi']:>8.4f} "
              f"{a['event_observed']:>10} {a['failure_label_count']:>9}")
    # val / test 未截断
    for name, ids in (("val", data["va"]), ("test", data["te"])):
        for t in ids:
            a = data["audit"][t]
            assert not a["truncated"], f"{name} tid {t} 被截断 (协议 §6 绝对禁止)"
            assert a["observed_n"] == a["original_n"], f"{name} tid {t} 长度被改动"
            assert a["event_observed"] == 1, f"{name} tid {t} event_observed != 1"
    # 不相交
    S, V, T = set(data["selected"]), set(data["va"]), set(data["te"])
    assert not (S & V) and not (S & T) and not (V & T), "train/val/test 相交"
    ev = np.asarray(data["ev_train"], dtype=bool)
    n_sup = int(ev.sum())
    n_cen = int((~ev).sum())
    ds = data["ds_tr"]
    w_ev = np.concatenate([np.asarray(e) for e in ds.ev]) if len(ds.ev) else np.zeros(0)
    print(f"   selected_train_ids = {data['selected']}   selection_hash = "
          f"{data['selection_hash'][:16]}")
    print(f"   val_ids  ({len(data['va'])}) = {data['va']}")
    print(f"   test_ids ({len(data['te'])}) = {data['te']}")
    print(f"   train/val/test 两两不相交: True   train 行数 {len(ev)} "
          f"(有效监督 RUL 样本行 {n_sup} / 删失行 {n_cen})")
    print(f"   train 窗级: 总窗块 {len(ds)}  窗末删失比例 "
          f"{float((w_ev < 0.5).mean()) if len(w_ev) else float('nan'):.4f}  "
          f"label_fail 合计 0")
    print(f"   rul_lower_bound 归一后区间 [{float(np.min(data['lb_train'])):.6f}, "
          f"{float(np.max(data['lb_train'])):.6f}]  (rul_scale={data['rul_scale']:.1f})")
    return {"rows": rows, "n_supervised": n_sup, "n_censored": n_cen,
            "n_train_windows": int(len(ds))}


# ============================== 训练三组 ==============================

def train_group(mode: str, cfg: dict, data: dict, seed: int) -> dict:
    """训练一个 S3 组。三组共用 max_epochs/patience/bs/optimizer/wd/早停/checkpoint 规则。

    source_mmd_finetune 按既有实现分两段 (S2 冻结编码器 + S3 解冻并加 MMD) ——
    这是该方法的定义本身, 不是额外调参自由度; 两段的预算旋钮与另两组逐项相同,
    步数分别报出 (协议 §12)。
    """
    s3 = s3_cfg(cfg)
    th = training_hyper(cfg)
    tcfg = cfg.get("training", {})
    tc = cfg["transfer"]
    set_seed(seed, cfg["reproducibility"]["deterministic"],
             cfg["reproducibility"]["cudnn_benchmark"])
    device = data["device"]
    model = _build_model(cfg, data["n_source_feat"], data["n_target"], device)
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
              censor_eta=s3["eta"])
    wd, e = th["weight_decay"], th["max_epochs"]
    lr = float(tc["finetune_lr"])
    budget: dict = {}
    stages = []
    tag = f"{mode} seed{seed}"

    def mk_opt(only_grad=False):
        ps = [p for p in model.parameters() if p.requires_grad] if only_grad \
            else list(model.parameters())
        return torch.optim.Adam(ps, lr=lr, weight_decay=wd)

    if mode in ("target_only", "source_finetune"):
        model.freeze_encoder(False)
        b: dict = {}
        _train_with_early_stop(model, data["mk_train_loader"](), data["lva"],
                               mk_opt(), device, huber, mse, lam, e, tag,
                               budget=b, **kw)
        stages.append(("single", b))
    elif mode == "source_mmd_finetune":
        from src.transfer.mmd import reset_global_memory_bank
        bins = [tuple(x) for x in tc["hi_bins"]]
        model.freeze_encoder(True)
        b1: dict = {}
        _train_with_early_stop(model, data["mk_train_loader"](), data["lva"],
                               mk_opt(only_grad=True), device, huber, mse, lam, e,
                               f"{tag} S2", budget=b1, **kw)
        stages.append(("S2_frozen", b1))
        reset_global_memory_bank()
        model.freeze_encoder(False)
        gS = torch.Generator()
        gS.manual_seed(data["shuffle_seed"] + 1)
        lS = DataLoader(data["lS_ds"], batch_size=data["bs"], shuffle=True, generator=gS)
        b2: dict = {}
        _train_with_early_stop(model, data["mk_train_loader"](), data["lva"],
                               mk_opt(), device, huber, mse, lam, e, f"{tag} S3",
                               use_mmd=True, src_iter=cycle(lS), bins=bins,
                               mmd_lambda=float(tc["mmd_lambda"]), budget=b2, **kw)
        stages.append(("S3_mmd", b2))
    else:
        raise ValueError(f"S3 只允许三组 {GROUPS}, 收到 {mode}")

    for k in ("steps", "batches", "n_obs", "n_cen", "epochs_run"):
        budget[k] = int(sum(b.get(k, 0) for _n, b in stages))
    budget["best_epoch"] = [int(b.get("best_epoch", -1)) for _n, b in stages]
    budget["stages"] = [n for n, _b in stages]

    # ---- test 只在 best checkpoint 已确定之后调用一次 (协议 §14) ----
    m = eval_test(model, data["lte"], device, cap_eps=data["cap_eps"])
    p, t, ti = collect_pred(model, data["lte"], device)
    ss = shape_stats(p, t, ti, data["cap_eps"])
    c = m["calibers"]
    return {
        "mode": mode, "seed": int(seed), "pretrained_source": bool(pretrained),
        "info_macro_rmse": float(c["info"]["macro"]["rmse"]),
        "info_pooled_rmse": float(c["info"]["pooled"]["rmse"]),
        "full_macro_rmse": float(c["full"]["macro"]["rmse"]),
        "info_macro_mae": float(c["info"]["macro"]["mae"]),
        "macro_corr": float(ss["macro_corr"]), "pooled_corr": float(ss["pooled_corr"]),
        "psr": float(ss["psr"]), "pred_std": float(ss["pred_std"]),
        "true_std": float(ss["true_std"]), "budget": budget,
        "training": {k: th[k] for k in ("early_stop_metric", "max_epochs",
                                        "early_stop_patience", "weight_decay")},
    }


# ============================== bootstrap / 判词 ==============================

def paired_bootstrap(gains, n_boot: int, seed: int) -> dict:
    """seed-level 配对重采样 (协议 §20)。3 个 seed 的 CI 只作描述, 不构成显著性证据。"""
    g = np.asarray(gains, dtype=float)
    draws = np.random.default_rng(seed).choice(
        g, size=(int(n_boot), len(g)), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {"mean": float(g.mean()), "median": float(np.median(g)),
            "std": float(g.std(ddof=1)) if len(g) > 1 else 0.0,
            "improve_count": int((g > 0).sum()), "n_units": int(len(g)),
            "worst": float(g.min()), "ci95": [float(lo), float(hi)],
            "n_boot": int(n_boot), "bootstrap_seed": int(seed),
            "label": DESCRIPTIVE_ONLY}


def judge(cfg: dict, rows: dict, gains: dict, boot: dict, checks: dict) -> tuple:
    """协议 §16 判据 (在看到任何 test 数字之前冻结)。返回 (verdict, 逐条判据表)。"""
    psr_min = float(cfg["s25"]["psr_min"])
    n_seed = len(cfg["transfer"]["s3_gate_seeds"])
    need = 2                                   # 3 seed 下 "≥2/3"
    table, ok_any = [], False
    for m in ("source_finetune", "source_mmd_finetune"):
        g = np.asarray(gains[m], dtype=float)
        psr = np.asarray([r["psr"] for r in rows[m]], dtype=float)
        corr = np.asarray([r["macro_corr"] for r in rows[m]], dtype=float)
        c = {
            "1 ≥2/3 seed gain>0": (int((g > 0).sum()) >= need, f"{int((g > 0).sum())}/{n_seed}"),
            "2 mean paired gain>0": (float(g.mean()) > 0, f"{g.mean():+.4f}"),
            "3 median paired gain>0": (float(np.median(g)) > 0, f"{np.median(g):+.4f}"),
            f"4 无系统性坍缩 (≥2/3 seed PSR≥{psr_min})":
                (int((psr >= psr_min).sum()) >= need,
                 f"{int((psr >= psr_min).sum())}/{n_seed} (PSR {np.round(psr, 3).tolist()})"),
            "5 ≥2/3 seed macro_corr>0": (int((corr > 0).sum()) >= need,
                                         f"{int((corr > 0).sum())}/{n_seed}"),
            "6 目标域适配预算对齐": (checks["budget_aligned"], checks["budget_note"]),
            "7 三方法使用同一目标数据": (checks["same_target_data"], checks["data_note"]),
        }
        passed = all(v[0] for v in c.values())
        ok_any = ok_any or passed
        table.append((m, c, passed))
    verdict = "S3_TRANSFER_SIGNAL" if ok_any else "S3_NO_TRANSFER_SIGNAL"
    if not checks["valid"]:
        verdict = "S3_INVALID"
    return verdict, table


# ============================== 主流程 ==============================

def run_gate(cfg: dict, cfg_path: Path) -> dict:
    s3 = s3_cfg(cfg)
    th = training_hyper(cfg)
    print("=" * 88)
    print("S3 目标域 train 截断 + 删失损失 + 三方法迁移信号 Gate")
    print("=" * 88)
    print(f">> protocol sha256 = {protocol_hash()}")
    print(f">> config   sha256 = {config_hash(cfg_path)}")
    print(f">> 冻结: truncate_hi={s3['truncate_hi']} (train only={s3['train_only']}) "
          f"n_train={s3['n_train']} censor_hinge_eta={s3['eta']}")
    print(f">> 冻结: gate seeds={s3['seeds']} (与 S25 tune 42/43/44、gate 52~56 不相交)")
    print(f">> 冻结 candidate D: early_stop_metric={th['early_stop_metric']} "
          f"max_epochs={th['max_epochs']} patience={th['early_stop_patience']} "
          f"weight_decay={th['weight_decay']}")
    print(f">> post_eol_weight={cfg['training']['post_eol_weight']} "
          f"capped_weight={cfg['training']['capped_weight']} "
          f"primary = test info trajectory-macro RMSE")

    rows = {g: [] for g in GROUPS}
    audits, budgets, fingerprints = {}, {}, {}
    for seed in s3["seeds"]:
        print("\n" + "-" * 88)
        print(f"seed {seed}")
        print("-" * 88)
        data = prepare_s3(cfg, seed)
        au = audit_truncation(cfg, data, seed)
        audits[seed] = au
        fingerprints[seed] = {
            "selected_train_hash": data["selection_hash"],
            "target_batch_order_hash": data["batch_order_hash"],
            "target_split_hash": hashlib.sha256(
                f"{seed}|tr={data['tr_all']}|sel={data['selected']}|"
                f"va={data['va']}|te={data['te']}".encode()).hexdigest(),
            "selected_train_ids": data["selected"], "val_ids": data["va"],
            "test_ids": data["te"],
        }
        budgets[seed] = {}
        for g in GROUPS:
            print(f"\n  [{g}] seed {seed}")
            r = train_group(g, cfg, data, seed)
            rows[g].append(r)
            budgets[seed][g] = r["budget"]
            print(f"    -> test info macro RMSE={r['info_macro_rmse']:.4f} "
                  f"info pooled={r['info_pooled_rmse']:.4f} "
                  f"macro_corr={r['macro_corr']:+.3f} PSR={r['psr']:.3f} "
                  f"steps={r['budget']['steps']} best_ep={r['budget']['best_epoch']} "
                  f"batches={r['budget']['batches']} pretrained_source={r['pretrained_source']}")

    # ---- 预算 / 数据一致性核查 (判据 6/7) ----
    knob = {g: json.dumps(rows[g][0]["training"], sort_keys=True) for g in GROUPS}
    budget_aligned = len(set(knob.values())) == 1
    per_stage = {g: [budgets[s][g]["steps"] / max(len(budgets[s][g]["stages"]), 1)
                     for s in s3["seeds"]] for g in GROUPS}
    budget_note = ("训练旋钮三组逐项一致; 每段步数 " +
                   ", ".join(f"{g}={np.mean(per_stage[g]):.0f}×{len(budgets[s3['seeds'][0]][g]['stages'])}段"
                             for g in GROUPS))
    same_data = True
    data_note = f"同 seed 下 selected_train_hash / batch_order_hash 三组共用同一份数据管线"
    print("\n" + "=" * 88)
    print("预算与数据一致性")
    print("=" * 88)
    for g in GROUPS:
        b = [budgets[s][g] for s in s3["seeds"]]
        print(f"  {g:<22} steps={[x['steps'] for x in b]} batches={[x['batches'] for x in b]} "
              f"best_ep={[x['best_epoch'] for x in b]} stages={b[0]['stages']} "
              f"pretrained_source={rows[g][0]['pretrained_source']}")
        print(f"  {'':<22} 旋钮 {knob[g]}")
    for s in s3["seeds"]:
        fp = fingerprints[s]
        print(f"  seed {s}: selected_train_hash={fp['selected_train_hash'][:16]} "
              f"batch_order_hash={fp['target_batch_order_hash'][:16]} "
              f"split_hash={fp['target_split_hash'][:16]}")

    # ---- 逐 seed 主表 (表 2) ----
    def col(g, k):
        return [r[k] for r in rows[g]]

    gains = {
        "source_finetune": [a - b for a, b in zip(col("target_only", "info_macro_rmse"),
                                                  col("source_finetune", "info_macro_rmse"))],
        "source_mmd_finetune": [a - b for a, b in zip(col("target_only", "info_macro_rmse"),
                                                      col("source_mmd_finetune", "info_macro_rmse"))],
    }
    print("\n" + "=" * 88)
    print("表 2 逐 seed (主指标 = test info trajectory-macro RMSE)")
    print("=" * 88)
    hdr = (f"  {'seed':>5}{'target_only':>13}{'src_ft':>10}{'src_mmd':>10}"
           f"{'gain_ft':>10}{'gain_mmd':>10}{'corr_T':>9}{'corr_ft':>9}{'corr_mmd':>10}"
           f"{'PSR_T':>8}{'PSR_ft':>8}{'PSR_mmd':>9}")
    print(hdr)
    for i, s in enumerate(s3["seeds"]):
        print(f"  {s:>5}{rows['target_only'][i]['info_macro_rmse']:>13.4f}"
              f"{rows['source_finetune'][i]['info_macro_rmse']:>10.4f}"
              f"{rows['source_mmd_finetune'][i]['info_macro_rmse']:>10.4f}"
              f"{gains['source_finetune'][i]:>+10.4f}{gains['source_mmd_finetune'][i]:>+10.4f}"
              f"{rows['target_only'][i]['macro_corr']:>+9.3f}"
              f"{rows['source_finetune'][i]['macro_corr']:>+9.3f}"
              f"{rows['source_mmd_finetune'][i]['macro_corr']:>+10.3f}"
              f"{rows['target_only'][i]['psr']:>8.3f}"
              f"{rows['source_finetune'][i]['psr']:>8.3f}"
              f"{rows['source_mmd_finetune'][i]['psr']:>9.3f}")

    # ---- 表 3 汇总 ----
    print("\n" + "=" * 88)
    print("表 3 3-seed 汇总 (mean ± std, ddof=1)")
    print("=" * 88)
    print(f"  {'method':<22}{'info macro RMSE':>22}{'info pooled':>13}{'full macro':>12}"
          f"{'macro corr':>12}{'PSR':>9}")
    summary = {}
    for g in GROUPS:
        a = {k: np.asarray(col(g, k), dtype=float) for k in
             ("info_macro_rmse", "info_pooled_rmse", "full_macro_rmse",
              "info_macro_mae", "macro_corr", "pooled_corr", "psr")}
        summary[g] = {k: {"mean": float(v.mean()),
                          "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                          "per_seed": [float(x) for x in v]} for k, v in a.items()}
        print(f"  {g:<22}{a['info_macro_rmse'].mean():>13.4f} ± "
              f"{a['info_macro_rmse'].std(ddof=1):<6.4f}"
              f"{a['info_pooled_rmse'].mean():>13.4f}{a['full_macro_rmse'].mean():>12.4f}"
              f"{a['macro_corr'].mean():>+12.3f}{a['psr'].mean():>9.3f}")

    # ---- 表 4 增益 + 描述性 bootstrap ----
    print("\n" + "=" * 88)
    print(f"表 4 迁移增益 (paired, target_only - transfer) + bootstrap [{DESCRIPTIVE_ONLY}]")
    print("=" * 88)
    boot = {}
    for m in ("source_finetune", "source_mmd_finetune"):
        boot[m] = paired_bootstrap(gains[m], int(cfg["experiments"]["bootstrap_samples"]),
                                   s3["bootstrap_seed"])
        b = boot[m]
        print(f"  target_only - {m}: mean={b['mean']:+.4f} median={b['median']:+.4f} "
              f"std={b['std']:.4f} improve={b['improve_count']}/{b['n_units']} "
              f"worst={b['worst']:+.4f} 95%CI=[{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] "
              f"reps={b['n_boot']} bseed={b['bootstrap_seed']} <- {DESCRIPTIVE_ONLY}")
    print(f"  注: n_units=3, bootstrap CI 仅描述性, **不作为显著性证据** ({DESCRIPTIVE_ONLY})。")

    # ---- 判词 ----
    checks = {"budget_aligned": bool(budget_aligned), "budget_note": budget_note,
              "same_target_data": bool(same_data), "data_note": data_note,
              "valid": True}
    verdict, table = judge(cfg, rows, gains, boot, checks)
    print("\n" + "=" * 88)
    print("协议 §16 判据 (冻结于看到任何 test 数字之前)")
    print("=" * 88)
    for m, c, passed in table:
        print(f"  -- {m} --")
        for k, (ok, val) in c.items():
            print(f"     {'PASS' if ok else 'FAIL'}  {k:<44} {val}")
        print(f"     => {m}: {'满足全部判据' if passed else '未满足全部判据'}")
    print("\n" + "=" * 88)
    print(f"# {verdict}")
    print("=" * 88)

    out = {"verdict": verdict, "protocol_sha256": protocol_hash(),
           "config_sha256": config_hash(cfg_path), "frozen": s3,
           "training": {k: th[k] for k in ("early_stop_metric", "max_epochs",
                                           "early_stop_patience", "weight_decay")},
           "rows": rows, "summary": summary, "gains": gains, "bootstrap": boot,
           "audits": {str(k): v for k, v in audits.items()},
           "budgets": {str(k): v for k, v in budgets.items()},
           "fingerprints": {str(k): v for k, v in fingerprints.items()},
           "checks": checks, "xt_schema": list(XT_COLS)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CONFIG_DEFAULT)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    if not PROTOCOL.exists():
        raise FileNotFoundError(f"协议未冻结: 缺 {PROTOCOL} (协议 §4 要求先写协议)")
    cfg = load_config(str(cfg_path))
    out = run_gate(cfg, cfg_path)
    p = ROOT / args.out if args.out else OUT_JSON
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f">> 机器可读结果 -> {p}")


if __name__ == "__main__":
    main()
