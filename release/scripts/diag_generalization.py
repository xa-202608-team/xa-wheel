"""scripts/diag_generalization.py — S2.5 候选调参 (严格 VAL-only) + 过拟合诊断

协议: docs/diagnostics_s25_protocol.md (先冻结, 再跑本脚本)。

两个 phase:
  --phase tune    A/B/C/D 四候选 × 3 个**调参** seed, **只看 train/val**。
                  本 phase 根本不构造 test loader、不计算任何 test 指标
                  (tests/test_generalization_gate.py::test_candidate_selection_never_reads_test
                   用 source 扫描 + 运行时 flag 双重钉死)。
  --phase select  按协议 §9 的冻结规则从 tune 结果里选出候选, 写回 configs/wheel.yaml,
                  打印 protocol hash / config hash。

为什么不复用 scripts/diag_common.prepare_target: 它无条件构造 lte (test loader),
而 S2.5 §7 要求调参阶段"禁止计算 test RMSE"。本脚本自带 with_test 开关的数据准备,
并把它导出给 scripts/run_s25_gate.py 复用, 保证 tune 与 gate 的数据管线逐行同源。
(diag_common 不在 S2.5 允许修改的文件白名单内, 故不改它。)

用法:
  python scripts/diag_generalization.py --config configs/wheel.yaml --phase tune
  python scripts/diag_generalization.py --config configs/wheel.yaml --phase select
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import load_config, set_seed                                # noqa: E402
from src.sim.build_hi import XT_COLS                                       # noqa: E402
from src.transfer.train_transfer import (                                  # noqa: E402
    TargetSeqDataset, split_trajectories_from_cfg, describe_eol_split, load_target)
from src.baselines.physical_extrap import (                                # noqa: E402
    caliber_metrics, caliber_mask, caliber_eps)
from src.experiments.run_groups import (                                   # noqa: E402
    _build_model, eval_test, _train_with_early_stop, training_hyper,
    EARLY_STOP_METRICS)

CONFIG_DEFAULT = "configs/wheel.yaml"
PROTOCOL = ROOT / "docs" / "diagnostics_s25_protocol.md"
TUNE_OUT = ROOT / "checkpoints" / "s25_tune.json"

# 候选之间唯一允许存在差异的四个 training 键 (协议 §6)
CANDIDATE_KEYS = ("early_stop_metric", "max_epochs", "early_stop_patience", "weight_decay")


# ============================== 配置 / 候选 ==============================

def s25_cfg(cfg: dict) -> dict:
    if "s25" not in cfg:
        raise KeyError("configs/wheel.yaml 缺 s25 段 (见 docs/diagnostics_s25_protocol.md)")
    return cfg["s25"]


def tuning_seeds(cfg: dict) -> list[int]:
    return [int(cfg["seed"]) + int(o) for o in s25_cfg(cfg)["tuning_seed_offsets"]]


def gate_seeds(cfg: dict) -> list[int]:
    return [int(cfg["seed"]) + int(o) for o in s25_cfg(cfg)["gate_seed_offsets"]]


def candidate_names(cfg: dict) -> list[str]:
    return list(s25_cfg(cfg)["candidates"].keys())


def apply_candidate(cfg: dict, name: str) -> dict:
    """返回 training.* 被候选 name 覆盖后的 cfg 深拷贝 (不落盘)。

    只覆盖 CANDIDATE_KEYS 四个键; 其余训练参数保持 config 原值 —— 候选之间
    除这四个键外禁止存在任何差异 (协议 §5/§6)。
    """
    cand = s25_cfg(cfg)["candidates"]
    if name not in cand:
        raise KeyError(f"未知候选 {name}; 可选 {sorted(cand)}")
    extra = set(cand[name]) - set(CANDIDATE_KEYS)
    if extra:
        raise ValueError(f"候选 {name} 含协议外的键 {sorted(extra)} (禁止)")
    out = copy.deepcopy(cfg)
    out.setdefault("training", {}).update({k: cand[name][k] for k in CANDIDATE_KEYS
                                           if k in cand[name]})
    return out


def candidate_signature(cfg: dict) -> str:
    """当前 cfg 的训练配置指纹 (供 gate 断言 5 个 seed 用同一冻结配置)。"""
    th = training_hyper(cfg)
    tcfg = cfg.get("training", {})
    payload = {
        **{k: th[k] for k in CANDIDATE_KEYS},
        "post_eol_weight": float(tcfg.get("post_eol_weight", 1.0)),
        "capped_weight": float(tcfg.get("capped_weight", 1.0)),
        "cap_eps": caliber_eps(cfg),
        "finetune_lr": float(cfg["transfer"]["finetune_lr"]),
        "batch_size": int(cfg["pretrain"]["batch_size"]),
        "encoder": cfg["model"]["encoder"],
        "L": int(cfg["model"]["input_len_L"]),
        "lam": [float(cfg["loss"]["lambda_hi"]), float(cfg["loss"]["lambda_mono"]),
                float(cfg["loss"]["lambda_smooth"])],
        "huber_delta": float(cfg["loss"]["huber_delta"]),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def config_hash(path: Path = None) -> str:
    p = Path(path or (ROOT / CONFIG_DEFAULT))
    return hashlib.sha256(p.read_bytes()).hexdigest()


def protocol_hash() -> str:
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


# ============================== 数据准备 ==============================

def prepare(cfg: dict, seed: int, with_test: bool = False, verbose: bool = True) -> dict:
    """目标域数据准备, 与 run_groups.run_one_group 逐行同口径。

    with_test=False (调参阶段): **不构造 test loader**, 返回的 dict 里 lte 键不存在 ——
    调用方在结构上拿不到 test, 满足协议 §9 "test 在候选冻结前不参与任何选择"。
    """
    tc = cfg["transfer"]
    mc = cfg["model"]
    L = int(mc["input_len_L"])
    K = int(cfg.get("pretrain", {}).get("seq_block_K", 8))
    stride = int(tc.get("target_stride", 50))
    h5 = ROOT / tc.get("target_feature_path",
                       "data/features/wheel/schema_v1/target_features.h5")
    set_seed(seed, cfg["reproducibility"]["deterministic"],
             cfg["reproducibility"]["cudnn_benchmark"])

    xT, hiT, rulT, tidT, n_traj = load_target(
        h5, bool(tc.get("target_has_nodes", False)), observed_only=True)
    tr, va, te, eol = split_trajectories_from_cfg(h5, n_traj, tc, seed, observed_only=True)
    if verbose:
        print(f">> 轨迹划分 train {len(tr)} / val {len(va)} / test {len(te)}"
              f"   (with_test={with_test})")
        describe_eol_split(eol, tr, va, te)

    rul_scale = max(float(np.nanmax(rulT[np.isin(tidT, tr)])), 1.0)
    rulT = rulT / rul_scale
    _m = np.isin(tidT, tr)                       # z-score 只用 train 统计量
    xT = (xT - xT[_m].mean(axis=0)) / (xT[_m].std(axis=0) + 1e-6)

    def mkDS(ids, block_k):
        m = np.isin(tidT, ids)
        return TargetSeqDataset(xT[m], hiT[m], rulT[m], tidT[m], L, block_k, stride=stride)

    bs = int(cfg["pretrain"]["batch_size"])
    out = {
        "ltr": DataLoader(mkDS(tr, K), batch_size=bs, shuffle=True),
        "lva": DataLoader(mkDS(va, 1), batch_size=bs, shuffle=False),
        "tr": tr, "va": va, "te": te, "eol": eol, "n_traj": n_traj,
        "rul_scale": rul_scale, "n_target": xT.shape[1],
        "cap_eps": caliber_eps(cfg),
        "device": ("cuda" if (cfg["pretrain"]["device"] == "cuda"
                              and torch.cuda.is_available()) else "cpu"),
    }
    if with_test:
        out["lte"] = DataLoader(mkDS(te, 1), batch_size=bs, shuffle=False)
    return out


# ============================== 训练 / 评估 ==============================

def train_candidate(cfg: dict, data: dict, seed: int, tag: str):
    """训一次 target_only (随机初始化 + 全参), 返回 (model, history, training_hyper)。

    与 run_groups.run_one_group(mode='target_only') 逐行等价, 含 S2.5 的
    early_stop_metric / patience / weight_decay。
    """
    th = training_hyper(cfg)
    tcfg = cfg.get("training", {})
    model = _build_model(cfg, data["n_target"], data["n_target"], data["device"])
    model.freeze_encoder(False)
    opt = torch.optim.Adam(model.parameters(),
                           lr=float(cfg["transfer"]["finetune_lr"]),
                           weight_decay=th["weight_decay"])
    huber = nn.HuberLoss(delta=float(cfg["loss"]["huber_delta"]))
    mse = nn.MSELoss()
    lam = (float(cfg["loss"]["lambda_hi"]), float(cfg["loss"]["lambda_mono"]),
           float(cfg["loss"]["lambda_smooth"]))
    history = []
    _train_with_early_stop(
        model, data["ltr"], data["lva"], opt, data["device"], huber, mse, lam,
        th["max_epochs"], tag,
        post_eol_weight=float(tcfg.get("post_eol_weight", 1.0)),
        huber_delta=float(cfg["loss"]["huber_delta"]),
        capped_weight=float(tcfg.get("capped_weight", 1.0)),
        cap_eps=data["cap_eps"],
        early_stop_metric=th["early_stop_metric"],
        early_stop_patience=th["early_stop_patience"],
        history=history)
    return model, history, th


@torch.no_grad()
def collect_pred(model, loader, device):
    """在 loader 上收集 (pred, true, tids)。loader 以 K=1 构造, 每样本 1 个预测点。"""
    model.eval()
    P, T, TI = [], [], []
    ds = loader.dataset
    st = np.asarray(getattr(ds, "sample_tids", np.zeros(len(ds))), dtype=np.int64)
    i = 0
    for x, _h, r in loader:
        x = x.to(device)
        B, Kk = x.size(0), x.size(1)
        _, rul_p, _ = model(x.reshape(B * Kk, x.size(2), x.size(3)))
        P.append(rul_p.cpu().numpy().reshape(-1))
        T.append(r.reshape(-1).numpy())
        TI.append(np.repeat(st[i:i + B], Kk))
        i += B
    return (np.concatenate(P), np.concatenate(T), np.concatenate(TI))


def macro_corr(pred, true, tids) -> float:
    """逐轨迹 Pearson 相关再对轨迹等权平均 (std 退化的轨迹跳过)。"""
    vals = []
    for t in np.unique(tids):
        m = tids == t
        p, y = pred[m], true[m]
        if len(p) < 3 or p.std() < 1e-12 or y.std() < 1e-12:
            continue
        vals.append(float(np.corrcoef(p, y)[0, 1]))
    return float(np.mean(vals)) if vals else float("nan")


def pooled_corr(pred, true) -> float:
    if len(pred) < 3 or pred.std() < 1e-12 or true.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(pred, true)[0, 1])


def shape_stats(pred, true, tids, cap_eps) -> dict:
    """info 掩码内的 PSR / macro_corr / pooled_corr (输出坍缩与趋势分辨力)。"""
    m = caliber_mask(true, "info", cap_eps)
    if not m.any():
        return {"pred_std": float("nan"), "true_std": float("nan"),
                "psr": float("nan"), "macro_corr": float("nan"),
                "pooled_corr": float("nan"), "n_info": 0}
    p, y, ti = pred[m], true[m], tids[m]
    ps, ys = float(p.std()), float(y.std())
    return {"pred_std": ps, "true_std": ys,
            "psr": float(ps / ys) if ys > 1e-12 else float("nan"),
            "macro_corr": macro_corr(p, y, ti), "pooled_corr": pooled_corr(p, y),
            "n_info": int(m.sum())}


def summarize_history(history: list, es_metric: str) -> dict:
    """从逐 epoch **val** 历史里抽过拟合诊断量 (协议 §13; 全是 val/train, 无 test)。

    generalization_gap proxy = 末 epoch 的 val_info_macro − best epoch 的 val_info_macro
      —— "最优点之后 val 还恶化了多少", 直接量化过拟合幅度 (同单位, 可跨候选比)。
    """
    if not history:
        return {}
    best_i = int(np.argmin([h["es_value"] for h in history]))
    b, last = history[best_i], history[-1]
    return {
        "n_epochs_run": len(history),
        "best_epoch": int(b["epoch"]),
        "train_loss_at_best": float(b["train_loss"]),
        "val_full_rmse": float(b["val_full_rmse"]),
        "val_info_pooled_rmse": float(b["val_info_pooled_rmse"]),
        "val_info_macro_rmse": float(b["val_info_macro_rmse"]),
        "es_metric": es_metric,
        "es_value_at_best": float(b["es_value"]),
        "final_val_info_macro_rmse": float(last["val_info_macro_rmse"]),
        "gap_proxy": float(last["val_info_macro_rmse"] - b["val_info_macro_rmse"]),
        "curve_val_info_macro": [float(h["val_info_macro_rmse"]) for h in history],
        "curve_train_loss": [float(h["train_loss"]) for h in history],
    }


# ============================== phase: tune ==============================

def phase_tune(cfg: dict, seeds: list[int], names: list[str]) -> dict:
    print("=" * 82)
    print("S2.5 phase=tune  候选调参 (严格 VAL-only; 本 phase 不构造 test loader, "
          "不计算任何 test 指标)")
    print("=" * 82)
    print(f">> protocol sha256 = {protocol_hash()}")
    print(f">> config   sha256 = {config_hash()}")
    print(f">> 调参 seeds = {seeds}   候选 = {names}")

    runs = {}
    for name in names:
        ccfg = apply_candidate(cfg, name)
        th = training_hyper(ccfg)
        sig = candidate_signature(ccfg)
        print(f"\n{'-' * 82}\n候选 {name}: early_stop_metric={th['early_stop_metric']} "
              f"max_epochs={th['max_epochs']} patience={th['early_stop_patience']} "
              f"weight_decay={th['weight_decay']}  sig={sig}\n{'-' * 82}")
        rows = []
        for s in seeds:
            data = prepare(ccfg, s, with_test=False, verbose=False)
            assert "lte" not in data, "调参阶段不得持有 test loader"
            model, hist, _ = train_candidate(ccfg, data, s, f"{name} seed{s}")
            hs = summarize_history(hist, th["early_stop_metric"])
            # best checkpoint 已恢复 → 下面的 val 指标即"按 val 选出的模型"在 val 上的表现
            vm = eval_test(model, data["lva"], data["device"], cap_eps=data["cap_eps"])
            pv, tv, iv = collect_pred(model, data["lva"], data["device"])
            ss = shape_stats(pv, tv, iv, data["cap_eps"])
            row = {"seed": int(s), "candidate": name, "sig": sig, **hs,
                   "val_info_macro_rmse_ckpt": float(vm["calibers"]["info"]["macro"]["rmse"]),
                   "val_info_pooled_rmse_ckpt": float(vm["calibers"]["info"]["pooled"]["rmse"]),
                   "val_full_rmse_ckpt": float(vm["calibers"]["full"]["pooled"]["rmse"]),
                   **{f"val_{k}": v for k, v in ss.items()}}
            rows.append(row)
            print(f"  seed {s}: best_ep={row['best_epoch']:>2d}/{row['n_epochs_run']:>2d} "
                  f"train_loss@best={row['train_loss_at_best']:.4f} "
                  f"val_full={row['val_full_rmse_ckpt']:.4f} "
                  f"val_info_pooled={row['val_info_pooled_rmse_ckpt']:.4f} "
                  f"val_info_macro={row['val_info_macro_rmse_ckpt']:.4f} "
                  f"PSR={row['val_psr']:.3f} macro_corr={row['val_macro_corr']:+.3f} "
                  f"gap={row['gap_proxy']:+.4f}")
        v = np.array([r["val_info_macro_rmse_ckpt"] for r in rows])
        runs[name] = {
            "rows": rows, "sig": sig,
            "training": {k: th[k] for k in CANDIDATE_KEYS},
            "mean_val_info_macro_rmse": float(v.mean()),
            "std_val_info_macro_rmse": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
        }

    print("\n" + "=" * 82)
    print("表 1  调参阶段 VAL-only 汇总 (3 seeds, mean±std)")
    print("=" * 82)
    print(f"  {'候选':<4}{'val info macro RMSE':>22}{'best_ep mean':>14}{'range':>10}"
          f"{'val macro_corr':>16}{'val PSR':>10}{'gap proxy':>12}")
    for name in names:
        rs = runs[name]["rows"]
        be = np.array([r["best_epoch"] for r in rs])
        print(f"  {name:<4}{runs[name]['mean_val_info_macro_rmse']:>13.4f} ± "
              f"{runs[name]['std_val_info_macro_rmse']:<6.4f}{be.mean():>14.1f}"
              f"{f'[{be.min()},{be.max()}]':>10}"
              f"{np.mean([r['val_macro_corr'] for r in rs]):>+16.3f}"
              f"{np.mean([r['val_psr'] for r in rs]):>10.3f}"
              f"{np.mean([r['gap_proxy'] for r in rs]):>+12.4f}")
    print("  注: 全部为 val 量; 本 phase 未计算任何 test 指标。")
    return {"phase": "tune", "seeds": seeds, "candidates": names, "runs": runs,
            "protocol_sha256": protocol_hash(), "config_sha256": config_hash()}


# ============================== phase: select ==============================

def select_candidate(runs: dict, names: list[str], tie_tol: float,
                     simplicity_order: list[str], verbose: bool = True):
    """协议 §9 冻结规则: mean → (差<tol) std → (差<tol) 更简单者; A 不参加简单者优先。"""
    stats = {n: (runs[n]["mean_val_info_macro_rmse"], runs[n]["std_val_info_macro_rmse"])
             for n in names}
    order = sorted(names, key=lambda n: stats[n][0])
    best = order[0]
    log = [f"步骤 1 按 mean(val_info_macro_rmse) 升序: " +
           ", ".join(f"{n}={stats[n][0]:.4f}" for n in order)]
    ties = [n for n in order if abs(stats[n][0] - stats[best][0]) < tie_tol]
    if len(ties) > 1:
        log.append(f"步骤 2 mean 差值 < {tie_tol} 的并列组 {ties}; 比 std: " +
                   ", ".join(f"{n}={stats[n][1]:.4f}" for n in ties))
        ties2 = sorted(ties, key=lambda n: stats[n][1])
        best = ties2[0]
        ties3 = [n for n in ties2 if abs(stats[n][1] - stats[best][1]) < tie_tol]
        if len(ties3) > 1:
            # A 只作历史基线, 不参加"更简单者优先"
            cand = [n for n in simplicity_order if n in ties3]
            log.append(f"步骤 3 std 也在 {tie_tol} 内并列 {ties3}; 按 simplicity_order "
                       f"{simplicity_order} 取更简单者 (A 不参加)")
            if cand:
                best = cand[0]
            else:
                log.append("  并列组内无 simplicity_order 成员 -> 保持 std 最低者")
    else:
        log.append(f"步骤 2 无并列 (次优 {order[1]} 与最优差 "
                   f"{abs(stats[order[1]][0] - stats[best][0]):.4f} >= {tie_tol})")
    log.append(f"=> selected_candidate = {best}   (未使用任何 test 指标)")
    if verbose:
        for line in log:
            print("  " + line)
    return best, log


_TOP_KEY = re.compile(r"^[A-Za-z_][\w]*:")


def _patch_scoped(text: str, block: str, kv: dict) -> str:
    """在顶层块 block 内替换 `key: value` 行 (保留注释与缩进)。

    必须限定作用域: `weight_decay` 在 pretrain 段也出现 (1.0e-5), 全局替换会误改。
    """
    lines = text.split("\n")
    start = next((i for i, ln in enumerate(lines) if ln.strip().startswith(f"{block}:")
                  and _TOP_KEY.match(ln)), None)
    if start is None:
        raise KeyError(f"config 缺顶层块 {block}:")
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].strip() and _TOP_KEY.match(lines[i]):
            end = i
            break
    remaining = dict(kv)
    for i in range(start + 1, end):
        m = re.match(r"^(\s+)([A-Za-z_][\w]*):(\s*)(.*?)(\s*#.*)?$", lines[i])
        if not m:
            continue
        key = m.group(2)
        if key in remaining:
            lines[i] = f"{m.group(1)}{key}: {remaining.pop(key)}{m.group(5) or ''}"
    if remaining:
        raise KeyError(f"块 {block} 内未找到键 {sorted(remaining)}")
    return "\n".join(lines)


def freeze_selection(cfg_path: Path, cfg: dict, selected: str, log: list) -> str:
    """把选中候选的四个键写入 configs/wheel.yaml 的 training.*, 并置 s25.selected_candidate。"""
    cand = s25_cfg(cfg)["candidates"][selected]
    fmt = {"early_stop_metric": lambda v: str(v),
           "max_epochs": lambda v: str(int(v)),
           "early_stop_patience": lambda v: str(int(v)),
           "weight_decay": lambda v: (f"{float(v):.1e}" if float(v) else "0.0")}
    text = cfg_path.read_text(encoding="utf-8")
    text = _patch_scoped(text, "training", {k: fmt[k](cand[k]) for k in CANDIDATE_KEYS})
    text = _patch_scoped(text, "s25", {"selected_candidate": selected})
    cfg_path.write_text(text, encoding="utf-8")

    # 协议追加 selected_candidate (协议 §10 要求)
    ap = ["\n\n---\n\n## selected_candidate (由 --phase select 追加, VAL-only)\n\n",
          f"- `selected_candidate: {selected}`\n",
          "- `selected_from_val_only: true`\n",
          f"- 冻结值: " + ", ".join(f"`{k} = {cand[k]}`" for k in CANDIDATE_KEYS) + "\n",
          "- 选择过程 (仅 val 指标):\n"]
    ap += [f"  {i + 1}. {ln}\n" for i, ln in enumerate(log)]
    with open(PROTOCOL, "a", encoding="utf-8", newline="\n") as f:
        f.write("".join(ap))
    return config_hash(cfg_path)


def _diag_overfit(runs: dict, base: str, sel: str):
    """协议 §13: A CURRENT vs selected 的过拟合诊断对比 + 四个明确回答。"""
    print("\n" + "=" * 82)
    print(f"过拟合诊断 (协议 §13): {base} CURRENT  vs  selected {sel}")
    print("=" * 82)

    def pack(n):
        rs = runs[n]["rows"]
        be = np.array([r["best_epoch"] for r in rs])
        return {
            "best_ep": be,
            "early_frac": float(np.mean(be <= 1)),
            "train_at_best": np.array([r["train_loss_at_best"] for r in rs]),
            "val_at_best": np.array([r["val_info_macro_rmse_ckpt"] for r in rs]),
            "val_final": np.array([r["final_val_info_macro_rmse"] for r in rs]),
            "gap": np.array([r["gap_proxy"] for r in rs]),
            "n_ep": np.array([r["n_epochs_run"] for r in rs]),
        }

    A, S = pack(base), pack(sel)
    print(f"  {'量':<34}{base:>16}{sel:>16}")
    rows = [
        ("best_epoch 分布", str(A["best_ep"].tolist()), str(S["best_ep"].tolist())),
        ("实跑 epoch 数", str(A["n_ep"].tolist()), str(S["n_ep"].tolist())),
        ("best_epoch ∈ {0,1} 的 seed 比例", f"{A['early_frac']:.0%}", f"{S['early_frac']:.0%}"),
        ("train loss @best (mean)", f"{A['train_at_best'].mean():.4f}",
         f"{S['train_at_best'].mean():.4f}"),
        ("val info macro @best (mean)", f"{A['val_at_best'].mean():.4f}",
         f"{S['val_at_best'].mean():.4f}"),
        ("val info macro @best (std)", f"{A['val_at_best'].std(ddof=1):.4f}",
         f"{S['val_at_best'].std(ddof=1):.4f}"),
        ("val info macro @末 epoch (mean)", f"{A['val_final'].mean():.4f}",
         f"{S['val_final'].mean():.4f}"),
        ("generalization_gap proxy (mean)", f"{A['gap'].mean():+.4f}", f"{S['gap'].mean():+.4f}"),
    ]
    for k, a, s in rows:
        print(f"  {k:<34}{a:>16}{s:>16}")

    print("\n  -- 四个明确回答 --")
    q1 = ("减轻" if S["early_frac"] < A["early_frac"] else
          "未减轻 (持平)" if S["early_frac"] == A["early_frac"] else "加重")
    print(f"  (1) ep0/ep1 集中现象是否减轻: **{q1}** "
          f"({A['early_frac']:.0%} -> {S['early_frac']:.0%}; 注意 selected 的 max_epochs="
          f"{runs[sel]['training']['max_epochs']}, 短训下 best_ep 早本身不算失败)")
    d = A["val_at_best"].mean() - S["val_at_best"].mean()
    print(f"  (2) val info 是否改善: **{'是' if d > 0 else '否'}** "
          f"({A['val_at_best'].mean():.4f} -> {S['val_at_best'].mean():.4f}, 改善 {d:+.4f})")
    ds = A["val_at_best"].std(ddof=1) - S["val_at_best"].std(ddof=1)
    print(f"  (3) 方差是否下降: **{'是' if ds > 0 else '否'}** "
          f"(std {A['val_at_best'].std(ddof=1):.4f} -> {S['val_at_best'].std(ddof=1):.4f}, "
          f"{ds:+.4f})")
    wd = float(runs[sel]["training"]["weight_decay"])
    if wd <= 0:
        print("  (4) 更强正则是否只降训练拟合而未提升 val: **不适用** "
              f"(selected {sel} 的 weight_decay=0, 未加正则)")
    else:
        wd_rows = [(n, runs[n]) for n in runs
                   if float(runs[n]["training"]["weight_decay"]) > 0]
        txt = ", ".join(f"{n}(wd={r['training']['weight_decay']}): "
                        f"train@best={np.mean([x['train_loss_at_best'] for x in r['rows']]):.4f} "
                        f"val={r['mean_val_info_macro_rmse']:.4f}" for n, r in wd_rows)
        print(f"  (4) 更强正则的效果: {txt}")
        print(f"      对比无正则 B: train@best="
              f"{np.mean([x['train_loss_at_best'] for x in runs['B']['rows']]):.4f} "
              f"val={runs['B']['mean_val_info_macro_rmse']:.4f}")


def phase_select(cfg: dict, cfg_path: Path) -> dict:
    if not TUNE_OUT.exists():
        raise FileNotFoundError(f"缺 {TUNE_OUT}; 先跑 --phase tune")
    tune = json.loads(TUNE_OUT.read_text(encoding="utf-8"))
    runs, names = tune["runs"], tune["candidates"]
    sc = s25_cfg(cfg)
    print("=" * 82)
    print("S2.5 phase=select  按协议 §9 冻结规则选择候选 (只用 val 指标)")
    print("=" * 82)
    if tune.get("protocol_sha256") != protocol_hash():
        print(f"!! 注意: tune 时 protocol sha256 = {tune.get('protocol_sha256')}, "
              f"当前 = {protocol_hash()} (协议在 tune 后被追加过)")
    sel, log = select_candidate(runs, names, float(sc["select_tie_tol"]),
                               list(sc["simplicity_order"]))
    _diag_overfit(runs, "A", sel)
    new_hash = freeze_selection(cfg_path, cfg, sel, log)
    print("\n" + "=" * 82)
    print(f">> selected_candidate = {sel}")
    print(f">> 冻结值 = " + ", ".join(
        f"{k}={s25_cfg(cfg)['candidates'][sel][k]}" for k in CANDIDATE_KEYS))
    print(f">> protocol sha256 (追加 selected_candidate 后) = {protocol_hash()}")
    print(f">> config   sha256 (写入冻结值后)             = {new_hash}")
    print(f">> gate seeds = {gate_seeds(cfg)} (与调参 seeds {tuning_seeds(cfg)} 不相交)")
    print("=" * 82)
    return {"phase": "select", "selected": sel, "log": log,
            "config_sha256": new_hash, "protocol_sha256": protocol_hash()}


# ============================== CLI ==============================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CONFIG_DEFAULT)
    ap.add_argument("--phase", choices=["tune", "select"], required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_config(str(cfg_path))
    if not PROTOCOL.exists():
        raise FileNotFoundError(f"协议未冻结: 缺 {PROTOCOL} (协议 §4 要求先写协议)")

    if args.phase == "tune":
        out = phase_tune(cfg, tuning_seeds(cfg), candidate_names(cfg))
        p = ROOT / args.out if args.out else TUNE_OUT
    else:
        out = phase_select(cfg, cfg_path)
        p = ROOT / args.out if args.out else ROOT / "checkpoints" / "s25_select.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f">> 结果 -> {p}")


if __name__ == "__main__":
    main()
