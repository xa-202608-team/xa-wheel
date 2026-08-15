#!/usr/bin/env python
"""scripts/basilisk_b6/run_formal_matrix.py

BASILISK-B6 §10/§11/§12/§24-5 —— 失效标签稀缺正式矩阵。

矩阵 = 4 个标签档 × 3 个训练方法 × 5 个正式 seed = **60 次训练**;
两个 eval-only 基线 (`damage_extrapolation`, `const_mean_info`) 逐档逐 seed 另评。

编排逐字沿用 B5 的 `run_group` 结构 (它本身沿用 B4X = `run_one_group` 的分支),
训练循环 / 损失 / 早停 / 模型 / MMD / evaluator / 基线全部 import 冻结实现。
B6 只换一件事: **train 的轨迹清单换成该档的失效标签子集** (data_b6.prepare_b6)。

§11 公平性: 同一 (档, seed) 内三组共享 train/val/test IDs、target batch 顺序、
预处理、optimizer、epoch 预算、早停、checkpoint 规则、删失损失、evaluator;
且架构完全一致故初始权重逐值相同。唯一差异 = 是否 load source + 是否走 MMD。

§12: checkpoint 只按 validation 选。禁止自动 retry / 换 seed / 因某组难看重跑。
训练崩塌也是正式结果; 只有确凿代码错误 / NaN 才把整个 cell 标 INVALID。

用法:
    python scripts/basilisk_b6/run_formal_matrix.py --config configs/wheel_basilisk_b6.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.basilisk_b11.calibrate_degradation import (  # noqa: E402
    load_b11_config as load_cfg,
)
from scripts.basilisk_b2.baselines_b2 import (  # noqa: E402
    EXCLUDED_ORACLE_METHODS,
    const_mean_info,
    const_pred_on,
    damage_extrapolation,
)
from scripts.basilisk_b2.eval_b2 import (  # noqa: E402
    b2_endpoint_axis,
    evaluate_arrays_b2,
    evaluate_b2,
    strip_private,
)
from scripts.basilisk_b4x.diagnose_transfer_stability import (  # noqa: E402
    catastrophic_rate,
    catastrophic_threshold,
)
from scripts.basilisk_b6.data_b6 import (  # noqa: E402
    batch_order_signature,
    load_b6_split,
    load_subset_manifest,
    prepare_b6,
    prepare_source,
    reset_target_batch_order,
    resolve_input_schema_b6,
    train_ids_signature,
)
from scripts.diag_generalization import summarize_history  # noqa: E402
from src.experiments.run_groups import (  # noqa: E402
    _build_model,
    _train_with_early_stop,
    training_hyper,
)
from src.transfer.mmd import reset_global_memory_bank  # noqa: E402

CFG_PATH = "configs/wheel_basilisk_b6.yaml"
INVALID = "B6_INVALID"

TRAINED_GROUPS = ("target_only", "source_finetune", "source_mmd_finetune")
SOURCE_GROUPS = ("source_finetune", "source_mmd_finetune")
EVAL_ONLY_GROUPS = ("const_mean_info", "damage_extrapolation")

MATRIX_PURPOSE = (
    "B6 §1: 唯一的科学问题是 —— 当目标域只有少量**完整失效轨迹** "
    "(event-observed failure labels) 可用于训练时, source 预训练是否比 "
    "Target-only 更有价值。减少的是失效标签条数, **不是**目标域轨迹总量: "
    "真实航天约束是失效标签稀缺, 退化 / 未失效运行数据仍可大量存在。"
    "因此每档始终保留 B2.1 train 中全部 24 条 censored 轨迹。"
    "B5_NO_POSITIVE_TRANSFER 是全数据确认性结论, **不得因 B6 被覆盖**; "
    "即使某个低标签档出现正增益, 也只能构成"
    "\"条件性低标签迁移收益\", 绝不构成\"总体正转移\"。"
)

ARCH_NOTE = (
    "三组统一 n_features=12 (source encoder 第一层宽度) / n_target=10, "
    "沿用 B5 §6.3 的架构公平性设定。B6 的 target_only 是**本阶段按该档 train "
    "重新训练**的结果, 其数值既不等于 B2.1 的 0.2441, 也不等于 B5 的 0.241024 "
    "(除 n=21 档的 train 轨迹与 B5 相同外, 其余档 train 更小)。"
    "gain 只在同一 (档, seed) 的三组之间计算, 绝不跨档或跨阶段相减。"
)

PAIRING_NOTE = (
    "§11 严格配对: 同一 (n_event_labeled, seed) 内三组共享 train/val/test IDs、"
    "target batch 顺序 (每组开训前重置 DataLoader generator)、target 预处理、"
    "optimizer、epoch 预算、早停指标、checkpoint 选择规则、删失损失、evaluator; "
    "架构一致故初始权重逐值相同。唯一允许的差异 = source 初始化 + MMD 项。"
    "不得因为某组失败而单独加 epoch。"
)

SUBSET_NOTE = (
    "rul_scale 与 z-score 的 mu/sd 由 prepare_b2 从**该档 train** 推导, "
    "故随标签预算变化 —— 这是标签预算的后果, 不是公平性破坏: "
    "同一 cell 内三组共用同一份, 只是不同档之间不同。每档均记录 rul_scale 以便复现。"
)

CHECKPOINT_NOTE = (
    "§12 checkpoint 只按 validation 选。_train_with_early_stop 的签名里没有 test "
    "loader —— 按 test 早停 / 按 test 选 checkpoint 在结构上不可达。"
    "禁止自动 retry, 禁止因某 seed 难看而重跑, 禁止换 seed。"
    "训练崩塌也是正式结果; 只有确凿代码错误 / NaN 才把整个 cell 标 INVALID。"
)


def _assert_frozen_hyper(cfg: dict) -> dict:
    """§2: 冻结超参必须与 B2 记录的期望值逐项一致, 不符即停。"""
    th = training_hyper(cfg)
    exp = cfg["b2_frozen_training_expected"]
    for k, want in (("early_stop_metric", exp["early_stop_metric"]),
                    ("max_epochs", int(exp["max_epochs"])),
                    ("early_stop_patience", int(exp["early_stop_patience"])),
                    ("weight_decay", float(exp["weight_decay"]))):
        if th[k] != want:
            raise SystemExit(
                f"!! {INVALID}: training.{k}={th[k]!r} != B2 冻结值 {want!r} "
                f"—— §2 禁止在本阶段改超参")
    return th


def _assert_formal_seeds(cfg: dict) -> list[int]:
    """§9: 正式 seed 必须全新, 与任何已观察过行为的 seed 无交集。"""
    b6 = cfg["b6"]
    seeds = [int(s) for s in b6["seeds"]]
    forbidden = {int(s) for s in b6["forbid_reused_seeds"]}
    bad = sorted(set(seeds) & forbidden)
    if bad:
        raise SystemExit(
            f"!! {INVALID}: 正式 seed 与已观察 seed 重叠 {bad} —— "
            f"B6 是 confirmatory, 不能消费行为已知的 seed")
    if len(set(seeds)) != len(seeds):
        raise SystemExit(f"!! 正式 seed 有重复: {seeds}")
    return seeds


def _assert_protocol_frozen(cfg: dict) -> dict:
    """§15/§24: protocol 必须已在出数字前冻结, 且哈希与当前文件一致。"""
    hp = ROOT / cfg["protocol"]["hash_path"]
    if not hp.exists():
        raise SystemExit(f"!! 缺 {hp.relative_to(ROOT)} —— 必须先跑 freeze_protocol.py")
    rec = json.loads(hp.read_text(encoding="utf-8"))
    now = hashlib.sha256((ROOT / cfg["protocol"]["path"]).read_bytes()).hexdigest()
    if now != str(rec["protocol_sha256"]):
        raise SystemExit(
            f"!! protocol.md 在冻结之后被改动 —— {INVALID}。"
            " 出数字前改请重新 freeze; 出数字后改 = 事后调门槛")
    if not bool(rec["frozen_before_any_b6_number"]):
        raise SystemExit("!! protocol_hash 未标记 frozen_before_any_b6_number")
    return rec


def _optimizer(cfg: dict, model, th: dict, trainable_only: bool = False):
    params = ([p for p in model.parameters() if p.requires_grad]
              if trainable_only else list(model.parameters()))
    return torch.optim.Adam(params, lr=float(cfg["transfer"]["finetune_lr"]),
                            weight_decay=th["weight_decay"])


def _state_sha256(model) -> str:
    """模型权重指纹 —— §12 要求逐组记录 checkpoint_sha256 / init_weights_sha256。"""
    h = hashlib.sha256()
    sd = model.state_dict()
    for k in sorted(sd.keys()):
        v = sd[k].detach().cpu().numpy()
        h.update(k.encode("utf-8"))
        h.update(np.ascontiguousarray(v, dtype=np.float64).tobytes())
    return h.hexdigest()


def run_group(cfg: dict, data: dict, src: dict, n_event: int, seed: int,
              group: str, th: dict, ckpt: Path, axis: dict,
              ckpt_dir: Path) -> dict:
    """训练一组并在 test 上评估。编排逐字沿用 B5。"""
    print(f"\n{'-' * 70}\n[B6 矩阵] n_event={n_event}  seed {seed}  group {group}"
          f"\n{'-' * 70}")
    t0 = time.time()
    # §11: DataLoader 的 generator 有状态, 每 epoch 推进。三组共用同一 loader 对象,
    # 若不在每组开训前重置, 第二、三组拿到的就是"接着上一组走"的 batch 顺序,
    # RMSE 差里会混进 batch 顺序差异。
    reset_target_batch_order(data)
    # 指纹在 reset 之后、训练之前量; 遍历会推进 generator, 故量完立刻再 reset。
    border_sig = batch_order_signature(data["ltr"])
    reset_target_batch_order(data)
    # 数据装载消耗的随机数不影响初始化 —— 三组因此拿到逐值相同的初始权重。
    torch.manual_seed(int(seed))
    model = _build_model(cfg, src["n_features"], data["n_target"], data["device"])
    init_sha = _state_sha256(model)

    loaded: dict = {"source_checkpoint_loaded": False}
    if group in SOURCE_GROUPS:
        if not ckpt.exists():
            raise SystemExit(f"!! 缺 source checkpoint {ckpt} —— {group} 无法运行 "
                             f"(§2 禁止改它, 也禁止跳过该组)")
        missing, unexpected = model.load_pretrained(str(ckpt), data["device"])
        loaded = {"source_checkpoint_loaded": True,
                  "source_checkpoint": str(ckpt.relative_to(ROOT)),
                  "source_checkpoint_sha256":
                      hashlib.sha256(ckpt.read_bytes()).hexdigest(),
                  "n_missing_keys": len(list(missing)),
                  "n_unexpected_keys": len(list(unexpected))}

    huber = nn.HuberLoss(delta=float(cfg["loss"]["huber_delta"]))
    mse = nn.MSELoss()
    lam = (float(cfg["loss"]["lambda_hi"]), float(cfg["loss"]["lambda_mono"]),
           float(cfg["loss"]["lambda_smooth"]))
    tcfg = cfg.get("training", {})
    e = int(th["max_epochs"])
    kw = dict(post_eol_weight=float(tcfg.get("post_eol_weight", 1.0)),
              huber_delta=float(cfg["loss"]["huber_delta"]),
              capped_weight=float(tcfg.get("capped_weight", 1.0)),
              cap_eps=data["cap_eps"],
              early_stop_metric=th["early_stop_metric"],
              early_stop_patience=th["early_stop_patience"],
              censor_eta=float(data["censor_eta"]))
    tag = f"b6_n{n_event}_{group}_s{seed}"
    history: list = []
    budget: dict = {}
    stages: list[dict] = []

    if group in ("target_only", "source_finetune"):
        model.freeze_encoder(False)
        opt = _optimizer(cfg, model, th)
        best = _train_with_early_stop(model, data["ltr"], data["lva"], opt,
                                      data["device"], huber, mse, lam, e, tag,
                                      history=history, budget=budget, **kw)
        stages.append({"stage": "single", "best_val_es": float(best),
                       "encoder_frozen": False, "use_mmd": False})
    elif group == "source_mmd_finetune":
        # S2: 冻结 encoder, 只训 adapter + heads
        model.freeze_encoder(True)
        opt = _optimizer(cfg, model, th, trainable_only=True)
        h2: list = []
        v2 = _train_with_early_stop(model, data["ltr"], data["lva"], opt,
                                    data["device"], huber, mse, lam, e,
                                    f"{tag} S2", history=h2, budget=budget, **kw)
        stages.append({"stage": "S2", "best_val_es": float(v2),
                       "encoder_frozen": True, "use_mmd": False,
                       "history": summarize_history(h2, th["early_stop_metric"])})
        # S3: 解冻 + HI 分箱 MMD。memory bank 先清空, 否则 S2 的 z 会漏进来。
        reset_global_memory_bank()
        model.freeze_encoder(False)
        opt = _optimizer(cfg, model, th)
        v3 = _train_with_early_stop(model, data["ltr"], data["lva"], opt,
                                    data["device"], huber, mse, lam, e,
                                    f"{tag} S3", use_mmd=True,
                                    src_iter=cycle(src["loader"]),
                                    bins=src["hi_bins"],
                                    mmd_lambda=src["mmd_lambda"],
                                    history=history, budget=budget, **kw)
        stages.append({"stage": "S3", "best_val_es": float(v3),
                       "encoder_frozen": False, "use_mmd": True,
                       "mmd_lambda": float(src["mmd_lambda"]),
                       "mmd_lambda_frozen": True,
                       "hi_bins": [list(b) for b in src["hi_bins"]]})
        best = v3
    else:
        raise ValueError(f"§10 只允许 {TRAINED_GROUPS}, 收到 {group!r}")

    m = evaluate_b2(model, data["lte"], cfg, axis, data,
                    tag=f"n{n_event}_{group}_s{seed}")
    # catastrophic 阈值只能来自该档 target_only 的 **validation** 分布 (§14),
    # 故每组都额外在 val 上评一次, 由分析脚本按组挑取。
    mv = evaluate_b2(model, data["lva"], cfg, axis, data,
                     tag=f"n{n_event}_{group}_s{seed}_val")
    m["_raw_val"] = mv["_raw"]

    cpath = ckpt_dir / f"n{n_event}_{group}_s{seed}.pt"
    torch.save(model.state_dict(), cpath)
    m["_meta"] = {
        "group": group,
        "n_event_labeled": int(n_event),
        "seed": int(seed),
        "n_features": int(src["n_features"]),
        "n_features_source_side": int(src["n_features"]),
        "n_target": int(data["n_target"]),
        "input_schema": str(data["input_schema"]),
        "split_sha256": str(data["split_sha256"]),
        "subset_manifest_sha256": str(data["subset_manifest_sha256"]),
        "n_train": int(len(data["train_tids_used"])),
        "n_train_event": int(len(data["train_event_tids"])),
        "n_train_censored": int(len(data["train_censored_tids"])),
        "train_ids_signature": train_ids_signature(data),
        "batch_order_generator_seed": int(data["batch_order_generator_seed"]),
        "batch_order_signature": border_sig,
        "max_epochs": int(th["max_epochs"]),
        "early_stop_patience": int(th["early_stop_patience"]),
        "early_stop_metric": str(th["early_stop_metric"]),
        "optimizer": "Adam",
        "lr": float(cfg["transfer"]["finetune_lr"]),
        "weight_decay": float(th["weight_decay"]),
        "post_eol_weight": float(tcfg.get("post_eol_weight", 1.0)),
        "capped_weight": float(tcfg.get("capped_weight", 1.0)),
        "use_mmd": bool(group == "source_mmd_finetune"),
        "init_from_source_checkpoint": bool(group in SOURCE_GROUPS),
        "source_generator_seed": int(src["source_generator_seed"]),
        "rul_scale": float(data["rul_scale"]),
        "hyper": dict(th),
        "budget": dict(budget),
        "stages": stages,
        "best_epoch": int(budget.get("best_epoch", -1)),
        "val_metric": {"name": str(th["early_stop_metric"]),
                       "value": float(best),
                       "selection_basis": "validation_only"},
        "checkpoint_sha256": _state_sha256(model),
        "checkpoint_path": str(cpath.relative_to(ROOT)),
        "init_weights_sha256": init_sha,
        "val_info_macro_rmse": float(mv["info_macro_rmse"]),
        **loaded,
        "secs": round(time.time() - t0, 1),
    }
    print(f"   [n{n_event} {group} s{seed}] test info macro RMSE = "
          f"{m['info_macro_rmse']:.6f}  (val {mv['info_macro_rmse']:.6f}, "
          f"best_ep {m['_meta']['best_epoch']}, {m['_meta']['secs']}s)")
    return m


def eval_only_baselines(cfg: dict, data: dict, axis: dict, n_event: int,
                        seed: int, model_tids: np.ndarray,
                        dmg_cache: dict | None) -> dict:
    """§10 的两个只评估不训练的组, 与模型走同一 evaluator、同一 test 取点。"""
    out: dict[str, dict] = {}
    cval = const_mean_info(cfg, axis, data)
    pa, ta, ia = const_pred_on(axis, data["te"], cval, cfg, data)
    out["const_mean_info"] = evaluate_arrays_b2(
        pa, ta, ia, cfg, axis, data, tag=f"n{n_event}_const_mean_info_s{seed}")
    out["const_mean_info_value"] = float(cval)

    if dmg_cache is not None and "pred" in dmg_cache:
        pb, tb, ib = dmg_cache["pred"], dmg_cache["true"], dmg_cache["tids"]
    else:
        pb, tb, ib = damage_extrapolation(cfg, axis, data, data["te"])
        if dmg_cache is not None:
            dmg_cache.update({"pred": pb, "true": tb, "tids": ib})
    out["damage_extrapolation"] = evaluate_arrays_b2(
        pb, tb, ib, cfg, axis, data,
        tag=f"n{n_event}_damage_extrapolation_s{seed}")

    # 取点必须逐点一致, 否则指标不可比 —— 直接报错, 不静默对齐
    for nm, arr in (("const_mean_info", ia), ("damage_extrapolation", ib)):
        if not np.array_equal(np.asarray(arr), np.asarray(model_tids)):
            raise SystemExit(f"!! {nm} 与模型 test 取点不一致, 拒绝比较")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CFG_PATH)
    ap.add_argument("--fast", action="store_true",
                    help="冒烟: 只跑 1 epoch (结果不得入正式报告)")
    ap.add_argument("--levels", default="",
                    help="仅跑指定档 (逗号分隔), 留空 = 全部。用于分批执行, "
                         "不改变任何数值")
    a = ap.parse_args()

    cfg = load_cfg(a.config)
    b6 = cfg["b6"]
    if str(b6["label"]) != "FAILURE_LABEL_SCARCITY_FORMAL_MATRIX":
        raise SystemExit(f"!! label 必须是 FAILURE_LABEL_SCARCITY_FORMAL_MATRIX, "
                         f"得到 {b6['label']!r}")
    if not bool(b6["formal_stage"]):
        raise SystemExit("!! b6.formal_stage 必须为 true")
    for flag in ("forbid_new_transfer_methods", "forbid_new_mmd_lambda",
                 "forbid_mission_feature_arm"):
        if not bool(b6[flag]):
            raise SystemExit(f"!! {flag} 必须为 true (§10)")
    if not bool(b6["checkpoint"]["forbid_auto_retry"]):
        raise SystemExit("!! forbid_auto_retry 必须为 true (§12)")

    proto = _assert_protocol_frozen(cfg)
    schema = resolve_input_schema_b6(cfg)
    split = load_b6_split(cfg)
    mf = load_subset_manifest(cfg)
    th = _assert_frozen_hyper(cfg)
    seeds = _assert_formal_seeds(cfg)

    groups = list(b6["trained_groups"])
    if groups != list(TRAINED_GROUPS):
        raise SystemExit(f"!! §10 训练组必须恰为 {TRAINED_GROUPS}, 得到 {groups}")
    if sorted(b6["eval_only_groups"]) != sorted(EVAL_ONLY_GROUPS):
        raise SystemExit(f"!! §10 eval-only 组必须恰为 {EVAL_ONLY_GROUPS}")

    levels = [int(x) for x in proto["label_scarcity"]["levels"]]
    if a.levels:
        want = [int(x) for x in a.levels.split(",") if x.strip()]
        bad = sorted(set(want) - set(levels))
        if bad:
            raise SystemExit(f"!! --levels 含未登记档 {bad}")
        levels = [n for n in levels if n in set(want)]
    primary = int(proto["primary"]["n_event_labeled"])

    if a.fast:
        cfg["training"]["max_epochs"] = 1
        cfg["b2_frozen_training_expected"]["max_epochs"] = 1
        th = training_hyper(cfg)
        print("!! --fast: max_epochs=1, 结果只能用于冒烟, 不得入正式报告")

    ckpt = ROOT / cfg["paths"]["source_checkpoint"]
    ckpt_dir = ROOT / cfg["paths"]["ckpt_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    axis = b2_endpoint_axis(cfg)
    mult = float(b6["catastrophic"]["multiplier"])
    per_level_thr = bool(b6["catastrophic"]["per_level_threshold"])

    n_cells = len(levels) * len(groups) * len(seeds)
    print(f">> B6 正式矩阵: 档 {levels} × 组 {groups} × seed {seeds} "
          f"= {n_cells} 次训练")
    print(f"   + eval-only {list(EVAL_ONLY_GROUPS)} (逐档逐 seed 另评)")
    print(f"   PRIMARY = n_event_labeled {primary} (唯一允许出正式判定的档)")
    print(f"   {ARCH_NOTE}")
    print(f"   protocol_sha256 = {proto['protocol_sha256'][:16]}... (已冻结)")
    print(f"   subset_manifest_sha256 = {mf['subset_manifest_sha256'][:16]}... (已冻结)")

    by_level: dict[str, dict] = {}
    raw_store: dict[str, dict] = {}
    t_all = time.time()
    for n_event in levels:
        L = mf["levels"][str(n_event)]
        print(f"\n{'=' * 70}\n>> 标签档 n_event_labeled = {n_event} "
              f"[{L['role']}]  train = event {L['n_event_labeled']} + "
              f"censored {L['n_censored']} = {L['n_train_total']}\n{'=' * 70}")
        rows: list[dict] = []
        # damage_extrapolation 只依赖 test 与物理外推, 与 train 无关 -> 每档缓存一次。
        dmg_cache: dict = {}
        for seed in seeds:
            data = prepare_b6(cfg, split, mf, n_event, seed, schema,
                              with_test=True, verbose=True)
            src = prepare_source(cfg, seed, verbose=False)
            cap_eps = float(data["cap_eps"])
            row: dict = {"seed": int(seed), "n_event_labeled": int(n_event),
                         "rul_scale": float(data["rul_scale"]),
                         "n_train": int(len(data["train_tids_used"]))}
            for g in groups:
                m = run_group(cfg, data, src, n_event, seed, g, th, ckpt,
                              axis, ckpt_dir)
                raw_store[f"n{n_event}_{g}_s{seed}"] = {**m["_raw"],
                                                       "val": m["_raw_val"]}
                row[g] = strip_private(m)
                row[f"{g}_meta"] = m["_meta"]
                # §14 catastrophic 阈值: 只由该档 target_only 的 validation 分布定,
                # 同 cell 内三组共用 (否则各组自带阈值 = 各自放宽标准);
                # 跨档不共用 (每档有自己的 target_only, 用别档的尺子无意义)。
                if g == "target_only":
                    row["catastrophic_threshold"] = catastrophic_threshold(
                        m["_raw_val"]["pred"], m["_raw_val"]["true"],
                        m["_raw_val"]["tids"], cap_eps, mult)
            thr = row["catastrophic_threshold"]
            row["catastrophic"] = {}
            for g in groups:
                r = raw_store[f"n{n_event}_{g}_s{seed}"]
                row["catastrophic"][g] = catastrophic_rate(
                    r["pred"], r["true"], r["tids"], cap_eps, thr["threshold"])

            base = eval_only_baselines(
                cfg, data, axis, n_event, seed,
                raw_store[f"n{n_event}_target_only_s{seed}"]["tids"], dmg_cache)
            row["const_mean_info_value"] = base["const_mean_info_value"]
            for g in EVAL_ONLY_GROUPS:
                row[g] = strip_private(base[g])
                raw_store[f"n{n_event}_{g}_s{seed}"] = dict(base[g]["_raw"])
                row["catastrophic"][g] = catastrophic_rate(
                    base[g]["_raw"]["pred"], base[g]["_raw"]["true"],
                    base[g]["_raw"]["tids"], cap_eps, thr["threshold"])

            t0v = float(row["target_only"]["info_macro_rmse"])
            ftv = float(row["source_finetune"]["info_macro_rmse"])
            mdv = float(row["source_mmd_finetune"]["info_macro_rmse"])
            dmv = float(row["damage_extrapolation"]["info_macro_rmse"])
            cmv = float(row["const_mean_info"]["info_macro_rmse"])
            row["gains"] = {
                "target_only": t0v, "source_finetune": ftv,
                "source_mmd_finetune": mdv, "damage_extrapolation": dmv,
                "const_mean_info": cmv,
                "gain_ft": t0v - ftv, "gain_mmd": t0v - mdv,
                # §18: 与物理基线的差, 负数 = 学习模型不如物理外推
                "target_minus_damage": t0v - dmv,
                "ft_minus_damage": ftv - dmv,
                "mmd_minus_damage": mdv - dmv,
                # §15 条件 10: source 方法至少要打赢 const
                "ft_minus_const": ftv - cmv,
                "mmd_minus_const": mdv - cmv,
                "definition": ("gain_ft = RMSE_target_only - "
                               "RMSE_source_finetune; gain_mmd = "
                               "RMSE_target_only - RMSE_source_mmd_finetune; "
                               "正数 = source 方法更好"),
            }
            # §11 公平性自检: 同 cell 内三组的 IDs / batch 顺序 / 初始权重必须同一。
            sigs = {g: (row[f"{g}_meta"]["train_ids_signature"],
                        row[f"{g}_meta"]["batch_order_signature"],
                        row[f"{g}_meta"]["init_weights_sha256"]) for g in groups}
            uniq = set(sigs.values())
            row["fairness"] = {
                "same_train_ids": len({s[0] for s in sigs.values()}) == 1,
                "same_batch_order": len({s[1] for s in sigs.values()}) == 1,
                "same_init_weights": len({s[2] for s in sigs.values()}) == 1,
                "n_distinct_signature_tuples": len(uniq),
            }
            row["fairness"]["pass"] = bool(
                row["fairness"]["same_train_ids"]
                and row["fairness"]["same_batch_order"]
                and row["fairness"]["same_init_weights"])
            if not row["fairness"]["pass"]:
                raise SystemExit(
                    f"!! n={n_event} seed {seed} 三组配对被破坏 {sigs} —— {INVALID}")
            print(f"   >> n={n_event} seed {seed}: gain_ft {t0v - ftv:+.6f} / "
                  f"gain_mmd {t0v - mdv:+.6f}  (t0 {t0v:.6f} ft {ftv:.6f} "
                  f"mmd {mdv:.6f} damage {dmv:.6f})")
            rows.append(row)

        gf = [r["gains"]["gain_ft"] for r in rows]
        gm = [r["gains"]["gain_mmd"] for r in rows]
        by_level[str(n_event)] = {
            "n_event_labeled": int(n_event),
            "role": str(L["role"]),
            "is_primary": bool(n_event == primary),
            "train_composition": {"event": int(L["n_event_labeled"]),
                                  "censored": int(L["n_censored"]),
                                  "total": int(L["n_train_total"])},
            "train_event_tids": list(L["event_tids"]),
            "event_bin_counts": dict(L["event_bin_counts"]),
            "removed_event_tids": list(L["removed_event_tids"]),
            "per_seed": rows,
        }
        print(f">> [n={n_event}] gain_ft  {[round(x, 6) for x in gf]}  "
              f"mean {np.mean(gf):+.6f}  improve {sum(1 for x in gf if x > 0)}/{len(gf)}")
        print(f">> [n={n_event}] gain_mmd {[round(x, 6) for x in gm]}  "
              f"mean {np.mean(gm):+.6f}  improve {sum(1 for x in gm if x > 0)}/{len(gm)}")

    out = {
        "stage": "BASILISK_B6",
        "label": str(b6["label"]),
        "formal_stage": True,
        "last_stage_allowed_to_produce_core_numbers":
            bool(b6["last_stage_allowed_to_produce_core_numbers"]),
        "purpose": MATRIX_PURPOSE,
        "question": str(b6["question"]),
        "scarcity_axis": str(b6["scarcity_axis"]),
        "scarcity_axis_is_not": str(b6["scarcity_axis_is_not"]),
        "architecture_note": ARCH_NOTE,
        "pairing_note": PAIRING_NOTE,
        "subset_note": SUBSET_NOTE,
        "checkpoint_note": CHECKPOINT_NOTE,
        "frozen_facts": dict(b6["frozen_facts"]),
        "b5_must_not_be_overwritten": True,
        "config": a.config,
        "config_sha256": str(proto["config_sha256"]),
        "protocol_sha256": str(proto["protocol_sha256"]),
        "subset_manifest_sha256": str(mf["subset_manifest_sha256"]),
        "fast": bool(a.fast),
        "trained_groups": groups,
        "eval_only_groups": list(EVAL_ONLY_GROUPS),
        "excluded_oracle_methods": list(EXCLUDED_ORACLE_METHODS),
        "input_schema": schema,
        "n_features": int(b6["input_schema"]["n_features"]),
        "split_sha256": str(split["split_sha256"]),
        "split_n": {k: int(split["splits"][k]["n"])
                    for k in ("train", "val", "test")},
        "val_test_never_touched": True,
        "hi_key": str(cfg["transfer"]["target_hi_key"]),
        "label_levels": levels,
        "primary_level": primary,
        "secondary_levels": [int(x) for x in proto["secondary"]["levels"]],
        "formal_seeds": seeds,
        "forbidden_seeds": sorted(int(s) for s in b6["forbid_reused_seeds"]),
        "seed_role": str(b6["seed_role"]),
        "source_checkpoint": str(ckpt.relative_to(ROOT)),
        "source_checkpoint_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
        "mmd_lambda": float(cfg["transfer"]["mmd_lambda"]),
        "mmd_lambda_frozen": True,
        "catastrophic": {"multiplier": mult, "per_level_threshold": per_level_thr,
                         "threshold_source":
                             str(b6["catastrophic"]["threshold_source"])},
        "n_cells_trained": n_cells,
        "by_level": by_level,
        "secs_total": round(time.time() - t_all, 1),
    }
    mp = ROOT / cfg["paths"]["metrics_json"]
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(out, indent=2, ensure_ascii=False,
                             default=float) + "\n", encoding="utf-8")
    print(f"\n>> 已写 {mp.relative_to(ROOT)}  ({out['secs_total']}s)")

    npz = ROOT / cfg["paths"]["raw_npz"]
    flat: dict[str, np.ndarray] = {}
    for k, r in raw_store.items():
        for f in ("pred", "true", "tids", "tau", "lb"):
            flat[f"{k}__{f}"] = np.asarray(r[f])
            if "val" in r:
                flat[f"{k}__val_{f}"] = np.asarray(r["val"][f])
    np.savez_compressed(npz, **flat)
    print(f">> 已写 {npz.relative_to(ROOT)} ({len(raw_store)} 组预测)")
    print(">> Gate 判定在 final_transfer_verdict.py, 本脚本只出数字, 不下结论")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
