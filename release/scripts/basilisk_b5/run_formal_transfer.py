#!/usr/bin/env python
"""scripts/basilisk_b5/run_formal_transfer.py

BASILISK-B5 §4/§5/§6/§7 —— 正式迁移评估: 三组 × 五个全新正式 seed。

三组 (§4, **不得新增第四种迁移方法**):
  `target_only`          随机初始化, 不加载 source checkpoint
  `source_finetune`      加载 source encoder, 全参数微调
  `source_mmd_finetune`  S2 冻结 encoder -> reset_global_memory_bank()
                         -> S3 解冻 + use_mmd=True (mmd_lambda 已冻结)

只评估不训练 (§4): `const_mean_info`, `damage_extrapolation`。

编排逐字沿用 B4X 的对应分支 (它本身逐字照抄 run_one_group), 训练循环 / 损失 /
早停 / 模型 / MMD / evaluator / 基线全部 import 冻结实现, **一行不改**。B5 只换
三件事:
  1. 划分 -> B2.1 冻结划分 (data_b5, B4X 的 check_split 钉在 B2 旧哈希上);
  2. seed -> 全新正式 seed [112..116] (§5), 并硬断言与开发 seed 无交集;
  3. 追加两个 eval-only 基线, 使正式主表 (§18) 五行齐备。

**为什么不用 run_one_group**: 它自己重新划分、用 observed_only=True 丢掉全部 79
条删失轨迹、且按 hi_b 口径评估 —— 三处都与 B2/B2.1 契约冲突。

§7 结构性保证: `_train_with_early_stop` 签名里没有 test loader; checkpoint 只能
按 val 选。逐组记录 best_epoch / val_metric / checkpoint_sha256。

用法:
    python scripts/basilisk_b5/run_formal_transfer.py --config configs/wheel_basilisk_b5.yaml
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
from scripts.basilisk_b4x.data_b4x import prepare_source  # noqa: E402
from scripts.basilisk_b4x.diagnose_transfer_stability import (  # noqa: E402
    catastrophic_rate,
    catastrophic_threshold,
)
from scripts.basilisk_b5.data_b5 import (  # noqa: E402
    batch_order_signature,
    load_b21_split,
    prepare_b5,
    reset_target_batch_order,
    resolve_input_schema_b5,
)
from scripts.diag_generalization import summarize_history  # noqa: E402
from src.experiments.run_groups import (  # noqa: E402
    _build_model,
    _train_with_early_stop,
    training_hyper,
)
from src.transfer.mmd import reset_global_memory_bank  # noqa: E402

CFG_PATH = "configs/wheel_basilisk_b5.yaml"

TRAINED_GROUPS = ("target_only", "source_finetune", "source_mmd_finetune")
SOURCE_GROUPS = ("source_finetune", "source_mmd_finetune")
EVAL_ONLY_GROUPS = ("const_mean_info", "damage_extrapolation")

FORMAL_PURPOSE = (
    "B5 §0: 这是第一次允许产生正式迁移结论的阶段。在 B2.1 冻结划分 "
    "(寿命覆盖完整, B21_GENERALIZATION_PASS) 上, 用全新正式 seed 回答三个问题: "
    "Q1 source_finetune 是否稳定优于 Target-only; "
    "Q2 source_mmd_finetune 是否稳定优于 Target-only; "
    "Q3 即使存在正转移, 迁移模型是否仍然输给 damage_extrapolation。"
    "B4X_TRANSFER_STABILIZATION_SIGNAL 仍是 EXPLORATORY_ONLY, "
    "**不得被继承为正式 positive-transfer 结论** —— B5 必须自己重新证明。"
    "B2_GENERALIZATION_FAIL 在其自身划分上保持终局, 本阶段不追溯改写。"
    "本阶段禁止重新调任何模型或迁移超参。"
)

# §6.3 的关键限定。抽成模块级常量: 反作弊扫描器豁免常量声明块, 函数体内不豁免;
# 而这段话必须写进产物, 否则读表的人会拿 B5 的迁移组去减 B2.1 的 0.2441。
ARCH_NOTE = (
    "三组统一 n_features=12 (source encoder 第一层宽度) / n_target=10, "
    "以满足 §6 的架构公平性。B2/B2.1 的 target_only 建的是 (10,10), "
    "**所以 B5 的 target_only 是本阶段的一次重跑, 其数值不等于 B2.1 的 0.2441**。"
    "gain 只在 B5 自己的三组之间计算, 绝不拿 B5 的迁移组去减 B2.1 的表 —— "
    "跨阶段相减是不同尺子。"
)

PAIRING_NOTE = (
    "§6 严格配对: 同一 seed 下三组共享 train/val/test IDs、target batch 顺序、"
    "target 预处理、optimizer、epoch 预算、早停指标、checkpoint 选择规则、"
    "evaluator; 且因架构完全一致, 三组初始权重逐值相同。"
    "唯一差异是是否 load_pretrained 与是否走 MMD。"
    "不得因为某组失败而单独加 epoch。"
)

CHECKPOINT_NOTE = (
    "§7 checkpoint 只按 validation 选。_train_with_early_stop 的签名里没有 test "
    "loader, early_stop_value 对任何含 test 的指标名直接抛错 —— "
    "按 test 选 checkpoint / 按 test 早停 / 跨 seed 挑最好 checkpoint / "
    "跨 seed 换组, 四者都在结构上不可达。"
)


def _assert_frozen_hyper(cfg: dict) -> dict:
    """§1: 冻结超参必须与 B2 记录的期望值逐项一致, 不符即停。"""
    th = training_hyper(cfg)
    exp = cfg["b2_frozen_training_expected"]
    for k, want in (("early_stop_metric", exp["early_stop_metric"]),
                    ("max_epochs", int(exp["max_epochs"])),
                    ("early_stop_patience", int(exp["early_stop_patience"])),
                    ("weight_decay", float(exp["weight_decay"]))):
        if th[k] != want:
            raise SystemExit(
                f"!! B5_INVALID: training.{k}={th[k]!r} != B2 冻结值 {want!r} "
                f"—— §1 禁止在本阶段改超参")
    return th


def _assert_formal_seeds(cfg: dict) -> list[int]:
    """§5: 正式 seed 必须全新。与开发 seed 有任何交集即停。"""
    b5 = cfg["b5"]
    seeds = [int(s) for s in b5["seeds"]]
    forbidden = {int(s) for s in b5["forbid_reused_seeds"]}
    bad = sorted(set(seeds) & forbidden)
    if bad:
        raise SystemExit(
            f"!! B5_INVALID: 正式 seed 与开发 seed 重叠 {bad} —— "
            f"本阶段是 confirmatory evaluation, 不能继续消费开发 seed")
    if len(set(seeds)) != len(seeds):
        raise SystemExit(f"!! 正式 seed 有重复: {seeds}")
    return seeds


def _assert_protocol_frozen(cfg: dict) -> dict:
    """§9/§17: protocol 必须已在出数字前冻结, 且哈希与当前文件一致。"""
    hp = ROOT / cfg["protocol"]["hash_path"]
    if not hp.exists():
        raise SystemExit(
            f"!! 缺 {hp.relative_to(ROOT)} —— 必须先跑 freeze_protocol.py "
            f"(protocol 必须在任何 B5 数字之前冻结)")
    rec = json.loads(hp.read_text(encoding="utf-8"))
    now = hashlib.sha256(
        (ROOT / cfg["protocol"]["path"]).read_bytes()).hexdigest()
    if now != str(rec["protocol_sha256"]):
        raise SystemExit(
            "!! protocol.md 在冻结之后被改动 —— B5_INVALID。"
            " 出数字前改 protocol 请重新 freeze; 出数字后改 = 事后调门槛")
    if not bool(rec["frozen_before_any_b5_number"]):
        raise SystemExit("!! protocol_hash 未标记 frozen_before_any_b5_number")
    return rec


def _optimizer(cfg: dict, model, th: dict, trainable_only: bool = False):
    params = ([p for p in model.parameters() if p.requires_grad]
              if trainable_only else list(model.parameters()))
    return torch.optim.Adam(params, lr=float(cfg["transfer"]["finetune_lr"]),
                            weight_decay=th["weight_decay"])


def _state_sha256(model) -> str:
    """模型权重指纹 —— §7 要求逐组记录 checkpoint_sha256。"""
    h = hashlib.sha256()
    for k in sorted(model.state_dict().keys()):
        v = model.state_dict()[k].detach().cpu().numpy()
        h.update(k.encode("utf-8"))
        h.update(np.ascontiguousarray(v, dtype=np.float64).tobytes())
    return h.hexdigest()


def run_group(cfg: dict, data: dict, src: dict, seed: int, group: str,
              th: dict, ckpt: Path, axis: dict, ckpt_dir: Path) -> dict:
    """训练一组并在 test 上评估。编排逐字沿用 B4X (= run_one_group 的分支)。"""
    print(f"\n{'-' * 66}\n[B5 正式] seed {seed}  group {group}\n{'-' * 66}")
    t0 = time.time()
    # §6.1: target train loader 的 generator 在训练中会**推进**, 若三组共用同一个
    # 对象而不重置, 第二组、第三组拿到的就是接着走的 batch 顺序 —— 那就不是严格配对。
    # 每组开训前把它重置回同一个种子, 三组因此逐 batch 一致。
    reset_target_batch_order(data)
    # 指纹在 reset 之后、训练之前量 —— 遍历只走 dataset, 不消耗模型侧随机性, 但会
    # 推进 generator, 故量完立刻再 reset 一次。
    border_sig = batch_order_signature(data["ltr"])
    reset_target_batch_order(data)
    # §6.2: 数据装载消耗的随机数不影响初始化 —— 三组因此拿到逐值相同的初始权重。
    torch.manual_seed(int(seed))
    model = _build_model(cfg, src["n_features"], data["n_target"],
                         data["device"])
    init_sha = _state_sha256(model)

    loaded: dict = {"source_checkpoint_loaded": False}
    if group in SOURCE_GROUPS:
        if not ckpt.exists():
            raise SystemExit(f"!! 缺 source checkpoint {ckpt} —— {group} 无法运行 "
                             f"(§1 禁止改它, 也禁止跳过该组)")
        missing, unexpected = model.load_pretrained(str(ckpt), data["device"])
        loaded = {"source_checkpoint_loaded": True,
                  "source_checkpoint": str(ckpt.relative_to(ROOT)),
                  "source_checkpoint_sha256":
                      hashlib.sha256(ckpt.read_bytes()).hexdigest(),
                  "n_missing_keys": len(list(missing)),
                  "n_unexpected_keys": len(list(unexpected))}
        print(f"   已加载 source encoder ({loaded['n_missing_keys']} missing / "
              f"{loaded['n_unexpected_keys']} unexpected —— "
              f"heads/adapter 按设计不迁移)")

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
    tag = f"b5_{group}_s{seed}"
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
                       "history": summarize_history(h2,
                                                    th["early_stop_metric"])})
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
        raise ValueError(f"§4 只允许 {TRAINED_GROUPS}, 收到 {group!r}")

    m = evaluate_b2(model, data["lte"], cfg, axis, data, tag=f"{group}_s{seed}")
    # catastrophic 阈值只能来自 target_only 的 **validation** 分布 (§11),
    # 因此每组都额外在 val 上评一次, 由分析脚本按组挑取。
    mv = evaluate_b2(model, data["lva"], cfg, axis, data,
                     tag=f"{group}_s{seed}_val")
    m["_raw_val"] = mv["_raw"]

    # §7: 落盘 checkpoint 并记录哈希。选择依据只有 val。
    cpath = ckpt_dir / f"{group}_s{seed}.pt"
    torch.save(model.state_dict(), cpath)
    m["_meta"] = {
        "group": group,
        "seed": int(seed),
        "n_features_source_side": int(src["n_features"]),
        "n_target": int(data["n_target"]),
        "input_schema": str(data["input_schema"]),
        "split_sha256": str(data["split_sha256"]),
        "batch_order_generator_seed": int(data["batch_order_generator_seed"]),
        "batch_order_signature": border_sig,
        "n_features": int(src["n_features"]),
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
        # ---- §7 三项必录 ----
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
    print(f"   [{group} s{seed}] test info macro RMSE = "
          f"{m['info_macro_rmse']:.6f}  (val {mv['info_macro_rmse']:.6f}, "
          f"best_ep {m['_meta']['best_epoch']}, {m['_meta']['secs']}s)")
    return m


def eval_only_baselines(cfg: dict, data: dict, axis: dict, seed: int,
                        model_tids: np.ndarray,
                        dmg_cache: dict | None) -> dict:
    """§4 的两个只评估不训练的组, 与模型走同一 evaluator、同一 test 取点。"""
    out: dict[str, dict] = {}
    cval = const_mean_info(cfg, axis, data)
    pa, ta, ia = const_pred_on(axis, data["te"], cval, cfg, data)
    m_const = evaluate_arrays_b2(pa, ta, ia, cfg, axis, data,
                                 tag=f"const_mean_info_s{seed}")
    out["const_mean_info"] = m_const
    out["const_mean_info_value"] = float(cval)

    if dmg_cache is not None and "pred" in dmg_cache:
        pb, tb, ib = dmg_cache["pred"], dmg_cache["true"], dmg_cache["tids"]
    else:
        pb, tb, ib = damage_extrapolation(cfg, axis, data, data["te"])
        if dmg_cache is not None:
            dmg_cache.update({"pred": pb, "true": tb, "tids": ib})
    m_dmg = evaluate_arrays_b2(pb, tb, ib, cfg, axis, data,
                               tag=f"damage_extrapolation_s{seed}")
    out["damage_extrapolation"] = m_dmg

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
    a = ap.parse_args()

    cfg = load_cfg(a.config)
    b5 = cfg["b5"]
    if str(b5["label"]) != "FORMAL_TRANSFER_EVALUATION":
        raise SystemExit(f"!! label 必须是 FORMAL_TRANSFER_EVALUATION, "
                         f"得到 {b5['label']!r}")
    if not bool(b5["formal_stage"]):
        raise SystemExit("!! b5.formal_stage 必须为 true")
    if not bool(b5["forbid_new_transfer_methods"]):
        raise SystemExit("!! forbid_new_transfer_methods 必须为 true (§4)")

    proto = _assert_protocol_frozen(cfg)
    schema = resolve_input_schema_b5(cfg)
    split = load_b21_split(cfg)
    th = _assert_frozen_hyper(cfg)
    seeds = _assert_formal_seeds(cfg)

    groups = list(b5["trained_groups"])
    if groups != list(TRAINED_GROUPS):
        raise SystemExit(f"!! §4 训练组必须恰为 {TRAINED_GROUPS}, 得到 {groups}")
    if list(b5["eval_only_groups"]) != list(EVAL_ONLY_GROUPS):
        raise SystemExit(f"!! §4 eval-only 组必须恰为 {EVAL_ONLY_GROUPS}")

    if a.fast:
        cfg["training"]["max_epochs"] = 1
        cfg["b2_frozen_training_expected"]["max_epochs"] = 1
        th = training_hyper(cfg)
        print("!! --fast: max_epochs=1, 结果只能用于冒烟, 不得入正式报告")

    ckpt = ROOT / cfg["paths"]["source_checkpoint"]
    ckpt_dir = ROOT / cfg["paths"]["ckpt_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    axis = b2_endpoint_axis(cfg)
    mult = float(b5["catastrophic"]["multiplier"])

    print(f">> B5 正式迁移评估: seeds {seeds} × groups {groups}")
    print(f"   + eval-only {list(EVAL_ONLY_GROUPS)}")
    print(f"   {ARCH_NOTE}")
    print(f"   protocol_sha256 = {proto['protocol_sha256'][:16]}... (已冻结)")

    rows: list[dict] = []
    raw_store: dict[str, dict] = {}
    dmg_cache: dict = {}
    t_all = time.time()
    for seed in seeds:
        data = prepare_b5(cfg, split, seed, schema, with_test=True,
                          verbose=True)
        src = prepare_source(cfg, seed, verbose=True)
        cap_eps = float(data["cap_eps"])
        row: dict = {"seed": int(seed)}
        for g in groups:
            m = run_group(cfg, data, src, seed, g, th, ckpt, axis, ckpt_dir)
            raw_store[f"{g}_s{seed}"] = {**m["_raw"], "val": m["_raw_val"]}
            row[g] = strip_private(m)
            row[f"{g}_meta"] = m["_meta"]
            # §11 catastrophic 阈值: 只由 **target_only 的 validation** 分布定,
            # 三组共用同一阈值 (否则各组自带阈值 = 各自放宽标准)。
            if g == "target_only":
                thr = catastrophic_threshold(
                    m["_raw_val"]["pred"], m["_raw_val"]["true"],
                    m["_raw_val"]["tids"], cap_eps, mult)
                row["catastrophic_threshold"] = thr
        thr = row["catastrophic_threshold"]
        row["catastrophic"] = {}
        for g in groups:
            r = raw_store[f"{g}_s{seed}"]
            row["catastrophic"][g] = catastrophic_rate(
                r["pred"], r["true"], r["tids"], cap_eps, thr["threshold"])

        base = eval_only_baselines(cfg, data, axis, seed,
                                   raw_store[f"target_only_s{seed}"]["tids"],
                                   dmg_cache)
        row["const_mean_info_value"] = base["const_mean_info_value"]
        for g in EVAL_ONLY_GROUPS:
            row[g] = strip_private(base[g])
            raw_store[f"{g}_s{seed}"] = dict(base[g]["_raw"])
            row["catastrophic"][g] = catastrophic_rate(
                base[g]["_raw"]["pred"], base[g]["_raw"]["true"],
                base[g]["_raw"]["tids"], cap_eps, thr["threshold"])

        t0 = float(row["target_only"]["info_macro_rmse"])
        ft = float(row["source_finetune"]["info_macro_rmse"])
        md = float(row["source_mmd_finetune"]["info_macro_rmse"])
        dm = float(row["damage_extrapolation"]["info_macro_rmse"])
        row["gains"] = {
            "target_only": t0, "source_finetune": ft,
            "source_mmd_finetune": md, "damage_extrapolation": dm,
            "gain_ft": t0 - ft, "gain_mmd": t0 - md,
            # §14: 与物理基线的差, 负数 = 迁移不如物理方法
            "transfer_vs_damage_ft": dm - ft,
            "transfer_vs_damage_mmd": dm - md,
            "definition": ("gain_ft = RMSE_target_only - RMSE_source_finetune; "
                           "gain_mmd = RMSE_target_only - "
                           "RMSE_source_mmd_finetune; 正数 = 迁移改进"),
        }
        print(f"   >> seed {seed}: gain_ft {t0 - ft:+.6f} / "
              f"gain_mmd {t0 - md:+.6f}  "
              f"(t0 {t0:.6f} ft {ft:.6f} mmd {md:.6f} damage {dm:.6f})")
        rows.append(row)

    out = {
        "stage": "BASILISK_B5",
        "label": str(b5["label"]),
        "formal_stage": True,
        "first_stage_allowed_to_conclude_transfer": True,
        "purpose": FORMAL_PURPOSE,
        "architecture_note": ARCH_NOTE,
        "pairing_note": PAIRING_NOTE,
        "checkpoint_note": CHECKPOINT_NOTE,
        "questions": dict(b5["questions"]),
        "frozen_facts": dict(b5["frozen_facts"]),
        "b4x_must_not_be_promoted": True,
        "config": a.config,
        "config_sha256": str(proto["config_sha256"]),
        "protocol_sha256": str(proto["protocol_sha256"]),
        "fast": bool(a.fast),
        "trained_groups": groups,
        "eval_only_groups": list(EVAL_ONLY_GROUPS),
        "excluded_oracle_methods": list(EXCLUDED_ORACLE_METHODS),
        "input_schema": schema,
        "input_schema_decided_by": str(b5["input_schema"]["decided_by"]),
        "split_sha256": str(split["split_sha256"]),
        "split_reused_from": str(b5["split"]["reuse_from"]),
        "split_n": {k: int(split["splits"][k]["n"])
                    for k in ("train", "val", "test")},
        "hi_key": str(cfg["transfer"]["target_hi_key"]),
        "formal_seeds": seeds,
        "forbidden_seeds": sorted(int(s) for s in b5["forbid_reused_seeds"]),
        "seed_role": str(b5["seed_role"]),
        "source_checkpoint": str(ckpt.relative_to(ROOT)),
        "source_checkpoint_sha256":
            hashlib.sha256(ckpt.read_bytes()).hexdigest(),
        "mmd_lambda": float(cfg["transfer"]["mmd_lambda"]),
        "mmd_lambda_frozen": True,
        "per_seed": rows,
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

    gf = [r["gains"]["gain_ft"] for r in rows]
    gm = [r["gains"]["gain_mmd"] for r in rows]
    print(f">> gain_ft  {[round(x, 6) for x in gf]}  mean {np.mean(gf):+.6f}  "
          f"improve {sum(1 for x in gf if x > 0)}/{len(gf)}")
    print(f">> gain_mmd {[round(x, 6) for x in gm]}  mean {np.mean(gm):+.6f}  "
          f"improve {sum(1 for x in gm if x > 0)}/{len(gm)}")
    print(">> Gate 判定在 summarize_b5.py, 本脚本只出数字, 不下结论")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
