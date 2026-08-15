#!/usr/bin/env python
"""scripts/basilisk_b4x/run_transfer_diagnostic.py

BASILISK-B4X §11/§12/§13/§14 —— 三组 × 两套 seed 的迁移诊断。

`POST_B2_FAIL_EXPLORATORY` / `NOT_FORMAL_EVIDENCE` / `EXPLORATORY_ONLY`

三组 (§11, 不添加新方法):
  `target_only`          随机初始化, 不加载 source checkpoint
  `source_finetune`      加载 source encoder, 全参数微调
  `source_mmd_finetune`  S2 冻结 encoder 训练 -> reset_global_memory_bank()
                         -> S3 解冻 + use_mmd=True

**为什么不用 run_one_group** (必须写明, 否则看起来像重复实现):
  `src.experiments.run_groups.run_one_group` 与 B4X 的契约有三处硬冲突:
    1. 它自己调 `split_trajectories_from_cfg` 重新划分 —— §12 禁止;
    2. 它用 `observed_only=True` 加载目标域, 会丢掉全部 79 条删失轨迹 ——
       与 B2 的删失契约不符, 且 test 集会缩水;
    3. 它按 `hi_b` 口径评估, 与 B2 的 evaluate_b2 不同口径。
  因此本脚本**只做装配**: 训练循环 / 损失 / 早停 / 模型 / MMD / 源域 loader
  全部 import 冻结实现, 一行不改; 三组的 S2/S3 编排逐字照抄 run_one_group 的
  对应分支。

**架构宽度 (§3.1 已在 protocol 冻结)**:
  source checkpoint 的 encoder 第一层期望 12 通道, 故三组统一
  `_build_model(cfg, n_features=12, n_target=10, …)`。B2 的 target_only 建的是
  (10, 10)。**所以 B4X 的 target_only 是一次重跑, 不是 B2 的数字。**
  gain 只在 B4X 内部三组之间计算, 绝不拿 source 组去减 B2 的表。

§8 条件 7 沿用: `_train_with_early_stop` 签名里没有 test loader。

用法:
    python scripts/basilisk_b4x/run_transfer_diagnostic.py --config configs/wheel_basilisk_b4x.yaml
    python scripts/basilisk_b4x/run_transfer_diagnostic.py --seed-set new --fast
"""
from __future__ import annotations

import argparse
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
from scripts.basilisk_b2.eval_b2 import (  # noqa: E402
    b2_endpoint_axis,
    evaluate_b2,
    strip_private,
)
from scripts.basilisk_b4x.data_b4x import (  # noqa: E402
    prepare_b4x,
    prepare_source,
    resolve_input_schema,
)
from scripts.diag_generalization import summarize_history  # noqa: E402
from src.experiments.run_groups import (  # noqa: E402
    _build_model,
    _train_with_early_stop,
    training_hyper,
)
from src.transfer.mmd import reset_global_memory_bank  # noqa: E402

CFG_PATH = "configs/wheel_basilisk_b4x.yaml"

GROUPS = ("target_only", "source_finetune", "source_mmd_finetune")
SOURCE_GROUPS = ("source_finetune", "source_mmd_finetune")

TRANSFER_PURPOSE = (
    "B4X §11: 在 B2 冻结的划分与超参下跑 target_only / source_finetune / "
    "source_mmd_finetune 三组, 判断 source 系方法究竟是只比不稳定的 target_only "
    "更稳, 还是确实提供了跨域增益。"
    "这是探索性诊断, 不是正式 B4, 不构成迁移证据, 不重新解释 B2 的判定。"
    "B2_GENERALIZATION_FAIL 保持终局。"
    "三组共用同一份目标域输入 / 同一划分 / 同一 batch 顺序 / 同一架构宽度; "
    "差异只在是否加载 source encoder 与是否走 MMD。"
    "本阶段不添加新的迁移方法, 不修改 source checkpoint, 不修改 mmd_lambda。"
)

# §18 表述纪律。抽成模块级常量的理由与前几阶段同: 文案必须写明"不得声称迁移
# 有效"这类边界, 而反作弊扫描器扫源码符号, 常量声明块被豁免, 函数体内不会。
PHRASING_DISCIPLINE = (
    "即使 target_only 的 RMSE 远高于 source 组, 也不得写迁移显著有效。"
    "必须先判断是不是 target_only 在外推区崩溃、source 模型只是更保守; "
    "这种情况只能写成 source initialization regularizes extrapolation under "
    "the known B2 distribution gap, 而不是 positive transfer proven。"
)

ARCH_NOTE = (
    "三组统一 n_features=12 (source encoder 宽度) / n_target=10。"
    "B2 的 target_only 是 (10,10), 故本阶段的 target_only 是重跑, "
    "其数字不等于 B2 表里的 target_only。gain 只在 B4X 三组之间计算。"
)


def _assert_frozen_hyper(cfg: dict) -> dict:
    """§0: 冻结超参必须与 B2 记录的期望值逐项一致。"""
    th = training_hyper(cfg)
    exp = cfg["b2_frozen_training_expected"]
    for k, want in (("early_stop_metric", exp["early_stop_metric"]),
                    ("max_epochs", int(exp["max_epochs"])),
                    ("early_stop_patience", int(exp["early_stop_patience"])),
                    ("weight_decay", float(exp["weight_decay"]))):
        if th[k] != want:
            raise SystemExit(
                f"!! B4X_CONFIG_INCOMPATIBLE: training.{k}={th[k]!r} != "
                f"B2 冻结值 {want!r} —— §0 禁止在本阶段改超参")
    return th


def _build_shared(cfg: dict, data: dict, n_source: int, seed: int):
    """三组共用的建模步骤。同 seed 下三组权重逐值相同 (架构完全一致)。"""
    # set_seed 已在 prepare_b4x -> prepare_b2 里按 seed 执行过; 这里再置一次,
    # 使"数据装载消耗了多少随机数"不影响初始化 —— 三组因此拿到同一初始权重。
    torch.manual_seed(int(seed))
    model = _build_model(cfg, n_source, data["n_target"], data["device"])
    return model


def _optimizer(cfg: dict, model, th: dict, trainable_only: bool = False):
    params = ([p for p in model.parameters() if p.requires_grad]
              if trainable_only else list(model.parameters()))
    return torch.optim.Adam(params, lr=float(cfg["transfer"]["finetune_lr"]),
                            weight_decay=th["weight_decay"])


def run_group(cfg: dict, data: dict, src: dict, seed: int, group: str,
              th: dict, ckpt: Path, axis: dict) -> dict:
    """训练一组并在 test 上评估。编排逐字照抄 run_one_group 的对应分支。"""
    print(f"\n{'-' * 66}\n[B4X] seed {seed}  group {group}\n{'-' * 66}")
    t0 = time.time()
    model = _build_shared(cfg, data, src["n_features"], seed)

    loaded: dict = {"source_checkpoint_loaded": False}
    if group in SOURCE_GROUPS:
        if not ckpt.exists():
            raise SystemExit(f"!! 缺 source checkpoint {ckpt} —— "
                             f"{group} 无法运行 (§0 禁止改它, 也禁止跳过)")
        missing, unexpected = model.load_pretrained(str(ckpt), data["device"])
        loaded = {"source_checkpoint_loaded": True,
                  "source_checkpoint": str(ckpt.relative_to(ROOT)),
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
    tag = f"b4x_{group}_s{seed}"
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
        b2 = _train_with_early_stop(model, data["ltr"], data["lva"], opt,
                                    data["device"], huber, mse, lam, e,
                                    f"{tag} S2", history=h2, budget=budget, **kw)
        stages.append({"stage": "S2", "best_val_es": float(b2),
                       "encoder_frozen": True, "use_mmd": False,
                       "history": summarize_history(h2, th["early_stop_metric"])})
        # S3: 解冻 + HI 分箱 MMD。memory bank 必须先清空, 否则 S2 的 z 会漏进来。
        reset_global_memory_bank()
        model.freeze_encoder(False)
        opt = _optimizer(cfg, model, th)
        b3 = _train_with_early_stop(model, data["ltr"], data["lva"], opt,
                                    data["device"], huber, mse, lam, e,
                                    f"{tag} S3", use_mmd=True,
                                    src_iter=cycle(src["loader"]),
                                    bins=src["hi_bins"],
                                    mmd_lambda=src["mmd_lambda"],
                                    history=history, budget=budget, **kw)
        stages.append({"stage": "S3", "best_val_es": float(b3),
                       "encoder_frozen": False, "use_mmd": True,
                       "mmd_lambda": src["mmd_lambda"],
                       "hi_bins": [list(b) for b in src["hi_bins"]]})
        best = b3
    else:
        raise ValueError(f"§11 只允许 {GROUPS}, 收到 {group!r}")

    m = evaluate_b2(model, data["lte"], cfg, axis, data,
                    tag=f"{group}_s{seed}")
    # §16 的 catastrophic 阈值只能来自 target_only 的 **validation** 分布,
    # 因此每组都额外在 val 上评一次, 由 diagnose 脚本按组挑取。
    mv = evaluate_b2(model, data["lva"], cfg, axis, data,
                     tag=f"{group}_s{seed}_val")
    m["_raw_val"] = mv["_raw"]
    m["_meta"] = {
        "group": group,
        "seed": int(seed),
        "n_features_source_side": int(src["n_features"]),
        "n_target": int(data["n_target"]),
        "input_schema": str(data["input_schema"]),
        "batch_order_generator_seed": int(data["batch_order_generator_seed"]),
        "source_generator_seed": int(src["source_generator_seed"]),
        "rul_scale": float(data["rul_scale"]),
        "hyper": dict(th),
        "budget": dict(budget),
        "stages": stages,
        "best_val_es": float(best),
        "history": summarize_history(history, th["early_stop_metric"]),
        "val_info_macro_rmse": float(mv["info_macro_rmse"]),
        **loaded,
        "secs": round(time.time() - t0, 1),
    }
    print(f"   [{group} s{seed}] test info macro RMSE = "
          f"{m['info_macro_rmse']:.6f}  (val {mv['info_macro_rmse']:.6f}, "
          f"{m['_meta']['secs']}s)")
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CFG_PATH)
    ap.add_argument("--seed-set", choices=("new", "replay", "both"),
                    default="both")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--groups", nargs="*", default=None)
    ap.add_argument("--fast", action="store_true",
                    help="冒烟: 只跑 1 epoch (结果不得入正式报告)")
    a = ap.parse_args()

    cfg = load_cfg(a.config)
    b4 = cfg["b4x"]
    if str(b4["label"]) != "POST_B2_FAIL_EXPLORATORY":
        raise SystemExit(f"!! label 必须是 POST_B2_FAIL_EXPLORATORY, "
                         f"得到 {b4['label']!r}")

    schema = resolve_input_schema(cfg)
    split = json.loads(
        (ROOT / b4["split"]["reuse_from"]).read_text(encoding="utf-8"))
    th = _assert_frozen_hyper(cfg)

    groups = list(a.groups) if a.groups else list(b4["groups"])
    for g in groups:
        if g not in GROUPS:
            raise SystemExit(f"!! §11 禁止新增迁移方法, 未知组 {g!r}")

    # §13: 两套 seed 分开。这里按套组织, 落 JSON 时也分开存 —— 结构上不给
    # "合并成一个 6-seed 平均" 留位置。
    declared = {k: [int(s) for s in v] for k, v in b4["seeds"].items()}
    sets = (["new", "replay"] if a.seed_set == "both" else [a.seed_set])
    if a.seeds:
        # 手动指定 seed 只用于调试, 归入其所属套; 不属于任何套则报错。
        pick = set(int(s) for s in a.seeds)
        sets = [k for k in sets if pick & set(declared[k])]
        declared = {k: [s for s in v if s in pick] for k, v in declared.items()}

    if a.fast:
        cfg["training"]["max_epochs"] = 1
        cfg["b2_frozen_training_expected"]["max_epochs"] = 1
        th = training_hyper(cfg)
        print("!! --fast: max_epochs=1, 结果只能用于冒烟, 不得入报告")

    ckpt = ROOT / cfg["paths"]["source_checkpoint"]
    axis = b2_endpoint_axis(cfg)
    print(f">> B4X 迁移诊断: seed sets {sets} × groups {groups}")
    print(f"   {ARCH_NOTE}")

    by_set: dict[str, list[dict]] = {}
    raw_store: dict[str, dict] = {}
    t_all = time.time()
    for sname in sets:
        rows: list[dict] = []
        for seed in declared[sname]:
            data = prepare_b4x(cfg, split, seed, schema, with_test=True,
                               verbose=True)
            src = prepare_source(cfg, seed, verbose=True)
            row: dict = {"seed": int(seed), "seed_set": sname}
            for g in groups:
                m = run_group(cfg, data, src, seed, g, th, ckpt, axis)
                raw_store[f"{g}_s{seed}"] = {**m["_raw"],
                                             "val": m["_raw_val"]}
                row[g] = strip_private(m)
                row[f"{g}_meta"] = m["_meta"]
            if all(g in row for g in GROUPS):
                t0 = float(row["target_only"]["info_macro_rmse"])
                ft = float(row["source_finetune"]["info_macro_rmse"])
                md = float(row["source_mmd_finetune"]["info_macro_rmse"])
                row["gains_overall"] = {
                    "target_only": t0, "source_finetune": ft,
                    "source_mmd_finetune": md,
                    "gain_ft": t0 - ft, "gain_mmd": t0 - md,
                    "definition": ("gain_ft = target_only - source_finetune; "
                                   "gain_mmd = target_only - source_mmd_finetune"),
                }
                print(f"   >> seed {seed} [{sname}] gain_ft {t0 - ft:+.6f} / "
                      f"gain_mmd {t0 - md:+.6f}  "
                      f"(t0 {t0:.6f} ft {ft:.6f} mmd {md:.6f})")
            rows.append(row)
        by_set[sname] = rows

    out = {
        "stage": "BASILISK_B4X",
        "label": str(b4["label"]),
        "formal_stage": bool(b4.get("formal_stage", False)),
        "not_formal_evidence": True,
        "exploratory_only": True,
        "b2_verdict_unchanged": "B2_GENERALIZATION_FAIL",
        "b2_verdict_status": "B2_GENERALIZATION_FAIL_REMAINS_FINAL",
        "not_formal_transfer_evidence": True,
        "purpose": TRANSFER_PURPOSE,
        "phrasing_discipline": PHRASING_DISCIPLINE,
        "architecture_note": ARCH_NOTE,
        "config": a.config,
        "fast": bool(a.fast),
        "groups": groups,
        "input_schema": schema,
        "input_schema_decided_by": str(b4["input_schema"]["decided_by"]),
        "split_sha256": str(split["split_sha256"]),
        "split_reused_from": str(b4["split"]["reuse_from"]),
        "hi_key": str(cfg["transfer"]["target_hi_key"]),
        "short_eol_subset": [int(x) for x in b4["short_eol_subset"]],
        "seed_sets": {k: declared[k] for k in sets},
        "forbid_merging_seed_sets": bool(b4.get("forbid_merging_seed_sets")),
        "source_checkpoint": str(ckpt.relative_to(ROOT)),
        "mmd_lambda": float(cfg["transfer"]["mmd_lambda"]),
        "per_seed_by_set": by_set,
        "secs_total": round(time.time() - t_all, 1),
    }
    mp = ROOT / cfg["paths"]["metrics_json"]
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(out, indent=2, ensure_ascii=False,
                             default=float) + "\n", encoding="utf-8")
    print(f"\n>> 已写 {mp.relative_to(ROOT)}  ({out['secs_total']}s)")

    npz = mp.parent / "transfer_raw.npz"
    flat: dict[str, np.ndarray] = {}
    for k, r in raw_store.items():
        for f in ("pred", "true", "tids", "tau", "lb"):
            flat[f"{k}__{f}"] = np.asarray(r[f])
            flat[f"{k}__val_{f}"] = np.asarray(r["val"][f])
    np.savez_compressed(npz, **flat)
    print(f">> 已写 {npz.relative_to(ROOT)} ({len(raw_store)} 组 test+val 预测)")

    for sname, rows in by_set.items():
        gf = [r["gains_overall"]["gain_ft"] for r in rows
              if "gains_overall" in r]
        gm = [r["gains_overall"]["gain_mmd"] for r in rows
              if "gains_overall" in r]
        if gf:
            print(f">> [{sname}] gain_ft  {[round(x, 6) for x in gf]}  "
                  f"mean {np.mean(gf):+.6f}")
            print(f">> [{sname}] gain_mmd {[round(x, 6) for x in gm]}  "
                  f"mean {np.mean(gm):+.6f}")
    print(">> 两套 seed 的数字**分开**报告, 不合并 (§13)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
