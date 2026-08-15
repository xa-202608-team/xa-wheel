#!/usr/bin/env python
"""scripts/basilisk_b4x/data_b4x.py

BASILISK-B4X §10/§12 —— 三组共用的目标域 / 源域 loader。

`POST_B2_FAIL_EXPLORATORY` / `NOT_FORMAL_EVIDENCE`

目标域装载**整段复用** `data_b2.prepare_b2`, 不复制它的归一化 / rul_scale /
删失契约逻辑。本模块只做三件 B4X 特有的事:

  1. **核对 split hash** —— §12 继续用同一 B2 划分, 不重新 stratify。
  2. **核对输入 schema** —— §10 规定 schema 由 B3X verdict 唯一决定, 规则在
     B3X 出数之前已写入 protocol。本模块只**读取** B3X summary 的
     `b4x_input_schema` 字段并与 config 里的规则表比对, 不允许按结果挑。
     `core_only` = B2 冻结的 10 列, 即 prepare_b2 的原样输入。
  3. **显式 batch generator** —— 三组必须拿到同一 batch 序列。
     `prepare_b2` 建的是 `DataLoader(shuffle=True)` 且没传 generator, 取样
     顺序来自全局 torch RNG; 而 `source_mmd_finetune` 在 S3 里还会
     `next(src_iter)` 从源域 loader 取 batch, 那个 loader 也 shuffle,
     同样消耗全局 RNG。若不隔离, MMD 组的目标域 batch 顺序会与另外两组
     静默错开, "三组看到同一份目标数据"就是假话。
     因此 target train loader 与 source loader 各自持有独立 generator。

诚实口径 (与 B3X 同): 三组的 RNG **种子与消耗顺序**相同, 且因为三组架构
完全一致 (n_features=12 / n_target=10), 初始化权重也逐值相同; 差异只来自
是否 `load_pretrained` 与是否走 MMD。这一点强于 B3X (B3X 两臂输入维度不同,
权重不可能逐值相同)。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.basilisk_b2.data_b2 import prepare_b2  # noqa: E402
from src.transfer.train_transfer import (  # noqa: E402
    SourceWindowDataset,
    load_source,
)

# §10 允许的输入 schema。`core_plus_mission` 只有在 B3X 给出
# B3X_STABILIZING_SIGNAL 时才会被选中 —— 实际 B3X 结论是
# B3X_NO_STABILIZING_SIGNAL, 故本阶段走 core_only。
SCHEMA_CORE_ONLY = "core_only"
SCHEMA_CORE_PLUS_MISSION = "core_plus_mission"

# 三组共用的 batch generator 播种规则。与 B3X 同式 (seed*1000+7), 使
# "同 seed 的 B3X core_only 与 B4X target_only 拿到同一 batch 序列" 这件事
# 至少在 loader 层面成立。源域 loader 另起偏移, 避免两者相位耦合。
TARGET_GEN_OFFSET = 7
SOURCE_GEN_OFFSET = 11


def resolve_input_schema(cfg: dict, verbose: bool = True) -> str:
    """按 §10 读取并核对 B4X 的输入 schema。

    schema **不是**本阶段的自由参数: 它由 B3X 的 verdict 经 config 里的
    规则表唯一决定。这里做三重核对, 任何一处不符即 abort:
      * B3X summary 的 verdict 在规则表里有对应项;
      * B3X summary 自己记录的 b4x_input_schema 与规则表一致;
      * 结果落在允许的 schema 集合内。
    """
    b4 = cfg["b4x"]["input_schema"]
    rule = {str(k): str(v) for k, v in b4["rule"].items()}
    sm_p = ROOT / cfg["paths"]["b3x_summary_json"]
    if not sm_p.exists():
        raise SystemExit(f"!! 缺 B3X summary: {sm_p} —— §10 的 schema 无从确定")
    sm = json.loads(sm_p.read_text(encoding="utf-8"))
    verdict = str(sm["verdict"])
    if verdict not in rule:
        raise SystemExit(
            f"!! B4X_SCHEMA_UNRESOLVED: B3X verdict={verdict!r} 不在规则表 "
            f"{sorted(rule)} 中 —— 规则表在 B3X 出数前已冻结, 不得事后补")
    want = rule[verdict]
    got = str(sm["b4x_input_schema"])
    if got != want:
        raise SystemExit(
            f"!! B4X_SCHEMA_MISMATCH: B3X summary 记录 {got!r}, 规则表按 "
            f"verdict={verdict!r} 要求 {want!r}")
    if want not in (SCHEMA_CORE_ONLY, SCHEMA_CORE_PLUS_MISSION):
        raise SystemExit(f"!! 未知 schema {want!r}")
    if want != SCHEMA_CORE_ONLY:
        # 本阶段的 loader 只实现 core_only (= prepare_b2 原样输入)。若将来
        # B3X 给出 STABILIZING_SIGNAL, 必须显式接入 data_b3x, 不允许默默降级。
        raise SystemExit(
            f"!! schema={want!r} 需要 mission 列, 本脚本只实现 "
            f"{SCHEMA_CORE_ONLY} —— 不得默默降级成 core_only")
    if verbose:
        print(f">> §10 输入 schema = {want}  (由 B3X verdict {verdict} 决定, "
              f"规则表冻结于 B3X 出数之前)")
    return want


def check_split(cfg: dict, split: dict) -> str:
    """§12: 划分必须逐字是 B2 的那一份。"""
    want = str(cfg["b4x"]["split"]["expected_sha256"])
    got = str(split["split_sha256"])
    if got != want:
        raise SystemExit(
            f"!! B4X_SPLIT_MISMATCH: split_sha256={got[:16]}… != "
            f"契约钉死的 B2 划分 {want[:16]}… —— §12 禁止重新划分")
    return got


def prepare_b4x(cfg: dict, split: dict, seed: int, schema: str,
                with_test: bool = True, verbose: bool = True) -> dict:
    """目标域数据。core_only 下与 prepare_b2 逐值等价, 只换 train loader 的
    取样 generator (§6 同源要求: 三组 batch 顺序必须一致)。"""
    if schema != SCHEMA_CORE_ONLY:
        raise ValueError(f"prepare_b4x 只支持 {SCHEMA_CORE_ONLY}, 收到 {schema!r}")
    check_split(cfg, split)
    d = prepare_b2(cfg, split, seed, with_test=with_test, verbose=verbose)

    gtr = torch.Generator()
    gtr.manual_seed(int(seed) * 1000 + TARGET_GEN_OFFSET)
    old = d["ltr"]
    d["ltr"] = DataLoader(old.dataset, batch_size=old.batch_size,
                          shuffle=True, generator=gtr)
    d["batch_order_generator_seed"] = int(seed) * 1000 + TARGET_GEN_OFFSET
    d["input_schema"] = schema
    d["split_sha256"] = str(split["split_sha256"])
    if verbose:
        print(f"   batch generator seed = {d['batch_order_generator_seed']} "
              f"(三组共用, 与源域 loader 隔离)")
    return d


def prepare_source(cfg: dict, seed: int, verbose: bool = True) -> dict:
    """源域窗数据 (只给 MMD 用)。stride / batch / 归一化全部沿用冻结实现。

    `load_source` 内部做的是 source-train z-score, 与 pretrain 同口径 ——
    §0 禁止修改, 这里只调用。
    """
    pc, tc = cfg["pretrain"], cfg["transfer"]
    L = int(cfg["model"]["input_len_L"])
    h5 = ROOT / pc["source_feature_path"]
    if not h5.exists():
        raise SystemExit(f"!! 缺源域特征 {h5}")
    sp = cfg.get("source", {}).get("split", {})
    val_ids = (sp.get("val_device_ids") or sp.get("val_bearing_ids") or [])
    feats, hi, bid, tidx = load_source(
        h5, pc.get("source_id_field", "bearing_id"), val_ids, train_only=True)
    ds = SourceWindowDataset(feats, hi, bid, tidx, L,
                             stride=max(1, len(feats) // 2000))
    gs = torch.Generator()
    gs.manual_seed(int(seed) * 1000 + SOURCE_GEN_OFFSET)
    ls = DataLoader(ds, batch_size=int(pc["batch_size"]), shuffle=True,
                    generator=gs)
    if verbose:
        print(f">> 源域: {feats.shape[0]} 行 × {feats.shape[1]} 维 -> "
              f"{len(ds)} 窗 (L={L})  generator seed="
              f"{int(seed) * 1000 + SOURCE_GEN_OFFSET}")
    return {"loader": ls, "n_features": int(feats.shape[1]),
            "n_rows": int(feats.shape[0]), "n_windows": int(len(ds)),
            "hi_bins": [tuple(b) for b in tc["hi_bins"]],
            "mmd_lambda": float(tc["mmd_lambda"]),
            "source_generator_seed": int(seed) * 1000 + SOURCE_GEN_OFFSET}
