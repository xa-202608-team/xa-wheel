#!/usr/bin/env python
"""scripts/basilisk_b5/data_b5.py

BASILISK-B5 数据装载 —— 钉在**冻结的 B2.1 划分**上。

为什么不能直接用 scripts/basilisk_b4x/data_b4x.py::prepare_b4x:
    `data_b4x.check_split` 硬断言 `expected_sha256 = 079296e5...113ae`, 那是
    **B2 的旧划分**, 不符即抛 B4X_SPLIT_MISMATCH。B5 用的是 B2.1 的划分
    23e2b944...5c932。改 data_b4x 属于 §1 禁止修改 B4X 产物, 所以 B5 自带一份
    只认 B2.1 划分的入口。

除此之外一切复用, 不重写:
    * 目标域预处理 / rul_scale / z-score / 删失契约 -> `prepare_b2`
    * 源域 loader / hi_bins / mmd_lambda / 源域 generator -> `prepare_source`
  重写它们等于悄悄换了预处理: 三组之间的公平性还在, 但与 B2.1 就不是同一把尺子。

§6 的两个公平性要点:
    * target train loader 持**显式 generator** (seed*1000+7), 与源域 loader 的
      generator (seed*1000+11, 在 prepare_source 内部) 隔离。于是 MMD 组在 S3
      里 `next(src_iter)` 消耗源域随机性时, 不会让它的 target batch 顺序与另外
      两组静默错开 —— 否则"三组看到同一份目标数据"是假话。
    * 三组统一 `n_features = 12` (source encoder 第一层宽度), 见 protocol §6.3:
      因此 B5 的 target_only 是本阶段重跑, 其数值**不等于** B2.1 的 0.2441。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts.basilisk_b2.data_b2 import prepare_b2  # noqa: E402
from scripts.basilisk_b4x.data_b4x import prepare_source  # noqa: E402

# §6.1 batch 顺序隔离的 generator 偏移。沿用 B4X 取值, 使"同一 seed 下 target
# batch 顺序"这一性质在两阶段之间可比 (B4X 的源域偏移 11 在 prepare_source 内)。
TARGET_GEN_OFFSET = 7

# B5 钉死的划分哈希 (§2)。与 B4X 的常量并列而非替换 —— 后者仍指向 B2 划分。
B5_EXPECTED_SPLIT_SHA256 = (
    "23e2b94445e698eab263ca17a36d99f7130eb2e8c5c3b9b8d0bd9ba772e5c932")

SPLIT_MISMATCH_LABEL = "B5_INVALID"
SCHEMA_CORE_ONLY = "core_only"


def load_b21_split(cfg: dict) -> dict:
    """读 B2.1 划分并逐项核对 (§2)。任何不符 -> B5_INVALID, 立即停止。"""
    man = json.loads(
        (ROOT / cfg["paths"]["b21_split_json"]).read_text(encoding="utf-8"))
    exp = str(cfg["b5"]["split"]["expected_sha256"])
    if exp != B5_EXPECTED_SPLIT_SHA256:
        raise SystemExit(
            f"!! config 的 expected_sha256 与脚本常量不符 —— {SPLIT_MISMATCH_LABEL}")
    if str(man["split_sha256"]) != exp:
        raise SystemExit(
            f"!! B2.1 split hash {str(man['split_sha256'])[:16]} != {exp[:16]} "
            f"—— {SPLIT_MISMATCH_LABEL}")
    if str(man["verdict"]) != "B21_SPLIT_VALID":
        raise SystemExit(
            f"!! B2.1 划分级判定 {man['verdict']} 非法 —— {SPLIT_MISMATCH_LABEL}")
    b21 = json.loads(
        (ROOT / cfg["paths"]["b21_summary_json"]).read_text(encoding="utf-8"))
    need = str(cfg["b5"]["split"]["require_b21_verdict"])
    if str(b21["verdict"]) != need:
        raise SystemExit(
            f"!! B2.1 阶段判定 {b21['verdict']} != {need} —— {SPLIT_MISMATCH_LABEL}")
    if bool(cfg["b5"]["split"]["forbid_restratify"]) is not True:
        raise SystemExit("!! forbid_restratify 必须为 true (§2)")
    return man


def resolve_input_schema_b5(cfg: dict) -> str:
    """§3: 由 B3X 判定推正式 schema, 与 config 规则表交叉核对, 不按结果挑。"""
    isch = cfg["b5"]["input_schema"]
    b3x = json.loads(
        (ROOT / cfg["paths"]["b3x_summary_json"]).read_text(encoding="utf-8"))
    v = str(b3x["verdict"])
    rule = {str(k): str(x) for k, x in isch["rule"].items()}
    if v not in rule:
        raise SystemExit(f"!! B3X 判定 {v} 不在规则表内 —— B5_SCHEMA_UNRESOLVED")
    want = rule[v].lower()
    if str(isch["formal"]).lower() != want:
        raise SystemExit(
            f"!! schema 不一致: B3X={v} 应得 {want}, config 写 {isch['formal']} "
            f"—— B5_SCHEMA_MISMATCH")
    if want != SCHEMA_CORE_ONLY:
        raise SystemExit(
            f"!! B5 正式 schema 必须是 {SCHEMA_CORE_ONLY} "
            f"(B3X 未给出稳定化信号), 得到 {want}")
    if not bool(isch["forbid_mission_features"]):
        raise SystemExit("!! forbid_mission_features 必须为 true (§3)")
    return want


def reset_target_batch_order(d: dict) -> int:
    """把 target train loader 的取样 generator 重置回本 seed 的固定种子。

    §6.1 的要点: DataLoader 的 generator 是**有状态**的, 每个 epoch 取样都会推进
    它。三个学习组在同一个 seed 下共用同一个 loader 对象, 若不在每组开训前重置,
    第二组、第三组拿到的就是"接着上一组走"的 batch 顺序 —— 那三组就不是严格配对,
    RMSE 差里会混进 batch 顺序的差异。每组开训前调这个函数, 三组逐 batch 一致。
    """
    s = int(d["batch_order_generator_seed"])
    d["ltr"].generator.manual_seed(s)
    return s


def prepare_b5(cfg: dict, split: dict, seed: int, schema: str,
               with_test: bool = True, verbose: bool = True) -> dict:
    """目标域数据。core_only 下与 prepare_b2 逐值等价, 只换 train loader 的
    取样 generator (§6: 三组 batch 顺序必须一致)。"""
    if schema != SCHEMA_CORE_ONLY:
        raise ValueError(f"prepare_b5 只支持 {SCHEMA_CORE_ONLY}, 收到 {schema!r}")
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
        print(f"   target batch generator seed = "
              f"{d['batch_order_generator_seed']} (三组共用, 与源域 loader 隔离)")
    return d


def batch_order_signature(loader: DataLoader, n_batches: int = 6) -> str:
    """前若干 batch 的数值指纹, 供测试核对"三组 batch 顺序逐 batch 一致"。

    只遍历 dataset, 不做前向, 因此不消耗模型侧随机性。用数值指纹而非样本 index
    是因为 Dataset 不暴露 index; 数值等价已足够 —— 顺序不同则指纹必不同。
    """
    parts = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        arr = np.asarray(x.detach().cpu().numpy(), dtype=np.float64)
        parts.append(f"{tuple(arr.shape)}:{np.nansum(arr):.10e}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
