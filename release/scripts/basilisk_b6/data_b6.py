#!/usr/bin/env python
"""scripts/basilisk_b6/data_b6.py

BASILISK-B6 数据装载 —— 在冻结的 B2.1 划分上做**失效标签稀缺**子集。

与 B5 的唯一差别: train 的 tid 列表被替换成该标签档的子集 (选中的 event +
全部 24 条 censored)。val / test 逐字节沿用 B2.1 (§5)。

为什么用"构造改写后的 split dict"而不是改 prepare_b2:
    `scripts/basilisk_b2/data_b2.py` 属于契约里的只读组 (§2 禁止修改 B2 产物),
    而它恰好只按 `split["splits"]["train"]["tids"]` 取 train 掩码。因此在**调用前**
    把 train tids 换成子集, 就能让 rul_scale / z-score / 删失契约全部按该档的
    train 重新推导, 而不动 prepare_b2 一行代码。

关于 rul_scale 与 z-score 随子集变化:
    prepare_b2 用 train 的 event 行算 `rul_scale = max(nanmax(rul[train]), 1.0)`,
    用 train 全行算 `mu, sd`。移除 event 轨迹会同时改动这两者 —— 这是**标签预算的
    后果, 不是公平性破坏**: 同一个 cell 内三组共用同一份 rul_scale / mu / sd,
    只是不同档之间不同。故每个 cell 都要把它们记进结果, 以便复现与核对。
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts.basilisk_b2.data_b2 import prepare_b2  # noqa: E402
from scripts.basilisk_b4x.data_b4x import prepare_source  # noqa: E402
from scripts.basilisk_b5.data_b5 import (  # noqa: E402
    TARGET_GEN_OFFSET, batch_order_signature, load_b21_split,
    resolve_input_schema_b5,
)

INVALID = "B6_INVALID"
SCHEMA_CORE_ONLY = "core_only"

# 与 B5 同一个划分哈希。B6 不改划分, 只在 train 内部做标签子集。
B6_EXPECTED_SPLIT_SHA256 = (
    "23e2b94445e698eab263ca17a36d99f7130eb2e8c5c3b9b8d0bd9ba772e5c932")

__all__ = ["prepare_b6", "reset_target_batch_order", "batch_order_signature",
           "load_b6_split", "load_subset_manifest", "subset_split",
           "resolve_input_schema_b6", "prepare_source", "B6_EXPECTED_SPLIT_SHA256"]


def load_b6_split(cfg: dict) -> dict:
    """复用 B5 的划分校验 (同一哈希、同一 B2.1 判定要求)。"""
    man = load_b21_split(_as_b5_view(cfg))
    if str(man["split_sha256"]) != B6_EXPECTED_SPLIT_SHA256:
        raise SystemExit(f"!! split hash 与 B6 常量不符 —— {INVALID}")
    return man


def resolve_input_schema_b6(cfg: dict) -> str:
    """§4: schema 由 B3X 判定决定, 且必须是 core_only / n_features=12。"""
    schema = resolve_input_schema_b5(_as_b5_view(cfg))
    isch = cfg["b6"]["input_schema"]
    if int(isch["n_features"]) != 12:
        raise SystemExit(f"!! §4 要求 n_features=12, 得到 {isch['n_features']}")
    if not bool(isch["forbid_restoring_mission_arm"]):
        raise SystemExit("!! forbid_restoring_mission_arm 必须为 true (§4)")
    return schema


def _as_b5_view(cfg: dict) -> dict:
    """把 B6 config 包装成 B5 校验函数期望的形状 (它们读 cfg["b5"][...])。

    只做键名转接, 不改任何数值 —— 于是划分/schema 的校验逻辑只有一处实现。
    """
    if "b5" in cfg and "split" in cfg.get("b5", {}):
        return cfg
    view = dict(cfg)
    view["b5"] = cfg["b6"]
    return view


def load_subset_manifest(cfg: dict) -> dict:
    """读已冻结的标签子集清单, 并核对其 sha256 与 protocol 一致性。"""
    p = ROOT / cfg["paths"]["subset_manifest_json"]
    if not p.exists():
        raise SystemExit(f"!! {p.relative_to(ROOT)} 缺失 —— "
                         f"必须先跑 build_label_subsets.py (§24-3)")
    mf = json.loads(p.read_text(encoding="utf-8"))
    core = {k: mf[k] for k in ("levels", "subset_seed", "bin_names",
                               "bin_edges", "per_level_bin_quota")}
    got = hashlib.sha256(
        json.dumps(core, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    if got != str(mf["subset_manifest_sha256"]):
        raise SystemExit(f"!! 子集清单内容与其 sha256 不符 (清单被改动过) —— {INVALID}")
    ph = json.loads((ROOT / cfg["protocol"]["hash_path"]).read_text("utf-8"))
    if str(mf["protocol_sha256"]) != str(ph["protocol_sha256"]):
        raise SystemExit(f"!! 子集清单登记的 protocol_sha256 与冻结件不符 —— {INVALID}")
    if str(mf["split_sha256"]) != B6_EXPECTED_SPLIT_SHA256:
        raise SystemExit(f"!! 子集清单登记的 split_sha256 不符 —— {INVALID}")
    return mf


def subset_split(split: dict, mf: dict, n_event: int) -> dict:
    """构造该标签档的 split 视图: train 换成子集, val / test 原样。

    同时保留原始 `split_sha256` —— 它标识的是 B2.1 划分本身, 而 B6 并未改动划分;
    档位信息另存在 `b6_label_level` 字段, 便于下游记录而不冒充新划分。
    """
    key = str(int(n_event))
    if key not in mf["levels"]:
        raise SystemExit(f"!! 清单内没有 n_event_labeled={n_event} 档 —— {INVALID}")
    L = mf["levels"][key]
    want = sorted(L["train_tids"])
    orig_tr = list(split["splits"]["train"]["tids"])
    if not set(want).issubset(set(orig_tr)):
        raise SystemExit(f"!! n={n_event} 的 train 子集含 B2.1 train 之外的轨迹 "
                         f"—— {INVALID}")
    if len(want) != int(L["n_train_total"]):
        raise SystemExit(f"!! n={n_event} train 条数 {len(want)} != "
                         f"清单 {L['n_train_total']} —— {INVALID}")

    out = copy.deepcopy(split)
    tr = out["splits"]["train"]
    idx_of = {t: int(i) for t, i in zip(orig_tr, split["splits"]["train"]["tid_indices"])}
    # 保持 B2.1 的书写顺序 (按原 train 顺序过滤), 使 Dataset 内样本排列可复现。
    kept = [t for t in orig_tr if t in set(want)]
    tr["tids"] = kept
    tr["tid_indices"] = [idx_of[t] for t in kept]
    tr["n"] = len(kept)
    tr["n_event"] = int(L["n_event_labeled"])
    tr["n_censored"] = int(L["n_censored"])
    tr["b6_label_level"] = int(n_event)
    tr["b6_removed_event_tids"] = list(L["removed_event_tids"])
    out["b6_label_level"] = int(n_event)
    out["b6_subset_manifest_sha256"] = str(mf["subset_manifest_sha256"])

    # val / test 必须逐条未动 —— 这里显式再验一次, 而不是"相信 deepcopy"。
    for name in ("val", "test"):
        if list(out["splits"][name]["tids"]) != list(split["splits"][name]["tids"]):
            raise SystemExit(f"!! {name} 被改动 —— {INVALID}")
    return out


def reset_target_batch_order(d: dict) -> int:
    """把 target train loader 的取样 generator 重置回本 cell 的固定种子。

    与 B5 同一纪律: DataLoader 的 generator 有状态, 每 epoch 推进。B6 有
    4 档 x 3 方法 x 5 seed = 60 个 cell, 每个 cell 内三组必须逐 batch 一致,
    因此每组开训前必须重置 —— 否则第二、三组是"接着上一组走"的顺序。
    """
    s = int(d["batch_order_generator_seed"])
    d["ltr"].generator.manual_seed(s)
    return s


def prepare_b6(cfg: dict, split: dict, mf: dict, n_event: int, seed: int,
               schema: str, with_test: bool = True,
               verbose: bool = True) -> dict:
    """该 (标签档, seed) 下的目标域数据。三组共用同一个返回值。"""
    if schema != SCHEMA_CORE_ONLY:
        raise ValueError(f"prepare_b6 只支持 {SCHEMA_CORE_ONLY}, 收到 {schema!r}")
    sub = subset_split(split, mf, n_event)
    d = prepare_b2(cfg, sub, seed, with_test=with_test, verbose=verbose)

    gtr = torch.Generator()
    gtr.manual_seed(int(seed) * 1000 + TARGET_GEN_OFFSET)
    old = d["ltr"]
    d["ltr"] = DataLoader(old.dataset, batch_size=old.batch_size,
                          shuffle=True, generator=gtr)
    d["batch_order_generator_seed"] = int(seed) * 1000 + TARGET_GEN_OFFSET
    d["input_schema"] = schema
    d["split_sha256"] = str(split["split_sha256"])
    d["n_event_labeled"] = int(n_event)
    d["subset_manifest_sha256"] = str(mf["subset_manifest_sha256"])
    L = mf["levels"][str(int(n_event))]
    d["train_event_tids"] = list(L["event_tids"])
    d["train_censored_tids"] = list(L["censored_tids"])
    d["train_tids_used"] = list(sub["splits"]["train"]["tids"])
    d["removed_event_tids"] = list(L["removed_event_tids"])
    # 标签预算改变了这两者 —— 记进结果, 供复现核对 (同 cell 内三组共用)。
    d["rul_scale_recorded"] = float(d["rul_scale"])
    if verbose:
        print(f"   n_event_labeled={n_event}: train {sub['splits']['train']['n']} 条 "
              f"(event {L['n_event_labeled']} + censored {L['n_censored']}), "
              f"rul_scale={float(d['rul_scale']):.6g}")
        print(f"   target batch generator seed = "
              f"{d['batch_order_generator_seed']} (三组共用, 与源域 loader 隔离)")
    return d


def train_ids_signature(d: dict) -> str:
    """train/val/test tid 集合的指纹, 供 §11 公平性核对 (三组必须相同)。"""
    payload = json.dumps({
        "train": list(d["train_tids_used"]),
        "n_event_labeled": int(d["n_event_labeled"]),
    }, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
