#!/usr/bin/env python
"""scripts/basilisk_b3x/run_mission_ablation.py

BASILISK-B3X §3/§5/§6/§7 —— 两臂 × 5 seed 配对消融。

`POST_B2_FAIL_EXPLORATORY` / `NOT_FORMAL_EVIDENCE`

臂 A `CORE_ONLY`         : B2 冻结的 10 列输入
臂 B `CORE_PLUS_MISSION` : A + 6 列 mission features (§3)

唯一差异 = 输入列。训练函数 (train_b2.train_target_only) / 评估器
(eval_b2.evaluate_b2) / 划分 / 删失契约 / 早停 / max_epochs / weight_decay
全部 import 复用 B2 的实现, 一行不改。

**关于与 B2 数字的可比性 (必须写明, 不得含糊)**:
  B3X 的 CORE_ONLY 臂**不会**逐值复现 B2 的 target_only 数字。原因是 §6 要求
  两臂 batch 顺序相同, 而 B2 的 DataLoader(shuffle=True) 没传 generator ——
  它的取样顺序来自被模型初始化消耗过的全局 RNG。两臂输入维度不同 (10 vs 16),
  初始化抽走的随机数个数就不同, 沿用 B2 的做法会让两臂 batch 顺序静默错开,
  那样"唯一差异是输入列"就是假话。
  取舍: 保证**臂间**配对严格成立 (这是本阶段的诊断对象), 代价是 CORE_ONLY
  相对 B2 多了一处 RNG 差异。因此:
    * gain 只在 B3X 内部两臂之间计算 (A - B), **绝不**拿 B 臂去减 B2 的表;
    * CORE_ONLY vs B2 的差异单独报告, 作为"B2 失败模式是否复现"的旁证。

§8 条件 7 沿用: _train_with_early_stop 签名里没有 test loader。

用法:
    python scripts/basilisk_b3x/run_mission_ablation.py --config configs/wheel_basilisk_b3x.yaml
    python scripts/basilisk_b3x/run_mission_ablation.py --seeds 72 --fast
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

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
from scripts.basilisk_b2.train_b2 import train_target_only  # noqa: E402
from scripts.basilisk_b3x.data_b3x import (  # noqa: E402
    b3x_input_cols,
    prepare_b3x,
)
from scripts.diag_generalization import summarize_history  # noqa: E402

CFG_PATH = "configs/wheel_basilisk_b3x.yaml"

ABLATION_PURPOSE = (
    "B3X §3: 在 B2 冻结的划分与超参下, 只改输入列, 判断 Basilisk mission "
    "features 是否缓解 B2 在短寿命轨迹 tid 19/109/111 上的失控。"
    "这是探索性诊断, 不是正式 B3, 不产生迁移结论, 不重新解释 B2 的判定。"
    "B2_GENERALIZATION_FAIL 保持终局。"
    "两臂唯一差异是输入列; 划分/超参/早停/删失契约/评估器全部沿用 B2 实现。"
    "本脚本不加载 source checkpoint, 不使用 MMD, 不做 truncation。"
)

ARMS = ("core_only", "core_plus_mission")


def run_arm(cfg: dict, split: dict, axis: dict, seed: int, arm: str) -> dict:
    """一个 (seed, 臂): 训练 + test 评估。返回带 _raw 的完整结果。"""
    print(f"\n{'-' * 66}\n[B3X] seed {seed}  arm {arm}\n{'-' * 66}")
    t0 = time.time()
    data = prepare_b3x(cfg, split, seed, arm, with_test=True, verbose=True)
    model, history, th, budget = train_target_only(
        cfg, data, seed, f"b3x_{arm}_s{seed}")
    m = evaluate_b2(model, data["lte"], cfg, axis, data, tag=f"{arm}_s{seed}")
    m["_meta"] = {
        "arm": arm,
        "seed": int(seed),
        "n_target": int(data["n_target"]),
        "xt_cols": list(data["xt_cols"]),
        "mission_cols": list(data["mission_cols"]),
        "batch_order_generator_seed": int(data["batch_order_generator_seed"]),
        "rul_scale": float(data["rul_scale"]),
        "hyper": dict(th),
        "budget": dict(budget),
        "history": summarize_history(history, th["early_stop_metric"]),
        "secs": round(time.time() - t0, 1),
    }
    print(f"   [{arm} s{seed}] test info macro RMSE = "
          f"{m['info_macro_rmse']:.6f}  ({m['_meta']['secs']}s)")
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CFG_PATH)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--fast", action="store_true",
                    help="冒烟: 只跑 1 epoch (结果不得入正式报告)")
    a = ap.parse_args()

    cfg = load_cfg(a.config)
    b3 = cfg["b3x"]
    if str(b3["label"]) != "POST_B2_FAIL_EXPLORATORY":
        raise SystemExit(f"!! label 必须是 POST_B2_FAIL_EXPLORATORY, "
                         f"得到 {b3['label']!r}")

    split_p = ROOT / b3["split"]["reuse_from"]
    split = json.loads(split_p.read_text(encoding="utf-8"))

    seeds = list(a.seeds) if a.seeds else [int(s) for s in b3["seeds"]]
    arms = list(a.arms) if a.arms else list(ARMS)
    for arm in arms:
        if arm not in ARMS:
            raise SystemExit(f"!! 未知臂 {arm!r}")

    if a.fast:
        cfg["training"]["max_epochs"] = 1
        cfg["b2_frozen_training_expected"]["max_epochs"] = 1
        print("!! --fast: max_epochs=1, 结果只能用于冒烟, 不得入报告")

    axis = b2_endpoint_axis(cfg)
    print(f">> B3X mission 消融: seeds {seeds} × arms {arms}")
    for arm in arms:
        print(f"   {arm}: {b3x_input_cols(cfg, arm)}")

    per_seed: list[dict] = []
    raw_store: dict[str, dict] = {}
    t_all = time.time()
    for seed in seeds:
        row: dict = {"seed": int(seed)}
        for arm in arms:
            m = run_arm(cfg, split, axis, seed, arm)
            # _raw 单独存 npz 供 diagnose 复用, 不进 JSON
            raw_store[f"{arm}_s{seed}"] = m["_raw"]
            row[arm] = strip_private(m)
            row[f"{arm}_meta"] = m["_meta"]
        if all(k in row for k in ARMS):
            ga = float(row["core_only"]["info_macro_rmse"])
            gb = float(row["core_plus_mission"]["info_macro_rmse"])
            # §8 D: gain = RMSE_core - RMSE_core_plus_mission (正 = B 臂更好)
            row["gain_mission"] = {
                "core_only": ga,
                "core_plus_mission": gb,
                "gain": ga - gb,
                "definition": "RMSE_core_only - RMSE_core_plus_mission",
            }
            print(f"   >> seed {seed} gain_mission = {ga - gb:+.6f} "
                  f"(core {ga:.6f} -> mission {gb:.6f})")
        per_seed.append(row)

    out = {
        "stage": "BASILISK_B3X",
        "label": str(b3["label"]),
        "formal_stage": bool(b3.get("formal_stage", False)),
        "not_formal_evidence": True,
        "b2_verdict_unchanged": "B2_GENERALIZATION_FAIL",
        "purpose": ABLATION_PURPOSE,
        "config": a.config,
        "fast": bool(a.fast),
        "seeds": seeds,
        "arms": arms,
        "split_sha256": str(split["split_sha256"]),
        "split_reused_from": str(b3["split"]["reuse_from"]),
        "hi_key": str(cfg["transfer"]["target_hi_key"]),
        "short_eol_subset": [int(x) for x in b3["short_eol_subset"]],
        "input_cols": {arm: b3x_input_cols(cfg, arm) for arm in arms},
        "comparability_note": (
            "CORE_ONLY 不逐值复现 B2 的 target_only: §6 要求两臂 batch 顺序相同, "
            "故 train loader 用显式 generator, 而 B2 用的是被模型初始化消耗过的"
            "全局 RNG。gain 只在 B3X 两臂之间计算, 绝不拿本表去减 B2 的表。"),
        "per_seed": per_seed,
        "secs_total": round(time.time() - t_all, 1),
    }
    mp = ROOT / cfg["paths"]["metrics_json"]
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(out, indent=2, ensure_ascii=False,
                             default=float) + "\n", encoding="utf-8")
    print(f"\n>> 已写 {mp.relative_to(ROOT)}  ({out['secs_total']}s)")

    npz = mp.parent / "ablation_raw.npz"
    flat: dict[str, np.ndarray] = {}
    for k, r in raw_store.items():
        for f in ("pred", "true", "tids", "tau", "lb"):
            flat[f"{k}__{f}"] = np.asarray(r[f])
    np.savez_compressed(npz, **flat)
    print(f">> 已写 {npz.relative_to(ROOT)} ({len(raw_store)} 组预测)")

    if all(k in per_seed[0] for k in ARMS) if per_seed else False:
        g = [r["gain_mission"]["gain"] for r in per_seed if "gain_mission" in r]
        print(f">> gain_mission per seed: "
              f"{[round(x, 6) for x in g]}  mean {np.mean(g):+.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
