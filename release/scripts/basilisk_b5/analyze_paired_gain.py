#!/usr/bin/env python
"""scripts/basilisk_b5/analyze_paired_gain.py

BASILISK-B5 §8/§10 —— seed 级配对 gain 与 paired bootstrap。

§8 符号定义 (在此固定, 由测试钉死):
    gain_ft(seed)  = RMSE_target_only(seed) - RMSE_source_finetune(seed)
    gain_mmd(seed) = RMSE_target_only(seed) - RMSE_source_mmd_finetune(seed)
    **正数 = 迁移带来改进。**

§10 bootstrap 纪律:
    单位 = **seed 级配对差值, n = 5**, 重采样 ≥ 5000, bootstrap seed 固定。
    禁止: endpoint bootstrap 替代 seed bootstrap; 把时间点当独立样本;
          用轨迹数量抬高显著性。
    paired_bootstrap 直接 import B2 的冻结实现 —— 它内部用局部
    np.random.default_rng(seed), 且返回 unit = one_paired_difference_per_seed。

同时算 §9 条件 5/6/7 所需的辅助量 (corr / catastrophic / warning miss),
判定本身留给 summarize_b5.py —— 本脚本只出数字。

用法:
    python scripts/basilisk_b5/analyze_paired_gain.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.basilisk_b11.calibrate_degradation import (  # noqa: E402
    load_b11_config as load_cfg,
)
from scripts.basilisk_b2.run_gate import paired_bootstrap, warning_valid  # noqa: E402
from scripts.basilisk_b3x.summarize_b3x import _stat  # noqa: E402

NAN = float("nan")

GAIN_DEFINITION = (
    "gain_ft(seed) = RMSE_target_only(seed) - RMSE_source_finetune(seed); "
    "gain_mmd(seed) = RMSE_target_only(seed) - RMSE_source_mmd_finetune(seed); "
    "主指标 = test information-zone trajectory-macro RMSE。"
    "**正数 = 迁移带来改进**。符号定义在 protocol 冻结时已固定, 不得反转。"
)

BOOTSTRAP_DISCIPLINE = (
    "重采样单位是 seed 级配对差值, n = 5, 一个 seed 一个数。"
    "绝不把时间点当独立样本; 绝不用 endpoint bootstrap 替代 seed bootstrap; "
    "绝不用轨迹数量抬高显著性。"
    "n = 5 的 bootstrap 只描述训练随机性下差值方向是否稳定, "
    "**不是泛化误差的置信区间** —— 这一点必须写进 limitations。"
)

TRANSFER_VS_DAMAGE_NOTE = (
    "transfer_vs_damage = RMSE_damage - RMSE_transfer。"
    "负数 = 迁移模型不如物理 damage extrapolation 基线。"
    "若为负, 必须原样写出: 迁移模型可能相对 Target-only 有增益, "
    "但未达到物理 damage extrapolation 基线。"
    "不得把存在正转移写成最佳方法 —— 这是两个不同命题。"
)


def _macro_corr(m: dict) -> float:
    """从已评估的指标块取宏 corr (evaluate_b2 放在 shape 下)。缺失即 NaN, 不补 0。"""
    v = m.get("shape", {}).get("macro_corr")
    if v is None:
        return NAN
    v = float(v)
    return v if np.isfinite(v) else NAN


def _psr(m: dict) -> dict:
    """§17: pred_std / true_std / PSR 三者同时给出, 不只给一个上界。"""
    sh = m.get("shape", {})
    out = {}
    for k in ("pred_std", "true_std", "psr"):
        v = sh.get(k)
        out[k] = (float(v) if v is not None and np.isfinite(float(v)) else NAN)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wheel_basilisk_b5.yaml")
    a = ap.parse_args()

    cfg = load_cfg(a.config)
    b5 = cfg["b5"]
    mp = ROOT / cfg["paths"]["metrics_json"]
    if not mp.exists():
        raise SystemExit(f"!! 缺 {mp.relative_to(ROOT)} —— 先跑 run_formal_transfer.py")
    M = json.loads(mp.read_text(encoding="utf-8"))
    if bool(M.get("fast")):
        raise SystemExit("!! metrics 来自 --fast 冒烟跑, 不得进入正式分析")

    bc = b5["bootstrap"]
    if str(bc["unit"]) != "seed_level_paired_gain":
        raise SystemExit(f"!! §10 bootstrap 单位必须是 seed_level_paired_gain")
    n_rep = int(bc["samples"])
    if n_rep < 5000:
        raise SystemExit(f"!! §10 要求重采样 ≥ 5000, config 写 {n_rep}")
    bseed = int(bc["seed"])

    rows = M["per_seed"]
    seeds = [int(r["seed"]) for r in rows]
    if len(rows) != int(bc["n"]):
        raise SystemExit(f"!! seed 数 {len(rows)} != 声明的 n={bc['n']}")

    per_seed: list[dict] = []
    for r in rows:
        g = r["gains"]
        w_t = r["target_only"]["warning"]
        per_seed.append({
            "seed": int(r["seed"]),
            "target_only": float(g["target_only"]),
            "source_finetune": float(g["source_finetune"]),
            "source_mmd_finetune": float(g["source_mmd_finetune"]),
            "damage_extrapolation": float(g["damage_extrapolation"]),
            "const_mean_info": float(r["const_mean_info"]["info_macro_rmse"]),
            "gain_ft": float(g["gain_ft"]),
            "gain_mmd": float(g["gain_mmd"]),
            "gain_ft_positive": bool(float(g["gain_ft"]) > 0),
            "gain_mmd_positive": bool(float(g["gain_mmd"]) > 0),
            "transfer_vs_damage_ft": float(g["transfer_vs_damage_ft"]),
            "transfer_vs_damage_mmd": float(g["transfer_vs_damage_mmd"]),
            "corr": {k: _macro_corr(r[k]) for k in
                     ("target_only", "source_finetune",
                      "source_mmd_finetune", "damage_extrapolation",
                      "const_mean_info")},
            "stability": {k: _psr(r[k]) for k in
                          ("target_only", "source_finetune",
                           "source_mmd_finetune", "damage_extrapolation",
                           "const_mean_info")},
            "catastrophic_rate": {
                k: (float(v["catastrophic_error_rate"])
                    if v.get("catastrophic_error_rate") is not None
                    and np.isfinite(float(v["catastrophic_error_rate"]))
                    else NAN)
                for k, v in r["catastrophic"].items()},
            "catastrophic_n": {k: v.get("n_catastrophic")
                               for k, v in r["catastrophic"].items()},
            "warning_miss": {
                k: (float(r[k]["warning"].get("miss_rate", NAN))
                    if r[k].get("warning") else NAN)
                for k in ("target_only", "source_finetune",
                          "source_mmd_finetune")},
            "warning_valid_target_only": bool(warning_valid(w_t)[0]),
            "best_epoch": {k: int(r[f"{k}_meta"]["best_epoch"])
                           for k in ("target_only", "source_finetune",
                                     "source_mmd_finetune")},
            "val_metric": {k: float(r[f"{k}_meta"]["val_metric"]["value"])
                           for k in ("target_only", "source_finetune",
                                     "source_mmd_finetune")},
            "checkpoint_sha256": {
                k: str(r[f"{k}_meta"]["checkpoint_sha256"])
                for k in ("target_only", "source_finetune",
                          "source_mmd_finetune")},
        })

    def _col(key: str) -> list[float]:
        return [float(p[key]) for p in per_seed]

    boots: dict[str, dict] = {}
    for name, key in (("gain_ft", "gain_ft"), ("gain_mmd", "gain_mmd")):
        d = _col(key)
        bs = paired_bootstrap(d, n_rep, bseed)
        boots[name] = {
            "per_seed": [round(x, 8) for x in d],
            "mean": float(np.mean(d)),
            "median": float(np.median(d)),
            "std": float(np.std(d, ddof=1)) if len(d) > 1 else NAN,
            "improve_count": int(sum(1 for x in d if x > 0)),
            "n_seeds": len(d),
            "bootstrap": bs,
            "ci95_lower": float(bs["ci_lower"]),
            "ci95_upper": float(bs["ci_upper"]),
            "ci_lower_positive": bool(float(bs["ci_lower"]) > 0),
        }

    # §9 条件 5/6/7 的辅助量 (判定在 summarize_b5.py)
    def _mean_of(getter) -> float:
        v = [getter(p) for p in per_seed]
        v = [x for x in v if x is not None and np.isfinite(float(x))]
        return float(np.mean(v)) if v else NAN

    corr_means = {g: _mean_of(lambda p, g=g: p["corr"][g])
                  for g in ("target_only", "source_finetune",
                            "source_mmd_finetune", "damage_extrapolation",
                            "const_mean_info")}
    cat_means = {g: _mean_of(lambda p, g=g: p["catastrophic_rate"].get(g))
                 for g in ("target_only", "source_finetune",
                           "source_mmd_finetune", "damage_extrapolation",
                           "const_mean_info")}
    miss_means = {g: _mean_of(lambda p, g=g: p["warning_miss"].get(g))
                  for g in ("target_only", "source_finetune",
                            "source_mmd_finetune")}

    agg = {g: _stat(_col(g)) for g in
           ("target_only", "source_finetune", "source_mmd_finetune",
            "damage_extrapolation", "const_mean_info")}
    agg["transfer_vs_damage_ft"] = _stat(_col("transfer_vs_damage_ft"))
    agg["transfer_vs_damage_mmd"] = _stat(_col("transfer_vs_damage_mmd"))

    # §17 诊断项: pred_std / true_std / PSR 三者同时报告, PSR > 3 标注告警。
    # **明确不进 Gate** —— 出数字后往 Gate 加条件就是事后调门槛。
    hv = float(b5["output_stability"]["high_variance_psr"])
    stability: dict[str, dict] = {}
    for g in ("target_only", "source_finetune", "source_mmd_finetune",
              "damage_extrapolation", "const_mean_info"):
        psr_v = [p["stability"][g]["psr"] for p in per_seed]
        fin = [x for x in psr_v if np.isfinite(x)]
        stability[g] = {
            "psr_per_seed": [round(x, 6) if np.isfinite(x) else None
                             for x in psr_v],
            "psr_mean": float(np.mean(fin)) if fin else NAN,
            "pred_std_mean": _stat([p["stability"][g]["pred_std"]
                                    for p in per_seed])["mean"],
            "true_std_mean": _stat([p["stability"][g]["true_std"]
                                    for p in per_seed])["mean"],
            "n_seeds_psr_above_threshold":
                int(sum(1 for x in fin if x > hv)),
            "high_variance_warning": bool(fin and np.mean(fin) > hv),
        }

    out = {
        "stage": "BASILISK_B5",
        "section": "§8/§10 paired gain",
        "gain_definition": GAIN_DEFINITION,
        "bootstrap_discipline": BOOTSTRAP_DISCIPLINE,
        "transfer_vs_damage_note": TRANSFER_VS_DAMAGE_NOTE,
        "primary_metric": str(b5["primary_metric"]),
        "formal_seeds": seeds,
        "split_sha256": str(M["split_sha256"]),
        "protocol_sha256": str(M["protocol_sha256"]),
        "bootstrap_config": {"unit": str(bc["unit"]), "n": int(bc["n"]),
                             "samples": n_rep, "seed": bseed,
                             "forbid_endpoint_bootstrap":
                                 bool(bc["forbid_endpoint_bootstrap"]),
                             "forbid_trajectory_count_significance":
                                 bool(bc["forbid_trajectory_count_significance"])},
        "per_seed": per_seed,
        "aggregate_rmse": agg,
        "output_stability": {
            "by_group": stability,
            "high_variance_psr": hv,
            "high_variance_label":
                str(b5["output_stability"]["high_variance_label"]),
            "in_gate": bool(b5["output_stability"]["in_gate"]),
            "reason": str(b5["output_stability"]["reason"]),
        },
        "gain": boots,
        "corr_mean_by_group": corr_means,
        "catastrophic_rate_mean_by_group": cat_means,
        "warning_miss_mean_by_group": miss_means,
        "gate_inputs": {
            "ft": {"improve_count": boots["gain_ft"]["improve_count"],
                   "mean": boots["gain_ft"]["mean"],
                   "median": boots["gain_ft"]["median"],
                   "ci95_lower": boots["gain_ft"]["ci95_lower"],
                   "corr_mean": corr_means["source_finetune"],
                   "corr_mean_target": corr_means["target_only"],
                   "catastrophic_mean": cat_means["source_finetune"],
                   "catastrophic_mean_target": cat_means["target_only"],
                   "warning_miss_mean": miss_means["source_finetune"],
                   "warning_miss_mean_target": miss_means["target_only"]},
            "mmd": {"improve_count": boots["gain_mmd"]["improve_count"],
                    "mean": boots["gain_mmd"]["mean"],
                    "median": boots["gain_mmd"]["median"],
                    "ci95_lower": boots["gain_mmd"]["ci95_lower"],
                    "corr_mean": corr_means["source_mmd_finetune"],
                    "corr_mean_target": corr_means["target_only"],
                    "catastrophic_mean": cat_means["source_mmd_finetune"],
                    "catastrophic_mean_target": cat_means["target_only"],
                    "warning_miss_mean": miss_means["source_mmd_finetune"],
                    "warning_miss_mean_target": miss_means["target_only"]},
        },
        "verdict_not_decided_here":
            "Gate 判定在 summarize_b5.py; 本脚本只出数字, 不下结论。",
    }
    op = ROOT / cfg["paths"]["gain_json"]
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2, ensure_ascii=False,
                             default=float) + "\n", encoding="utf-8")
    print(f">> 已写 {op.relative_to(ROOT)}")
    for name in ("gain_ft", "gain_mmd"):
        b = boots[name]
        print(f"   {name}: {[round(x, 6) for x in b['per_seed']]}")
        print(f"      mean {b['mean']:+.6f}  median {b['median']:+.6f}  "
              f"std {b['std']:.6f}  improve {b['improve_count']}/{b['n_seeds']}")
        print(f"      bootstrap CI95 [{b['ci95_lower']:+.6f}, "
              f"{b['ci95_upper']:+.6f}]  (n={b['bootstrap'].get('n_rep', n_rep)}, "
              f"unit={b['bootstrap'].get('unit')})")
    print(f"   corr mean: target {corr_means['target_only']:.4f} / "
          f"ft {corr_means['source_finetune']:.4f} / "
          f"mmd {corr_means['source_mmd_finetune']:.4f}")
    print(f"   transfer_vs_damage: ft {agg['transfer_vs_damage_ft']['mean']:+.6f} / "
          f"mmd {agg['transfer_vs_damage_mmd']['mean']:+.6f} "
          f"(负数 = 不如物理基线)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
