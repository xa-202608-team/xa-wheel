"""S4 统一评价入口 (协议 docs/diagnostics_s4_protocol.md §14/§15/§18)。

在**完全相同的 evaluator** 上打印五个方法的完整 S4 指标:

  S3 三个迁移组 : target_only / source_finetune / source_mmd_finetune
                  (逐 endpoint 预测取自 checkpoints/s4_stage_diagnostics.json ——
                   那批预测已通过与冻结 S3 数字的逐位校验; 本脚本不重训、不写 .pt)
  基线          : physical_extrap / const_mean_info (另附 const_mean / hi_extrap)

所有方法都走 `src.experiments.run_groups.s4_metrics_block` 这**同一个函数**,
口径不可能分叉。persistence 保持 ORACLE_LIKE, 不进入可部署排名。

不重跑调参。不修改 S3。不进入 S5。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines.physical_extrap import evaluate_physical                # noqa: E402
from src.baselines.trivial import (GATE_BASELINE, METHODS as TRIVIAL_METHODS,  # noqa: E402
                                   ORACLE_LIKE, evaluate_trivial)
from src.experiments.metrics import prognostics_cfg                        # noqa: E402
from src.experiments.run_groups import s4_metrics_block                    # noqa: E402
from src.utils import load_config                                          # noqa: E402

S3_JSON = ROOT / "checkpoints" / "s3_gate_metrics.json"
STAGE_JSON = ROOT / "checkpoints" / "s4_stage_diagnostics.json"
S4_PROTOCOL = ROOT / "docs" / "diagnostics_s4_protocol.md"
OUT_JSON = ROOT / "checkpoints" / "s4_metrics.json"
FROZEN_S3_METRICS_SHA = ("f11f46b107168fe0d21e16aa5c286789580a0eec073f8cd879c4"
                         "ac6f9528f68e")
# 正式可部署排名里出现的方法 (persistence 因 ORACLE_LIKE 被排除, 协议 §15)
DEPLOYABLE_BASELINES = ("physical_extrap", GATE_BASELINE)


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _f(v, w=9, nd=4):
    return f"{'  n/a':>{w}}" if v is None or not np.isfinite(v) else f"{v:>{w}.{nd}f}"


def _mean_nan(vals):
    v = [float(x) for x in vals if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def average_blocks(blocks: list[dict], bins) -> dict:
    """把多个 seed 的 S4 指标块按 seed 等权平均 (忽略 NaN), 与 S3 三 seed 汇总同做法。

    只平均标量与逐阶段标量; per_trajectory 明细不跨 seed 合并 (轨迹划分随 seed 变化,
    强行合并会造出不存在的"平均轨迹")。逐 seed 明细完整落在 JSON 里。
    """
    nb = len(bins)
    out = {"n_seeds": len(blocks)}
    for k in ("full_macro_rmse", "info_macro_rmse", "info_pooled_rmse"):
        out[k] = _mean_nan([b.get(k) for b in blocks])
    out["stage_bins"] = []
    for i in range(nb):
        rec = {"bin": list(bins[i])}
        for k in ("macro_rmse", "macro_mae", "macro_corr",
                  "pred_std_true_std_ratio", "pooled_rmse",
                  "pred_mean", "pred_std", "true_mean", "true_std"):
            rec[k] = _mean_nan([b["stage_metrics"]["bins"][i][k] for b in blocks])
        rec["n_points"] = int(_mean_nan(
            [b["stage_metrics"]["bins"][i]["n_points"] for b in blocks]) or 0)
        rec["n_trajectories"] = int(_mean_nan(
            [b["stage_metrics"]["bins"][i]["n_trajectories"] for b in blocks]) or 0)
        out["stage_bins"].append(rec)
    out["macro_ph"] = _mean_nan([b["prognostic_horizon"]["macro_ph"] for b in blocks])
    out["n_observable"] = int(_mean_nan(
        [b["prognostic_horizon"]["n_observable"] for b in blocks]) or 0)
    out["n_censored"] = int(_mean_nan(
        [b["prognostic_horizon"]["n_censored"] for b in blocks]) or 0)
    out["alpha_lambda"] = {}
    for lam in blocks[0]["alpha_lambda"]["by_lambda"]:
        out["alpha_lambda"][lam] = {
            "accuracy": _mean_nan([b["alpha_lambda"]["by_lambda"][lam]["accuracy"]
                                   for b in blocks]),
            "eligible_count": int(_mean_nan(
                [b["alpha_lambda"]["by_lambda"][lam]["eligible_count"]
                 for b in blocks]) or 0)}
    out["warning"] = {k: _mean_nan([b["warning"][k] for b in blocks])
                      for k in ("warning_coverage", "miss_rate",
                                "coverage_before_eol", "miss_rate_before_eol",
                                "false_alarm_rate", "conditional_lead",
                                "conditional_lead_before_eol")}
    for k in ("n_observed", "n_censored", "n_warned_observed",
              "n_warned_before_eol"):
        out["warning"][k] = int(_mean_nan([b["warning"][k] for b in blocks]) or 0)
    out["convergence"] = {k: _mean_nan([b["convergence"][k] for b in blocks])
                          for k in ("macro_convergence", "macro_late_convergence")}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="S4 统一评价 (协议 §14/§15/§18)")
    ap.add_argument("--config", default="configs/wheel.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    pcfg = prognostics_cfg(cfg)
    bins = pcfg["hi_bins"]
    labels = [f"[{lo:g},{hi:g})" for lo, hi in bins]

    print("=" * 96)
    print("S4 统一评价: S3 三个迁移组 + physical_extrap + trivial 基线 (同一 evaluator)")
    print("=" * 96)
    sha_before = sha256(S3_JSON)
    print(f">> S4 protocol sha256     : {sha256(S4_PROTOCOL)}")
    print(f">> S3 metrics sha256 (前) : {sha_before}")
    if sha_before != FROZEN_S3_METRICS_SHA:
        raise SystemExit("!! S3 指标文件已改动, 与 S4 协议 §1 指纹不符, 拒绝出数")
    frozen = json.loads(S3_JSON.read_text(encoding="utf-8"))
    print(f"   与协议 §1 冻结指纹一致 [OK]   S3 判词 (只读) = {frozen['verdict']}")

    if not STAGE_JSON.exists():
        raise SystemExit(f"!! 缺 {STAGE_JSON}; 先跑 python scripts/diag_s3_stages.py")
    stage = json.loads(STAGE_JSON.read_text(encoding="utf-8"))
    if stage["s3_metrics_sha256_before"] != sha_before:
        raise SystemExit("!! 缓存的逐 endpoint 预测对应的 S3 指纹不一致, 拒绝复用")
    seeds = [int(s) for s in stage["seeds"]]
    print(f">> 逐 endpoint 预测来自 {STAGE_JSON.name} (已通过与冻结 S3 的逐位校验)")
    print(f">> alpha={pcfg['alpha']} floor={pcfg['absolute_floor']} "
          f"lambdas={pcfg['lambdas']} warn<= {pcfg['rul_threshold']} "
          f"persistence={pcfg['persistence']} late_from={pcfg['late_from']}")

    per_method_blocks: dict[str, list[dict]] = {}
    per_seed_detail: dict[str, dict] = {}

    # ---- 1) S3 三个迁移组: 复用已校验的预测, 走同一 s4_metrics_block ----
    for m in stage["methods"]:
        blocks = []
        for s in seeds:
            r = stage["raw_predictions"][str(s)][m]
            blk = s4_metrics_block(np.asarray(r["pred"], dtype=float),
                                   np.asarray(r["true"], dtype=float),
                                   np.asarray(r["tids"], dtype=np.int64), cfg)
            blocks.append(blk)
            per_seed_detail.setdefault(m, {})[str(s)] = blk
        per_method_blocks[m] = blocks
        print(f">> {m:<22} 已评价 {len(blocks)} seeds")

    # ---- 2) physical_extrap + trivial 基线: 同一函数, 同一阈值 ----
    for s in seeds:
        ph = evaluate_physical(cfg, s, return_raw=True)
        if ph and "_raw" in ph:
            r = ph["_raw"]
            blk = s4_metrics_block(r["pred"], r["true"], r["tids"], cfg)
            per_method_blocks.setdefault("physical_extrap", []).append(blk)
            per_seed_detail.setdefault("physical_extrap", {})[str(s)] = blk
        tv = evaluate_trivial(cfg, s, return_raw=True)
        if tv and "_raw" in tv:
            raw = tv["_raw"]
            for name in TRIVIAL_METHODS:
                blk = s4_metrics_block(raw["pred"][name], raw["true"],
                                       raw["tids"], cfg)
                per_method_blocks.setdefault(name, []).append(blk)
                per_seed_detail.setdefault(name, {})[str(s)] = blk
    for name in ("physical_extrap",) + tuple(TRIVIAL_METHODS):
        n = len(per_method_blocks.get(name, []))
        tagg = "  [ORACLE_LIKE / NOT_DEPLOYABLE]" if name in ORACLE_LIKE else ""
        print(f">> {name:<22} 已评价 {n} seeds{tagg}")

    agg = {m: average_blocks(b, bins) for m, b in per_method_blocks.items() if b}
    order = list(stage["methods"]) + ["physical_extrap"] + list(TRIVIAL_METHODS)
    order = [m for m in order if m in agg]

    # ============================== 表 1 分阶段 RMSE ==============================
    print("\n" + "=" * 96)
    print(f"表1  分阶段 macro RMSE (按 HI 健康阶段, {len(seeds)} seeds 等权平均)")
    print("=" * 96)
    print(f"  {'method':<22}" + "".join(f"{l:>13}" for l in labels)
          + f"{'info_macro':>13}")
    for m in order:
        a = agg[m]
        row = "".join(_f(b["macro_rmse"], 13) for b in a["stage_bins"])
        tag = " *" if m in ORACLE_LIKE else ""
        print(f"  {m:<22}{row}{_f(a['info_macro_rmse'], 13)}{tag}")
    print("  * = ORACLE_LIKE / NOT_DEPLOYABLE, 不进入可部署方法排名 (协议 §15)")

    # ============================== 表 2 可解释性 ==============================
    print("\n" + "=" * 96)
    print("表2  可解释性: Prognostic Horizon / alpha-lambda / convergence")
    print("=" * 96)
    lam_keys = list(agg[order[0]]["alpha_lambda"].keys())
    print(f"  {'method':<22}{'PH(macro)':>11}"
          + "".join(f"{'a-l@' + k:>10}" for k in lam_keys)
          + f"{'converge':>11}{'late_conv':>11}{'n_obs':>7}{'n_cen':>7}")
    for m in order:
        a = agg[m]
        al = "".join(_f(a["alpha_lambda"][k]["accuracy"], 10) for k in lam_keys)
        tag = " *" if m in ORACLE_LIKE else ""
        print(f"  {m:<22}{_f(a['macro_ph'], 11)}{al}"
              f"{_f(a['convergence']['macro_convergence'], 11)}"
              f"{_f(a['convergence']['macro_late_convergence'], 11)}"
              f"{a['n_observable']:>7d}{a['n_censored']:>7d}{tag}")
    print("  PH / convergence 越大越好 / 越小越好; alpha-lambda 分母 = 轨迹数 (非时间点数)")

    # ============================== 表 3 预警 ==============================
    print("\n" + "=" * 96)
    print(f"表3  预警能力 (pred_rul <= {pcfg['rul_threshold']} 连续 "
          f"{pcfg['persistence']} 个 endpoint)")
    print("=" * 96)
    print(f"  {'method':<22}{'cover_all':>11}{'cover<EOL':>11}{'miss<EOL':>10}"
          f"{'false_alarm':>13}{'lead<EOL':>10}{'n_warn<EOL':>12}{'n_obs':>7}{'n_cen':>7}")
    for m in order:
        w = agg[m]["warning"]
        tag = " *" if m in ORACLE_LIKE else ""
        print(f"  {m:<22}{_f(w['warning_coverage'], 11)}"
              f"{_f(w['coverage_before_eol'], 11)}{_f(w['miss_rate_before_eol'], 10)}"
              f"{_f(w['false_alarm_rate'], 13)}"
              f"{_f(w['conditional_lead_before_eol'], 10)}"
              f"{w['n_warned_before_eol']:>12d}{w['n_observed']:>7d}"
              f"{w['n_censored']:>7d}{tag}")
    print("  whole-risk (coverage/miss_rate/false_alarm) 与 conditional lead 必须同表读:")
    print("  lead 只在报了警的轨迹上平均, 单看它会把漏报藏起来 (协议 §9)。")
    print("  cover_all 计入**失效后才报出**的警报; cover<EOL / miss<EOL / lead<EOL 只计")
    print("  EOL 之前报出的, 才是可部署性口径 —— 本数据 tau 可达中位 4.85 (EOL 后仍有")
    print("  采样点), 两者差异正是'事后报警'的规模。")
    print(f"  test 划分全部为 event-observed (n_cen=0) ⇒ false_alarm 分母为空 = n/a;")
    print("  删失语义由 tests/test_metrics_s4.py 的构造场景 E/F 独立验证。")

    # ============================== 表 4 动态范围 ==============================
    print("\n" + "=" * 96)
    print("表4  输出动态范围 PSR = pred_std / true_std (阈值 "
          f"{cfg['s25']['psr_min']})")
    print("=" * 96)
    print(f"  {'method':<22}" + "".join(f"{l:>13}" for l in labels)
          + f"{'early(HI<.5)':>14}{'late(HI>=.5)':>14}")
    psr_min = float(cfg["s25"]["psr_min"])
    for m in order:
        a = agg[m]
        vals = [b["pred_std_true_std_ratio"] for b in a["stage_bins"]]
        early = _mean_nan(vals[:2])
        late = _mean_nan(vals[2:])
        row = "".join(_f(v, 13) for v in vals)
        tag = " *" if m in ORACLE_LIKE else ""
        print(f"  {m:<22}{row}{_f(early, 14)}{_f(late, 14)}{tag}")
    below = [m for m in order if m not in ORACLE_LIKE and all(
        (not np.isfinite(b["pred_std_true_std_ratio"]))
        or b["pred_std_true_std_ratio"] < psr_min for b in agg[m]["stage_bins"])]
    print(f"  全阶段 PSR < {psr_min} 的可部署方法: {below}")

    sha_after = sha256(S3_JSON)
    print(f"\n>> S3 metrics sha256 (后) : {sha_after}")
    print(f"   与运行前一致: {sha_before == sha_after}")
    if sha_before != sha_after:
        raise SystemExit("!! S3 指标文件在评价过程中被改动 —— 违反 S4 协议")

    OUT_JSON.write_text(json.dumps(
        {"s4_protocol_sha256": sha256(S4_PROTOCOL),
         "s3_metrics_sha256_before": sha_before,
         "s3_metrics_sha256_after": sha_after,
         "s3_verdict_readonly": frozen["verdict"], "seeds": seeds,
         "prognostics_config": pcfg, "psr_min": psr_min,
         "deployable_baselines": list(DEPLOYABLE_BASELINES),
         "oracle_like": list(ORACLE_LIKE),
         "aggregate": agg, "per_seed": per_seed_detail},
        indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f">> S4 指标 -> {OUT_JSON}")
    print(">> S3 判词保持不变: " + frozen["verdict"])


if __name__ == "__main__":
    main()
