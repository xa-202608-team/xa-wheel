"""S3 分阶段只读诊断 (S4 协议 §12/§13)。

**只读**已冻结的 S3 结果: 不重训调参、不写任何新 checkpoint、不改任何 S3 数字。

--- 为什么需要"确定性复现"而不是"读 checkpoint" ---
S3 gate 运行时没有保存 per-group model checkpoint (`checkpoints/` 只有
`source_tcn_pretrain.pt`; `_train_with_early_stop` 的 best state 只存在内存)。
所以协议 §12 字面要求的"读冻结 best checkpoint + 校验 hash"没有可读对象。

替代路径 (协议 §12 已记录并实测):
  1. 用**冻结的** prepare_s3 + train_group 在内存里确定性复现 best model;
  2. **硬校验** 复现出的 info_macro_rmse 与 checkpoints/s3_gate_metrics.json 里
     同 (method, seed) 行**逐位相等**, 不等立即报错退出;
  3. 逐 endpoint prediction 通过包裹 `collect_pred` **捕获**得到 ——
     捕获的就是 S3 当时用来出数的那一批预测, 不做二次推理。
不写任何 .pt。不修改 scripts/run_s3_gate.py (只在本脚本内做函数包裹)。

回答协议 §13 的四个问题:
  Q1 PSR<0.30 是全阶段还是集中在 early/late?
  Q2 source_mmd 的 +0.0613 平均 RMSE gain 主要来自哪个 HI 阶段?
  Q3 source_mmd 在早期 (HI<0.5) 是否优于 target_only?
  Q4 source_mmd 是否只是输出更接近某个常数?
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

import scripts.run_s3_gate as s3g                                  # noqa: E402
from src.baselines.physical_extrap import caliber_mask             # noqa: E402
from src.experiments.metrics import (attach_axis, prognostics_cfg,  # noqa: E402
                                     target_endpoint_axis)
from src.experiments.run_groups import s4_metrics_block            # noqa: E402
from src.utils import load_config                                  # noqa: E402

# stage_block 需要 cfg 但签名已被上游固定; 用单元素列表在 main 里注入, 避免全局可变状态
_CFG: list = [None]

S3_JSON = ROOT / "checkpoints" / "s3_gate_metrics.json"
S4_PROTOCOL = ROOT / "docs" / "diagnostics_s4_protocol.md"
OUT_JSON = ROOT / "checkpoints" / "s4_stage_diagnostics.json"
# 协议 §1 记录的冻结指纹 —— 不匹配即拒绝出数
FROZEN_S3_METRICS_SHA = ("f11f46b107168fe0d21e16aa5c286789580a0eec073f8cd879c4"
                         "ac6f9528f68e")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def capture_s3_predictions(cfg: dict, seed: int, mode: str, frozen_row: dict) -> dict:
    """确定性复现 (mode, seed) 的 S3 best model, 捕获其逐 endpoint prediction。

    包裹 s3g.collect_pred: train_group 在 best checkpoint 确定后调用它一次, 我们借此
    拿到 (pred, true, tids) —— 与 S3 出数用的完全是同一批数组, 不做二次推理。
    完成后无条件恢复原函数 (即使中途抛错), 避免污染同进程后续调用。
    """
    box = {}
    orig = s3g.collect_pred

    def wrapped(model, loader, device):
        out = orig(model, loader, device)
        box["pred"], box["true"], box["tids"] = out
        return out

    s3g.collect_pred = wrapped
    try:
        data = s3g.prepare_s3(cfg, seed, verbose=False)
        row = s3g.train_group(mode, cfg, data, seed)
    finally:
        s3g.collect_pred = orig

    # ---- 硬校验: 复现值必须与冻结值逐位相等 (代替不存在的 checkpoint hash 校验) ----
    for k in ("info_macro_rmse", "info_pooled_rmse", "full_macro_rmse", "psr"):
        got, exp = float(row[k]), float(frozen_row[k])
        if repr(got) != repr(exp):
            raise SystemExit(
                f"!! 复现与冻结 S3 不一致 ({mode} seed={seed} {k}): "
                f"repro={got!r} frozen={exp!r} delta={abs(got - exp)!r}\n"
                f"   诊断必须建立在与冻结数字逐位一致的模型上, 拒绝出数。")
    if "pred" not in box:
        raise SystemExit(f"!! 未捕获到 {mode} seed={seed} 的逐 endpoint prediction")
    return {"pred": box["pred"], "true": box["true"], "tids": box["tids"],
            "row": row, "cap_eps": data["cap_eps"]}


def stage_block(pred, true, tids, hi, tau, ev, pcfg, cap_eps) -> dict:
    """分阶段 + 预后性指标。

    直接委托 `run_groups.s4_metrics_block` —— 与 `scripts/run_s4_metrics.py` 及所有
    基线**共用同一个函数**, 不在本脚本内维护第二份口径 (协议 §1 纪律 5)。
    分阶段指标走 info 掩码 (与 S3 表 3 的 PSR 同口径), 预后性指标走全部 endpoint。
    """
    blk = s4_metrics_block(pred, true, tids, _CFG[0], hi=hi)
    return {"n_info": int(caliber_mask(np.asarray(true, dtype=float), "info",
                                      cap_eps).sum()),
            "stage_metrics": blk["stage_metrics"],
            "stage_caliber": blk["stage_caliber"],
            "prognostics": {k: blk[k] for k in ("prognostic_horizon",
                                                "alpha_lambda", "warning",
                                                "convergence")}}


def _fmt(v, w=8, nd=4):
    return f"{'   n/a':>{w}}" if v is None or not np.isfinite(v) else f"{v:>{w}.{nd}f}"


def print_stage_table(title: str, per_method: dict, bins) -> None:
    """per_method[m]["bins"] = 逐 HI 阶段的指标 dict 列表。"""
    labels = [f"[{lo:g},{hi:g})" for lo, hi in bins]
    for key, head in (("macro_rmse", "macro RMSE"), ("macro_mae", "macro MAE"),
                      ("macro_corr", "macro corr"),
                      ("pred_std_true_std_ratio", "PSR")):
        print(f"\n  -- {title}: {head} (按 HI 阶段) --")
        print(f"  {'method':<22}" + "".join(f"{l:>14}" for l in labels))
        for m, blk in per_method.items():
            row = "".join(_fmt(b[key], 14) for b in blk["bins"])
            print(f"  {m:<22}{row}")
    print(f"\n  -- {title}: 样本数 / 轨迹数 (按 HI 阶段) --")
    print(f"  {'method':<22}" + "".join(f"{l:>14}" for l in labels))
    for m, blk in per_method.items():
        row = "".join(f"{b['n_points']:>7d}/{b['n_trajectories']:<6d}"
                      for b in blk["bins"])
        print(f"  {m:<22}{row}")


def answer_questions(agg: dict, bins, psr_min: float) -> dict:
    """协议 §13 Q1–Q4 —— 用数字回答, 不做修辞。"""
    labels = [f"[{lo:g},{hi:g})" for lo, hi in bins]
    ans = {}

    # Q1: PSR<psr_min 是全阶段还是集中在某些阶段
    q1 = {}
    for m, blk in agg.items():
        per = [b["pred_std_true_std_ratio"] for b in blk["bins"]]
        fin = [(l, v) for l, v in zip(labels, per) if np.isfinite(v)]
        q1[m] = {"per_stage": {l: v for l, v in zip(labels, per)},
                 "all_stages_below": bool(fin) and all(v < psr_min for _l, v in fin),
                 "stages_above": [l for l, v in fin if v >= psr_min],
                 "max_stage": max(fin, key=lambda kv: kv[1])[0] if fin else None,
                 "max_value": max(v for _l, v in fin) if fin else float("nan")}
    ans["Q1_psr_collapse_scope"] = q1

    # Q2: source_mmd 的 RMSE gain 按阶段分解 (gain = target_only - source_mmd, >0 = 更好)
    tgt = agg.get("target_only", {}).get("bins", [])
    mmd = agg.get("source_mmd_finetune", {}).get("bins", [])
    q2 = {}
    for l, a, b in zip(labels, tgt, mmd):
        ga = (a["macro_rmse"] - b["macro_rmse"]
              if np.isfinite(a["macro_rmse"]) and np.isfinite(b["macro_rmse"])
              else float("nan"))
        q2[l] = {"target_only_macro_rmse": a["macro_rmse"],
                 "source_mmd_macro_rmse": b["macro_rmse"], "gain": ga,
                 "n_points": b["n_points"], "n_trajectories": b["n_trajectories"]}
    fin = [(l, v["gain"]) for l, v in q2.items() if np.isfinite(v["gain"])]
    ans["Q2_gain_by_stage"] = {
        "per_stage": q2,
        "largest_gain_stage": max(fin, key=lambda kv: kv[1])[0] if fin else None,
        "largest_loss_stage": min(fin, key=lambda kv: kv[1])[0] if fin else None}

    # Q3: 早期 (HI<0.5, 即前两个箱) source_mmd 是否优于 target_only
    early = [l for l in labels if float(l[1:l.index(",")]) < 0.5]
    e_gain = [q2[l]["gain"] for l in early if np.isfinite(q2[l]["gain"])]
    ans["Q3_early_advantage"] = {
        "early_stages": early, "early_gains": e_gain,
        "mean_early_gain": float(np.mean(e_gain)) if e_gain else float("nan"),
        "source_mmd_better_early": bool(e_gain) and float(np.mean(e_gain)) > 0}

    # Q4: 是否只是逼近某个常数 —— 逐阶段 pred/true 的 mean/std
    q4 = {}
    for m, blk in agg.items():
        q4[m] = {l: {"pred_mean": b["pred_mean"], "pred_std": b["pred_std"],
                     "true_mean": b["true_mean"], "true_std": b["true_std"]}
                 for l, b in zip(labels, blk["bins"])}
        pm = [b["pred_mean"] for b in blk["bins"] if np.isfinite(b["pred_mean"])]
        q4[m]["_pred_mean_spread_across_stages"] = (
            float(max(pm) - min(pm)) if pm else float("nan"))
    ans["Q4_near_constant"] = q4
    return ans


def aggregate_over_seeds(per_seed: dict, methods, bins) -> dict:
    """把逐 seed 的分阶段指标按 seed 等权平均 (忽略 NaN); 与 S3 三 seed 汇总同做法。"""
    nb = len(bins)
    out = {}
    for m in methods:
        rows = []
        for b in range(nb):
            acc = {}
            for k in ("macro_rmse", "macro_mae", "macro_corr",
                      "pred_std_true_std_ratio", "pooled_rmse", "pooled_mae",
                      "pred_mean", "pred_std", "true_mean", "true_std"):
                v = [per_seed[s][m]["stage_metrics"]["bins"][b][k]
                     for s in per_seed if m in per_seed[s]]
                v = [x for x in v if np.isfinite(x)]
                acc[k] = float(np.mean(v)) if v else float("nan")
            n = [per_seed[s][m]["stage_metrics"]["bins"][b] for s in per_seed
                 if m in per_seed[s]]
            acc["n_points"] = int(np.mean([x["n_points"] for x in n])) if n else 0
            acc["n_trajectories"] = (int(np.mean([x["n_trajectories"] for x in n]))
                                     if n else 0)
            acc["bin"] = list(bins[b])
            rows.append(acc)
        out[m] = {"bins": rows}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="S3 分阶段只读诊断 (S4 §12/§13)")
    ap.add_argument("--config", default="configs/wheel.yaml")
    ap.add_argument("--no-reuse", action="store_true",
                    help="忽略已缓存的逐 endpoint 预测, 强制重新确定性复现")
    args = ap.parse_args()

    cfg = load_config(args.config)
    _CFG[0] = cfg
    pcfg = prognostics_cfg(cfg)
    psr_min = float(cfg["s25"]["psr_min"])

    print("=" * 78)
    print("S3 分阶段只读诊断 (S4 协议 §12/§13) —— 不重训调参 / 不写 checkpoint / 不改 S3")
    print("=" * 78)
    if not S3_JSON.exists():
        raise SystemExit(f"!! 缺 {S3_JSON}; 先跑 python scripts/run_s3_gate.py")
    sha_before = sha256(S3_JSON)
    print(f">> S4 protocol sha256      : {sha256(S4_PROTOCOL)}")
    print(f">> S3 metrics sha256 (前)  : {sha_before}")
    if sha_before != FROZEN_S3_METRICS_SHA:
        raise SystemExit("!! S3 指标文件已被改动, 与 S4 协议 §1 记录的指纹不符, 拒绝出数")
    print("   与 S4 协议 §1 冻结指纹一致 [OK]")

    frozen = json.loads(S3_JSON.read_text(encoding="utf-8"))
    seeds = [int(s) for s in cfg["transfer"]["s3_gate_seeds"]]
    methods = list(s3g.GROUPS)
    print(f">> S3 判词 (只读, 不改)    : {frozen['verdict']}")
    print(f">> seeds {seeds}  methods {methods}")
    print(f">> HI 阶段分箱 {pcfg['hi_bins']}   PSR 阈值 {psr_min}")

    axis = target_endpoint_axis(cfg)

    # 复用已缓存的逐 endpoint 预测: 缓存里记录了当时的 S3 指标指纹, 只有指纹一致才复用
    # —— 这不是"跳过校验", 而是把已经通过逐位校验的那批预测直接拿来重算指标。
    cache = {}
    if OUT_JSON.exists() and not args.no_reuse:
        old = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        if old.get("s3_metrics_sha256_before") == sha_before:
            cache = old.get("raw_predictions", {})
            if cache:
                print(">> 复用缓存的逐 endpoint 预测 (S3 指纹一致); "
                      "加 --no-reuse 可强制重新复现")

    per_seed, raw_store = {}, {}
    for s in seeds:
        per_seed[str(s)] = {}
        for m in methods:
            fr = [r for r in frozen["rows"][m] if int(r["seed"]) == s]
            if not fr:
                raise SystemExit(f"!! 冻结 JSON 里没有 {m} seed={s} 的行")
            hit = cache.get(str(s), {}).get(m)
            if hit:
                pred = np.asarray(hit["pred"], dtype=float)
                true = np.asarray(hit["true"], dtype=float)
                tid = np.asarray(hit["tids"], dtype=np.int64)
                cap_eps = float(cfg["evaluation"]["capped_eps"])
                print(f">> 复用 {m} seed={s} ({len(pred)} endpoints)")
            else:
                print(f"\n>> 复现并捕获 {m} seed={s} ...", flush=True)
                cap = capture_s3_predictions(cfg, s, m, fr[0])
                print(f"   逐位校验通过: "
                      f"info_macro_rmse={cap['row']['info_macro_rmse']!r}"
                      f" == frozen {fr[0]['info_macro_rmse']!r}")
                pred, true, tid = cap["pred"], cap["true"], cap["tids"]
                cap_eps = cap["cap_eps"]
            tau, hi, ev = attach_axis(tid, axis)
            per_seed[str(s)][m] = stage_block(pred, true, tid, hi, tau, ev,
                                             pcfg, cap_eps)
            raw_store.setdefault(str(s), {})[m] = {
                "pred": np.asarray(pred).tolist(),
                "true": np.asarray(true).tolist(),
                "tids": np.asarray(tid).tolist()}

    agg = aggregate_over_seeds(per_seed, methods, pcfg["hi_bins"])
    print("\n" + "=" * 78)
    print("三 seed 等权平均的分阶段指标")
    print("=" * 78)
    print_stage_table("3-seed 平均", agg, pcfg["hi_bins"])

    ans = answer_questions(agg, pcfg["hi_bins"], psr_min)
    labels = [f"[{lo:g},{hi:g})" for lo, hi in pcfg["hi_bins"]]

    print("\n" + "=" * 78)
    print("协议 §13 必答: 坍缩在哪里")
    print("=" * 78)
    print(f"\nQ1 PSR < {psr_min} 是全阶段还是集中在 early/late?")
    for m, v in ans["Q1_psr_collapse_scope"].items():
        print(f"  {m:<22} 全阶段低于阈值={str(v['all_stages_below']):<5} "
              f"超阈阶段={v['stages_above'] or '无'}  "
              f"最大 PSR={_fmt(v['max_value'], 7)} @ {v['max_stage']}")

    print("\nQ2 source_mmd 的 RMSE gain 主要来自哪个 HI 阶段? "
          "(gain = target_only − source_mmd, >0 = source_mmd 更好)")
    print(f"  {'stage':<14}{'target_only':>14}{'source_mmd':>14}{'gain':>12}"
          f"{'n_pts':>8}{'n_traj':>8}")
    for l in labels:
        v = ans["Q2_gain_by_stage"]["per_stage"][l]
        print(f"  {l:<14}{_fmt(v['target_only_macro_rmse'], 14)}"
              f"{_fmt(v['source_mmd_macro_rmse'], 14)}{_fmt(v['gain'], 12)}"
              f"{v['n_points']:>8d}{v['n_trajectories']:>8d}")
    print(f"  最大增益阶段 = {ans['Q2_gain_by_stage']['largest_gain_stage']}   "
          f"最大退化阶段 = {ans['Q2_gain_by_stage']['largest_loss_stage']}")

    q3 = ans["Q3_early_advantage"]
    print(f"\nQ3 source_mmd 在早期 (HI<0.5, 阶段 {q3['early_stages']}) 是否优于 "
          f"target_only?  {'是' if q3['source_mmd_better_early'] else '否'}"
          f"  (早期平均 gain = {_fmt(q3['mean_early_gain'], 7)})")

    print("\nQ4 source_mmd 是否只是输出更接近某个常数? (按健康阶段分列 pred/true)")
    print(f"  {'method':<22}{'stage':<14}{'pred_mean':>11}{'pred_std':>11}"
          f"{'true_mean':>11}{'true_std':>11}")
    for m in methods:
        for l in labels:
            v = ans["Q4_near_constant"][m][l]
            print(f"  {m:<22}{l:<14}{_fmt(v['pred_mean'], 11)}{_fmt(v['pred_std'], 11)}"
                  f"{_fmt(v['true_mean'], 11)}{_fmt(v['true_std'], 11)}")
        sp = ans["Q4_near_constant"][m]["_pred_mean_spread_across_stages"]
        print(f"  {'':<22}→ pred_mean 跨阶段极差 = {_fmt(sp, 7)} "
              f"(越接近 0 越像常数预测器)")

    sha_after = sha256(S3_JSON)
    print(f"\n>> S3 metrics sha256 (后)  : {sha_after}")
    print(f"   与运行前一致: {sha_before == sha_after}")
    if sha_before != sha_after:
        raise SystemExit("!! S3 指标文件在诊断过程中被改动 —— 违反 S4 协议")

    OUT_JSON.write_text(json.dumps(
        {"s4_protocol_sha256": sha256(S4_PROTOCOL),
         "s3_metrics_sha256_before": sha_before,
         "s3_metrics_sha256_after": sha_after,
         "s3_verdict_readonly": frozen["verdict"],
         "seeds": seeds, "methods": methods, "hi_bins": pcfg["hi_bins"],
         "psr_min": psr_min, "per_seed": per_seed, "aggregate": agg,
         "answers": ans, "raw_predictions": raw_store},
        indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f">> 分阶段诊断 (含逐 endpoint 预测) -> {OUT_JSON}")
    print(">> S3 判词保持不变: " + frozen["verdict"])


if __name__ == "__main__":
    main()
