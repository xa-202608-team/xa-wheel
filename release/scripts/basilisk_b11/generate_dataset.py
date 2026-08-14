#!/usr/bin/env python
"""scripts/basilisk_b11/generate_dataset.py

Basilisk-B1.1 §13/§14 —— 用**冻结**的 candidate 正式生成 150 条 B1.1 数据集。

  输出 data/sim/wheel_basilisk_b11/seed_<seed>_<candidate>/wheel_all.h5
  硬拒绝写入 v1 / B1 / analytic 任何命名空间。

前置条件 (硬校验): checkpoints/basilisk_b11/frozen_calibration.json 存在, 且当前
config 重算出的 calibration_hash 与冻结值完全一致 —— 否则拒绝运行 (§12)。

§14 正式数据 Gate (10 条) 在此评估。**若 150 条因随机波动跌出 candidate 区间,
不自动重调任何参数** —— 直接判 `B11_DATA_NOT_READY` 并以非零码退出。

本阶段不运行任何 RUL 模型 (§17)。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "basilisk_b11"))

import calibrate_degradation as cal                      # noqa: E402
from generate_candidates import (content_hash, guard_out,    # noqa: E402
                                 traj_row, mode_conditioned_wear)
from audit_candidates import (mode_severity_from_p95_reference,  # noqa: E402
                              spearman)
from src.sim.basilisk_bridge import DUTY_KEYS, MODES      # noqa: E402

FROZEN = ROOT / "checkpoints" / "basilisk_b11" / "frozen_calibration.json"

# §14 的 10 条 gate 名称 (顺序固定)
DATA_GATE_NAMES = (
    "1 event-observed fraction >= 下限",
    "2 censored fraction >= 下限",
    "3 event-observed 轨迹数 >= 下限",
    "4 early-EOL fraction < 上限",
    "5 EOL IQR > 0 (event-observed)",
    "6 p95(initial current ratio) <= 上限",
    "7 无 NaN / Inf",
    "8 physics range 在硬界内",
    "9 mode-conditioned wear ordering 一致",
    "10 dataset 确定性 (content hash 可复算)",
)


def eval_data_gates(rows: list[dict], cfg: dict, mcw: dict,
                    order: list[str], det_ok: bool, det_detail: str) -> dict:
    """§14 的 10 条正式数据 Gate。阈值全部走 config, 不硬编码。"""
    dg = cfg["data_gate"]
    g = cfg["gate"]
    n = len(rows)
    obs = [r for r in rows if r["failed"]]
    n_obs = len(obs)
    ff = n_obs / n
    cf = (n - n_obs) / n
    out = {}

    out[DATA_GATE_NAMES[0]] = (
        bool(ff >= float(dg["event_observed_frac_min"])),
        f"event-observed fraction = {ff:.4f} >= {dg['event_observed_frac_min']}")
    out[DATA_GATE_NAMES[1]] = (
        bool(cf >= float(dg["censored_frac_min"])),
        f"censored fraction = {cf:.4f} >= {dg['censored_frac_min']}")
    out[DATA_GATE_NAMES[2]] = (
        bool(n_obs >= int(dg["min_event_observed_traj"])),
        f"n_event_observed = {n_obs} >= {dg['min_event_observed_traj']}")

    n_early = sum(r["early_eol"] for r in rows)
    fe = n_early / n
    out[DATA_GATE_NAMES[3]] = (
        bool(fe < float(dg["early_eol_frac_max"])),
        f"early-EOL fraction = {fe:.4f} < {dg['early_eol_frac_max']} "
        f"({n_early}/{n}; 早期 = EOL < {g['early_eol_horizon_frac']}×horizon)")

    if obs:
        e = np.array([r["eol_idx"] for r in obs], dtype=float)
        iqr = float(np.percentile(e, 75) - np.percentile(e, 25))
        ok5, d5 = bool(iqr > float(g["eol_iqr_min"])), \
            f"event-observed EOL IQR = {iqr:.1f} > {g['eol_iqr_min']}"
    else:
        iqr, ok5, d5 = float("nan"), False, "无 event-observed 轨迹 (n=0, 不伪造 0)"
    out[DATA_GATE_NAMES[4]] = (ok5, d5)

    imr = np.array([r["initial_margin_ratio"] for r in rows], dtype=float)
    p95 = float(np.percentile(imr, 95))
    out[DATA_GATE_NAMES[5]] = (
        bool(p95 <= float(dg["initial_margin_ratio_max"])),
        f"p95(initial current ratio) = {p95:.4f} <= "
        f"{dg['initial_margin_ratio_max']} (median={np.median(imr):.4f}, "
        f"max={imr.max():.4f})")

    n_nan = sum(r["n_nan"] for r in rows)
    out[DATA_GATE_NAMES[6]] = (
        bool(n_nan == 0 and all(r["all_finite"] for r in rows)),
        f"NaN 总数 {n_nan}, all_finite {sum(r['all_finite'] for r in rows)}/{n}")

    Tmin = min(r["T_min"] for r in rows); Tmax = max(r["T_max"] for r in rows)
    Imax = max(r["Im_absmax"] for r in rows)
    out[DATA_GATE_NAMES[7]] = (
        bool(Tmin > float(g["temperature_min_K"])
             and Tmax < float(g["temperature_max_K"])
             and Imax < float(g["current_absmax_A"])),
        f"T ∈ [{Tmin:.1f}, {Tmax:.1f}] K (界 [{g['temperature_min_K']}, "
        f"{g['temperature_max_K']}]), |I_m|max = {Imax:.4f} A "
        f"(界 {g['current_absmax_A']})")

    present = [m for m in order if mcw[m]["n_windows"] > 0]
    if len(present) >= 3:
        rank_sev = np.array([order.index(m) for m in present], float)
        meas = np.array([mcw[m]["mean_g_duty"] for m in present], float)
        rho = spearman(-rank_sev, meas)
        ok9 = bool(np.isfinite(rho) and rho > float(g["wear_ordering_min_spearman"]))
        d9 = (f"spearman(严酷度排序, 实测 mean g_duty) = {rho:.4f} > "
              f"{g['wear_ordering_min_spearman']}; 覆盖 mode {present}")
    else:
        rho, ok9 = float("nan"), False
        d9 = f"仅 {len(present)} 个 mode 有窗 (n={len(present)})"
    out[DATA_GATE_NAMES[8]] = (ok9, d9)

    out[DATA_GATE_NAMES[9]] = (bool(det_ok), det_detail)

    return {"gates": out, "all_pass": all(v[0] for v in out.values()),
            "derived": {
                "event_observed_fraction": float(ff),
                "censored_fraction": float(cf),
                "n_event_observed": int(n_obs), "n_censored": int(n - n_obs),
                "early_eol_fraction": float(fe), "n_early_eol": int(n_early),
                "eol_iqr_observed": iqr,
                "initial_margin_ratio_p95": p95,
                "initial_margin_ratio_median": float(np.median(imr)),
                "initial_margin_ratio_max": float(imr.max()),
                "n_nan_total": int(n_nan),
                "T_min_K": float(Tmin), "T_max_K": float(Tmax),
                "Im_absmax_A": float(Imax),
                "wear_ordering_spearman": rho}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wheel_basilisk_b11.yaml")
    ap.add_argument("--n-traj", type=int, default=150)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--json", default="checkpoints/basilisk_b11/dataset_stats.json")
    ap.add_argument("--md", default="docs/basilisk_b11/simulation_report.md")
    a = ap.parse_args()

    import h5py

    if not FROZEN.exists():
        print(f"!! 缺 {FROZEN.relative_to(ROOT)}; 先跑 freeze_candidate.py (§12)")
        print(">> B11_DATA_NOT_READY (无冻结标定, 拒绝生成)")
        return 1
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))

    cfg = cal.load_b11_config(a.config)
    seed = a.seed if a.seed is not None else int(cfg["seed"])
    cname = frozen["selected_candidate"]
    hs = float(frozen["protocol"]["horizon_scale"])
    n_traj = int(a.n_traj)

    # ---- §12 冻结校验: 重算 calibration_hash 必须一致 ----
    from freeze_candidate import build_protocol
    cand = json.loads((ROOT / "checkpoints/basilisk_b11/candidates.json")
                      .read_text(encoding="utf-8"))
    audit = json.loads((ROOT / "checkpoints/basilisk_b11/candidate_audit.json")
                       .read_text(encoding="utf-8"))
    rec = json.loads((ROOT / "checkpoints/basilisk_b11/protocol_hash.json")
                     .read_text(encoding="utf-8"))
    h_now = cal.calibration_protocol_hash(build_protocol(cfg, cand, audit, rec))
    if h_now != frozen["calibration_hash"]:
        print(f"!! calibration_hash 变了: {frozen['calibration_hash'][:16]} "
              f"-> {h_now[:16]}")
        print("!! §12 冻结后禁止改标定, 拒绝生成数据集")
        return 1

    ctx = cal.prepare_context(cfg, [hs])
    n_win = int(round(ctx["n_base"] * hs))
    out_dir = ROOT / cfg["paths"]["sim_dir"] / f"seed_{seed}_{cname}"
    guard_out(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spy = 365.25 * 24 * 3600
    dt = float(cfg["sim"]["sample_period_s"])
    print("=" * 78)
    print(f"Basilisk-B1.1 §13 正式数据集生成  n_traj={n_traj}")
    print(f"  candidate            = {cname} (冻结)")
    print(f"  calibration hash     = {h_now[:32]}... [MATCH]")
    print(f"  reference caliber    = {ctx['ref']['caliber']}")
    print(f"  horizon              = {n_win} 窗 × {dt:.0f}s = {n_win*dt/spy:.2f} 年")
    print(f"  derived b0_scale     = {ctx['calib']['b0_scale']:.9f}")
    print(f"  每退化窗             = {ctx['n_per_window']} 个 1 s Basilisk 点")
    print(f"  输出                 = {out_dir.relative_to(ROOT)}")
    print("=" * 78)

    t0 = time.time()
    dfs, rows, plist, drives_all, duty_all = [], [], [], [], []
    for i in range(n_traj):
        df, eol, failed, p, duty, drives = cal.build_trajectory(i, seed, cfg, ctx, hs)
        dfs.append(df); plist.append(p); drives_all.append(drives)
        duty_all.append(duty)
        rows.append(traj_row(i, df, eol, failed, p, duty, drives, n_win, cfg))
        if (i + 1) % 25 == 0:
            print(f"   traj {i+1:3d}/{n_traj}  "
                  f"observed={sum(r['failed'] for r in rows)}")
    elapsed = time.time() - t0

    ch = content_hash(dfs, plist)
    # 确定性证据: 对前 3 条重建并比对 content hash (完整 150 条重跑代价过高;
    # 逐轨迹 RNG 只依赖 (seed, traj_id), 前缀一致即全体一致)
    n_det = min(3, n_traj)
    det_dfs, det_p = [], []
    for i in range(n_det):
        d2, _, _, p2, _, _ = cal.build_trajectory(i, seed, cfg, ctx, hs)
        det_dfs.append(d2); det_p.append(p2)
    det_ok = content_hash(det_dfs, det_p) == content_hash(dfs[:n_det], plist[:n_det])
    det_detail = (f"重建前 {n_det} 条轨迹, content hash "
                  f"{'逐位一致' if det_ok else '**不一致**'} "
                  f"(逐轨迹 RNG 只依赖 (seed, traj_id), 无全局 seed)")

    h5p = out_dir / "wheel_all.h5"
    with h5py.File(h5p, "w") as f:
        for i, df in enumerate(dfs):
            g = f.create_group(f"traj_{i:03d}")
            for c in df.columns:
                g.create_dataset(c, data=df[c].values)
            for k, v in plist[i].items():
                g.attrs[k] = float(v)
            mg = g.create_group("mission_features")
            for k in DUTY_KEYS:
                mg.create_dataset(k, data=duty_all[i]["stats"][k].astype(np.float32))
            mg.create_dataset("mode_id", data=duty_all[i]["mode_id"].astype(np.int16))
            for m, wv in duty_all[i]["mode_weights"].items():
                mg.attrs[f"weight_{m}"] = float(wv)
            mg.attrs["n_per_window"] = int(duty_all[i]["n_per_window"])
        f.attrs["lineage"] = "basilisk_b11"
        f.attrs["candidate"] = cname
        f.attrs["horizon_scale"] = hs
        f.attrs["seed"] = int(seed)
        f.attrs["n_traj"] = n_traj
        f.attrs["content_sha256"] = ch
        f.attrs["calibration_hash"] = h_now
        f.attrs["reference_caliber"] = str(ctx["ref"]["caliber"])
        f.attrs["b0_scale"] = float(ctx["calib"]["b0_scale"])
        f.attrs["contains_rul_label"] = False
        f.attrs["n_per_window"] = int(ctx["n_per_window"])
    (out_dir / "params.json").write_text(json.dumps(
        [{k: float(v) for k, v in p.items()} for p in plist], indent=2),
        encoding="utf-8")

    # ---------- §13 审计 ----------
    n_obs = sum(r["failed"] for r in rows)
    obs_rows = [r for r in rows if r["failed"]]
    eols = np.array([r["eol_idx"] for r in rows], dtype=float)
    e_obs = (np.array([r["eol_idx"] for r in obs_rows], dtype=float)
             if obs_rows else np.array([]))
    total_rows = int(sum(r["n"] for r in rows))
    mode_frac = {m: float(np.mean([r["mode_fractions"][m] for r in rows]))
                 for m in MODES}
    n_nan = sum(r["n_nan"] for r in rows)
    post_eol = sum(r["n_post_eol_rows"] for r in rows)
    imr = np.array([r["initial_margin_ratio"] for r in rows], dtype=float)
    n_early = sum(r["early_eol"] for r in rows)
    mcw = mode_conditioned_wear(drives_all, duty_all, ctx["ref"], cfg["wear_drive"])
    order, mode_gd = mode_severity_from_p95_reference(rec, cfg)

    print()
    print(f">> trajectory count       : {n_traj}")
    print(f">> event-observed         : {n_obs}")
    print(f">> censored               : {n_traj - n_obs}")
    print(f">> failure fraction       : {n_obs/n_traj:.4f}")
    print(f">> censored fraction      : {(n_traj-n_obs)/n_traj:.4f}")
    print(f">> early-EOL fraction     : {n_early/n_traj:.4f} ({n_early}/{n_traj})")
    print(f">> initial margin p95     : {np.percentile(imr, 95):.4f} "
          f"(median {np.median(imr):.4f}, max {imr.max():.4f})")
    qa = np.percentile(eols, [5, 25, 50, 75, 95])
    print(">> EOL quantiles (all)    : " + " / ".join(f"{q:.0f}" for q in qa))
    if e_obs.size:
        qo = np.percentile(e_obs, [5, 25, 50, 75, 95])
        print(">> EOL quantiles (observed): " + " / ".join(f"{q:.0f}" for q in qo))
        print(f">> EOL IQR (observed)     : "
              f"{np.percentile(e_obs,75)-np.percentile(e_obs,25):.1f}")
    else:
        qo = None
        print(">> EOL quantiles (observed): n=0 (无 event-observed, 不伪造数值)")
    print(f">> total rows             : {total_rows:,}")
    print(f">> physical range: T [{min(r['T_min'] for r in rows):.1f}, "
          f"{max(r['T_max'] for r in rows):.1f}] K  "
          f"|I_m|max {max(r['Im_absmax'] for r in rows):.4f} A  "
          f"|omega|max {max(r['omega_absmax'] for r in rows):.1f} rad/s")
    print(">> b_final/b_init p5/p50/p95: " + " / ".join(
        f"{np.percentile([r['b_ratio'] for r in rows], q):.3f}" for q in (5, 50, 95)))
    print(">> mode distribution      : " +
          "  ".join(f"{m}={mode_frac[m]:.3f}" for m in MODES))
    print(f">> NaN/Inf                : NaN={n_nan}, all_finite="
          f"{sum(r['all_finite'] for r in rows)}/{n_traj}")
    print(f">> post-EOL rows          : {post_eol:,}  label 单向 "
          f"{sum(r['label_monotone'] for r in rows)}/{n_traj}")
    print(f">> dataset content SHA256 : {ch}")
    print(f">> wall clock             : {elapsed:.1f}s")

    # ---------- §14 Gate ----------
    dgate = eval_data_gates(rows, cfg, mcw, order, det_ok, det_detail)
    print("\n-- §14 正式数据 Gate (10 条) --")
    for name, (ok, detail) in dgate["gates"].items():
        print(f"   [{'PASS' if ok else 'FAIL'}] {name}\n          {detail}")
    verdict = "B11_DATA_READY" if dgate["all_pass"] else "B11_DATA_NOT_READY"
    if not dgate["all_pass"]:
        print("\n>> B11_DATA_NOT_READY —— §14 明令: 正式 150 条跌出区间时**不自动重调**,")
        print("   停止并如实报告。禁止改 b0_scale / Im_rated / Gate / 重跑碰运气。")

    payload = {
        "stage": "BASILISK_B11_DATASET",
        "config": a.config, "seed": seed, "candidate": cname,
        "calibration_hash": h_now,
        "protocol_sha256": rec["protocol_sha256"],
        "reference_caliber": str(ctx["ref"]["caliber"]),
        "b0_scale": float(ctx["calib"]["b0_scale"]),
        "horizon_scale": hs, "n_windows": n_win, "n_traj": n_traj,
        "out_dir": out_dir.relative_to(ROOT).as_posix(),
        "content_sha256": ch,
        "file_sha256": hashlib.sha256(h5p.read_bytes()).hexdigest(),
        "n_event_observed": int(n_obs), "n_censored": int(n_traj - n_obs),
        "failure_fraction": float(n_obs / n_traj),
        "censored_fraction": float((n_traj - n_obs) / n_traj),
        "n_early_eol": int(n_early),
        "early_eol_fraction": float(n_early / n_traj),
        "initial_margin_ratio_p95": float(np.percentile(imr, 95)),
        "initial_margin_ratio_median": float(np.median(imr)),
        "initial_margin_ratio_max": float(imr.max()),
        "eol_quantiles_all": {f"p{q}": float(v)
                              for q, v in zip((5, 25, 50, 75, 95), qa)},
        "eol_quantiles_observed": ({f"p{q}": float(v) for q, v in
                                    zip((5, 25, 50, 75, 95), qo)}
                                   if qo is not None else None),
        "eol_iqr_observed": (float(np.percentile(e_obs, 75)
                                   - np.percentile(e_obs, 25))
                             if e_obs.size else None),
        "total_rows": total_rows,
        "physical_range": {
            "T_min_K": float(min(r["T_min"] for r in rows)),
            "T_max_K": float(max(r["T_max"] for r in rows)),
            "Im_absmax_A": float(max(r["Im_absmax"] for r in rows)),
            "Im_p95_A": float(np.percentile([r["Im_p95"] for r in rows], 95)),
            "omega_absmax_rad_s": float(max(r["omega_absmax"] for r in rows)),
            "b_ratio_p5_p50_p95": [float(np.percentile(
                [r["b_ratio"] for r in rows], q)) for q in (5, 50, 95)],
        },
        "mode_distribution": mode_frac,
        "mode_conditioned_wear": mcw,
        "mode_severity_order": order,
        "mode_reference_g_duty": mode_gd,
        "n_nan_total": int(n_nan),
        "all_finite_count": int(sum(r["all_finite"] for r in rows)),
        "post_eol_rows_total": int(post_eol),
        "label_monotone_count": int(sum(r["label_monotone"] for r in rows)),
        "determinism_ok": bool(det_ok), "determinism_detail": det_detail,
        "data_gates": [{"name": n, "pass": bool(ok), "detail": d}
                       for n, (ok, d) in dgate["gates"].items()],
        "data_gates_all_pass": bool(dgate["all_pass"]),
        "data_gate_derived": dgate["derived"],
        "verdict": verdict,
        "per_traj": rows, "wall_clock_s": round(elapsed, 2),
        "ran_rul_model": False, "used_model_metric": False,
    }
    jp = ROOT / a.json
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                  encoding="utf-8")
    print(f">> 统计写入 {jp.relative_to(ROOT)}")
    write_md(ROOT / a.md, payload, cfg)
    print(f">> 报告写入 {(ROOT / a.md).relative_to(ROOT)}")
    print(f">> {verdict}")
    return 0 if dgate["all_pass"] else 1


def write_md(path: Path, p: dict, cfg: dict) -> None:
    spy = 365.25 * 24 * 3600
    dt = float(cfg["sim"]["sample_period_s"])
    pr = p["physical_range"]
    L = ["# Basilisk-B1.1 §13/§14 —— 150 条正式数据集报告", "",
         "本文件由 `scripts/basilisk_b11/generate_dataset.py` 自动生成。",
         f"**本阶段未运行任何 RUL 模型** (`ran_rul_model: {p['ran_rul_model']}`, "
         f"`used_model_metric: {p['used_model_metric']}`)。", "",
         f"> 结论: **{p['verdict']}**", "",
         "## 1. 配置", "", "| 项 | 值 |", "|---|---|",
         f"| candidate (冻结) | `{p['candidate']}` |",
         f"| `calibration_hash` | `{p['calibration_hash']}` |",
         f"| `protocol.md` sha256 | `{p['protocol_sha256']}` |",
         f"| reference 口径 | `{p['reference_caliber']}` |",
         f"| derived `b0_scale` | {p['b0_scale']:.9f} |",
         f"| horizon_scale | ×{p['horizon_scale']} |",
         f"| n_windows | {p['n_windows']} (= {p['n_windows']*dt/spy:.2f} 年) |",
         f"| n_traj | {p['n_traj']} |", f"| seed | {p['seed']} |",
         f"| 输出目录 | `{p['out_dir']}` (独立于 v1 / B1) |",
         f"| dataset **content** sha256 | `{p['content_sha256']}` |",
         f"| dataset file sha256 | `{p['file_sha256']}` |", "",
         "> content hash 只对数值内容取, 不含 HDF5 容器元数据 (h5py 会写创建时间), "
         "故它才是可复现口径; file hash 仅作记录。", "",
         "## 2. 失效 / 删失分布", "", "| 项 | 值 |", "|---|---|",
         f"| trajectory count | {p['n_traj']} |",
         f"| event-observed | **{p['n_event_observed']}** |",
         f"| censored | **{p['n_censored']}** |",
         f"| failure fraction | **{p['failure_fraction']:.4f}** |",
         f"| censored fraction | **{p['censored_fraction']:.4f}** |",
         f"| early-EOL fraction | **{p['early_eol_fraction']:.4f}** "
         f"({p['n_early_eol']}/{p['n_traj']}) |",
         f"| initial margin ratio p95 | **{p['initial_margin_ratio_p95']:.4f}** |",
         f"| initial margin ratio median / max | "
         f"{p['initial_margin_ratio_median']:.4f} / "
         f"{p['initial_margin_ratio_max']:.4f} |", "",
         "## 3. EOL quantiles", "",
         "| 口径 | p5 | p25 | p50 | p75 | p95 | IQR |",
         "|---|---|---|---|---|---|---|"]
    qa = p["eol_quantiles_all"]
    L.append("| 全部轨迹 | " + " | ".join(f"{qa[f'p{q}']:.0f}"
                                          for q in (5, 25, 50, 75, 95)) + " | - |")
    qo = p["eol_quantiles_observed"]
    if qo:
        L.append("| event-observed | " + " | ".join(
            f"{qo[f'p{q}']:.0f}" for q in (5, 25, 50, 75, 95)) +
            f" | {p['eol_iqr_observed']:.1f} |")
    else:
        L.append("| event-observed | n=0 | n=0 | n=0 | n=0 | n=0 | n=0 |")
    L += ["", "> 删失轨迹的 `eol_idx = n-1` 是**约定**而非观测到的失效; 涉及 EOL 分布的",
          "> 判断一律只用 event-observed 子集, 否则删失比例会被误算成 EOL 堆积。", "",
          "## 4. 物理范围审计", "", "| 量 | 值 | 界 |", "|---|---|---|",
          f"| T | [{pr['T_min_K']:.1f}, {pr['T_max_K']:.1f}] K | "
          f"[{cfg['gate']['temperature_min_K']}, "
          f"{cfg['gate']['temperature_max_K']}] |",
          f"| \\|I_m\\|max | {pr['Im_absmax_A']:.4f} A | "
          f"< {cfg['gate']['current_absmax_A']} |",
          f"| I_m p95 | {pr['Im_p95_A']:.4f} A | "
          f"(Im_rated = {cfg['sim']['failure']['Im_rated_A']}) |",
          f"| \\|omega\\|max | {pr['omega_absmax_rad_s']:.1f} rad/s | "
          f"(HR16 Omega_max = 628.32) |",
          "| b_final/b_init p5/p50/p95 | " +
          "/".join(f"{v:.3f}" for v in pr["b_ratio_p5_p50_p95"]) + " | > 1 |",
          f"| total rows | {p['total_rows']:,} | - |", "",
          "## 5. Mode 分布 (逐轨迹窗占比的均值)", "",
          "| mode | 平均占比 | mode-conditioned mean g_duty |", "|---|---|---|"]
    for m, v in p["mode_distribution"].items():
        mc = p["mode_conditioned_wear"][m]
        L.append(f"| {m} | {v:.4f} | {mc['mean_g_duty']:.4f} "
                 f"(n={mc['n_windows']}) |")
    L += ["", f"严酷度排序 (p95 口径推导): **{' > '.join(p['mode_severity_order'])}**",
          "", "## 6. 数值稳定性 / post-EOL 处理", "", "| 项 | 值 |", "|---|---|",
          f"| NaN 总数 | {p['n_nan_total']} |",
          f"| all_finite 轨迹 | {p['all_finite_count']}/{p['n_traj']} |",
          f"| post-EOL 行数 (label=1 之后) | {p['post_eol_rows_total']:,} |",
          f"| label_fail 单向 (置 1 后不回 0) | "
          f"{p['label_monotone_count']}/{p['n_traj']} |",
          f"| determinism | {p['determinism_detail']} |", "",
          "> post-EOL 行保留在仿真数据里, 但 §15 的标签派生对 `k >= eol` 一律置 "
          "`rul = 0`, 不产生 post-EOL 非法监督; 删失轨迹只给 `rul_lower_bound`。", "",
          "## 7. §14 正式数据 Gate (10 条)", "",
          "| # | gate | 结果 | 细节 |", "|---|---|---|---|"]
    for i, g in enumerate(p["data_gates"], 1):
        L.append(f"| {i} | {g['name']} | {'PASS' if g['pass'] else 'FAIL'} | "
                 f"{g['detail']} |")
    L += ["", f"- `data_gates_all_pass = {p['data_gates_all_pass']}`",
          f"- **结论: {p['verdict']}**", "",
          "> §14 明令: 若正式 150 条因随机波动跌出 candidate 区间, **不自动重调**任何",
          "> 参数, 直接判 `B11_DATA_NOT_READY` 并停止。", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
