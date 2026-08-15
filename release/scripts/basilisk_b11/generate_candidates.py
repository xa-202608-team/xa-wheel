#!/usr/bin/env python
"""scripts/basilisk_b11/generate_candidates.py

Basilisk-B1.1 §7/§8 —— 生成三个**预先登记**的 candidate (各 60 条, CRN 对齐)。

  A11 = P95_UTILIZATION_ONLY           horizon × 1.0  (3.0 年)
  B11 = P95_UTILIZATION_DURATION_1P5   horizon × 1.5  (4.5 年)
  C11 = P95_UTILIZATION_DURATION_2P0   horizon × 2.0  (6.0 年)

三者退化参数**完全相同** (同 p95 reference / 同 derived b0_scale / 同 g_duty),
唯一差异 = mission horizon。§7 明令不得新增 D/E —— registry 由
configs/wheel_basilisk_b11.yaml candidates.registry 冻结, 本脚本硬校验。

§8 common-random-number: trajectory i 在 A/B/C 中拥有同 mode weights、同初始物理
参数、同 profile slice、同噪声 seed。实现见 calibrate_degradation.build_trajectory:
duty 按最长 horizon 生成一次, 短 candidate 取逐位相同前缀。

比 B1 多记录两个量 (Gate 11/12 所需):
  * early_eol (EOL < 0.1 × horizon) —— 左侧堆积检测
  * initial_margin_ratio (初始健康窗 p95(|I_m|)/Im_rated) —— 初始裕度

禁令: 本脚本不计算 RUL / 不训练任何模型 / 不读取任何模型指标 (§17)。
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

import calibrate_degradation as cal          # noqa: E402
from src.sim.basilisk_bridge import DUTY_KEYS, MODES   # noqa: E402

# §7 冻结的 candidate 集合 (代码侧再钉一遍, 与 config registry 必须一致)
FROZEN_REGISTRY = ("P95_UTILIZATION_ONLY", "P95_UTILIZATION_DURATION_1P5",
                   "P95_UTILIZATION_DURATION_2P0")

# 硬隔离: 绝不写入这些命名空间 (v1 正式 lineage + B1 归档)
FORBIDDEN_OUT = ("data/simulated/wheel/sim_v1", "data/features/wheel/schema_v1",
                 "data/sim/wheel_basilisk_v1", "data/features/wheel/basilisk_v1",
                 "data/sim/wheel_basilisk_b1", "data/features/wheel/basilisk_b1")


def guard_out(out_dir: Path) -> None:
    rel = (out_dir.relative_to(ROOT).as_posix() if out_dir.is_relative_to(ROOT)
           else out_dir.as_posix())
    for bad in FORBIDDEN_OUT:
        # 前缀比对需按路径段边界, 否则 .../wheel_basilisk_b11 会被
        # .../wheel_basilisk_b1 的前缀误伤 (B1.1 自己的目录被当成 B1 拒绝)
        if rel == bad or rel.startswith(bad + "/"):
            raise SystemExit(f"!! 拒绝写入 {rel} (§0 硬隔离: 不覆盖 v1 / B1 lineage)")


def check_protocol(cfg: dict, ctx: dict) -> dict:
    """复验 §2 协议 hash 与 §4 推导值 —— 与 audit_peak_reference 的冻结记录一致。"""
    p = ROOT / cfg["paths"]["ckpt_dir"] / "protocol_hash.json"
    if not p.exists():
        raise SystemExit(f"!! 缺少 {p.relative_to(ROOT)}; 必须先运行 "
                         "audit_peak_reference.py (§19 顺序)")
    rec = json.loads(p.read_text(encoding="utf-8"))
    md = ROOT / "docs" / "basilisk_b11" / "protocol.md"
    now = hashlib.sha256(md.read_bytes()).hexdigest()
    if now != rec["protocol_sha256"]:
        raise SystemExit(f"!! protocol.md 在冻结后被修改 ({rec['protocol_sha256'][:12]}"
                         f" -> {now[:12]}); §2 禁止改协议, 停止")
    exp = float(rec["derivation"]["derived_b0_scale"])
    got = float(ctx["calib"]["b0_scale"])
    if abs(got - exp) > 1e-12 * max(abs(exp), 1.0):
        raise SystemExit(f"!! b0_scale 与冻结推导值不一致 {got!r} != {exp!r}, 停止")
    return rec


def content_hash(dfs, params_list) -> str:
    """数值内容 hash (不含 HDF5 容器元数据, 故可跨次复现)。"""
    h = hashlib.sha256()
    for df, p in zip(dfs, params_list):
        for c in df.columns:
            h.update(c.encode())
            h.update(np.ascontiguousarray(df[c].values, dtype=np.float64).tobytes())
        for k in sorted(p.keys()):
            h.update(f"{k}={p[k]!r};".encode())
    return h.hexdigest()


def traj_row(i: int, df, eol: int, failed: bool, params: dict, duty: dict,
             drives: dict, n_win: int, cfg: dict) -> dict:
    om = df["omega"].values; im = df["I_m"].values
    T = df["T"].values; b = df["b_true"].values
    lf = df["label_fail"].values
    vals = df[["omega", "I_m", "T", "T_cmd", "b_true"]].values
    n = len(df)
    # Gate 11: EOL 是否落在前 early_eol_horizon_frac 的 horizon 内。
    # 只对 event-observed 有意义 —— censored 的 eol_idx = n-1 是约定而非观测,
    # 恒不满足 "< 0.1·horizon", 计入分母但永不计入分子。
    ef = float(cfg["gate"]["early_eol_horizon_frac"])
    return dict(
        idx=i, n=n, n_windows=n_win,
        failed=bool(failed), eol_idx=int(eol),
        at_horizon_end=bool(int(eol) == n - 1),
        early_eol=bool(failed and int(eol) < ef * n),
        eol_frac_of_horizon=float(int(eol) / max(n - 1, 1)),
        initial_margin_ratio=cal.initial_margin_ratio(df, cfg),
        b_init=float(b[0]), b_final=float(b[-1]),
        b_ratio=float(b[-1] / max(b[0], 1e-30)),
        b_monotone_frac=float(np.mean(np.diff(b) >= -1e-18)),
        label_monotone=bool(np.all(np.diff(lf) >= 0)),
        n_post_eol_rows=int(n - 1 - int(eol)) if failed else 0,
        omega_absmax=float(np.abs(om).max()),
        Im_absmax=float(np.abs(im).max()),
        Im_p95=float(np.percentile(np.abs(im), 95)),
        T_min=float(T.min()), T_max=float(T.max()),
        T_p95=float(np.percentile(T, 95)),
        n_nan=int(np.isnan(vals).sum()),
        all_finite=bool(np.isfinite(vals).all()),
        g_duty=float(params["_b1_g_duty"]),
        speed_util_mean=float(np.mean(drives["speed_util"])),
        torque_util_mean=float(np.mean(drives["torque_util"])),
        b0_b11=float(params["b0"]), tau_years_b11=float(params["tau_years"]),
        omega0_b11=float(params["omega0"]),
        mode_weight_sum=float(sum(duty["mode_weights"].values())),
        n_modes_used=int(len(np.unique(duty["mode_id"]))),
        mode_fractions={m: float(np.mean(duty["mode_id"] == k))
                        for k, m in enumerate(MODES)},
    )


def mode_conditioned_wear(drives_all: list[dict], duty_all: list[dict],
                          ref: dict, wd: dict) -> dict:
    """§9(9) mode-conditioned wear rate: 按窗归组算 g_duty 均值。

    空组返回 NaN + n=0, 绝不伪造 0 (项目纪律)。
    """
    acc: dict[str, list] = {m: [] for m in MODES}
    for dr, dt in zip(drives_all, duty_all):
        g = cal.g_duty(dr, ref, wd)
        mid = dt["mode_id"]
        for k, m in enumerate(MODES):
            sel = mid == k
            if sel.any():
                acc[m].append(g[sel])
    out = {}
    for m in MODES:
        if acc[m]:
            v = np.concatenate(acc[m])
            out[m] = {"mean_g_duty": float(v.mean()), "n_windows": int(v.size)}
        else:
            out[m] = {"mean_g_duty": float("nan"), "n_windows": 0}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wheel_basilisk_b11.yaml")
    ap.add_argument("--n-traj", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--json", default="checkpoints/basilisk_b11/candidates.json")
    a = ap.parse_args()

    import h5py

    cfg = cal.load_b11_config(a.config)
    seed = a.seed if a.seed is not None else int(cfg["seed"])
    ccfg = cfg["candidates"]
    n_traj = a.n_traj if a.n_traj is not None else int(ccfg["n_traj"])

    # ---- §7 registry 冻结校验 ----
    reg = tuple(ccfg["registry"])
    if reg != FROZEN_REGISTRY:
        raise SystemExit(f"!! candidate registry 被改动: {reg} != {FROZEN_REGISTRY}; "
                         "§7 明令只允许这三个, 禁止新增 D/E")
    if set(ccfg["spec"].keys()) != set(FROZEN_REGISTRY):
        raise SystemExit(f"!! candidates.spec 键集不等于 registry: {set(ccfg['spec'])}")

    # ---- A/B/C 除 horizon 外必须逐键相同 (§7) ----
    for c in reg[1:]:
        sa = {k: v for k, v in ccfg["spec"][reg[0]].items() if k != "horizon_scale"}
        sb = {k: v for k, v in ccfg["spec"][c].items() if k != "horizon_scale"}
        if sa != sb:
            raise SystemExit(f"!! candidate {c} 与 {reg[0]} 除 horizon 外还有差异: "
                             f"{sa} vs {sb}; §7 禁止")

    scales = [float(ccfg["spec"][c]["horizon_scale"]) for c in reg]
    ctx = cal.prepare_context(cfg, scales)
    rec = check_protocol(cfg, ctx)
    wd = cfg["wear_drive"]

    out_root = ROOT / cfg["paths"]["candidate_dir"]
    guard_out(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"Basilisk-B1.1 §7/§8 candidate 生成  seed={seed}  n_traj={n_traj}/candidate")
    print(f"  registry (冻结) : {' | '.join(reg)}")
    print(f"  horizon scales  : {scales}")
    print(f"  reference 口径  : {ctx['ref']['caliber']} "
          f"(speed={ctx['ref']['speed_stat']}, torque={ctx['ref']['torque_stat']})")
    print(f"  n_base          = {ctx['n_base']} 窗  n_max = {ctx['n_max']} 窗")
    print(f"  derived b0_scale= {ctx['calib']['b0_scale']:.9f}  "
          f"(su_ref_p95={ctx['ref']['speed_util']:.9f})")
    print(f"  [对比] B1 rms   = {ctx['calib_rms']['b0_scale']:.9f}  "
          f"(su_ref_rms={ctx['ref_rms']['speed_util']:.9f})")
    print(f"  protocol sha256 = {rec['protocol_sha256'][:24]}...")
    print("=" * 78)

    duty_cache: dict[int, dict] = {}
    results = {}
    for cname in reg:
        hs = float(ccfg["spec"][cname]["horizon_scale"])
        if not bool(ccfg["spec"][cname]["use_utilization_wear"]):
            raise SystemExit(f"!! candidate {cname} 关闭了 utilization wear; §7 三者"
                             "必须同退化尺度")
        n_win = int(round(ctx["n_base"] * hs))
        cdir = out_root / cname
        guard_out(cdir)
        cdir.mkdir(parents=True, exist_ok=True)

        yrs = n_win * float(cfg["sim"]["sample_period_s"]) / (365.25 * 24 * 3600)
        print(f"\n-- {cname}  horizon×{hs}  n_win={n_win} ({yrs:.2f} 年) --")
        t0 = time.time()
        dfs, rows, plist, drives_all, duty_all = [], [], [], [], []
        for i in range(n_traj):
            df, eol, failed, p, duty, drives = cal.build_trajectory(
                i, seed, cfg, ctx, hs, duty_cache)
            dfs.append(df); plist.append(p); drives_all.append(drives)
            duty_all.append(duty)
            rows.append(traj_row(i, df, eol, failed, p, duty, drives, n_win, cfg))
            if (i + 1) % 20 == 0:
                print(f"   traj {i+1:3d}/{n_traj}  "
                      f"fail={sum(r['failed'] for r in rows)}")
        elapsed = time.time() - t0

        ch = content_hash(dfs, plist)
        h5p = cdir / "wheel_all.h5"
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
                mg.create_dataset("mode_id",
                                  data=duty_all[i]["mode_id"].astype(np.int16))
                for m, wv in duty_all[i]["mode_weights"].items():
                    mg.attrs[f"weight_{m}"] = float(wv)
            f.attrs["lineage"] = f"basilisk_b11/candidate/{cname}"
            f.attrs["candidate"] = cname
            f.attrs["horizon_scale"] = hs
            f.attrs["seed"] = int(seed)
            f.attrs["n_traj"] = int(n_traj)
            f.attrs["content_sha256"] = ch
            f.attrs["reference_caliber"] = str(ctx["ref"]["caliber"])
            f.attrs["b0_scale"] = float(ctx["calib"]["b0_scale"])
            f.attrs["contains_rul_label"] = False   # §17 机器可读声明
        n_fail = sum(r["failed"] for r in rows)
        obs = [r for r in rows if r["failed"]]
        eols = np.array([r["eol_idx"] for r in rows], dtype=float)
        imr = np.array([r["initial_margin_ratio"] for r in rows], dtype=float)
        n_early = sum(r["early_eol"] for r in rows)
        print(f"   observed {n_fail}/{n_traj} (ff={n_fail/n_traj:.4f})  "
              f"early={n_early} ({n_early/n_traj:.4f})  "
              f"margin_p95={np.percentile(imr, 95):.4f}  "
              f"content={ch[:16]}  {elapsed:.1f}s")

        # EOL 分位数只在 event-observed 上算 (censored 的 eol_idx 是约定, 不是观测)
        obs_eol = np.array([r["eol_idx"] for r in obs], dtype=float)
        results[cname] = {
            "candidate": cname, "horizon_scale": hs, "n_windows": n_win,
            "horizon_years": float(yrs),
            "n_traj": n_traj, "seed": seed,
            "out": h5p.relative_to(ROOT).as_posix(),
            "content_sha256": ch,
            "n_event_observed": int(n_fail), "n_censored": int(n_traj - n_fail),
            "failure_fraction": float(n_fail / n_traj),
            "censored_fraction": float((n_traj - n_fail) / n_traj),
            "n_early_eol": int(n_early),
            "early_eol_fraction": float(n_early / n_traj),
            "early_eol_horizon_frac": float(cfg["gate"]["early_eol_horizon_frac"]),
            "initial_margin_ratio_p95": float(np.percentile(imr, 95)),
            "initial_margin_ratio_median": float(np.median(imr)),
            "initial_margin_ratio_max": float(imr.max()),
            "eol_quantiles_all": {f"p{q}": float(np.percentile(eols, q))
                                  for q in (5, 25, 50, 75, 95)},
            "eol_quantiles_observed": (
                {f"p{q}": float(np.percentile(obs_eol, q))
                 for q in (5, 25, 50, 75, 95)} if obs_eol.size else None),
            "eol_iqr": float(np.percentile(eols, 75) - np.percentile(eols, 25)),
            "mode_conditioned_wear": mode_conditioned_wear(
                drives_all, duty_all, ctx["ref"], wd),
            "per_traj": rows,
            "wall_clock_s": round(elapsed, 2),
        }

    # ---- §8 CRN 证据 ----
    print("\n-- §8 common-random-number 对齐证据 --")
    pth = cal.paired_trajectory_hash(seed, cfg, ctx, n_traj)
    print(f"   paired_trajectory_hash = {pth}")
    print("   (params + base-horizon duty 的联合 hash; 与 horizon 无关, 故三 candidate")
    print("    共享同一值 —— 若 CRN 被破坏, 逐 candidate 前缀比对会失败)")
    crn_checks = []
    with h5py.File(out_root / reg[0] / "wheel_all.h5", "r") as fa:
        for cname in reg[1:]:
            with h5py.File(out_root / cname / "wheel_all.h5", "r") as fb:
                same = True
                for i in range(min(n_traj, 10)):
                    k = f"traj_{i:03d}"
                    for at in ("b0", "tau_years", "omega0", "Kt", "Tc", "J",
                               "Delta", "seed_traj", "_b1_g_duty"):
                        if abs(float(fa[k].attrs[at]) - float(fb[k].attrs[at])) > 0:
                            same = False
                    na = fa[k]["omega_cmd"].shape[0]
                    if not np.array_equal(fa[k]["omega_cmd"][:],
                                          fb[k]["omega_cmd"][:na]):
                        same = False
                crn_checks.append(
                    (f"{reg[0]} vs {cname}: 退化尺度参数 + duty 前缀逐位相同",
                     bool(same)))
    for name, ok in crn_checks:
        print(f"   [{'PASS' if ok else 'FAIL'}] {name}")
    crn_ok = all(o for _, o in crn_checks)

    payload = {
        "stage": "BASILISK_B11_CANDIDATES",
        "config": a.config, "seed": seed, "n_traj_per_candidate": n_traj,
        "registry_frozen": list(FROZEN_REGISTRY),
        "horizon_scales": {c: float(ccfg["spec"][c]["horizon_scale"]) for c in reg},
        "n_base_windows": ctx["n_base"], "n_max_windows": ctx["n_max"],
        "protocol_sha256": rec["protocol_sha256"],
        "calibration_core_hash": rec["calibration_core_hash"],
        "reference_caliber": str(ctx["ref"]["caliber"]),
        "reference_duty_p95": {k: ctx["ref"][k] for k in cal.DRIVE_KEYS},
        "reference_duty_rms_for_comparison": {k: ctx["ref_rms"][k]
                                              for k in cal.DRIVE_KEYS},
        "b0_calibration": ctx["calib"],
        "b0_calibration_rms_for_comparison": ctx["calib_rms"],
        "wear_drive_cfg": {k: (float(v) if isinstance(v, (int, float)) else v)
                           for k, v in wd.items()},
        "paired_trajectory_hash": pth,
        "crn_prefix_checks": [{"name": n, "pass": bool(o)} for n, o in crn_checks],
        "crn_ok": bool(crn_ok),
        "candidates": results,
        "used_rul_metric": False, "used_model_metric": False,
    }
    jp = ROOT / a.json
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                  encoding="utf-8")
    print(f"\n>> 写入 {jp.relative_to(ROOT)}")
    print(">> " + ("B11_CANDIDATES_GENERATED" if crn_ok
                   else "B11_CRN_ALIGNMENT_FAIL"))
    return 0 if crn_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
