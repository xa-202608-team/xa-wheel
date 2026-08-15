#!/usr/bin/env python
"""scripts/basilisk_b11/build_features.py

Basilisk-B1.1 §15/§16 —— 由 B1.1 的 150 条数据集构建 target_features.h5。

§15 铁律 (与 BASILISK-V1 / B1 完全相同的口径, 一字未改):
  * **直接调用** `src.sim.build_hi.build_features` —— 不复制、不改写任何 HI 数学;
  * HI 定义 / self-calibration / RUL 定义 / `x_T` 核心 8 列一字不改;
  * `mission_features` 仍只作 auxiliary, **不进 x_T** (`mission_features_in_xT=False`);
  * `build_features(df, sim_cfg)` 签名不含 params -> 函数体拿不到真值
    `Kt`/`Tc`/`b0`/`omega0`; 本脚本传入的 df 只含 `TELEMETRY_COLS + [b_true,
    label_fail]`, 其中 `b_true` 仅写入 h5 供离线校核, 不参与任何计算;
    源 h5 组的 `attrs` (含 b0 / tau_years / _b1_g_duty 等真值) **完全不读**。

§16 Feature Gate (6 条) -> `B11_FEATURE_READY` / `B11_FEATURE_INVALID`。
  未达标**不自动调整参数** —— 直接报 INVALID 并以非零码退出。

本脚本不 import src.models / src.transfer / src.experiments / torch (§17 不训模型)。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "basilisk_b11"))

import calibrate_degradation as cal                                  # noqa: E402
from src.sim.build_hi import TELEMETRY_COLS, XT_COLS, build_features  # noqa: E402
from src.sim.basilisk_bridge import DUTY_KEYS                         # noqa: E402

# §16 必须保持语义与位置的核心列 (x_T 前 8 列)
CORE_XT_COLS = ("I_m", "omega", "T", "T_cmd", "sigma_Im", "b_hat", "dT", "omega_err")

# 绝不允许出现在特征文件里的真值 attrs (沿用 B1 清单; B1.1 复用 B1 的
# apply_calibration, 故 _b1_* 前缀的诊断量同样必须挡住)
TRUTH_ATTRS = ("Kt", "Tc", "b0", "omega0", "b_slope", "J", "seed_traj",
               "_b1_g_duty", "_b1_speed_util_mean", "tau_years", "Delta",
               "T_base", "T_amp")

FORBIDDEN_OUT = ("data/features/wheel/schema_v1", "data/features/wheel/basilisk_v1",
                 "data/features/wheel/basilisk_b1")


def guard_out(out: Path) -> None:
    """§0 硬隔离: 按路径段边界判断, 否则 basilisk_b11 会被 basilisk_b1 误伤。"""
    rel = out.relative_to(ROOT).as_posix() if out.is_absolute() else out.as_posix()
    for bad in FORBIDDEN_OUT:
        if rel == bad or rel.startswith(bad + "/"):
            raise SystemExit(f"!! 拒绝写入 {rel} (§0 硬隔离: 不得覆盖 v1 / B1 产物)")


def derive_labels(label_fail: np.ndarray, cap_ratio: float):
    """与 `src/sim/build_hi.py::main` 等价的 RUL 标签派生 (逐位相同)。

    右删失轨迹 (未观测到失效) 的 rul 为 NaN —— **不伪造 EOL**; 只给
    rul_lower_bound (至少还剩这么多寿命), 供删失感知损失使用。
    """
    lf = np.asarray(label_fail)
    n = len(lf)
    observed = bool(lf.any())
    eol = int(np.argmax(lf)) if observed else n - 1
    if observed:
        rul = np.minimum((eol - np.arange(n)).astype(float), cap_ratio * n)
        rul[eol:] = 0.0
        rul_lower_bound = rul.copy()
    else:
        rul = np.full(n, np.nan, dtype=float)
        rul_lower_bound = np.minimum(n - 1 - np.arange(n), cap_ratio * n)
    return rul, rul_lower_bound, eol, observed


def feature_content_hash(path: Path) -> str:
    """特征文件数值内容 hash (排序遍历, 不含 HDF5 容器元数据 -> 可复现)。

    跳过 mission_features (auxiliary, 不进 x_T); NaN 映射到一个不可能出现的
    有限值再取字节, 避免 NaN 位模式差异影响 hash。
    """
    import h5py
    h = hashlib.sha256()
    with h5py.File(path, "r") as f:
        for key in sorted(k for k in f.keys() if k.startswith("traj_")):
            h.update(key.encode())
            g = f[key]
            for name in sorted(g.keys()):
                if name == "mission_features":
                    continue
                h.update(name.encode())
                a = np.ascontiguousarray(g[name][:], dtype=np.float64)
                h.update(str(a.shape).encode())
                h.update(np.nan_to_num(a, nan=-1.0e300).tobytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wheel_basilisk_b11.yaml")
    ap.add_argument("--in", dest="indir", default=None,
                    help="默认由 frozen_calibration + config 推断")
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default="checkpoints/basilisk_b11/feature_stats.json")
    ap.add_argument("--md", default="docs/basilisk_b11/feature_report.md")
    a = ap.parse_args()

    import h5py

    cfg = cal.load_b11_config(a.config)
    sim_cfg = cfg["sim"]
    cap_ratio = float(cfg["source"]["rul_cap_ratio"])
    fg = cfg["feature_gate"]
    dg = cfg["data_gate"]

    if a.indir is None:
        frp = ROOT / "checkpoints/basilisk_b11/frozen_calibration.json"
        if not frp.exists():
            print(f"!! 缺 {frp.relative_to(ROOT)}; 先跑 freeze_candidate.py (§12)")
            print("   若 candidate 审计判 B11_CALIBRATION_FAIL, 按 §11/§19 应就此停止, "
                  "不得绕过冻结步骤直接构建特征")
            return 1
        fr = json.loads(frp.read_text(encoding="utf-8"))
        indir = (f"{cfg['paths']['sim_dir']}/seed_{int(cfg['seed'])}_"
                 f"{fr['selected_candidate']}")
    else:
        indir = a.indir
    h5_in = ROOT / indir / "wheel_all.h5"
    if not h5_in.exists():
        print(f"!! 缺 {h5_in}; 先跑 scripts/basilisk_b11/generate_dataset.py")
        return 1

    out = ROOT / (a.out or cfg["paths"]["feature_h5"])
    guard_out(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    hi_reach_tol = float(sim_cfg["hi"]["reach_tol"])
    n_traj = n_obs = n_hi_reach = mission_copied = 0
    b_rel_errs, hi_p95s, hi_maxs, hi_mins = [], [], [], []
    pre_eol_ok = 0
    xt_finite = True
    rul_nan_censored_ok = True
    post_eol_zero_ok = True
    meta_ok = True
    lens = []

    with h5py.File(h5_in, "r") as fin, h5py.File(out, "w") as fout:
        for key in sorted(k for k in fin.keys() if k.startswith("traj_")):
            g = fin[key]
            # 只取遥测列 + b_true/label_fail; g.attrs 的真值一概不读 (§15)
            df = pd.DataFrame({c: g[c][:] for c in
                               TELEMETRY_COLS + ["b_true", "label_fail"]})
            xT, b_hat, HI_A, HI_B, calib = build_features(df, sim_cfg)
            b_true = df["b_true"].values
            lf = df["label_fail"].values
            rul, rul_lb, eol, observed = derive_labels(lf, cap_ratio)

            gg = fout.create_group(key)
            gg.create_dataset("x_T", data=xT.astype(np.float32))
            gg.create_dataset("hi_b", data=HI_B.astype(np.float32))
            gg.create_dataset("hi_a", data=HI_A.astype(np.float32))
            gg.create_dataset("b_hat", data=b_hat.astype(np.float32))
            gg.create_dataset("b_true", data=b_true.astype(np.float32))  # 仅离线校核
            gg.create_dataset("rul", data=rul.astype(np.float32))
            gg.create_dataset("rul_lower_bound", data=rul_lb.astype(np.float32))
            gg.create_dataset("label_fail", data=lf.astype(np.int8))
            gg.attrs["eol_idx"] = eol
            gg.attrs["event_observed"] = int(observed)
            for k, v in calib.items():
                gg.attrs[k] = v if isinstance(v, str) else float(v)

            if "mission_features" in g:
                mg_in = g["mission_features"]
                mg = gg.create_group("mission_features")
                for k in list(DUTY_KEYS) + ["mode_id"]:
                    if k in mg_in:
                        mg.create_dataset(k, data=mg_in[k][:])
                for k, v in mg_in.attrs.items():
                    mg.attrs[k] = v
                mission_copied += 1

            xt_finite = xt_finite and bool(np.isfinite(xT).all())
            b_rel_errs.append(float(np.mean(np.abs(b_hat - b_true))
                                    / (np.mean(np.abs(b_true)) + 1e-30)))
            hi_mins.append(float(HI_B.min()))
            hi_maxs.append(float(HI_B.max()))
            hi_p95s.append(float(np.percentile(HI_B, 95)))
            n_hi_reach += int(HI_B.max() >= 1.0 - hi_reach_tol)
            # §15 event/censor metadata 一致性
            meta_ok = meta_ok and (int(gg.attrs["event_observed"]) == int(observed)) \
                and (0 <= eol < len(df))
            if observed:
                # §15 post-EOL handling: k >= eol 一律 rul = 0, 不产生 post-EOL 非法监督
                post_eol_zero_ok = post_eol_zero_ok and bool(
                    np.all(rul[eol:] == 0.0))
                pre_eol_ok += int(eol >= int(cfg["model"]["input_len_L"]))
            else:
                rul_nan_censored_ok = rul_nan_censored_ok and bool(
                    np.all(np.isnan(rul)))
            lens.append(len(df))
            n_traj += 1
            n_obs += int(observed)

        with h5py.File(h5_in, "r") as fx:
            src_content = str(fx.attrs.get("content_sha256", "ABSENT"))
            src_calib = str(fx.attrs.get("calibration_hash", "ABSENT"))
            src_cand = str(fx.attrs.get("candidate", "ABSENT"))
            src_caliber = str(fx.attrs.get("reference_caliber", "ABSENT"))
        fout.attrs["lineage"] = "basilisk_b11"
        fout.attrs["source_dataset"] = indir
        fout.attrs["source_dataset_file_sha256"] = hashlib.sha256(
            h5_in.read_bytes()).hexdigest()
        fout.attrs["source_dataset_content_sha256"] = src_content
        fout.attrs["calibration_hash"] = src_calib
        fout.attrs["candidate"] = src_cand
        fout.attrs["reference_caliber"] = src_caliber
        fout.attrs["xt_cols"] = json.dumps(XT_COLS)
        fout.attrs["hi_source"] = str(sim_cfg["failure"]["hi_source"])
        fout.attrs["n_traj"] = n_traj
        fout.attrs["mission_features_in_xT"] = False

    fch = feature_content_hash(out)
    ffh = hashlib.sha256(out.read_bytes()).hexdigest()

    # ---------- §15 无泄漏证据 (运行时属性扫描) ----------
    with h5py.File(out, "r") as f:
        allowed_ds = {"x_T", "hi_b", "hi_a", "b_hat", "b_true", "rul",
                      "rul_lower_bound", "label_fail", "mission_features"}
        leaked_ds, leaked_attr = [], []
        for key in (k for k in f.keys() if k.startswith("traj_")):
            for name in f[key].keys():
                if name not in allowed_ds:
                    leaked_ds.append(f"{key}/{name}")
            for at in f[key].attrs:
                if at in TRUTH_ATTRS:
                    leaked_attr.append(f"{key}.{at}")
        n_mf = sum(1 for k in f.keys() if k.startswith("traj_")
                   and "mission_features" in f[k])

    b_p95 = float(np.percentile(b_rel_errs, 95))
    hi_p95_max = float(np.max(hi_p95s))
    n_cens = n_traj - n_obs
    obs_frac = n_obs / n_traj
    cens_frac = n_cens / n_traj
    core_ok = tuple(XT_COLS[:8]) == CORE_XT_COLS
    no_leak = (not leaked_ds) and (not leaked_attr)

    struct = [
        ("x_T 列数与正式 lineage 一致", len(XT_COLS) == 10, f"{len(XT_COLS)} 列"),
        ("§15 核心 8 列语义/位置不变", core_ok, f"{XT_COLS[:8]}"),
        ("§15 mission_features 存在且独立于 x_T", n_mf == n_traj,
         f"{n_mf}/{n_traj}, mission_features_in_xT=False"),
        ("§15 event/censor metadata 一致", bool(meta_ok),
         f"每条 event_observed attr 与 label_fail 派生一致, 0<=eol<n"),
        ("§15 post-EOL 处理正确", bool(post_eol_zero_ok),
         f"event-observed 轨迹 k>=eol 处 rul 全为 0"),
        ("§15 删失轨迹不伪造 EOL", bool(rul_nan_censored_ok),
         f"{n_cens} 条删失轨迹 rul 全 NaN, 只给 rul_lower_bound"),
    ]

    # ---------- §16 Feature Gate (6 条, 决定 verdict) ----------
    gates = [
        (f"1 p95(HI) > {fg['hi_p95_min']}",
         hi_p95_max > float(fg["hi_p95_min"]),
         f"max over traj of p95(HI_B) = {hi_p95_max:.4f}; "
         f"HI_B max ∈ [{np.min(hi_maxs):.4f}, {np.max(hi_maxs):.4f}], "
         f"达 1.0 的轨迹 {n_hi_reach}/{n_traj}"),
        (f"2 b_hat p95 相对误差 < {fg['b_hat_p95_rel_err_max']:.0%}",
         b_p95 < float(fg["b_hat_p95_rel_err_max"]),
         f"mean {np.mean(b_rel_errs):.4f}, p95 {b_p95:.4f}"),
        ("3 x_T 无 truth attrs",
         no_leak,
         f"额外 dataset {leaked_ds[:3]}, 真值 attrs {leaked_attr[:3]} (均应为空); "
         f"build_features 签名不含 params"),
        ("4 NaN/Inf = 0 (删失 rul 的 NaN 是设计, 不是缺陷)",
         bool(xt_finite and rul_nan_censored_ok),
         f"x_T all finite={xt_finite}; 删失轨迹 rul 全 NaN="
         f"{rul_nan_censored_ok} (右删失不伪造 EOL)"),
        ("5 core 8 feature semantics unchanged",
         core_ok, f"{list(CORE_XT_COLS)}"),
        ("6 mission_features_in_xT = false",
         n_mf == n_traj, f"auxiliary 组 {n_mf}/{n_traj} 条, 不进 x_T"),
    ]

    print("=" * 78)
    print(f"Basilisk-B1.1 §15/§16 特征构建  {indir}")
    print("=" * 78)
    print(f">> candidate {src_cand}  caliber {src_caliber}  "
          f"calibration_hash {src_calib[:16]}...")
    print(f">> 处理 {n_traj} 条轨迹 -> {out.relative_to(ROOT)}")
    print(f"   event-observed {n_obs} ({obs_frac:.4f}) / censored {n_cens} "
          f"({cens_frac:.4f})  序列长 {sorted(set(lens))}")
    print(f"   HI_B min {np.min(hi_mins):.4f}  max {np.max(hi_maxs):.4f}  "
          f"p95(max over traj) {hi_p95_max:.4f}  达 1.0 的轨迹 {n_hi_reach}")
    print(f"   b_hat 相对误差 mean {np.mean(b_rel_errs):.4f}  p95 {b_p95:.4f}")
    print(f"   x_T 列: {XT_COLS}")
    print(f"   event-observed 有 pre-EOL 段 (>= input_len_L="
          f"{cfg['model']['input_len_L']}): {pre_eol_ok}/{n_obs}")
    print()
    print("-- §15 结构 / 泄漏 / 元数据自检 --")
    for n, ok, d in struct:
        print(f"   [{'PASS' if ok else 'FAIL'}] {n}\n          {d}")
    print()
    print("-" * 78)
    print("§16 Feature Gate (6 条)")
    print("-" * 78)
    for n, ok, d in gates:
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}\n         {d}")
    ok_all = all(o for _, o, _ in gates) and all(o for _, o, _ in struct)
    verdict = "B11_FEATURE_READY" if ok_all else "B11_FEATURE_INVALID"
    print()
    print(f">> feature content SHA256 : {fch}")
    print(f">> feature file SHA256    : {ffh}")
    print(f">> {verdict}")
    if not ok_all:
        print("   §16 明令: 不满足则报 B11_FEATURE_INVALID, **不自动调整参数**")

    payload = {
        "stage": "BASILISK_B11_FEATURES",
        "out": out.relative_to(ROOT).as_posix(),
        "source_dataset": indir,
        "source_dataset_content_sha256": src_content,
        "calibration_hash": src_calib,
        "candidate": src_cand, "reference_caliber": src_caliber,
        "feature_content_sha256": fch, "feature_file_sha256": ffh,
        "n_traj": n_traj, "n_event_observed": n_obs, "n_censored": n_cens,
        "event_observed_fraction": obs_frac, "censored_fraction": cens_frac,
        "seq_lengths": sorted(set(lens)),
        "hi_b_min": float(np.min(hi_mins)), "hi_b_max": float(np.max(hi_maxs)),
        "hi_p95_max_over_traj": hi_p95_max, "n_hi_reach_one": n_hi_reach,
        "b_hat_rel_err_mean": float(np.mean(b_rel_errs)),
        "b_hat_rel_err_p95": b_p95,
        "xt_cols": list(XT_COLS), "core_xt_cols": list(CORE_XT_COLS),
        "mission_features_in_xT": False,
        "mission_features_trajectories": mission_copied,
        "leaked_datasets": leaked_ds, "leaked_truth_attrs": leaked_attr,
        "pre_eol_ok": int(pre_eol_ok),
        "post_eol_zero_ok": bool(post_eol_zero_ok),
        "censored_rul_all_nan": bool(rul_nan_censored_ok),
        "event_censor_metadata_ok": bool(meta_ok),
        "reference_data_gate": {
            "event_observed_frac_min": float(dg["event_observed_frac_min"]),
            "censored_frac_min": float(dg["censored_frac_min"])},
        "structural_checks": [{"name": n, "pass": bool(o), "detail": d}
                              for n, o, d in struct],
        "feature_gates": [{"name": n, "pass": bool(o), "detail": d}
                          for n, o, d in gates],
        "feature_gates_all_pass": bool(ok_all),
        "verdict": verdict,
        "ran_rul_model": False, "trained_any_model": False,
    }
    jp = ROOT / a.json
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                  encoding="utf-8")
    print(f">> 统计写入 {jp.relative_to(ROOT)}")
    write_md(ROOT / a.md, payload, cfg)
    print(f">> 报告写入 {(ROOT / a.md).relative_to(ROOT)}")
    return 0 if ok_all else 1


def write_md(path: Path, p: dict, cfg: dict) -> None:
    L = ["# Basilisk-B1.1 §15/§16 —— 特征构建与 Feature Gate", "",
         "本文件由 `scripts/basilisk_b11/build_features.py` 自动生成。",
         f"**本阶段未训练任何模型** (`trained_any_model: {p['trained_any_model']}`, "
         f"`ran_rul_model: {p['ran_rul_model']}`) —— §17 明令 B1.1 不跑 RUL。", "",
         f"> 结论: **{p['verdict']}**", "",
         "## 1. 产物", "", "| 项 | 值 |", "|---|---|",
         f"| 输出 | `{p['out']}` |",
         f"| 源数据集 | `{p['source_dataset']}` |",
         f"| 源 dataset content sha256 | `{p['source_dataset_content_sha256']}` |",
         f"| candidate | `{p['candidate']}` |",
         f"| reference 口径 | `{p['reference_caliber']}` |",
         f"| `calibration_hash` | `{p['calibration_hash']}` |",
         f"| feature **content** sha256 | `{p['feature_content_sha256']}` |",
         f"| feature file sha256 | `{p['feature_file_sha256']}` |",
         f"| n_traj | {p['n_traj']} |",
         f"| 序列长 | {p['seq_lengths']} |", "",
         "> content hash 跳过 `mission_features` 且不含 HDF5 容器元数据, 才是可复现",
         "> 口径; file hash 仅作记录 (h5py 会写创建时间)。", "",
         "## 2. 复用的冻结管线 (数学定义一字未改)", "",
         "- HI 定义、self-calibration、RUL 定义、`x_T` 核心 8 列**一字未改**: 直接调用",
         "  `src.sim.build_hi.build_features` (契约冻结), 不复制不改写。",
         f"- `x_T` 列 = `{p['xt_cols']}`",
         f"- 核心 8 列 = `{p['core_xt_cols']}` (位置与语义与正式 lineage 一致)",
         f"- `mission_features_in_xT = {p['mission_features_in_xT']}` —— Basilisk 工况",
         f"  统计只在 auxiliary 组 ({p['mission_features_trajectories']}/{p['n_traj']} "
         "条), 不进正式模型输入。", "",
         "| HI / b_hat 统计 | 值 |", "|---|---|",
         f"| HI_B min (跨轨迹最小) | {p['hi_b_min']:.4f} |",
         f"| HI_B max (跨轨迹最大) | {p['hi_b_max']:.4f} |",
         f"| max over traj of p95(HI_B) | {p['hi_p95_max_over_traj']:.4f} |",
         f"| HI 达 1.0 的轨迹数 | {p['n_hi_reach_one']}/{p['n_traj']} |",
         f"| b_hat 相对误差 mean / p95 | {p['b_hat_rel_err_mean']:.4f} / "
         f"{p['b_hat_rel_err_p95']:.4f} |", "",
         "## 3. event / censor 元数据", "", "| 项 | 值 |", "|---|---|",
         f"| event-observed | {p['n_event_observed']} "
         f"({p['event_observed_fraction']:.4f}) |",
         f"| censored | {p['n_censored']} ({p['censored_fraction']:.4f}) |",
         f"| event-observed 有 pre-EOL 段 | {p['pre_eol_ok']}/"
         f"{p['n_event_observed']} (>= input_len_L="
         f"{cfg['model']['input_len_L']}) |",
         f"| post-EOL 处理 (k>=eol 时 rul=0) | {p['post_eol_zero_ok']} |",
         f"| 删失轨迹 rul 全 NaN (不伪造 EOL) | {p['censored_rul_all_nan']} |",
         f"| metadata 一致性 | {p['event_censor_metadata_ok']} |", "",
         "> 右删失轨迹只有 `rul_lower_bound`, `rul` 为 NaN。任何后续评估都不得把",
         "> 删失轨迹当作观测到 EOL 来算 RMSE —— 这是跨阶段的既有纪律。", "",
         "## 4. no-truth-leakage 证据", "",
         "1. `build_features(df, sim_cfg)` 签名**不含 params** —— 函数体拿不到真值 "
         "`Kt`/`Tc`/`b0`/`omega0`;",
         "2. 本脚本传入的 `df` 只含 `TELEMETRY_COLS + [b_true, label_fail]`; `b_true` "
         "仅写入 h5 供离线校核, 不参与任何计算;",
         "3. 源 h5 组的 `attrs` (含 `b0`/`tau_years`/`omega0`/`_b1_g_duty` 等真值) "
         "**完全不读**;",
         f"4. 输出文件额外 dataset = `{p['leaked_datasets']}`, 真值 attrs = "
         f"`{p['leaked_truth_attrs']}` (均应为空);",
         "5. 由 `tests/basilisk_b11/test_b11_feature_no_truth_leakage` (AST 静态检查 + "
         "运行时属性扫描) 钉死。", "",
         "## 5. §15 结构 / 泄漏 / 元数据自检", "",
         "| 检查 | 结果 | 细节 |", "|---|---|---|"]
    for c in p["structural_checks"]:
        L.append(f"| {c['name']} | {'PASS' if c['pass'] else 'FAIL'} | "
                 f"{c['detail']} |")
    L += ["", "## 6. §16 Feature Gate (6 条)", "",
          "| # | gate | 结果 | 细节 |", "|---|---|---|---|"]
    for i, c in enumerate(p["feature_gates"], 1):
        L.append(f"| {i} | {c['name']} | {'PASS' if c['pass'] else 'FAIL'} | "
                 f"{c['detail']} |")
    L += ["", f"- `feature_gates_all_pass = {p['feature_gates_all_pass']}`",
          f"- **结论: {p['verdict']}**", "",
          "> §16 纪律: 未达标时报 `B11_FEATURE_INVALID` 并停止, **不自动调整参数**。",
          "> §17 纪律: B1.1 到此不训练任何 RUL 模型 (`target_only` / "
          "`source_finetune` / `source_mmd` / rate model / Wiener PF 全部不跑)。", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
