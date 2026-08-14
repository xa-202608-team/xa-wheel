#!/usr/bin/env python
"""scripts/basilisk_b11/audit_peak_reference.py

Basilisk-B1.1 §3/§4 —— 峰值口径审计 + b0_scale 物理推导。

做三件事:
  1. 从已冻结的 profiles.h5 同时算出 **rms 口径** 与 **p95 口径** 的 reference,
     给出 p95/rms 比值 —— 这是 B1 失败的定量根因。
  2. 由 p95 reference **推导** 新的 b0_scale (公式与 B1 同一物理不变量,
     只是代入的 reference 换了口径)。**不硬编码任何目标值。**
  3. 冻结 §2 协议 hash 与推导结果 -> checkpoints/basilisk_b11/protocol_hash.json,
     下游脚本 (generate_candidates / generate_dataset / build_features) 全部复验。

禁令: 本脚本不计算 RUL / 不训练任何模型 / 不读取任何模型指标 (§17)。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "basilisk_b11"))

import calibrate_degradation as cal          # noqa: E402
from src.sim.basilisk_bridge import MODES     # noqa: E402

PROTOCOL_MD = ROOT / "docs" / "basilisk_b11" / "protocol.md"


def protocol_sha256() -> str:
    """§2 协议文件 hash (以字节为准, 冻结后不得改动)。"""
    if not PROTOCOL_MD.exists():
        raise SystemExit(f"!! 协议文件不存在: {PROTOCOL_MD} —— §2 要求先冻结协议")
    return hashlib.sha256(PROTOCOL_MD.read_bytes()).hexdigest()


def pointwise_crest_factors(lib: dict, rated: dict) -> dict:
    """逐点序列的波峰因数 (不做窗聚合) —— 独立佐证 p95/rms 差异不是窗口效应。"""
    out = {}
    for m in MODES:
        sp = np.concatenate([np.abs(r["speed"]) for r in lib[m]])
        tq = np.concatenate([np.abs(r["torque"]) for r in lib[m]])
        out[m] = {
            "omega_rms": float(np.sqrt(np.mean(sp ** 2))),
            "omega_p95": float(np.percentile(sp, 95)),
            "omega_max": float(np.max(sp)),
            "torque_rms": float(np.sqrt(np.mean(tq ** 2))),
            "torque_p95": float(np.percentile(tq, 95)),
            "n_samples": int(sp.size),
        }
        out[m]["omega_crest_p95_over_rms"] = (
            out[m]["omega_p95"] / max(out[m]["omega_rms"], 1e-30))
        out[m]["torque_crest_p95_over_rms"] = (
            out[m]["torque_p95"] / max(out[m]["torque_rms"], 1e-30))
    return out


def initial_current_ratio_at_t0(cfg: dict, rated: dict, ref_sim: dict,
                                calib: dict, caliber: str) -> dict:
    """解析估算 t=0 处的电流占额定比, 不跑仿真。

    I_m(0) ≈ (T_cmd + T_c·sgn(ω) + b0·ω) / K_t

    **关键**: `ref_sim` 必须始终是 **rms 口径** reference —— 因为契约冻结的
    `basilisk_bridge.duty_to_sim_inputs` 喂给仿真的 `omega_cmd`/`T_cmd` 恒为窗 rms,
    B1.1 没有改它 (§3 只换标定参考尺度, 不换仿真工作点)。两个口径下唯一变化的是
    `b0 = b0_range × b0_scale`。

    早期版本错误地把工作点一起换成 p95, 结果得出"p95 口径电流更高"的反向结论 ——
    那是把"参考尺度"与"仿真输入"混为一谈。此处显式保留该注记以免再犯。

    仅为量级说明, 不用于任何 Gate。
    """
    fc = cfg["sim"]["failure"]
    kt, tc, im_r = float(fc["Kt_nom"]), float(fc["Tc_nom"]), float(fc["Im_rated_A"])
    b0_lo, b0_hi = [float(v) for v in cfg["sim"]["physics"]["b0_range"]]
    s = float(calib["b0_scale"])
    om_op = float(ref_sim["speed_util"]) * float(rated["omega_rated_rad_s"])
    tq_op = float(ref_sim["torque_util"]) * float(rated["torque_rated_Nm"])
    out = {"caliber": caliber, "b0_scale": s,
           "sim_operating_caliber": "rms (duty_to_sim_inputs, 契约冻结)",
           "omega_operating_rad_s": om_op, "torque_operating_Nm": tq_op}
    for tag, b0 in (("b0_lo", b0_lo * s), ("b0_mid", 0.5 * (b0_lo + b0_hi) * s),
                    ("b0_hi", b0_hi * s)):
        im = (tq_op + tc + b0 * om_op) / kt
        out[f"Im_over_rated_{tag}"] = float(im / im_r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wheel_basilisk_b11.yaml")
    a = ap.parse_args()

    cfg = cal.load_b11_config(a.config)
    scales = [float(v["horizon_scale"]) for v in cfg["candidates"]["spec"].values()]
    print(f">> config chain: {cfg['_config_chain']}")
    print(f">> reference.caliber = {cfg['reference']['caliber']}  "
          f"(speed_stat={cfg['reference']['speed_stat']}, "
          f"torque_stat={cfg['reference']['torque_stat']})")

    p_hash = protocol_sha256()
    print(f">> protocol.md sha256 = {p_hash}")

    ctx = cal.prepare_context(cfg, scales)
    rated, ref_p95, ref_rms = ctx["rated"], ctx["ref"], ctx["ref_rms"]
    calib_p95, calib_rms = ctx["calib"], ctx["calib_rms"]

    print("\n== HR16 rated 复核 (Basilisk rwFactory 运行时读数) ==")
    for name, ok, msg in ctx["rated_checks"]:
        print(f"   [{'OK' if ok else 'FAIL'}] {name:26s} {msg}")

    # ---------- §5: Im_rated provenance 检查 (只警告, 绝不修改) ----------
    fc = cfg["sim"]["failure"]
    im_rated = float(fc["Im_rated_A"])
    kt_nom, tc_nom = float(fc["Kt_nom"]), float(fc["Tc_nom"])
    tq_capacity = kt_nom * im_rated                     # 该电流能产生的力矩
    tq_rated = float(rated["torque_rated_Nm"])
    warn = None
    if tq_capacity < tq_rated:
        warn = ("IM_RATED_PROVENANCE_WARNING: Kt_nom·Im_rated = "
                f"{tq_capacity:.6g} N·m < HR16 u_max = {tq_rated:.6g} N·m —— "
                "即额定电流产生不了额定力矩, 说明 Im_rated 缺真实 provenance。"
                "§5 明令: 本阶段不得修改, 仅记录。")
        print(f"\n!! {warn}")
    else:
        print(f"\n>> Im_rated 一致性: Kt_nom·Im_rated={tq_capacity:.6g} >= "
              f"u_max={tq_rated:.6g} N·m, 无警告")

    # ---------- §4 必须打印的 8 个量 ----------
    old_su = float(ref_rms["speed_util"])
    new_su = float(ref_p95["speed_util"])
    old_tu = float(ref_rms["torque_util"])
    new_tu = float(ref_p95["torque_util"])
    r_su = new_su / old_su
    r_tu = new_tu / old_tu
    old_b0s = float(calib_rms["b0_scale"])
    new_b0s = float(calib_p95["b0_scale"])

    print("\n" + "=" * 74)
    print("§4 峰值口径推导 (全部由冻结 profiles.h5 运行时算出, 无硬编码)")
    print("=" * 74)
    print(f"  old_su_ref_rms      = {old_su!r}")
    print(f"  new_su_ref_p95      = {new_su!r}")
    print(f"  ratio_p95_to_rms    = {r_su!r}          (speed)")
    print(f"  old_tu_ref_rms      = {old_tu!r}")
    print(f"  new_tu_ref_p95      = {new_tu!r}")
    print(f"  ratio_p95_to_rms    = {r_tu!r}          (torque)")
    print(f"  old_b0_scale        = {old_b0s!r}")
    print(f"  derived_b0_scale    = {new_b0s!r}")
    print(f"  check: old/new      = {old_b0s / new_b0s!r}  (应等于 speed ratio)")
    assert abs(old_b0s / new_b0s - r_su) < 1e-9, "b0_scale 比值与 su ratio 不一致"

    # 与 B1 归档值交叉核对 (不是硬编码目标, 是"旧口径必须复现"的证据)
    b1_json = ROOT / "checkpoints" / "basilisk_b1" / "candidates.json"
    b1_match = None
    if b1_json.exists():
        b1 = json.loads(b1_json.read_text(encoding="utf-8"))
        b1_su = float(b1["reference_duty"]["speed_util"])
        b1_b0s = float(b1["b0_calibration"]["b0_scale"])
        b1_match = {"b1_su_ref": b1_su, "b1_b0_scale": b1_b0s,
                    "su_reproduced": bool(abs(b1_su - old_su) < 1e-12),
                    "b0_scale_reproduced": bool(abs(b1_b0s - old_b0s) < 1e-9)}
        print(f"\n>> B1 归档交叉核对: su_ref {'OK' if b1_match['su_reproduced'] else 'MISMATCH'}"
              f" ({b1_su!r}), b0_scale "
              f"{'OK' if b1_match['b0_scale_reproduced'] else 'MISMATCH'} ({b1_b0s!r})")

    # ---------- 逐模式波峰因数 ----------
    pw = pointwise_crest_factors(ctx["lib"], rated)
    print("\n== 逐模式波峰因数 (逐点序列, 非窗聚合) ==")
    print(f"  {'mode':12s} {'ω p95/rms':>11s} {'τ p95/rms':>11s} {'n_samples':>11s}")
    for m in MODES:
        print(f"  {m:12s} {pw[m]['omega_crest_p95_over_rms']:11.4f} "
              f"{pw[m]['torque_crest_p95_over_rms']:11.4f} {pw[m]['n_samples']:11d}")

    print("\n== 窗级 reference 逐模式 (p95 口径) ==")
    print(f"  {'mode':12s} {'su_p95':>10s} {'tu_p95':>10s} {'su_rms':>10s} "
          f"{'tu_rms':>10s} {'n_win':>7s}")
    for m in MODES:
        a95, arms = ref_p95["per_mode"][m], ref_rms["per_mode"][m]
        print(f"  {m:12s} {a95['speed_util']:10.5f} {a95['torque_util']:10.5f} "
              f"{arms['speed_util']:10.5f} {arms['torque_util']:10.5f} "
              f"{a95['n_windows']:7d}")

    # ---------- t=0 电流占比闭式估算 ----------
    # 两者都用 **rms 工作点** (仿真输入口径), 只有 b0_scale 不同 —— 这才是
    # B1 -> B1.1 的真实差异。
    ic_p95 = initial_current_ratio_at_t0(cfg, rated, ref_rms, calib_p95, "p95")
    ic_rms = initial_current_ratio_at_t0(cfg, rated, ref_rms, calib_rms, "rms")
    print("\n== t=0 电流占额定比 (闭式估算; 工作点恒为 rms 口径, 只有 b0_scale 变) ==")
    for tag in ("b0_lo", "b0_mid", "b0_hi"):
        print(f"  {tag:8s} rms口径={ic_rms[f'Im_over_rated_{tag}']:.4f}  "
              f"p95口径={ic_p95[f'Im_over_rated_{tag}']:.4f}")

    print("\n== horizon ==")
    print(f"  n_base = {ctx['n_base']} 窗 = {cfg['sim']['duration_years']} 年")
    print(f"  n_max  = {ctx['n_max']} 窗 (horizon_scale max = {max(scales)})")

    # ---------- 冻结 ----------
    payload = {
        "stage": "BASILISK_B1.1",
        "protocol_sha256": p_hash,
        "config": cfg["_config_path"],
        "config_chain": cfg["_config_chain"],
        "seed": int(cfg["seed"]),
        "reference_caliber": str(cfg["reference"]["caliber"]),
        "reference_stats": {"speed": ref_p95["speed_stat"],
                            "torque": ref_p95["torque_stat"]},
        "aggregate_rule": ref_p95["aggregate"],
        "rated": rated,
        "rated_all_pass": all(ok for _, ok, _ in ctx["rated_checks"]),
        "im_rated_A": im_rated,
        "im_rated_provenance_warning": warn,
        "reference_p95": {k: ref_p95[k] for k in cal.DRIVE_KEYS},
        "reference_rms": {k: ref_rms[k] for k in cal.DRIVE_KEYS},
        "reference_p95_per_mode": ref_p95["per_mode"],
        "reference_rms_per_mode": ref_rms["per_mode"],
        "derivation": {
            "old_su_ref_rms": old_su, "new_su_ref_p95": new_su,
            "speed_ratio_p95_to_rms": r_su,
            "old_tu_ref_rms": old_tu, "new_tu_ref_p95": new_tu,
            "torque_ratio_p95_to_rms": r_tu,
            "old_b0_scale": old_b0s, "derived_b0_scale": new_b0s,
            "b0_scale_formula": "omega_ref_design / (su_ref · Omega_rated)",
            "hardcoded_target": False,
        },
        "b0_calibration_p95": calib_p95,
        "b0_calibration_rms": calib_rms,
        "b1_cross_check": b1_match,
        "pointwise_crest_factors": pw,
        "initial_current_estimate": {"p95": ic_p95, "rms": ic_rms},
        "n_per_window": ctx["n_per_window"],
        "n_base_windows": ctx["n_base"],
        "n_max_windows": ctx["n_max"],
        "wear_drive_cfg": cfg["wear_drive"],
        "ran_rul_model": False,
        "used_model_metric": False,
    }
    # calibration_protocol_hash 只覆盖"标定决定性内容", 不含逐模式明细/波峰因数
    core = {k: payload[k] for k in (
        "protocol_sha256", "reference_caliber", "reference_stats",
        "aggregate_rule", "im_rated_A", "reference_p95", "derivation",
        "wear_drive_cfg", "n_per_window", "n_base_windows", "n_max_windows")}
    payload["calibration_core_hash"] = cal.calibration_protocol_hash(core)
    print(f"\n>> calibration_core_hash = {payload['calibration_core_hash']}")

    out_json = ROOT / cfg["paths"]["ckpt_dir"] / "protocol_hash.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f">> 写入 {out_json.relative_to(ROOT)}")

    write_md(ROOT / cfg["paths"]["doc_dir"] / "peak_reference_report.md",
             payload, cfg, pw, ref_p95, ref_rms)
    return 0


def write_md(path: Path, p: dict, cfg: dict, pw: dict,
             ref95: dict, refrms: dict) -> None:
    d = p["derivation"]
    L = ["# Basilisk-B1.1 峰值口径审计报告 (§3/§4)", "",
         f"- 协议 `docs/basilisk_b11/protocol.md` SHA256: `{p['protocol_sha256']}`",
         f"- `calibration_core_hash`: `{p['calibration_core_hash']}`",
         f"- reference 口径: **{p['reference_caliber']}** "
         f"(speed=`{p['reference_stats']['speed']}`, "
         f"torque=`{p['reference_stats']['torque']}`)",
         f"- 窗聚合规则: `{p['aggregate_rule']}` (与 B1 完全一致)",
         f"- HR16 rated 复核: {'全部通过' if p['rated_all_pass'] else '**失败**'}",
         "", "## 1. 口径切换的定量结果", "",
         "| 量 | RMS 口径 (B1) | p95 口径 (B1.1) | p95/rms |",
         "|---|---|---|---|",
         f"| `su_ref` (speed utilization) | {d['old_su_ref_rms']:.9f} | "
         f"{d['new_su_ref_p95']:.9f} | **{d['speed_ratio_p95_to_rms']:.6f}** |",
         f"| `tu_ref` (torque utilization) | {d['old_tu_ref_rms']:.9f} | "
         f"{d['new_tu_ref_p95']:.9f} | **{d['torque_ratio_p95_to_rms']:.6f}** |",
         f"| `b0_scale` | {d['old_b0_scale']:.9f} | "
         f"**{d['derived_b0_scale']:.9f}** | "
         f"{d['derived_b0_scale'] / d['old_b0_scale']:.6f} |",
         "",
         f"`b0_scale` 公式 `{d['b0_scale_formula']}` 与 B1 **逐字相同**, 唯一变化是",
         "代入的 `su_ref` 换成了 p95 口径。口径比值与 `b0_scale` 比值互为倒数",
         "(脚本内 `assert` 强制), 说明这确实是一次纯口径变换, 没有引入额外自由度。", "",
         f"**推导值不是目标值**: `derived_b0_scale = {d['derived_b0_scale']:.6f}` 完全由",
         "冻结的 `profiles.h5` 统计决定; 源码中没有任何数值目标 "
         "(`hardcoded_target = false`)。", ""]
    if p["b1_cross_check"]:
        c = p["b1_cross_check"]
        L += ["### 与 B1 归档值的交叉核对", "",
              f"- B1 归档 `su_ref` = {c['b1_su_ref']:.12f} -> 本次 rms 口径复现: "
              f"**{'一致' if c['su_reproduced'] else '不一致'}**",
              f"- B1 归档 `b0_scale` = {c['b1_b0_scale']:.9f} -> 本次 rms 口径复现: "
              f"**{'一致' if c['b0_scale_reproduced'] else '不一致'}**", "",
              "旧口径能逐位复现, 证明 B1.1 与 B1 的差异**只**来自口径, 不来自任何",
              "其它环境/实现漂移。", ""]
    L += ["## 2. 逐模式波峰因数 (逐点序列, 非窗聚合)", "",
          "| mode | ω p95/rms | τ p95/rms | n_samples |", "|---|---|---|---|"]
    for m in MODES:
        L.append(f"| `{m}` | {pw[m]['omega_crest_p95_over_rms']:.4f} | "
                 f"{pw[m]['torque_crest_p95_over_rms']:.4f} | {pw[m]['n_samples']} |")
    L += ["", "波峰因数显著 > 1 说明 Basilisk 工况不是准正弦 —— 这是 RMS 标定与 peak",
          "判据不兼容的物理来源, 且它在**逐点**口径上就已成立, 不是窗口聚合的产物。",
          "", "## 3. 窗级 reference 逐模式", "",
          "| mode | su_p95 | tu_p95 | su_rms | tu_rms | n_win |",
          "|---|---|---|---|---|---|"]
    for m in MODES:
        a, b = ref95["per_mode"][m], refrms["per_mode"][m]
        L.append(f"| `{m}` | {a['speed_util']:.5f} | {a['torque_util']:.5f} | "
                 f"{b['speed_util']:.5f} | {b['torque_util']:.5f} | "
                 f"{a['n_windows']} |")
    ic = p["initial_current_estimate"]
    L += ["", "`zero_crossing_rate` / `maneuver_fraction` 两个 reference 分量在两种",
          "口径下**完全相同** (它们不是幅值量, 无 rms/peak 之分), 故未列出。", "",
          "## 4. t=0 电流占额定比 (闭式估算)", "",
          "两列都用 **rms 工作点** (契约冻结的 `duty_to_sim_inputs` 喂给仿真的",
          "`omega_cmd` 恒为窗 rms, B1.1 未改), 唯一差异是 `b0_scale`:", "",
          "| b0 取值 | RMS 标定 (B1) | p95 标定 (B1.1) |", "|---|---|---|"]
    for tag in ("b0_lo", "b0_mid", "b0_hi"):
        L.append(f"| `{tag}` | {ic['rms'][f'Im_over_rated_{tag}']:.4f} | "
                 f"{ic['p95'][f'Im_over_rated_{tag}']:.4f} |")
    L += ["", "仅为量级说明, **不参与任何 Gate**。真实的初始裕度由 Gate 12 在",
          "仿真轨迹上实测 (`initial_margin_ratio_p95`)。", "",
          "## 5. Im_rated provenance (§5)", ""]
    if p["im_rated_provenance_warning"]:
        L += [f"> **{p['im_rated_provenance_warning']}**", "",
              f"本阶段 `Im_rated = {p['im_rated_A']} A` **保持不变**。在缺少 HR16 真实",
              "额定电流 provenance 的情况下, 为凑失效率而改动它等于把物理常数当旋钮。",
              "该警告记入 `docs/basilisk_b11/limitations.md`, 留待后续阶段用真实",
              "datasheet 解决。"]
    else:
        L += [f"`Im_rated = {p['im_rated_A']} A`, 无 provenance 警告。"]
    L += ["", "## 6. horizon", "",
          f"- `n_per_window` = {p['n_per_window']}",
          f"- `n_base` = {p['n_base_windows']} 窗 = {cfg['sim']['duration_years']} 年",
          f"- `n_max` = {p['n_max_windows']} 窗 (CRN: duty 按 n_max 生成一次, "
          "短 candidate 取逐位相同前缀)", "",
          f"- 本阶段运行 RUL 模型: **{p['ran_rul_model']}**; "
          f"使用模型指标: **{p['used_model_metric']}**", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f">> 写入 {path.relative_to(ROOT)}")


if __name__ == "__main__":
    raise SystemExit(main())
