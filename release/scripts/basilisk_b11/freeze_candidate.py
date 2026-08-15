#!/usr/bin/env python
"""scripts/basilisk_b11/freeze_candidate.py

Basilisk-B1.1 §12 —— 冻结选中的 candidate, 计算 calibration_hash。

写 checkpoints/basilisk_b11/frozen_calibration.json 与
docs/basilisk_b11/selected_calibration.md, 记录 §12 要求的全部内容:
  selected candidate / derived p95 reference / b0_scale / horizon /
  Gate 1-12 结果 / protocol hash / selected_before_rul_training = true

若 audit 判定无 eligible candidate (B11_CALIBRATION_FAIL), 本脚本**拒绝冻结**并
以非零码退出 —— 这是 §11 停止链的第一环。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "basilisk_b11"))

import calibrate_degradation as cal          # noqa: E402

FROZEN = ROOT / "checkpoints" / "basilisk_b11" / "frozen_calibration.json"


def build_protocol(cfg: dict, cand: dict, audit: dict, rec: dict) -> dict:
    """标定协议 payload —— hash 的全部输入, 顺序无关 (json sort_keys)。"""
    sel = audit["selected_candidate"]
    if sel is None:
        raise SystemExit("!! 无 eligible candidate (B11_CALIBRATION_FAIL); 不得冻结。"
                         "§11: 禁止改 b0_scale / Im_rated / threshold / 加 candidate / 重跑")
    return {
        "candidate": sel,
        "horizon_scale": float(cfg["candidates"]["spec"][sel]["horizon_scale"]),
        "n_windows": int(cand["candidates"][sel]["n_windows"]),
        "seed": int(cand["seed"]),
        "sample_period_s": float(cfg["sim"]["sample_period_s"]),
        "duration_years_base": float(cfg["sim"]["duration_years"]),
        "rated": dict(rec["rated"]),
        "reference_caliber": str(rec["reference_caliber"]),
        "reference_stats": dict(rec["reference_stats"]),
        "reference_duty_p95": cand["reference_duty_p95"],
        "reference_aggregate_rule": str(rec["aggregate_rule"]),
        "b0_calibration": cand["b0_calibration"],
        "derivation": dict(rec["derivation"]),
        "wear_drive": cand["wear_drive_cfg"],
        "g_duty_min": cal.G_DUTY_MIN,
        "drive_keys": list(cal.DRIVE_KEYS),
        "physics_ranges": {k: [float(x) for x in v]
                           for k, v in cfg["sim"]["physics"].items()},
        "disturbance": {k: float(v) if isinstance(v, (int, float)) else v
                        for k, v in cfg["sim"]["disturbance"].items()},
        "failure": {k: float(v) if isinstance(v, (int, float)) else v
                    for k, v in cfg["sim"]["failure"].items()},
        "profile_h5": cfg["paths"]["profile_h5"],
        "profile_content_sha256": json.loads(
            (ROOT / "data/mission_profile/basilisk_v1/provenance.json")
            .read_text(encoding="utf-8"))["profiles_sha256"],
        "protocol_sha256": str(rec["protocol_sha256"]),
        "calibration_core_hash": str(rec["calibration_core_hash"]),
        "params_rng_derivation": "sha256('basilisk_b1_params|{seed}|{traj_id}')[:8]",
        "duty_rng_derivation": "sha256('basilisk_bridge|{seed}|{traj_id}')[:8]",
        "degradation_code": "src/sim/wheel_sim.py (契约冻结, 未改一字)",
        "wear_structure_code": ("scripts/basilisk_b1/calibrate_degradation.py "
                               "的 g_duty / b0_scale / apply_calibration "
                               "(B1.1 直接 import, 未改写)"),
        "calibration_applies_to": ["b0", "tau_years", "omega0"],
        "im_rated_A": float(rec["im_rated_A"]),
        "im_rated_provenance_warning": rec["im_rated_provenance_warning"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wheel_basilisk_b11.yaml")
    ap.add_argument("--candidates",
                    default="checkpoints/basilisk_b11/candidates.json")
    ap.add_argument("--audit",
                    default="checkpoints/basilisk_b11/candidate_audit.json")
    ap.add_argument("--protocol",
                    default="checkpoints/basilisk_b11/protocol_hash.json")
    ap.add_argument("--md", default="docs/basilisk_b11/selected_calibration.md")
    ap.add_argument("--check", action="store_true",
                    help="只复核已冻结 hash 是否仍成立 (不重写)")
    a = ap.parse_args()

    cfg = cal.load_b11_config(a.config)
    cand = json.loads((ROOT / a.candidates).read_text(encoding="utf-8"))
    audit = json.loads((ROOT / a.audit).read_text(encoding="utf-8"))
    rec = json.loads((ROOT / a.protocol).read_text(encoding="utf-8"))

    proto = build_protocol(cfg, cand, audit, rec)
    h = cal.calibration_protocol_hash(proto)

    if a.check:
        if not FROZEN.exists():
            print("!! 尚未冻结")
            return 1
        ref = json.loads(FROZEN.read_text(encoding="utf-8"))
        ok = ref["calibration_hash"] == h
        print(f"frozen  = {ref['calibration_hash']}")
        print(f"current = {h}")
        print(">> " + ("B11_CALIBRATION_FROZEN_OK" if ok
                       else "B11_CALIBRATION_PROTOCOL_CHANGED —— §12 禁止, 停止"))
        return 0 if ok else 1

    if FROZEN.exists():
        ref = json.loads(FROZEN.read_text(encoding="utf-8"))
        if ref["calibration_hash"] != h:
            print(f"!! 已冻结 hash {ref['calibration_hash'][:16]} != 当前 {h[:16]}")
            print("!! §12 冻结后禁止再改标定; 拒绝覆盖")
            return 1
        print(">> 已冻结且一致, 无需重写")

    sel = proto["candidate"]
    payload = {
        "stage": "BASILISK_B11_FROZEN_CALIBRATION",
        "calibration_hash": h,
        "selected_candidate": sel,
        "selection_rule": audit["selection_rule"],
        "selected_before_rul_training": True,
        "protocol_sha256": rec["protocol_sha256"],
        "candidate_report": "docs/basilisk_b11/candidate_report.md",
        "candidate_audit_json": a.audit,
        "eligible_map": {c: audit["per_candidate"][c]["eligible"]
                         for c in audit["priority"]},
        "gates_1_to_12": audit["per_candidate"][sel]["gates"],
        "gate_derived": audit["per_candidate"][sel]["derived"],
        "paired_trajectory_hash": cand["paired_trajectory_hash"],
        "crn_ok": cand["crn_ok"],
        "protocol": proto,
    }
    FROZEN.parent.mkdir(parents=True, exist_ok=True)
    FROZEN.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    d = proto["derivation"]
    print("=" * 78)
    print("Basilisk-B1.1 §12 冻结标定")
    print("=" * 78)
    print(f">> selected candidate        = {sel}")
    print(f">> horizon_scale             = {proto['horizon_scale']}  "
          f"(n_windows = {proto['n_windows']})")
    print(f">> reference caliber         = {proto['reference_caliber']} "
          f"({proto['reference_stats']})")
    print(f">> su_ref_p95                = {d['new_su_ref_p95']!r}")
    print(f">> tu_ref_p95                = {d['new_tu_ref_p95']!r}")
    print(f">> derived b0_scale          = {d['derived_b0_scale']!r}  "
          f"(old rms {d['old_b0_scale']!r})")
    print(f">> Gate 1-12                 = "
          f"{sum(1 for g in payload['gates_1_to_12'] if g['pass'])}/12 PASS")
    print(f">> selected_before_rul_training = True")
    print(f">> calibration_hash          = {h}")
    print(f">> 写入 {FROZEN.relative_to(ROOT)}")
    write_md(ROOT / a.md, payload, audit)
    print(f">> 报告写入 {(ROOT / a.md).relative_to(ROOT)}")
    print(">> B11_CALIBRATION_FROZEN")
    return 0


def write_md(path: Path, p: dict, audit: dict) -> None:
    pr = p["protocol"]; bc = pr["b0_calibration"]; r = pr["rated"]
    d = pr["derivation"]
    L = ["# Basilisk-B1.1 §12 —— 冻结的标定协议", "",
         "本文件由 `scripts/basilisk_b11/freeze_candidate.py` 自动生成。",
         "**冻结后禁止再改** —— 任何改动都会让 `calibration_hash` 变化, "
         "`generate_dataset.py` 会拒绝运行。", "",
         f"- **selected candidate**: `{p['selected_candidate']}`",
         f"- **calibration_hash**: `{p['calibration_hash']}`",
         f"- **protocol.md sha256**: `{p['protocol_sha256']}`",
         f"- **selected_before_rul_training**: "
         f"`{p['selected_before_rul_training']}`",
         f"- selection rule: {p['selection_rule']}",
         f"- candidate report: [candidate_report.md](candidate_report.md)",
         f"- eligible map: `{p['eligible_map']}`",
         f"- §8 `paired_trajectory_hash`: `{p['paired_trajectory_hash']}` "
         f"(`crn_ok = {p['crn_ok']}`)", "",
         "## 1. derived p95 reference (§3/§4 的核心产物)", "",
         "| 量 | B1 (rms 口径) | B1.1 (p95 口径) | 比值 |",
         "|---|---|---|---|",
         f"| `su_ref` | {d['old_su_ref_rms']:.9f} | "
         f"**{d['new_su_ref_p95']:.9f}** | {d['speed_ratio_p95_to_rms']:.6f} |",
         f"| `tu_ref` | {d['old_tu_ref_rms']:.9f} | "
         f"**{d['new_tu_ref_p95']:.9f}** | {d['torque_ratio_p95_to_rms']:.6f} |",
         f"| `b0_scale` | {d['old_b0_scale']:.9f} | "
         f"**{d['derived_b0_scale']:.9f}** | "
         f"{d['derived_b0_scale']/d['old_b0_scale']:.6f} |", "",
         f"- reference 统计量: speed = `{pr['reference_stats']['speed']}`, "
         f"torque = `{pr['reference_stats']['torque']}`",
         f"- 窗聚合规则: `{pr['reference_aggregate_rule']}` (与 B1 完全一致)",
         f"- `hardcoded_target = {d['hardcoded_target']}` —— `b0_scale` 由冻结 "
         "profile 统计运行时推导, 源码无任何数值目标", "",
         "| reference 分量 | p95 口径值 |", "|---|---|"]
    for k in pr["drive_keys"]:
        L.append(f"| `{k}` | {pr['reference_duty_p95'][k]:.9f} |")
    L += ["", "## 2. HR16 rated values (provenance: Basilisk `rwFactory`)", "",
          "| 量 | 值 |", "|---|---|"]
    for k in ("omega_rated_rad_s", "torque_rated_Nm", "rotor_inertia_Js_kgm2",
              "rotor_mass_kg", "max_momentum_Nms", "u_s_Nm", "u_d_Nm"):
        if k in r:
            L.append(f"| `{k}` | {r[k]} |")
    L += ["", f"来源: `{r['source']}`", "",
          f"### `Im_rated` (§5: 本阶段不得修改)", "",
          f"`Im_rated = {pr['im_rated_A']} A`, 保持 B1 值不变。"]
    if pr["im_rated_provenance_warning"]:
        L += ["", f"> **{pr['im_rated_provenance_warning']}**", "",
              "该警告只记录, 不修改 —— 在缺少 HR16 真实额定电流 provenance 的情况下,",
              "为凑失效率改动它等于把物理常数当旋钮。"]
    L += ["", "## 3. Horizon", "",
          f"- `horizon_scale` = **{pr['horizon_scale']}**",
          f"- `n_windows` = {pr['n_windows']}  (× {pr['sample_period_s']:.0f} s = "
          f"{pr['n_windows']*pr['sample_period_s']/(365.25*24*3600):.2f} 年)",
          f"- 基准 `duration_years` = {pr['duration_years_base']}", "",
          "## 4. Calibration logic", "",
          "退化**结构**完全保留 `configs/wheel.yaml` 的", "", "```",
          "b(t) = b0·[1 + Δ·(1 - e^{-integ/τ})],   integ = ∫ accel(T) dt",
          "I_m = (T_cmd + T_c·sgn(ω) + b(t)·ω) / K_t", "```", "",
          f"B1.1 只改写三个**调用侧**尺度参数 (`{pr['calibration_applies_to']}`):", "",
          f"- 退化物理代码: `{pr['degradation_code']}`",
          f"- wear 结构代码: `{pr['wear_structure_code']}`", "",
          "| 参数 | 标定式 | 值 |", "|---|---|---|",
          f"| `b0` | `b0_raw · b0_scale` | `b0_scale` = "
          f"**{bc['b0_scale']:.9f}** |",
          "| `tau_years` | `tau_raw / g_duty` | `g_duty` 逐轨迹, reference 处 = 1.0 |",
          f"| `omega0` | `speed_util_traj · omega_rated` | "
          f"omega_rated = {r['omega_rated_rad_s']:.2f} rad/s |", "",
          "### 4.1 `b0_scale` 的不变量", "", f"{bc['rationale']}", "", "```",
          "beta(b, ω) = b·ω / (K_t,nom·Im_rated - T_c,nom)   # 无量纲摩擦裕度消耗率",
          "b0_scale = omega_ref_design / (speed_util_ref · omega_rated)",
          f"         = {bc['omega_ref_design_rad_s']:.1f} / "
          f"({bc['speed_util_ref']:.9f} · {bc['omega_rated_rad_s']:.2f})",
          f"         = {bc['b0_scale']:.9f}", "```", "",
          f"公式与 B1 **逐字相同**, 唯一变化是代入的 `speed_util_ref` 换成 p95 口径。",
          "", f"- `b0_range`: {bc['b0_range_original']} → {bc['b0_range_b1']}",
          f"- `beta_range`: 设计 {[round(v,5) for v in bc['beta_range_design']]} → "
          f"标定后 {[round(v,5) for v in bc['beta_range_b1']]} (保持不变)", "",
          "### 4.2 `g_duty` (磨损时钟 duty 调制, **结构与 B1 完全相同**)", "", "```",
          "g_duty = w_s·(su/su_ref)^p_s + w_t·(tu/tu_ref)^p_t",
          "         + w_z·(zcr/zcr_ref) + w_m·(mf/mf_ref)",
          f"clip 到 [{pr['g_duty_min']}, {pr['wear_drive']['g_duty_max']}]", "```", "",
          "| 权重 / 指数 | 值 |", "|---|---|"]
    for k, v in pr["wear_drive"].items():
        L.append(f"| `{k}` | {v} |")
    L += ["", f"reference 完全由已冻结的 `{pr['profile_h5']}` (content sha256 "
          f"`{pr['profile_content_sha256'][:16]}`) 决定, 不是可调旋钮。", "",
          "## 5. Gate 1-12 结果 (selected candidate)", "",
          "| # | gate | 结果 | 细节 |", "|---|---|---|---|"]
    for i, g in enumerate(p["gates_1_to_12"], 1):
        L.append(f"| {i} | {g['name']} | {'PASS' if g['pass'] else 'FAIL'} | "
                 f"{g['detail']} |")
    L += ["", "## 6. 确定性来源", "",
          f"- 物理参数 RNG: `{pr['params_rng_derivation']}`",
          f"- duty RNG: `{pr['duty_rng_derivation']}`",
          "- 无任何全局 `np.random.seed()`", "",
          "## 7. Selection rule (§10)", "",
          f"- 优先顺序: **{' > '.join(audit['priority'])}** (优先最短 horizon)",
          f"- 失效率带宽 (§9 预先登记): {audit['failure_fraction_band']}",
          f"- {audit['band_rationale']}",
          f"- `used_rul_metric = {audit['used_rul_metric']}`, "
          f"`used_model_metric = {audit['used_model_metric']}`",
          "- 未按 RMSE 选择; 未按哪个 candidate 的 ff 更接近 65% 选择", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
