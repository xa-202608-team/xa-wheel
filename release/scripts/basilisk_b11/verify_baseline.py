"""scripts/basilisk_b11/verify_baseline.py

Basilisk-B1.1 §1 —— baseline contract。

比 B1 的契约更严: 在 B1 的 44 项之上追加
  * B1 candidate A/B/C 三个 content hash (数值口径, 不含 HDF5 时间戳);
  * B1 verdict = B1_CALIBRATION_FAIL;
  * B1 paired_trajectory_hash;
  * configs/wheel_basilisk_b1.yaml 与 B1 全部 docs / checkpoints 文件 hash。

也就是说: **B1.1 干活期间, B1 的失败结论必须原样保留**。B1 不是被"修好"的,
它是一条已归档的负结果; B1.1 是另一条独立分支。

用法:
  --write             计算并写 docs/basilisk_b11/baseline_contract.json
  --verify --tag X    重新计算并逐项比对; 任何一项变化 -> 退出码 1

MISSING 也是契约的一部分: 现在不存在的文件若在任务结束时出现, 同样算契约破坏
(这就是为什么 B1 那些"未生成"的产物路径也列进来 —— 它们必须继续不存在)。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "docs" / "basilisk_b11" / "baseline_contract.json"

# ---------------------------------------------------------------------------
# 冻结文件清单。B1.1 无 mutable 组 —— 标定口径改动全部落在 B1.1 自己的新文件里。
# ---------------------------------------------------------------------------
FROZEN_GROUPS: dict[str, list[str]] = {
    "config": [
        "configs/wheel.yaml",
        "configs/wheel_basilisk.yaml",
        # B1 config 进入不可变组 (§0 禁止修改)
        "configs/wheel_basilisk_b1.yaml",
    ],
    "analytic_features": [
        "data/features/wheel/schema_v1/target_features.h5",
        "data/features/wheel/schema_v1/source_features.h5",
    ],
    "frozen_metrics": [
        "checkpoints/s25_gate.json",
        "checkpoints/s25_select.json",
        "checkpoints/s25_tune.json",
        "checkpoints/s3_gate_metrics.json",
        "checkpoints/s4_metrics.json",
        "checkpoints/s4_stage_diagnostics.json",
        "checkpoints/s5_rate_metrics.json",
        "checkpoints/s5b_wiener_pf_metrics.json",
    ],
    "frozen_docs": [
        "docs/diagnostics_s25_protocol.md",
        "docs/diagnostics_s25_results.md",
        "docs/diagnostics_s3_protocol.md",
        "docs/diagnostics_s3_results.md",
        "docs/diagnostics_s4_protocol.md",
        "docs/diagnostics_s4_results.md",
        "docs/diagnostics_s5b_protocol.md",
        "docs/diagnostics_s5b_results.md",
        "docs/results.md",
    ],
    # B1.1 不得修改的算法代码。wheel_sim.py / basilisk_bridge.py 在此组内 ——
    # 这正是 §3 "p95 统计量已天然存在于 bridge 输出" 的意义: 换口径无需碰它们。
    "frozen_code": [
        "src/sim/wheel_sim.py",
        "src/sim/build_hi.py",
        "src/sim/basilisk_bridge.py",
        "src/sim/basilisk_profile.py",
        "src/experiments/metrics.py",
        "src/experiments/run_groups.py",
        "src/transfer/train_transfer.py",
        "src/models/rate_head.py",
        "src/baselines/physical_extrap.py",
        "src/baselines/trivial.py",
        "src/baselines/wiener_pf.py",
        "src/data/preprocess/wheel_features.py",
    ],
    "basilisk_v1_artifacts": [
        "data/mission_profile/basilisk_v1/profiles.h5",
        "data/sim/wheel_basilisk_v1/seed_20260808_basilisk/wheel_all.h5",
        "data/features/wheel/basilisk_v1/target_features.h5",
        "docs/basilisk_v1/baseline_contract.json",
        "STATUS_BASILISK_V1.md",
    ],
    # B1 全部产物 (含 scripts —— B1 代码也不许改)
    "basilisk_b1_artifacts": [
        "checkpoints/basilisk_b1/candidates.json",
        "checkpoints/basilisk_b1/candidate_audit.json",
        "checkpoints/basilisk_b1/physics_audit.json",
        "docs/basilisk_b1/baseline_contract.json",
        "docs/basilisk_b1/protocol.md",
        "docs/basilisk_b1/physics_audit.md",
        "docs/basilisk_b1/candidate_report.md",
        "docs/basilisk_b1/limitations.md",
        "docs/basilisk_b1/REPRODUCE.md",
        "STATUS_BASILISK_B1.md",
        "scripts/basilisk_b1/calibrate_degradation.py",
        "scripts/basilisk_b1/generate_candidates.py",
        "scripts/basilisk_b1/audit_candidates.py",
        "scripts/basilisk_b1/freeze_candidate.py",
        "scripts/basilisk_b1/generate_dataset.py",
        "scripts/basilisk_b1/build_features.py",
        "scripts/basilisk_b1/verify_baseline.py",
    ],
    # B1 **必须继续不存在**的产物 (candidate 全灭 -> 停止链)。
    # 若 B1.1 期间任何一项冒出来, 说明有人越界重跑了 B1 -> 契约破坏。
    "basilisk_b1_must_stay_absent": [
        "checkpoints/basilisk_b1/frozen_calibration.json",
        "docs/basilisk_b1/selected_calibration.md",
        "docs/basilisk_b1/simulation_report.md",
        "docs/basilisk_b1/feature_report.md",
        "data/features/wheel/basilisk_b1/target_features.h5",
    ],
}

# 数值 content hash (跨平台可复现口径, 不含 HDF5 容器时间戳)。
V1_CONTENT_HASHES: dict[str, tuple[str, str]] = {
    "v1_profile_content_sha256": (
        "data/mission_profile/basilisk_v1/provenance.json", "profiles_sha256"),
    "v1_dataset_file_sha256": (
        "checkpoints/basilisk_v1/feature_stats.json", "source_dataset_sha256"),
    "v1_feature_file_sha256": (
        "checkpoints/basilisk_v1/feature_stats.json", "feature_sha256"),
}

V1_EXPECTED = {
    "v1_profile_content_sha256":
        "21aab0036e651239a31de724ab24d119d82ed8cd2786620506900068e8b0ebaf",
    "v1_dataset_file_sha256":
        "138b117a0b826aadd779ee5d64b872b413647f98322a7d8e7ef84d73c7b379c6",
    "v1_feature_file_sha256":
        "c59983d13187b1b0734be4321d20fca78566888f3e434dba61d6debf262a222a",
}

# --- B1 数值结论 (§1 要求逐项冻结) ---
# candidate content hash 从 candidates.json 的 candidates[<name>].content_sha256 读出;
# verdict 从 candidate_audit.json 读出。三个 hash 的期望值硬写在此 (B1 §18 已公布),
# 防止 json 被整体替换后仍"自洽"。
B1_CANDIDATE_NAMES = ("UTILIZATION_ONLY", "UTILIZATION_PLUS_DURATION",
                      "UTILIZATION_PLUS_DURATION_LONG")
B1_EXPECTED = {
    "b1_candidate_A_content_sha256":
        "9e84de4084339cfa057a959617e73dd836d78b03f98e0117f7b60b0564250fda",
    "b1_candidate_B_content_sha256":
        "f1cc45014158f0497632c048da51db6d185e3d98a85c3774aeea154b9b046025",
    "b1_candidate_C_content_sha256":
        "97394e6501a5dd07206f115dca91345358c467dcaa1b17fd98e57bc58de8c2f0",
    "b1_paired_trajectory_hash":
        "ff1de559b2746dcb394c4de6d70bda4d96ca594d1b953f58a4fa390366f41cf6",
    "b1_verdict": "B1_CALIBRATION_FAIL",
    "b1_selected_candidate": "None",
    # B1 的旧 RMS 口径标定尺度 —— B1.1 §4 要用它算 ratio, 必须确认未被篡改
    "b1_b0_scale": "7.225772358824458",
    "b1_speed_util_ref_rms": "0.33039016838980756",
    "b1_torque_util_ref_rms": "0.11694081305399358",
}


def sha256_of(rel: str) -> str:
    p = ROOT / rel
    if not p.exists():
        return "MISSING"
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _json(rel: str):
    p = ROOT / rel
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def read_v1_content_hashes() -> dict:
    out = {}
    for name, (rel, field) in V1_CONTENT_HASHES.items():
        d = _json(rel)
        if d is None:
            out[name] = "MISSING"
            continue
        try:
            out[name] = str(d[field])
        except Exception as e:                                  # noqa: BLE001
            out[name] = f"ERROR:{type(e).__name__}"
    return out


def read_b1_results() -> dict:
    """读出 B1 的数值结论 (candidate hashes / verdict / 旧 RMS 标定)。"""
    out: dict[str, str] = {}
    cand = _json("checkpoints/basilisk_b1/candidates.json")
    keys = ("b1_candidate_A_content_sha256", "b1_candidate_B_content_sha256",
            "b1_candidate_C_content_sha256")
    for key, name in zip(keys, B1_CANDIDATE_NAMES):
        if cand is None:
            out[key] = "MISSING"
            continue
        try:
            out[key] = str(cand["candidates"][name]["content_sha256"])
        except Exception as e:                                  # noqa: BLE001
            out[key] = f"ERROR:{type(e).__name__}"
    if cand is None:
        out["b1_paired_trajectory_hash"] = "MISSING"
        out["b1_b0_scale"] = "MISSING"
        out["b1_speed_util_ref_rms"] = "MISSING"
        out["b1_torque_util_ref_rms"] = "MISSING"
    else:
        out["b1_paired_trajectory_hash"] = str(
            cand.get("paired_trajectory_hash", "ABSENT"))
        out["b1_b0_scale"] = repr(
            cand.get("b0_calibration", {}).get("b0_scale", "ABSENT"))
        out["b1_speed_util_ref_rms"] = repr(
            cand.get("reference_duty", {}).get("speed_util", "ABSENT"))
        out["b1_torque_util_ref_rms"] = repr(
            cand.get("reference_duty", {}).get("torque_util", "ABSENT"))
    # repr() 对 float 给出全精度且无引号问题, 但字符串会带引号 -> 统一剥掉
    for k in ("b1_b0_scale", "b1_speed_util_ref_rms", "b1_torque_util_ref_rms"):
        out[k] = out[k].strip("'\"")

    audit = _json("checkpoints/basilisk_b1/candidate_audit.json")
    if audit is None:
        out["b1_verdict"] = "MISSING"
        out["b1_selected_candidate"] = "MISSING"
    else:
        out["b1_verdict"] = str(audit.get("verdict", "ABSENT"))
        out["b1_selected_candidate"] = str(audit.get("selected_candidate", "ABSENT"))
    return out


def compute() -> dict:
    groups = {g: {rel: sha256_of(rel) for rel in files}
              for g, files in FROZEN_GROUPS.items()}
    flat = {rel: h for g in groups.values() for rel, h in g.items()}
    return {
        "stage": "BASILISK_B1.1",
        "purpose": ("冻结 analytic lineage + BASILISK_V1 产物 + BASILISK_B1 全部产物"
                    "与失败结论 (B1_CALIBRATION_FAIL)。B1.1 是独立分支, 不修补 B1。"
                    "B1.1 无 mutable 组: 口径改动只落在 b11 新文件内。"),
        "frozen_chain": {
            "S2.5": "S25_PASS", "S3": "S3_NO_TRANSFER_SIGNAL",
            "S4": "frozen metrics", "S5": "rate model done",
            "S5B": "S5B_BASELINE_WEAK", "BASILISK_V1": "BASILISK_V1_READY",
            "BASILISK_B1": "B1_CALIBRATION_FAIL",
        },
        "env": {"python": sys.version.split()[0], "platform": platform.platform()},
        "groups": groups,
        "flat_sha256": flat,
        "v1_content_hashes": read_v1_content_hashes(),
        "v1_expected": dict(V1_EXPECTED),
        "b1_results": read_b1_results(),
        "b1_expected": dict(B1_EXPECTED),
        "n_files": len(flat),
    }


def verify(tag: str) -> int:
    if not CONTRACT.exists():
        print(f"[{tag}] 契约文件不存在: {CONTRACT}")
        return 1
    ref = json.loads(CONTRACT.read_text(encoding="utf-8"))
    bad = []
    for rel, h_ref in ref["flat_sha256"].items():
        h_now = sha256_of(rel)
        if h_now != h_ref:
            bad.append((rel, h_ref, h_now))

    now_v1 = read_v1_content_hashes()
    for name, h_ref in ref["v1_content_hashes"].items():
        if now_v1.get(name) != h_ref:
            bad.append((f"[v1_content] {name}", h_ref, now_v1.get(name, "ABSENT")))
    for name, h_exp in V1_EXPECTED.items():
        if now_v1.get(name) != h_exp:
            bad.append((f"[v1_expected] {name}", h_exp, now_v1.get(name, "ABSENT")))

    now_b1 = read_b1_results()
    for name, h_ref in ref["b1_results"].items():
        if now_b1.get(name) != h_ref:
            bad.append((f"[b1_result] {name}", h_ref, now_b1.get(name, "ABSENT")))
    for name, h_exp in B1_EXPECTED.items():
        if now_b1.get(name) != h_exp:
            bad.append((f"[b1_expected] {name}", h_exp, now_b1.get(name, "ABSENT")))

    n_total = (ref["n_files"] + len(ref["v1_content_hashes"]) + len(V1_EXPECTED)
               + len(ref["b1_results"]) + len(B1_EXPECTED))
    print(f"[{tag}] 契约项数 {n_total}, 不变 {n_total - len(bad)}")
    for rel, a, b in bad:
        print(f"[{tag}] CHANGED(禁止) {rel}\n         {str(a)[:24]} -> {str(b)[:24]}")
    if bad:
        print(f"[{tag}] B11_BASELINE_CONTRACT_VIOLATION —— 必须立即停止")
        return 1
    print(f"[{tag}] B11_BASELINE_CONTRACT_OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--tag", default="b11_baseline")
    a = ap.parse_args()
    if a.write:
        CONTRACT.parent.mkdir(parents=True, exist_ok=True)
        c = compute()
        CONTRACT.write_text(json.dumps(c, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
        n_extra = (len(c["v1_content_hashes"]) + len(V1_EXPECTED)
                   + len(c["b1_results"]) + len(B1_EXPECTED))
        print(f">> 写入 {CONTRACT.relative_to(ROOT)}  ({c['n_files']} 文件 + "
              f"{n_extra} 数值项)")
        for g, d in c["groups"].items():
            n_missing = sum(1 for v in d.values() if v == "MISSING")
            print(f"   {g:32s} n={len(d):2d} missing={n_missing}")
        for rel, h in c["flat_sha256"].items():
            if h == "MISSING":
                print(f"   [MISSING-预期] {rel}")
        for k, v in c["v1_content_hashes"].items():
            mark = "OK" if v == V1_EXPECTED.get(k) else "MISMATCH"
            print(f"   {k:32s} {v[:16]}  [{mark}]")
        for k, v in c["b1_results"].items():
            mark = "OK" if v == B1_EXPECTED.get(k) else "MISMATCH"
            print(f"   {k:32s} {v[:24]:24s}  [{mark}]")
        return 0
    if a.verify:
        return verify(a.tag)
    ap.error("需要 --write 或 --verify")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
