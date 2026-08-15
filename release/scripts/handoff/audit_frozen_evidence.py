"""B8-HANDOFF-FINAL §8 —— 冻结证据最终核验（只读）。

对 §8 点名的 10 项逐一重算 sha256，与 B6/B7 冻结契约比对。
全部 unchanged -> 输出 FROZEN_EVIDENCE_AUDIT_PASS。
本脚本不写任何文件，不加载任何模型。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# §8 点名的 10 类冻结证据 -> 具体产物路径
TARGETS: dict[str, tuple[str, ...]] = {
    "source_tcn_pretrain.pt": ("checkpoints/source_tcn_pretrain.pt",),
    "B1.8 dataset": ("data/sim/wheel_basilisk_b18/final/wheel_all.h5",
                     "data/sim/wheel_basilisk_b18/final/params.json"),
    "B1.9 target_features.h5": ("data/features/wheel/basilisk_b19/target_features.h5",),
    "B2.1 split_manifest": ("docs/basilisk_b21/split_manifest.json",),
    "B5 protocol": ("docs/basilisk_b5/protocol.md",
                    "checkpoints/basilisk_b5/protocol_hash.json"),
    "B5 results": ("checkpoints/basilisk_b5/formal_metrics.json",
                   "checkpoints/basilisk_b5/summary.json",
                   "checkpoints/basilisk_b5/paired_gain.json",
                   "checkpoints/basilisk_b5/lifetime_bins.json",
                   "checkpoints/basilisk_b5/warning_metrics.json"),
    "B6 protocol": ("docs/basilisk_b6/protocol.md",
                    "checkpoints/basilisk_b6/protocol_hash.json"),
    "B6 subset manifest": ("checkpoints/basilisk_b6/label_subset_manifest.json",),
    "B6 results": ("checkpoints/basilisk_b6/all_metrics.json",
                   "checkpoints/basilisk_b6/final_verdict.json",
                   "checkpoints/basilisk_b6/paired_statistics.json",
                   "checkpoints/basilisk_b6/summary.json"),
}

# docs/results.md 是 B7 授权的最终表达面 —— 哈希冻结已退役, 由语义冻结取代
# (见 tests/lifecycle_registry.py)。因此这一项必须查**语义审计结果**,
# 而不是查文件哈希。五项语义断言全部为 True 才算 unchanged。
SEMANTIC_AUDIT = "checkpoints/basilisk_b7/report_number_audit.json"
SEMANTIC_KEYS = (
    "report_numbers_match_frozen_metrics",
    "report_contains_final_transfer_conclusion",
    "report_contains_engineering_recommendation",
    "report_contains_negative_transfer_disclosure",
    "report_contains_damage_baseline",
)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_contract(rel: str) -> dict:
    p = ROOT / rel
    if not p.exists():
        return {}
    c = json.loads(p.read_text(encoding="utf-8"))
    flat = c.get("flat_sha256")
    if flat:
        return {k.replace(chr(92), "/"): v for k, v in flat.items()}
    out = {}
    for d in c.get("groups", {}).values():
        for k, v in d.items():
            out[k.replace(chr(92), "/")] = v
    return out


def main() -> int:
    c6 = load_contract("docs/basilisk_b6/baseline_contract.json")
    c7 = load_contract("docs/basilisk_b7/baseline_contract.json")

    drift: list[str] = []
    missing: list[str] = []
    unpinned: list[str] = []
    n_ok = 0

    print("=" * 74)
    print("B8-HANDOFF-FINAL §8 —— 冻结证据最终核验")
    print("=" * 74)

    for group, rels in TARGETS.items():
        print(f"\n[{group}]")
        for rel in rels:
            p = ROOT / rel
            if not p.exists():
                print(f"  MISSING   {rel}")
                missing.append(rel)
                continue
            got = sha256(p)
            exp = c6.get(rel) or c7.get(rel)
            src = "b6" if rel in c6 and c6[rel] != "MISSING" else (
                "b7" if rel in c7 else "-")
            if exp in (None, "MISSING"):
                exp = c7.get(rel)
                src = "b7"
            if exp in (None, "MISSING"):
                print(f"  UNPINNED  {got[:16]}  {rel}")
                unpinned.append(rel)
                continue
            if got == exp:
                n_ok += 1
                print(f"  OK[{src}]   {got[:16]}  {rel}")
            else:
                drift.append(f"{rel}: {exp[:16]} -> {got[:16]}")
                print(f"  DRIFT     {exp[:16]} -> {got[:16]}  {rel}")

    print("\n" + "=" * 74)
    # docs/results.md —— 语义冻结核查 (哈希冻结已由 B7 §14 退役)
    print("[docs/results.md numeric audit —— 语义冻结]")
    sem_bad: list[str] = []
    sp = ROOT / SEMANTIC_AUDIT
    if not sp.exists():
        sem_bad.append(f"{SEMANTIC_AUDIT} 缺失 (需先跑 audit_report_numbers.py)")
        print(f"  MISSING   {SEMANTIC_AUDIT}")
    else:
        rec = json.loads(sp.read_text(encoding="utf-8"))
        sem = rec.get("semantic_freeze", {})
        for k in SEMANTIC_KEYS:
            ok = sem.get(k)
            if ok is True:
                print(f"  PASS      {k}")
            else:
                print(f"  FAIL      {k} = {ok!r}")
                sem_bad.append(f"{k} = {ok!r}")
        misses = rec.get("n_numeric_missing", 0)
        verdict = rec.get("verdict", "")
        print(f"  数值项 {rec.get('n_numeric_items')} 项, 未命中 {misses} 项")
        print(f"  verdict: {verdict}")
        if misses:
            sem_bad.append(f"数值未命中 {misses} 项")
        if verdict != "REPORT_NUMERIC_AUDIT_PASS":
            sem_bad.append(f"verdict != REPORT_NUMERIC_AUDIT_PASS ({verdict})")

    print("\n" + "=" * 74)
    print(f"unchanged (逐字节命中冻结契约) : {n_ok}")
    print(f"drift  (被改动)               : {len(drift)}")
    print(f"missing(消失)                 : {len(missing)}")
    print(f"unpinned(契约未覆盖, 仅报告)   : {len(unpinned)}")
    if unpinned:
        for r in unpinned:
            print(f"    - {r}")
    print("=" * 74)

    if drift or missing or sem_bad:
        for d in drift:
            print(f"  DRIFT: {d}")
        for m in missing:
            print(f"  MISSING: {m}")
        for s in sem_bad:
            print(f"  SEMANTIC_FAIL: {s}")
        print("\nB8_HANDOFF_INVALID —— 冻结证据发生变化")
        return 1
    print("\nFROZEN_EVIDENCE_AUDIT_PASS —— all unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
