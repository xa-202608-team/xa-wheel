#!/usr/bin/env python
"""scripts/basilisk_b4x/summarize_b4x.py

BASILISK-B4X §15/§17/§18 —— gain 汇总、崩塌归因与探索性分类。

`POST_B2_FAIL_EXPLORATORY` / `NOT_FORMAL_EVIDENCE` / `EXPLORATORY_ONLY`
`B2_GENERALIZATION_FAIL_REMAINS_FINAL` / `NOT_FORMAL_TRANSFER_EVIDENCE`

只允许三个标签 (§17):
    B4X_TRANSFER_STABILIZATION_SIGNAL   A 组四条判据同时成立
    B4X_GENERIC_TRANSFER_SIGNAL         A 之外额外满足 B 组三条
    B4X_NO_TRANSFER_SIGNAL              以上均不成立
三者都必须同时写 EXPLORATORY_ONLY / B2_GENERALIZATION_FAIL_REMAINS_FINAL /
NOT_FORMAL_TRANSFER_EVIDENCE。

§18 崩塌归因 (本脚本最重要的部分):
  source 组 RMSE 更低这件事**本身不构成迁移证据**。必须先判断:
  优势是不是几乎全部来自 target_only 在外推区崩溃, 而 source 模型只是更保守?
  判断材料 = catastrophic_error_rate + normal-EOL gain + PSR + pred std。
  若是, 结论只能写成
      "source initialization regularizes extrapolation under the known
       B2 distribution gap"
  而不是 positive transfer proven。本脚本把这句判断机器化输出为
  `collapse_attribution.reading`, 不留给事后叙述。

§13 两套 seed **分开**统计, 不合并成一个 6-seed 平均或显著性数字。

用法:
    python scripts/basilisk_b4x/summarize_b4x.py --config configs/wheel_basilisk_b4x.yaml
"""
from __future__ import annotations

import argparse
import hashlib
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
from scripts.basilisk_b3x.summarize_b3x import paired_bootstrap, _stat  # noqa: E402

CFG_PATH = "configs/wheel_basilisk_b4x.yaml"
NAN = float("nan")

LABEL_A = "B4X_TRANSFER_STABILIZATION_SIGNAL"
LABEL_B = "B4X_GENERIC_TRANSFER_SIGNAL"
LABEL_C = "B4X_NO_TRANSFER_SIGNAL"
ALLOWED_LABELS = (LABEL_A, LABEL_B, LABEL_C)

MANDATORY_QUALIFIERS = ("EXPLORATORY_ONLY",
                        "B2_GENERALIZATION_FAIL_REMAINS_FINAL",
                        "NOT_FORMAL_TRANSFER_EVIDENCE")

SOURCE_METHODS = ("source_finetune", "source_mmd_finetune")
GAIN_KEY = {"source_finetune": "gain_ft", "source_mmd_finetune": "gain_mmd"}

SUMMARY_PURPOSE = (
    "B4X §15/§17/§18: 汇总三个子集上的 gain_ft / gain_mmd, 做崩塌归因, "
    "给出探索性分类。本阶段不是正式 B4, 不构成迁移证据, 不宣称迁移有效。"
    "B2_GENERALIZATION_FAIL 保持终局, 不被本阶段覆盖或重新解释。"
    "两套 seed 分开统计, 不合并成一个 6-seed 平均。"
)

# §18 表述纪律 (模块级常量, 扫描器豁免)。
REGULARIZATION_PHRASING = (
    "source initialization regularizes extrapolation under the known "
    "B2 distribution gap")
FORBIDDEN_CLAIM_NOTE = (
    "禁止写 positive transfer proven / 迁移显著有效。source 组 RMSE 更低"
    "可能只是 target_only 在外推区崩溃而 source 模型更保守, 那不是跨域增益。"
    "只有 normal-EOL 也持续改善且 corr 同时上升时才谈得上 generic signal, "
    "而即便如此它仍然只是探索性信号, 不是正式迁移证据。")


def _num(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return NAN
    return f


def _finite(v) -> bool:
    return np.isfinite(_num(v))


def method_block(rows: list[dict], method: str, vc: dict) -> dict:
    """一个 source 方法在一套 seed 内的完整统计。"""
    gk = GAIN_KEY[method]
    ci = vc["_ci"]
    out: dict = {"method": method, "gain_key": gk, "per_seed": {}}
    per_sub: dict[str, list[float]] = {"overall": [], "short_eol": [],
                                       "normal_eol": []}
    cat_reduced, cat_detail = [], {}
    corr_delta, warn_delta, psr, pstd = [], [], [], []
    for r in rows:
        s = int(r["seed"])
        g = r["gains"]
        row = {sub: _num(g[sub][gk]) for sub in per_sub}
        for sub in per_sub:
            per_sub[sub].append(row[sub])
        row["rel_gain_overall"] = _num(g["overall"][f"rel_{gk}"])
        row["rel_gain_short_eol"] = _num(g["short_eol"][f"rel_{gk}"])
        row["rel_gain_normal_eol"] = _num(g["normal_eol"][f"rel_{gk}"])
        # §16: catastrophic rate 是否降低
        c0 = _num(r["target_only"]["catastrophic"]["catastrophic_error_rate"])
        cm = _num(r[method]["catastrophic"]["catastrophic_error_rate"])
        row["catastrophic_rate_target_only"] = c0
        row["catastrophic_rate_method"] = cm
        row["catastrophic_reduced"] = bool(np.isfinite(c0) and np.isfinite(cm)
                                           and cm < c0)
        cat_reduced.append(row["catastrophic_reduced"])
        cat_detail[str(s)] = {
            "target_only": c0, method: cm,
            "n_catastrophic_target_only":
                r["target_only"]["catastrophic"]["n_catastrophic"],
            f"n_catastrophic_{method}":
                r[method]["catastrophic"]["n_catastrophic"],
            "threshold": _num(r["catastrophic_threshold"]["threshold"]),
        }
        # corr / warning / PSR / pred std —— §17 B 的判据 + §18 归因材料
        d_corr = (_num(r[method]["overall"]["corr"])
                  - _num(r["target_only"]["overall"]["corr"]))
        wm0 = _num(r["target_only"]["overall"]["warning"]
                   ["miss_rate_before_eol"])
        wmm = _num(r[method]["overall"]["warning"]["miss_rate_before_eol"])
        d_warn = wmm - wm0
        row.update({"delta_macro_corr": d_corr,
                    "warning_miss_target_only": wm0,
                    "warning_miss_method": wmm,
                    "delta_warning_miss": d_warn,
                    "psr_target_only": _num(r["target_only"]["overall"]["psr"]),
                    "psr_method": _num(r[method]["overall"]["psr"]),
                    "pred_std_target_only":
                        _num(r["target_only"]["overall"]["pred_std"]),
                    "pred_std_method": _num(r[method]["overall"]["pred_std"])})
        corr_delta.append(d_corr)
        warn_delta.append(d_warn)
        psr.append(_num(r[method]["overall"]["psr"])
                   - _num(r["target_only"]["overall"]["psr"]))
        pstd.append(_num(r[method]["overall"]["pred_std"])
                    - _num(r["target_only"]["overall"]["pred_std"]))
        out["per_seed"][str(s)] = row

    out["gain_stat"] = {sub: _stat(v) for sub, v in per_sub.items()}
    out["gain_per_seed"] = {sub: {str(int(r["seed"])): _num(r["gains"][sub][gk])
                                  for r in rows}
                            for sub in per_sub}
    out["improve_count"] = {sub: int(sum(1 for x in v
                                         if np.isfinite(x) and x > 0))
                            for sub, v in per_sub.items()}
    out["n_seeds"] = len(rows)
    out["descriptive_ci"] = {
        sub: paired_bootstrap(v, ci["bootstrap_samples"], ci["bootstrap_seed"])
        for sub, v in per_sub.items()}
    out["catastrophic"] = {
        "n_reduced": int(sum(cat_reduced)),
        "n_seeds": len(cat_reduced),
        "per_seed": cat_detail,
    }
    out["delta_macro_corr"] = _stat(corr_delta)
    out["delta_warning_miss"] = _stat(warn_delta)
    out["delta_psr"] = _stat(psr)
    out["delta_pred_std"] = _stat(pstd)
    return out


def attribute_collapse(mb: dict, vc: dict) -> dict:
    """§18: 判断 source 优势是"避免了崩溃"还是"正常区域也改善"。

    这不是叙述性判断, 而是把 protocol 的判定规则机器化:
      * catastrophic 明显减少 + short-EOL 大幅改善 + normal-EOL gain ≈ 0/为负
        + PSR / pred std 明显下降  -> regularizes_extrapolation
      * normal-EOL 也持续改善 + corr 上升 -> normal_region_also_improves
    """
    st = vc["stabilization"]
    gen = vc["generic"]
    g_ov = mb["gain_stat"]["overall"]["mean"]
    g_sh = mb["gain_stat"]["short_eol"]["mean"]
    g_no = mb["gain_stat"]["normal_eol"]["mean"]
    n_imp_no = mb["improve_count"]["normal_eol"]
    n_seeds = mb["n_seeds"]
    d_corr = mb["delta_macro_corr"]["mean"]
    d_psr = mb["delta_psr"]["mean"]
    d_pstd = mb["delta_pred_std"]["mean"]
    n_cat = mb["catastrophic"]["n_reduced"]

    normal_improves = bool(np.isfinite(g_no) and g_no > 0
                           and n_imp_no > n_seeds / 2.0)
    corr_up = bool(np.isfinite(d_corr) and d_corr > 0)
    conservative = bool((np.isfinite(d_psr) and d_psr < 0)
                        or (np.isfinite(d_pstd) and d_pstd < 0))
    # short-EOL 主导: short-EOL 的平均 gain 在数量级上压过 normal-EOL
    short_dominates = bool(np.isfinite(g_sh) and np.isfinite(g_no)
                           and abs(g_sh) > 10.0 * abs(g_no))

    if normal_improves and corr_up and not short_dominates:
        reading = "normal_region_also_improves"
        phrase = ("normal-EOL 也有持续改善且 macro corr 上升, 且 short-EOL 未"
                  "在数量级上主导 —— 具备 generic signal 的材料, 但仍然只是"
                  "探索性信号, 不是正式迁移证据")
    elif n_cat > 0 or short_dominates or conservative:
        reading = "regularizes_extrapolation"
        phrase = REGULARIZATION_PHRASING
    else:
        reading = "no_advantage_to_attribute"
        phrase = "source 组没有稳定优势可归因"
    return {
        "reading": reading,
        "phrasing": phrase,
        "forbidden_claim_note": FORBIDDEN_CLAIM_NOTE,
        "evidence": {
            "mean_gain_overall": g_ov,
            "mean_gain_short_eol": g_sh,
            "mean_gain_normal_eol": g_no,
            "normal_eol_improve_count": n_imp_no,
            "n_seeds": n_seeds,
            "mean_delta_macro_corr": d_corr,
            "mean_delta_psr": d_psr,
            "mean_delta_pred_std": d_pstd,
            "n_seeds_catastrophic_reduced": n_cat,
            "short_eol_dominates_normal_eol": short_dominates,
            "short_over_normal_gain_ratio": (
                float(abs(g_sh) / abs(g_no))
                if np.isfinite(g_sh) and np.isfinite(g_no) and g_no != 0
                else NAN),
            "short_dominates_multiplier": 10.0,
            "source_more_conservative": conservative,
            "normal_region_improves": normal_improves,
            "corr_increases": corr_up,
        },
        "thresholds": {
            "normal_eol_max_rel_degradation":
                float(st["normal_eol_max_rel_degradation"]),
            "corr_worse_tol": float(gen["corr_worse_tol"]),
            "warning_miss_worse_tol": float(gen["warning_miss_worse_tol"]),
        },
    }


def evaluate_conditions(mb_replay: dict, mb_new: dict, vc: dict) -> dict:
    """§17 A / B 判据。A 只看 replay 套, B 额外要求 new 套。"""
    st, gen = vc["stabilization"], vc["generic"]
    g_ov = mb_replay["gain_stat"]["overall"]["mean"]
    g_sh = mb_replay["gain_stat"]["short_eol"]["mean"]
    g_no_rel = [v["rel_gain_normal_eol"]
                for v in mb_replay["per_seed"].values()]
    # "没有明显系统性变差": normal-EOL 平均相对恶化不超过阈值
    # rel_gain 为正 = 改善, 为负 = 恶化。取平均恶化幅度。
    deg = [-x for x in g_no_rel if np.isfinite(x)]
    mean_deg = float(np.mean(deg)) if deg else NAN
    max_deg = float(st["normal_eol_max_rel_degradation"])

    A = [
        {"id": "A1",
         "text": (f"replay seeds 中 catastrophic_error_rate 降低的数量 >= "
                  f"{st['min_replay_seeds_catastrophic_reduced']}"),
         "value": mb_replay["catastrophic"]["n_reduced"],
         "threshold": int(st["min_replay_seeds_catastrophic_reduced"]),
         "passed": (mb_replay["catastrophic"]["n_reduced"]
                    >= int(st["min_replay_seeds_catastrophic_reduced"]))},
        {"id": "A2", "text": "replay short-EOL mean gain > 0",
         "value": g_sh, "threshold": 0.0,
         "passed": bool(np.isfinite(g_sh) and g_sh > 0)
         if st["require_short_eol_improved"] else True},
        {"id": "A3", "text": "replay overall mean gain > 0",
         "value": g_ov, "threshold": 0.0,
         "passed": bool(np.isfinite(g_ov) and g_ov > 0)
         if st["require_overall_mean_gain_positive"] else True},
        {"id": "A4",
         "text": f"replay normal-EOL 平均相对恶化 <= {max_deg:.0%}",
         "value": mean_deg, "threshold": max_deg,
         "passed": bool(np.isfinite(mean_deg) and mean_deg <= max_deg)},
    ]
    n_new_pos = mb_new["improve_count"]["overall"]
    d_corr = mb_new["delta_macro_corr"]["mean"]
    d_warn = mb_new["delta_warning_miss"]["mean"]
    g_no_new = mb_new["gain_stat"]["normal_eol"]["mean"]
    B = [
        {"id": "B1",
         "text": f"new seeds 中 gain>0 的数量 >= {gen['min_new_seeds_gain_positive']}",
         "value": n_new_pos,
         "threshold": int(gen["min_new_seeds_gain_positive"]),
         "passed": n_new_pos >= int(gen["min_new_seeds_gain_positive"])},
        {"id": "B2", "text": "new seeds normal-EOL mean gain > 0",
         "value": g_no_new, "threshold": 0.0,
         "passed": bool(np.isfinite(g_no_new) and g_no_new > 0)
         if gen["require_normal_eol_improved"] else True},
        {"id": "B3",
         "text": f"macro corr 不恶化 (容差 {gen['corr_worse_tol']})",
         "value": d_corr, "threshold": -float(gen["corr_worse_tol"]),
         "passed": bool(np.isfinite(d_corr)
                        and d_corr >= -float(gen["corr_worse_tol"]))
         if gen["require_corr_not_worse"] else True},
        {"id": "B4",
         "text": (f"warning miss 不恶化 (容差 "
                  f"{gen['warning_miss_worse_tol']})"),
         "value": d_warn, "threshold": float(gen["warning_miss_worse_tol"]),
         "passed": bool(np.isfinite(d_warn)
                        and d_warn <= float(gen["warning_miss_worse_tol"]))
         if gen["require_warning_not_worse"] else True},
    ]
    return {"A": A, "B": B,
            "A_passed": all(c["passed"] for c in A),
            "B_passed": all(c["passed"] for c in B),
            "n_A_passed": sum(1 for c in A if c["passed"]),
            "n_B_passed": sum(1 for c in B if c["passed"])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CFG_PATH)
    a = ap.parse_args()

    cfg = load_cfg(a.config)
    b4 = cfg["b4x"]
    vc = dict(b4["verdict"])
    vc["_ci"] = b4["descriptive_ci"]

    sp = ROOT / cfg["paths"]["stability_json"]
    if not sp.exists():
        raise SystemExit(f"!! 先跑 diagnose_transfer_stability.py: 缺 {sp}")
    stab = json.loads(sp.read_text(encoding="utf-8"))
    tm = json.loads(
        (ROOT / cfg["paths"]["metrics_json"]).read_text(encoding="utf-8"))
    if tm.get("fast"):
        raise SystemExit("!! transfer metrics 是 --fast 冒烟产物, 不得进汇总")

    sets = list(stab["per_seed_by_set"].keys())
    for need in ("new", "replay"):
        if need not in sets:
            raise SystemExit(f"!! 缺 seed 套 {need!r} —— §13 要求两套都跑")

    # §13: 两套各自独立统计。结构上不给合并留位置。
    by_set: dict[str, dict] = {}
    for sname in ("new", "replay"):
        rows = stab["per_seed_by_set"][sname]
        by_set[sname] = {
            "seeds": [int(r["seed"]) for r in rows],
            "methods": {m: method_block(rows, m, vc) for m in SOURCE_METHODS},
            "target_only_overall_rmse": {
                str(int(r["seed"])): _num(r["target_only"]["overall"]["rmse"])
                for r in rows},
        }

    # §17: A 只看 replay, B 额外看 new。每个 source 方法各判一次;
    # "至少一个 source 方法满足"即成立。
    per_method_conditions: dict[str, dict] = {}
    for m in SOURCE_METHODS:
        cond = evaluate_conditions(by_set["replay"]["methods"][m],
                                   by_set["new"]["methods"][m], vc)
        cond["collapse_attribution_replay"] = attribute_collapse(
            by_set["replay"]["methods"][m], vc)
        cond["collapse_attribution_new"] = attribute_collapse(
            by_set["new"]["methods"][m], vc)
        per_method_conditions[m] = cond

    a_ok = [m for m in SOURCE_METHODS if per_method_conditions[m]["A_passed"]]
    b_ok = [m for m in SOURCE_METHODS
            if per_method_conditions[m]["A_passed"]
            and per_method_conditions[m]["B_passed"]]
    if b_ok:
        verdict = LABEL_B
    elif a_ok:
        verdict = LABEL_A
    else:
        verdict = LABEL_C
    if verdict not in ALLOWED_LABELS:
        raise SystemExit(f"!! 非法标签 {verdict!r}")
    allowed_cfg = tuple(str(x) for x in vc["allowed_labels"])
    if verdict not in allowed_cfg:
        raise SystemExit(f"!! 标签 {verdict!r} 不在 config 允许集 {allowed_cfg}")

    protocol = ROOT / cfg["protocol"]["path"]
    ptext = protocol.read_bytes()

    out = {
        "stage": "BASILISK_B4X",
        "label_discipline": str(b4["label"]),
        "formal_stage": bool(b4.get("formal_stage", False)),
        "not_formal_evidence": True,
        "mandatory_qualifiers": list(MANDATORY_QUALIFIERS),
        "exploratory_only": True,
        "b2_verdict_unchanged": "B2_GENERALIZATION_FAIL",
        "b2_verdict_status": "B2_GENERALIZATION_FAIL_REMAINS_FINAL",
        "not_formal_transfer_evidence": True,
        "purpose": SUMMARY_PURPOSE,
        "forbid_positive_transfer_claim": bool(
            vc.get("forbid_positive_transfer_claim", True)),
        "regularization_phrasing": REGULARIZATION_PHRASING,
        "forbidden_claim_note": FORBIDDEN_CLAIM_NOTE,
        "allowed_labels": list(ALLOWED_LABELS),
        "input_schema": str(tm["input_schema"]),
        "input_schema_decided_by": str(tm["input_schema_decided_by"]),
        "architecture_note": str(tm["architecture_note"]),
        "split_sha256": str(tm["split_sha256"]),
        "groups": list(tm["groups"]),
        "short_eol_subset": list(stab["short_eol_subset"]),
        "n_normal_eol": len(stab["normal_eol_subset"]),
        "catastrophic_definition": dict(stab["catastrophic"]),
        "seed_sets": {k: v["seeds"] for k, v in by_set.items()},
        "forbid_merging_seed_sets": True,
        "descriptive_ci": dict(b4["descriptive_ci"]),
        "by_seed_set": by_set,
        "conditions_by_method": per_method_conditions,
        "methods_passing_A": a_ok,
        "methods_passing_A_and_B": b_ok,
        "verdict": verdict,
        "verdict_rule": {
            LABEL_B: "至少一个 source 方法同时满足 A 与 B",
            LABEL_A: "至少一个 source 方法满足 A (但没有方法同时满足 B)",
            LABEL_C: "没有方法满足 A",
        },
        "thresholds_frozen_before_run": True,
        "threshold_source": (
            "docs/basilisk_b4x/protocol.md §9 + "
            "configs/wheel_basilisk_b4x.yaml::b4x.verdict"),
        "b2_1_next_step_question": (
            "是否值得做新的 B2.1 split-coverage robustness study —— "
            "由本阶段的 short-EOL 覆盖问题与 B3X 发现的 batch-order 敏感性共同"
            "决定, 结论写在 STATUS_BASILISK_B4X.md"),
    }
    op = ROOT / cfg["paths"]["summary_json"]
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2, ensure_ascii=False,
                             default=float) + "\n", encoding="utf-8")

    hp = ROOT / cfg["protocol"]["hash_path"]
    hp.write_text(json.dumps({
        "protocol": str(protocol.relative_to(ROOT)),
        "sha256": hashlib.sha256(ptext).hexdigest(),
        "n_bytes": len(ptext),
        "frozen_before_any_b4x_number": True,
        "label": str(b4["label"]),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n{'=' * 66}")
    for sname in ("new", "replay"):
        print(f"[{sname}] seeds {by_set[sname]['seeds']}")
        for m in SOURCE_METHODS:
            mb = by_set[sname]["methods"][m]
            print(f"   {m}: overall mean gain "
                  f"{mb['gain_stat']['overall']['mean']:+.6f} "
                  f"(improve {mb['improve_count']['overall']}/{mb['n_seeds']}) | "
                  f"short-EOL {mb['gain_stat']['short_eol']['mean']:+.6f} | "
                  f"normal {mb['gain_stat']['normal_eol']['mean']:+.6f} | "
                  f"catastrophic reduced "
                  f"{mb['catastrophic']['n_reduced']}/{mb['n_seeds']}")
    print(f"{'=' * 66}")
    for m in SOURCE_METHODS:
        c = per_method_conditions[m]
        print(f"[{m}] A {c['n_A_passed']}/4  B {c['n_B_passed']}/4  "
              f"归因(replay)={c['collapse_attribution_replay']['reading']}")
        for grp in ("A", "B"):
            for x in c[grp]:
                print(f"   {x['id']} {'PASS' if x['passed'] else 'FAIL'}  "
                      f"{x['text']}  值={x['value']}")
    print(f"{'=' * 66}\nB4X verdict = {verdict}")
    print("   " + " / ".join(MANDATORY_QUALIFIERS))
    print(f">> 已写 {op.relative_to(ROOT)} 与 {hp.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
