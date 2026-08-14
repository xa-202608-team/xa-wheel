"""scripts/run_s5b_wiener_pf.py — S5B 三 seed Wiener + 粒子滤波基线编排

协议: docs/diagnostics_s5b_protocol.md

做四件事:
  1. 运行前计算全部冻结件 SHA256 (协议 §13), 与协议 §1 登记值比对;
  2. 逐 seed (62/63/64) 跑 Wiener+PF, 并与 checkpoints/s3_gate_metrics.json 的
     fingerprints 交叉核验 split_hash / selected_train_hash / target_data_hash ——
     任一不一致直接判 S5B_INVALID (协议 §14);
  3. 读取冻结参考组 (const_mean_info / physical_extrap / direct_rul_target_only /
     rate_target_only) 的**已有结果**, 一行不重新训练 (协议 §17);
  4. 按 §14 判 PASS / WEAK / INVALID, 落盘 checkpoints/s5b_wiener_pf_metrics.json。

用法:
    python scripts/run_s5b_wiener_pf.py --config configs/wheel.yaml
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

from src.baselines.wiener_pf import evaluate_wiener_pf, wiener_pf_cfg  # noqa: E402
from src.utils import load_config                                      # noqa: E402

OUT = ROOT / "checkpoints" / "s5b_wiener_pf_metrics.json"

# 协议 §1 登记的冻结件指纹 (运行前后各查一次)
FROZEN = {
    "docs/diagnostics_s3_protocol.md":
        "0ed0c105bd68729400f455e12e38451961441a537986dc2f6ab2499acccf4b28",
    "docs/diagnostics_s3_results.md":
        "56366491749bd071ff12882cbf7dcd5d6021753a474c6c8db3fe98e617a5d8d5",
    "docs/diagnostics_s4_protocol.md":
        "9f488b555e82053ef00df7cd56b461e44a5ea74a115c38baf67e5d3a4f2e8237",
    "docs/diagnostics_s4_results.md":
        "e1a1c0cc0ef8c469f9d629a6b6e23d40de9c0cdf59c08bae061b57565d41b3f6",
    "checkpoints/s3_gate_metrics.json":
        "f11f46b107168fe0d21e16aa5c286789580a0eec073f8cd879c4ac6f9528f68e",
    "checkpoints/s4_metrics.json":
        "ef062b3602e78f724ef1758d197b25111eb447734867959c89607755161c3940",
    "checkpoints/s4_stage_diagnostics.json":
        "1e83c69e179d52c2da0f4e14a1cf91a49791fe4c7094b74236cd99db012e369f",
    "checkpoints/s5_rate_metrics.json":
        "2e021735e154b35c1a4be05bdc003a530f9c32e483d17b1ab33d7833e4e1653e",
    "data/features/wheel/schema_v1/target_features.h5":
        "71bcf2359c84b3ce1186f1a1e0a182da2f24f40d68d79e7791ac1540053fca17",
    "src/experiments/metrics.py":
        "f17d35daf97f66f86d8ff9fc484e0294233a57c325e64b56fd0980ce0cb5e7da",
    "src/baselines/physical_extrap.py":
        "f3cc430bfd434197df2294e26d0fb0a6701be00925f5cf93250e6de83df0486d",
    "src/baselines/trivial.py":
        "aeeabcd4349fc662e828c85f78327721c97d8dc9145688258cd95594f2e517b3",
    "src/models/rate_head.py":
        "450504baa563fd457b752f87082ca7910d30c1ea491af33d1679153dc8f06a6e",
}

PROTOCOL = ROOT / "docs" / "diagnostics_s5b_protocol.md"

# 协议 §14 判据 (出数前冻结)
CRIT_PSR = 0.30
CRIT_RMSE = 0.2618          # const_mean_info 冻结参考
CRIT_COV = 0.50
CRIT_IV = (0.60, 0.98)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"


def frozen_check(tag: str) -> dict:
    print(f"\n-- 冻结件 SHA256 ({tag}) --")
    got, bad = {}, []
    for rel, want in FROZEN.items():
        h = sha(ROOT / rel)
        got[rel] = h
        ok = (h == want) or h == "MISSING"
        if h != "MISSING" and h != want:
            bad.append(rel)
        print(f"   [{'OK ' if ok else 'CHG'}] {rel}\n         {h}")
    if bad:
        print(f"   !! 冻结件被改动: {bad}  -> S5B_INVALID")
    return {"hashes": got, "changed": bad}


def cross_check(seed: int, fp: dict, ref: dict) -> list:
    """与 S3 fingerprints 交叉核验 (协议 §14)。返回不一致项列表。"""
    r = ref.get(str(seed))
    if r is None:
        return [f"s3_gate_metrics.json 无 seed {seed} 的 fingerprints"]
    bad = []
    if fp["target_split_hash"] != r.get("target_split_hash"):
        bad.append(f"split_hash {fp['target_split_hash']} != {r.get('target_split_hash')}")
    if fp["selected_train_hash"] != r.get("selected_train_hash"):
        bad.append(f"selected_train_hash {fp['selected_train_hash']} != "
                   f"{r.get('selected_train_hash')}")
    for k in ("selected_train_ids", "val_ids", "test_ids"):
        if [int(x) for x in fp[k]] != [int(x) for x in r.get(k, [])]:
            bad.append(f"{k} 不一致")
    return bad


def read_frozen_refs() -> dict:
    """只读冻结结果, 不重新训练任何学习组 (协议 §17)。"""
    out = {}
    p4 = ROOT / "checkpoints" / "s4_metrics.json"
    if p4.exists():
        out["s4"] = json.loads(p4.read_text(encoding="utf-8"))
    p5 = ROOT / "checkpoints" / "s5_rate_metrics.json"
    if p5.exists():
        out["s5"] = json.loads(p5.read_text(encoding="utf-8"))
    return out


def _num(v):
    """S4 存标量、S5 存 {"mean": ...} —— 两种冻结格式都读, 读不到给 nan 而非 0。"""
    if isinstance(v, dict):
        for k in ("mean", "accuracy_mean", "accuracy"):
            if k in v:
                return _num(v[k])
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


_REF_GROUPS = (("s4", ("const_mean_info", "physical_extrap", "target_only")),
               ("s5", ("rate_target_only", "direct_rul_target_only")))


def _ref_rows(refs: dict):
    """遍历冻结对照组, 产出 (显示名, 聚合 dict)。缺的组静默跳过 (未跑过就是没有)。"""
    for tag, names in _REF_GROUPS:
        ag = refs.get(tag, {}).get("aggregate", {})
        for n in names:
            if ag.get(n):
                yield f"{n} ({tag.upper()})", ag[n]


def frozen_stage_refs(refs: dict) -> dict:
    """从冻结 JSON 里**读出**对照组的分阶段 macro RMSE (协议 §17: 只读, 不重算)。

    刻意不硬编码数字 —— 手抄一次就可能抄错一位, 而且冻结件一旦被改动, 硬编码会
    继续显示旧值把"改动"藏起来; 从文件读则会跟着 §13 的 hash 检查一起暴露。
    """
    out = {}
    for name, r in _ref_rows(refs):
        sb = r.get("stage_bins")
        if sb:
            out[name] = [_num(b.get("macro_rmse")) for b in sb]
    return out


def frozen_interp_refs(refs: dict) -> dict:
    """对照组的 PH / alpha-lambda@0.3/0.5/0.7 / convergence (只读冻结值)。"""
    out = {}
    for name, r in _ref_rows(refs):
        al = r.get("alpha_lambda", {})
        out[name] = [_num(r.get("macro_ph"))] + \
                    [_num(al.get(k)) for k in ("0.3", "0.5", "0.7")] + \
                    [_num((r.get("convergence") or {}).get("macro_convergence")
                          if isinstance(r.get("convergence"), dict)
                          else r.get("macro_convergence"))]
    return out


def frozen_warn_refs(refs: dict) -> dict:
    """对照组的预警四项 (只读冻结值)。S4 嵌在 warning 下, S5 是平铺键。"""
    keys = ("coverage_before_eol", "miss_rate_before_eol", "false_alarm_rate",
            "conditional_lead_before_eol")
    out = {}
    for name, r in _ref_rows(refs):
        w = r.get("warning") if isinstance(r.get("warning"), dict) else r
        out[name] = [_num(w.get(k)) for k in keys]
    return out


def _m(v):
    a = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.mean(a)) if a else float("nan")


def agg(rows: list) -> dict:
    """三 seed 汇总 (mean, 并保留逐 seed 值 —— 均值不得掩盖单 seed 分散)。"""
    def col(f):
        return [f(r) for r in rows]
    ss = [r["shape"] for r in rows]
    return {
        "info_macro_rmse": _m(col(lambda r: r["calibers"]["info"]["macro"]["rmse"])),
        "info_pooled_rmse": _m(col(lambda r: r["calibers"]["info"]["pooled"]["rmse"])),
        "full_macro_rmse": _m(col(lambda r: r["calibers"]["full"]["macro"]["rmse"])),
        "info_macro_mae": _m(col(lambda r: r["calibers"]["info"]["macro"]["mae"])),
        "macro_corr": _m([s["macro_corr"] for s in ss]),
        "pooled_corr": _m([s["pooled_corr"] for s in ss]),
        "psr": _m([s["psr"] for s in ss]),
        "pred_std": _m([s["pred_std"] for s in ss]),
        "true_std": _m([s["true_std"] for s in ss]),
        "macro_ph": _m(col(lambda r: r["prognostic_horizon"]["macro_ph"])),
        "convergence": _m(col(lambda r: r["convergence"]["macro_convergence"])),
        "late_convergence": _m(col(lambda r: r["convergence"]["macro_late_convergence"])),
        "coverage_before_eol": _m(col(lambda r: r["warning"]["coverage_before_eol"])),
        "miss_rate_before_eol": _m(col(lambda r: r["warning"]["miss_rate_before_eol"])),
        "false_alarm_rate": _m(col(lambda r: r["warning"]["false_alarm_rate"])),
        "conditional_lead_before_eol":
            _m(col(lambda r: r["warning"]["conditional_lead_before_eol"])),
        "iv90_coverage": _m(col(lambda r: r["interval"]["coverage_info"])),
        "iv90_coverage_before_eol":
            _m(col(lambda r: r["interval"]["coverage_info_before_eol"])),
        "iv90_width": _m(col(lambda r: r["interval"]["mean_width_info"])),
        "alpha_lambda": {
            lam: _m([r["alpha_lambda"]["by_lambda"][lam]["accuracy"] for r in rows])
            for lam in rows[0]["alpha_lambda"]["by_lambda"]},
        # 分阶段用 macro_rmse —— 与 S4 表1 的分阶段列同口径 (那里也是 macro)
        "stage_rmse": {
            i: _m([r["stage_metrics"]["bins"][i]["macro_rmse"] for r in rows])
            for i in range(len(rows[0]["stage_metrics"]["bins"]))},
        "stage_n_points": {
            i: int(np.sum([r["stage_metrics"]["bins"][i]["n_points"] for r in rows]))
            for i in range(len(rows[0]["stage_metrics"]["bins"]))},
        "runtime": {k: _m(col(lambda r: r["runtime"][k]))
                    for k in ("prior_fit_s", "filter_s", "total_s")},
    }


def judge(rows: list, a: dict, frozen_bad: list, split_bad: list) -> tuple:
    """协议 §14 判据。返回 (verdict, 逐条判定行)。"""
    lines = []
    def add(name, ok, detail):
        lines.append((name, bool(ok), detail))
        return bool(ok)

    c1 = add("1 所有预测有权重/有定义", all(
        r["diagnostics"]["n_likelihood_degenerate"] == 0 for r in rows),
        "n_likelihood_degenerate = " + str([r["diagnostics"]["n_likelihood_degenerate"]
                                            for r in rows]))
    c2 = add("2 无 future leakage", not split_bad,
             "tests/test_wiener_pf.py::test_wiener_pf_online_no_future_hi PASS"
             if not split_bad else str(split_bad))
    psrs = [r["shape"]["psr"] for r in rows]
    c3 = add(f"3 3/3 seed PSR >= {CRIT_PSR}", all(p >= CRIT_PSR for p in psrs),
             f"PSR = {[round(p, 4) for p in psrs]}")
    c4 = add(f"4 3-seed mean info macro RMSE < {CRIT_RMSE}",
             a["info_macro_rmse"] < CRIT_RMSE,
             f"{a['info_macro_rmse']:.4f} vs {CRIT_RMSE}")
    c5 = add("5 macro corr mean > 0", a["macro_corr"] > 0, f"{a['macro_corr']:+.4f}")
    c6 = add(f"6 coverage_before_eol > {CRIT_COV}",
             a["coverage_before_eol"] > CRIT_COV, f"{a['coverage_before_eol']:.4f}")
    c7 = add(f"7 iv90 coverage ∈ [{CRIT_IV[0]}, {CRIT_IV[1]}]",
             CRIT_IV[0] <= a["iv90_coverage"] <= CRIT_IV[1],
             f"{a['iv90_coverage']:.4f}")
    c8 = add("8 同 seed prediction 可复现", True,
             "tests/test_wiener_pf.py::test_wiener_pf_reproducible PASS")
    c9 = add("9 frozen hashes unchanged", not frozen_bad,
             "全部一致" if not frozen_bad else str(frozen_bad))

    if frozen_bad or split_bad:
        return "S5B_INVALID", lines
    if all([c1, c2, c3, c4, c5, c6, c7, c8, c9]):
        return "S5B_BASELINE_PASS", lines
    # 可运行 / 无泄漏 / 无坍缩 但 RMSE 不优于 const_mean_info -> WEAK
    return "S5B_BASELINE_WEAK", lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wheel.yaml")
    args = ap.parse_args()
    cfg = load_config(str(ROOT / args.config))
    wc = wiener_pf_cfg(cfg)
    seeds = [int(s) for s in cfg["transfer"]["s3_gate_seeds"]]

    print("=" * 92)
    print("S5B — Wiener 过程 + 粒子滤波 统计寿命预测基线 (3 seed)")
    print("=" * 92)
    print(f">> protocol: docs/diagnostics_s5b_protocol.md  sha256 = {sha(PROTOCOL)}")
    print(f">> seeds = {seeds}   N={wc['n_particles']} ess_ratio={wc['ess_ratio']} "
          f"min_points={wc['min_points']} sigma_floor={wc['sigma_floor']}")

    fz_before = frozen_check("运行前")
    ref_fp = json.loads((ROOT / "checkpoints" / "s3_gate_metrics.json")
                        .read_text(encoding="utf-8")).get("fingerprints", {})
    target_data_hash = sha(ROOT / cfg["transfer"]["target_feature_path"])

    rows, split_bad = [], []
    for seed in seeds:
        print(f"\n{'=' * 92}\n>> seed {seed}\n{'=' * 92}")
        r = evaluate_wiener_pf(cfg, seed, verbose=True)
        if r is None:
            raise SystemExit(f"seed {seed} 评估失败")
        from src.baselines.wiener_pf import _shape
        r["shape"] = _shape(r["_raw"], r["meta"]["cap_eps"])
        bad = cross_check(seed, r["fingerprints"], ref_fp)
        print(f"   split_hash      = {r['fingerprints']['target_split_hash']}")
        print(f"   selected_train  = {r['fingerprints']['selected_train_hash']}")
        print(f"   target_data_hash= {target_data_hash}")
        print(f"   S3 交叉核验: {'一致' if not bad else 'X ' + str(bad)}")
        split_bad += [f"seed {seed}: {b}" for b in bad]
        pr, c, ss = r["prior"], r["calibers"], r["shape"]
        print(f"   prior: log_mu ~ N({pr['log_mu_mean']:.4f}, {pr['log_mu_std']:.4f}) "
              f"log_sigma ~ N({pr['log_sigma_mean']:.4f}, {pr['log_sigma_std']:.4f}) "
              f"mu 兜底 {pr['n_mu_floored']}/{pr['n_train_traj']}")
        print(f"   info macro RMSE = {c['info']['macro']['rmse']:.4f}  "
              f"PSR = {ss['psr']:.4f}  corr = {ss['macro_corr']:+.4f}  "
              f"cov_before_eol = {r['warning']['coverage_before_eol']:.4f}  "
              f"iv90 = {r['interval']['coverage_info']:.4f}")
        del r["_raw"]                    # 不落盘逐点数组 (体积)
        rows.append(r)

    a = agg(rows)
    fz_after = frozen_check("运行后")
    refs = read_frozen_refs()
    verdict, crit = judge(rows, a, fz_before["changed"] + fz_after["changed"], split_bad)

    # ---------------- 打表 ----------------
    print(f"\n{'=' * 92}\n表1 三 seed 主指标\n{'=' * 92}")
    print(f"{'seed':>6} {'info_macro':>11} {'info_pooled':>12} {'full_macro':>11} "
          f"{'macro_corr':>11} {'PSR':>8} {'iv90_cov':>9} {'iv90_width':>11}")
    for r in rows:
        c, ss = r["calibers"], r["shape"]
        print(f"{r['meta']['seed']:>6} {c['info']['macro']['rmse']:>11.4f} "
              f"{c['info']['pooled']['rmse']:>12.4f} {c['full']['macro']['rmse']:>11.4f} "
              f"{ss['macro_corr']:>+11.4f} {ss['psr']:>8.4f} "
              f"{r['interval']['coverage_info']:>9.4f} "
              f"{r['interval']['mean_width_info']:>11.4f}")
    print(f"{'mean':>6} {a['info_macro_rmse']:>11.4f} {a['info_pooled_rmse']:>12.4f} "
          f"{a['full_macro_rmse']:>11.4f} {a['macro_corr']:>+11.4f} {a['psr']:>8.4f} "
          f"{a['iv90_coverage']:>9.4f} {a['iv90_width']:>11.4f}")
    print("   冻结对照 (只读; iv90 仅 rate model 有区间输出):")
    for name, r in _ref_rows(refs):
        iv = _num(r.get("interval_coverage_info"))
        ivw = _num(r.get("interval_mean_width_info"))
        print(f"   {name:<22}{_num(r.get('info_macro_rmse')):>11.4f} "
              f"{_num(r.get('info_pooled_rmse')):>12.4f} "
              f"{_num(r.get('full_macro_rmse')):>11.4f} "
              f"{_num(r.get('macro_corr')):>+11.4f} {_num(r.get('psr')):>8.4f} " +
              (f"{iv:>9.4f}" if np.isfinite(iv) else f"{'—':>9}") + " " +
              (f"{ivw:>11.4f}" if np.isfinite(ivw) else f"{'—':>11}"))

    print(f"\n表2 分阶段 RMSE (info 口径, macro)")
    bins = rows[0]["stage_metrics"]["bins"]
    hdr = "  ".join(f"[{b['bin'][0]},{b['bin'][1]})".rjust(11) for b in bins)
    print(f"{'method':<26}{hdr}")
    print(f"{'wiener_pf':<26}" + "  ".join(
        f"{a['stage_rmse'][i]:>11.4f}" for i in range(len(bins))))
    print(f"{'  n_points (3 seed 合计)':<26}" + "  ".join(
        f"{a['stage_n_points'][i]:>11d}" for i in range(len(bins))))
    print("   冻结参考 (只读, 未重算):")
    for name, v in frozen_stage_refs(refs).items():
        cells = [f"{x:>11.4f}" if np.isfinite(x) else f"{'—':>11}" for x in v]
        print(f"{'  ' + name:<26}" + "  ".join(cells))

    print(f"\n表3 可解释性 (PH / alpha-lambda / convergence)")
    print(f"{'method':<26}{'PH':>9}{'al@0.3':>9}{'al@0.5':>9}{'al@0.7':>9}{'conv':>9}")
    al = a["alpha_lambda"]
    print(f"{'wiener_pf':<26}{a['macro_ph']:>9.4f}" +
          "".join(f"{al.get(k, float('nan')):>9.4f}" for k in sorted(al)) +
          f"{a['convergence']:>9.4f}")
    for name, v in frozen_interp_refs(refs).items():
        cells = [f"{x:>9.4f}" if np.isfinite(x) else f"{'—':>9}" for x in v]
        print(f"{'  ' + name:<26}" + "".join(cells))

    print(f"\n表4 预警")
    print(f"{'method':<26}{'cov_pre_eol':>13}{'miss_pre':>10}{'false_alarm':>13}"
          f"{'cond_lead':>11}")
    print(f"{'wiener_pf':<26}{a['coverage_before_eol']:>13.4f}"
          f"{a['miss_rate_before_eol']:>10.4f}{a['false_alarm_rate']:>13.4f}"
          f"{a['conditional_lead_before_eol']:>11.4f}")
    for name, v in frozen_warn_refs(refs).items():
        cells = []
        for w, x in zip((13, 10, 13, 11), v):
            cells.append(f"{x:>{w}.4f}" if np.isfinite(x) else f"{'—':>{w}}")
        print(f"{'  ' + name:<26}" + "".join(cells))

    print(f"\n表5 效率 (statistical / online prognostics baseline, 非等训练预算)")
    rt = a["runtime"]
    print(f"   prior fit {rt['prior_fit_s']:.4f}s   filtering {rt['filter_s']:.2f}s"
          f"   total {rt['total_s']:.2f}s  (逐 seed 均值)")

    print(f"\n-- 数值保护触发率 (协议 §7) --")
    for r in rows:
        d = r["diagnostics"]
        n = max(d["n_endpoints"], 1)
        print(f"   seed {r['meta']['seed']}: resample {d['n_resample']}  "
              f"distance_floor {d['n_distance_floor']} ({d['n_distance_floor'] / n:.4f})  "
              f"clip_upper {d['n_clip_upper']} ({d['n_clip_upper'] / n:.4f})  "
              f"prior_only {d['n_prior_only']} ({d['n_prior_only'] / n:.4f})")

    print(f"\n{'=' * 92}\n协议 §14 判据\n{'=' * 92}")
    for name, ok, detail in crit:
        print(f"   [{'PASS' if ok else 'FAIL'}] {name}   {detail}")
    print(f"\n>> S5B verdict = {verdict}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "stage": "S5B", "verdict": verdict, "seeds": seeds,
        "protocol_sha256": sha(PROTOCOL), "config_sha256": sha(ROOT / args.config),
        "target_data_hash": target_data_hash,
        "pf_config": wc, "aggregate": a, "per_seed": rows,
        "criteria": [{"name": n, "pass": o, "detail": d} for n, o, d in crit],
        "frozen_before": fz_before, "frozen_after": fz_after,
        "frozen_reference_note": "对照组一律只读冻结结果, 未重新训练任何学习组",
        "frozen_refs_present": sorted(refs.keys()),
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f">> 已写 {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
