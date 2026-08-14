"""S5 报告表格提取 (从 checkpoints/s5_rate_metrics.json 读, 不重跑训练)。"""
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
d = json.load(io.open(ROOT / "checkpoints" / "s5_rate_metrics.json", encoding="utf-8"))
G = d["groups"]
A = d["aggregate"]


def bn(b):
    return f"[{b[0]:.1f},{b[1]:.1f})"


print("=== 表A 判据 (3 seeds 均值, 信息区口径) ===")
print(f"{'group':<24}{'PSR':>8}{'info_macro':>12}{'cov_pre_eol':>13}{'iv90':>8}"
      f"{'PH':>8}{'corr':>8}")
for g in G:
    a = A[g]
    iv = a.get("interval_coverage_info", {}).get("mean", float("nan"))
    print(f"{g:<24}{a['psr']['mean']:>8.4f}{a['info_macro_rmse']['mean']:>12.4f}"
          f"{a['coverage_before_eol']['mean']:>13.4f}{iv:>8.4f}"
          f"{a['macro_ph']['mean']:>8.4f}{a['macro_corr']['mean']:>8.4f}")

print("\n=== 表B 逐 seed ===")
print(f"{'group':<24}{'s62':>10}{'s63':>10}{'s64':>10}   |  PSR per seed")
for g in G:
    a = A[g]
    print(f"{g:<24}" + "".join(f"{v:>10.4f}" for v in a["info_macro_rmse"]["per_seed"])
          + "   |  " + " ".join(f"{v:.3f}" for v in a["psr"]["per_seed"]))

print("\n=== 表C 分阶段 macro RMSE (S4 表1 同格式, 3 seeds 均值) ===")
print(f"{'bin':<14}" + "".join(f"{g[:19]:>21}" for g in G))
for i in range(len(A[G[0]]["stage_bins"])):
    row = f"{bn(A[G[0]]['stage_bins'][i]['bin']):<14}"
    for g in G:
        row += f"{A[g]['stage_bins'][i]['macro_rmse']:>21.4f}"
    print(row)

print("\n=== 表D 分阶段 pred/true 统计 ===")
for g in G:
    print(f"-- {g}")
    print(f"   {'bin':<14}{'n_traj':>7}{'pred_mean':>11}{'pred_std':>10}"
          f"{'true_mean':>11}{'true_std':>10}{'PSR':>8}{'corr':>8}")
    pm, ps = [], []
    for sb in A[g]["stage_bins"]:
        pm.append(sb["pred_mean"])
        ps.append(sb["pred_std"])
        print(f"   {bn(sb['bin']):<14}{sb['n_trajectories']:>7.1f}"
              f"{sb['pred_mean']:>11.4f}{sb['pred_std']:>10.4f}"
              f"{sb['true_mean']:>11.4f}{sb['true_std']:>10.4f}"
              f"{sb['pred_std_true_std_ratio']:>8.4f}{sb['macro_corr']:>8.4f}")
    print(f"   极差: pred_mean {max(pm) - min(pm):.4f} ({min(pm):.4f}~{max(pm):.4f})"
          f"   pred_std {max(ps) - min(ps):.4f} ({min(ps):.4f}~{max(ps):.4f})")

print("\n=== 表E alpha-lambda accuracy (3 seeds 均值) ===")
lams = list(A[G[0]]["alpha_lambda"])
print(f"{'group':<24}" + "".join(f"{'lam=' + l:>12}" for l in lams))
for g in G:
    print(f"{g:<24}" + "".join(
        f"{A[g]['alpha_lambda'][l]['accuracy_mean']:>12.4f}" for l in lams))

print("\n=== 表F 预警 / 收敛 / 区间 / 速率 ===")
print(f"{'group':<24}{'lead_pre_eol':>13}{'miss_pre_eol':>13}{'n_warn':>8}"
      f"{'conv':>8}{'late_conv':>10}{'iv_width':>10}{'mu_mean':>9}{'mu@floor':>9}")
for g in G:
    a = A[g]
    def f_(k):
        return a.get(k, {}).get("mean", float("nan"))
    print(f"{g:<24}{a['conditional_lead_before_eol']['mean']:>13.4f}"
          f"{a['miss_rate_before_eol']['mean']:>13.4f}"
          f"{a['n_warned_before_eol']['mean']:>8.1f}"
          f"{a['macro_convergence']['mean']:>8.4f}"
          f"{a['macro_late_convergence']['mean']:>10.4f}"
          f"{f_('interval_mean_width_info'):>10.4f}{f_('mu_mean'):>9.4f}"
          f"{f_('frac_mu_at_floor'):>9.4f}")

print("\n=== 表G full 口径 / pooled 对照 ===")
print(f"{'group':<24}{'full_macro':>12}{'info_pooled':>13}{'pred_std':>10}{'true_std':>10}")
for g in G:
    a = A[g]
    print(f"{g:<24}{a['full_macro_rmse']['mean']:>12.4f}"
          f"{a['info_pooled_rmse']['mean']:>13.4f}"
          f"{a['pred_std']['mean']:>10.4f}{a['true_std']['mean']:>10.4f}")

print("\n=== 判词 ===")
print(f"verdict = {d['verdict']}   frozen_unchanged = {d['frozen_unchanged']}")
for t in d["judge_table"]:
    print(f"-- {t['group']}  {'PASS' if t['passed'] else 'FAIL'}")
    for k, v in t["checks"].items():
        print(f"   [{'x' if v['ok'] else ' '}] {k:<42} = {v['value']}")
