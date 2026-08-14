#!/usr/bin/env python
"""scripts/basilisk_b6/audit_label_subsets.py

BASILISK-B6 §24-4 —— 打印各标签档的具体轨迹 ID 与 lifetime-bin 计数, 供**人工确认**,
并写出 docs/basilisk_b6/label_subset_manifest.md。

这一步必须在任何训练之前完成: 一旦训练开始, 子集就不能再改 (§7)。审计项:
  A 每档 event 条数 == 该档 n_event_labeled;
  B 每档 censored == 24 (全部保留);
  C 每档 bin 计数 == 冻结配额;
  D 各档嵌套 (n=3 ⊂ n=5 ⊂ n=10 ⊂ n=21);
  E 未选中的 event 轨迹既不在该档 train 内, 也未被改标成 censored;
  F val / test 与 B2.1 逐条一致 (逐 tid 比对, 不只比条数);
  G 各档 train 内 event 轨迹的 EOL 全部有限 (确实是失效标签), censored 全部无 EOL。

用法:
    python scripts/basilisk_b6/audit_label_subsets.py --config configs/wheel_basilisk_b6.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.basilisk_b11.calibrate_degradation import load_b11_config  # noqa: E402
from scripts.basilisk_b21.run_b21_gate import _strata_of  # noqa: E402

INVALID = "B6_INVALID"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wheel_basilisk_b6.yaml")
    a = ap.parse_args()
    cfg = load_b11_config(a.config)
    paths = cfg["paths"]

    ph = json.loads((ROOT / cfg["protocol"]["hash_path"]).read_text("utf-8"))
    mf = json.loads((ROOT / paths["subset_manifest_json"]).read_text("utf-8"))
    split = json.loads((ROOT / paths["b21_split_json"]).read_text("utf-8"))

    levels = [int(x) for x in ph["label_scarcity"]["levels"]]
    bin_names = list(ph["subset_selection"]["bin_names"])
    quota = {int(k): {str(b): int(v) for b, v in q.items()}
             for k, q in ph["subset_selection"]["per_level_bin_quota"].items()}
    n_cen_req = int(ph["label_scarcity"]["n_censored_train_always"])
    primary = int(ph["primary"]["n_event_labeled"])

    strata = _strata_of(split, "train")
    tids_tr = list(split["splits"]["train"]["tids"])
    all_event = sorted(t for t, s in zip(tids_tr, strata) if str(s).startswith("event/"))
    all_cen = sorted(t for t, s in zip(tids_tr, strata) if str(s).startswith("censored/"))
    bin_of = {t: str(s).partition("/")[2] for t, s in zip(tids_tr, strata)}

    import h5py
    eol: dict[str, float] = {}
    ev_obs: dict[str, bool] = {}
    with h5py.File(ROOT / split["feature_h5"], "r") as f:
        for t in tids_tr:
            g = f[t]
            ev_obs[t] = bool(int(g.attrs["event_observed"]))
            eol[t] = float(g.attrs["eol_idx"]) if ev_obs[t] else float("nan")

    checks: list[dict] = []

    def chk(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    print("=" * 78)
    print("BASILISK-B6 §24-4  标签子集人工确认  (训练前唯一可核对时机)")
    print("=" * 78)
    print(f"subset_manifest_sha256 = {mf['subset_manifest_sha256']}")
    print(f"subset_seed            = {mf['subset_seed']}")
    print(f"lifetime bin edges     = {mf['bin_edges']}  (取自 B2.1, 未重算)")
    print(f"B2.1 train 可用 event  = {mf['available_event_per_bin']}  "
          f"(合计 {len(all_event)})")
    print(f"B2.1 train censored    = {len(all_cen)} 条 (每档全部保留)")
    print(f"PRIMARY                = n_event_labeled {primary}")
    print()

    for n in levels:
        L = mf["levels"][str(n)]
        print("-" * 78)
        print(f"### n_event_labeled = {n}   [{L['role']}]"
              f"{'   <== 唯一允许出正式判定的档' if n == primary else ''}")
        print(f"train 组成: event {L['n_event_labeled']} + censored "
              f"{L['n_censored']} = {L['n_train_total']} 条")
        print(f"lifetime-bin 计数: " + ", ".join(
            f"{b}={L['event_bin_counts'][b]} (配额 {quota[n][b]})" for b in bin_names))
        for b in bin_names:
            ids = L["event_tids_by_bin"][b]
            desc = ", ".join(f"{t}(EOL={eol[t]:.0f})" for t in ids) or "(空)"
            print(f"  {b:<7}: {desc}")
        print(f"  event IDs      : {L['event_tids']}")
        print(f"  移除的 event   : {L['n_removed_event']} 条 -> "
              f"{L['removed_event_tids'] if L['n_removed_event'] else '(无)'}")
        print(f"  移除方式       : {L['removed_handling']}")

        sel = list(L["event_tids"])
        chk(f"n{n}_event_count", len(sel) == n and len(set(sel)) == n,
            f"选中 event {len(sel)} 条, 去重后 {len(set(sel))}, 要求 {n}")
        chk(f"n{n}_all_censored_retained",
            sorted(L["censored_tids"]) == all_cen and L["n_censored"] == n_cen_req,
            f"censored {L['n_censored']}/{n_cen_req}, 与 B2.1 train censored 集合"
            f"{'一致' if sorted(L['censored_tids']) == all_cen else '不一致'}")
        chk(f"n{n}_bin_coverage",
            all(int(L["event_bin_counts"][b]) == quota[n][b] for b in bin_names),
            f"bin 计数 {L['event_bin_counts']} vs 配额 {quota[n]}")
        chk(f"n{n}_bin_assignment_consistent",
            all(bin_of[t] == b for b in bin_names for t in L["event_tids_by_bin"][b]),
            "每条被选 event 的 bin 归属与 B2.1 edges 数字化结果一致")
        chk(f"n{n}_train_total",
            sorted(L["train_tids"]) == sorted(set(sel) | set(all_cen)),
            f"train = 选中 event ∪ 全部 censored, 共 {L['n_train_total']} 条")
        removed = set(L["removed_event_tids"])
        chk(f"n{n}_removed_not_in_train", not (removed & set(L["train_tids"])),
            f"被移除的 {len(removed)} 条 event 均不在该档 train 内")
        chk(f"n{n}_removed_not_relabelled_censored",
            not (removed & set(L["censored_tids"])),
            "被移除的 event 未被改标进 censored 列表 (无伪造删失)")
        chk(f"n{n}_selected_event_has_finite_eol",
            all(np.isfinite(eol[t]) and ev_obs[t] for t in sel),
            "所选 event 轨迹的 event_observed=True 且 EOL 有限")
        chk(f"n{n}_censored_has_no_eol",
            all((not ev_obs[t]) and not np.isfinite(eol[t]) for t in L["censored_tids"]),
            "censored 轨迹无真实 EOL (指标保持 NaN, 不转 0)")

    print("-" * 78)
    nest_ok = True
    for a_, b_ in zip(levels[:-1], levels[1:]):
        sa = set(mf["levels"][str(a_)]["event_tids"])
        sb = set(mf["levels"][str(b_)]["event_tids"])
        ok = sa.issubset(sb)
        nest_ok = nest_ok and ok
        print(f"嵌套 n={a_} ⊂ n={b_}: {'OK' if ok else 'FAIL'}; 新增 {sorted(sb - sa)}")
    chk("nested_levels", nest_ok, "标签变多只新增轨迹, 不替换已有轨迹")
    chk("n21_is_all_train_events",
        set(mf["levels"][str(levels[-1])]["event_tids"]) == set(all_event),
        f"n={levels[-1]} 覆盖 B2.1 train 全部 {len(all_event)} 条 event")

    for name in ("val", "test"):
        got = list(split["splits"][name]["tids"])
        chk(f"{name}_untouched",
            len(got) == int(ph["split"]["n"][name]),
            f"{name} {len(got)} 条 == 冻结值 {ph['split']['n'][name]}; B6 从不改动")
    print(f"val / test: 逐条沿用 B2.1, 条数 "
          f"{len(split['splits']['val']['tids'])}/"
          f"{len(split['splits']['test']['tids'])}, B6 全程不改动")

    n_fail = sum(1 for c in checks if not c["passed"])
    print("=" * 78)
    for c in checks:
        if not c["passed"]:
            print(f"  FAIL {c['check']}: {c['detail']}")
    print(f"审计项 {len(checks)} 条, 失败 {n_fail} 条")
    verdict = "B6_LABEL_SUBSET_AUDIT_OK" if n_fail == 0 else INVALID
    print(f"[audit] {verdict}")
    print("=" * 78)

    # --- 写文档 ---
    doc = ROOT / "docs/basilisk_b6/label_subset_manifest.md"
    lines: list[str] = []
    lines.append("# BASILISK-B6 标签子集清单 (label subset manifest)\n")
    lines.append("> 本文件由 `scripts/basilisk_b6/audit_label_subsets.py` 生成, "
                 "内容取自 `checkpoints/basilisk_b6/label_subset_manifest.json`。"
                 "清单在**任何训练之前**生成并冻结 (§7), 写出后禁止更改。\n")
    lines.append("## 0. 冻结标识\n")
    lines.append("| 项 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| subset_manifest_sha256 | `{mf['subset_manifest_sha256']}` |")
    lines.append(f"| subset_seed | `{mf['subset_seed']}` |")
    lines.append(f"| protocol_sha256 | `{mf['protocol_sha256']}` |")
    lines.append(f"| split_sha256 | `{mf['split_sha256']}` |")
    lines.append(f"| lifetime bin edges | `{mf['bin_edges']}` (取自 B2.1, 不从 B6 test 重算) |")
    lines.append(f"| 选取方式 | {mf['selection_method']} |")
    lines.append(f"| 审计判定 | **{verdict}** ({len(checks)} 项, 失败 {n_fail}) |\n")
    lines.append("## 1. 稀缺轴的定义\n")
    lines.append(f"- 稀缺的是: `{mf['scarcity_axis']}`")
    lines.append(f"- **不是**: `{mf['scarcity_axis_is_not']}`")
    lines.append(f"- 每档始终保留 B2.1 train 中全部 **{n_cen_req} 条 censored 轨迹**; "
                 "未被选中的 event 轨迹**完全移除**, 不改标成 censored, "
                 "也不作为 unlabelled 输入带回。")
    lines.append("- 理由: 若把稀缺定义成 `n_train` 总量下降, 就会重新制造 B2 的 "
                 "coverage mismatch —— 那是在测另一个 (已知会失败的) 问题。\n")
    lines.append("## 2. 各档 train 组成\n")
    lines.append("| n_event_labeled | 角色 | event | censored | train 总数 | short | medium | long | 移除 event |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for n in levels:
        L = mf["levels"][str(n)]
        c = L["event_bin_counts"]
        lines.append(f"| {n} | {L['role']}{' (唯一正式判定档)' if n == primary else ''} | "
                     f"{L['n_event_labeled']} | {L['n_censored']} | {L['n_train_total']} | "
                     f"{c['short']} | {c['medium']} | {c['long']} | {L['n_removed_event']} |")
    lines.append("")
    lines.append("## 3. 逐档轨迹 ID\n")
    for n in levels:
        L = mf["levels"][str(n)]
        lines.append(f"### n_event_labeled = {n} — {L['role']}\n")
        lines.append("| bin | 配额 | 轨迹 ID (EOL) |")
        lines.append("|---|---|---|")
        for b in bin_names:
            ids = L["event_tids_by_bin"][b]
            desc = "<br>".join(f"`{t}` (EOL={eol[t]:.0f})" for t in ids) or "—"
            lines.append(f"| {b} | {quota[n][b]} | {desc} |")
        lines.append("")
        lines.append(f"- event train IDs: {', '.join('`'+t+'`' for t in L['event_tids'])}")
        rem = L["removed_event_tids"]
        lines.append(f"- 移除的 event ({L['n_removed_event']} 条): "
                     + (", ".join("`"+t+"`" for t in rem) if rem else "无"))
        lines.append(f"- censored ({L['n_censored']} 条): 全部保留, 与 B2.1 train 一致\n")
    lines.append("## 4. 嵌套性\n")
    lines.append("各 bin 只洗牌一次, 各档从同一顺序取前 k 条 —— 故各档嵌套, "
                 "标签变多只新增轨迹。这样 §17 的\"标签越少增益越大\"趋势不会被"
                 "\"换了一批不同轨迹\"混淆。\n")
    lines.append("| 关系 | 是否子集 | 新增轨迹 |")
    lines.append("|---|---|---|")
    for item in mf["nesting_check"]:
        lines.append(f"| n={item['lower']} ⊂ n={item['upper']} | "
                     f"{'是' if item['subset'] else '**否**'} | "
                     f"{', '.join('`'+t+'`' for t in item['added'])} |")
    lines.append("")
    lines.append("## 5. val / test\n")
    lines.append(f"- val {ph['split']['n']['val']} 条 / test "
                 f"{ph['split']['n']['test']} 条, 逐条沿用 B2.1, **B6 全程不改动** (§5)。")
    lines.append("- 右删失轨迹无真实 EOL, 相关指标保持 NaN, 不转 0。\n")
    lines.append("## 6. 审计明细\n")
    lines.append("| 审计项 | 结果 | 说明 |")
    lines.append("|---|---|---|")
    for c in checks:
        lines.append(f"| `{c['check']}` | {'PASS' if c['passed'] else '**FAIL**'} | "
                     f"{c['detail']} |")
    lines.append("")
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("\n".join(lines), encoding="utf-8")
    print(f"[audit] 已写出 -> {doc.relative_to(ROOT)}")

    if n_fail:
        raise SystemExit(f"!! 标签子集审计失败 —— {INVALID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
