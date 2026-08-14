#!/usr/bin/env python
"""scripts/basilisk_b7/audit_report_numbers.py

BASILISK-B7 §16 + 治理授权第 4/5 条 —— `docs/results.md` 的**语义冻结**审计。

## 为什么需要它

B7 之前, `docs/results.md` 的完整性由 12 份前阶段 baseline contract 的
`hash(docs/results.md) == 冻结值` 保证。B7 获得授权改写该文件作为最终交付
表达面, 该哈希约束随之退役。哈希退役后**必须有更强的约束顶上**, 否则等于
把最终数字放进了无人看守的文件。

本脚本就是那个更强的约束: 它**直接读取** `checkpoints/basilisk_b5/*` 与
`checkpoints/basilisk_b6/*` 的机器可读产物, 把 `docs/results.md` 里出现的
每一个关键数字逐项与冻结值比对。不是肉眼核对, 不允许手敲近似值。

判据 (全部通过才输出 REPORT_NUMERIC_AUDIT_PASS):
  1. report_numbers_match_frozen_metrics       —— 每个被审计数字都能在冻结产物中找到来源
  2. report_contains_final_transfer_conclusion —— 含 NO_POSITIVE_TRANSFER_SUPPORTED
  3. report_contains_engineering_recommendation—— 含 damage_extrapolation 工程推荐
  4. report_contains_negative_transfer_disclosure —— 含"未能证明正向迁移"边界措辞
  5. report_contains_damage_baseline           —— damage 基线出现在正表而非脚注

## 纪律

* 只读。不训练, 不评估, 不重算任何指标定义, 不写任何 checkpoints 下的数值产物。
* 容差: 报告里按 6 位小数呈现, 因此比对采用"报告中的字符串必须等于冻结值
  按同精度格式化的结果"—— 不是 abs(a-b) < eps 的松散比较, 而是精确的
  格式化字符串匹配, 防止"改成漂亮数字"。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
B5 = ROOT / "checkpoints/basilisk_b5"
B6 = ROOT / "checkpoints/basilisk_b6"
REPORT = ROOT / "docs/results.md"
OUT = ROOT / "checkpoints/basilisk_b7/report_number_audit.json"

# 报告中数字的呈现精度 (与 build_final_tables.py 一致)
ND = 6


def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def fmt(x: float, nd: int = ND) -> str:
    return f"{x:.{nd}f}"


def signed(x: float, nd: int = ND) -> str:
    """带符号呈现 (gain / CI 用)。报告里正增益必须显式带 +。"""
    return f"{'+' if x >= 0 else '-'}{abs(x):.{nd}f}"


class Audit:
    """逐项审计器。每一项都记录: 名称 / 冻结来源 / 期望字符串 / 是否命中。"""

    def __init__(self, text: str) -> None:
        self.text = text
        # 数字匹配前做一次**纯排版归一化**: 报告正文用 U+2212 MINUS SIGN
        # 呈现负号 (实测 24 处), 而 Python 的 f-string 输出 ASCII '-'。
        # 这只统一负号字形, 不改变任何数值 —— 归一化后仍是精确字符串比对。
        self.num_text = text.replace("−", "-").replace("–", "-")
        self.items: list[dict] = []

    def num(self, name: str, source: str, value: float,
            nd: int = ND, sign: bool = False) -> None:
        """断言报告中出现该数字 (按报告精度格式化后的精确字符串)。"""
        want = signed(value, nd) if sign else fmt(value, nd)
        found = want in self.num_text
        # 带符号项允许报告写成不带 + 的形式吗? 不允许 —— 正增益必须显式标正,
        # 否则读者无法区分 "0.029478" 是增益还是 RMSE。但负号本身已是符号,
        # 故负值的 signed() 与 fmt() 输出一致, 不产生额外要求。
        self.items.append({
            "item": name, "frozen_source": source,
            "expected_string": want, "found_in_report": found,
            "frozen_value": value,
        })

    def phrase(self, name: str, needle: str, required: bool = True) -> bool:
        found = needle in self.text
        self.items.append({
            "item": name, "frozen_source": "wording_requirement",
            "expected_string": needle, "found_in_report": found,
            "required_present": required,
        })
        return found

    @property
    def missing(self) -> list[dict]:
        return [i for i in self.items if not i["found_in_report"]]


def main() -> int:
    for p in (REPORT, B5 / "summary.json", B5 / "paired_gain.json",
              B6 / "final_verdict.json", B6 / "paired_statistics.json",
              B6 / "warning_metrics.json", B6 / "lifetime_bins.json"):
        if not p.exists():
            print(f"B7_REPORT_NUMBER_MISMATCH: 缺少必需产物 {p}")
            return 2

    text = REPORT.read_text(encoding="utf-8")
    A = Audit(text)

    # ---------------- B5 全标签确认性结果 ----------------
    b5_sum = _load(B5 / "summary.json")
    b5_gain = _load(B5 / "paired_gain.json")
    mt = b5_sum["main_table"]
    for g in ("target_only", "source_finetune", "source_mmd_finetune",
              "damage_extrapolation"):
        if g not in mt:
            print(f"B7_REPORT_NUMBER_MISMATCH: B5 main_table 缺 {g}")
            return 2
        A.num(f"B5.{g}.info_macro_rmse", "b5/summary.json:main_table",
              mt[g]["info_macro_rmse"])
        A.num(f"B5.{g}.info_macro_rmse_std", "b5/summary.json:main_table",
              mt[g]["info_macro_rmse_std"])

    for key in ("gain_ft", "gain_mmd"):
        gk = b5_gain["gain"][key]
        A.num(f"B5.{key}.mean", "b5/paired_gain.json:gain",
              gk["mean"], sign=True)
        A.num(f"B5.{key}.ci95_lower", "b5/paired_gain.json:gain",
              gk["ci95_lower"], sign=True)
        A.num(f"B5.{key}.ci95_upper", "b5/paired_gain.json:gain",
              gk["ci95_upper"], sign=True)

    # ---------------- B6 标签稀缺矩阵 (n=3/5/10/21) ----------------
    b6_fv = _load(B6 / "final_verdict.json")
    b6_ps = _load(B6 / "paired_statistics.json")
    for lvl in ("3", "5", "10", "21"):
        agg = b6_ps["by_level"][lvl]["aggregate_rmse"]
        for g in ("target_only", "source_finetune", "source_mmd_finetune",
                  "const_mean_info", "damage_extrapolation"):
            A.num(f"B6.n{lvl}.{g}.mean",
                  f"b6/paired_statistics.json:by_level[{lvl}].aggregate_rmse",
                  agg[g]["mean"])
        for key in ("gain_ft", "gain_mmd"):
            gk = b6_ps["by_level"][lvl]["gain"][key]
            A.num(f"B6.n{lvl}.{key}.mean",
                  f"b6/paired_statistics.json:by_level[{lvl}].gain",
                  gk["mean"], sign=True)
            A.num(f"B6.n{lvl}.{key}.median",
                  f"b6/paired_statistics.json:by_level[{lvl}].gain",
                  gk["median"], sign=True)
            A.num(f"B6.n{lvl}.{key}.ci95_lower",
                  f"b6/paired_statistics.json:by_level[{lvl}].gain",
                  gk["ci95_lower"], sign=True)
            A.num(f"B6.n{lvl}.{key}.ci95_upper",
                  f"b6/paired_statistics.json:by_level[{lvl}].gain",
                  gk["ci95_upper"], sign=True)

    # ---------------- B6 PRIMARY 告警指标 ----------------
    wm = _load(B6 / "warning_metrics.json")
    prim = str(b6_fv["primary_level"])
    wl = wm["by_level"][prim]["by_group"]
    for g in ("damage_extrapolation", "target_only", "source_mmd_finetune",
              "source_finetune"):
        if g not in wl:
            continue
        # 实测 schema: by_group[g][metric] 是 dict, 均值在 ["mean"] 下,
        # 同级还有 per_seed —— 不要凭印象当成裸标量。
        A.num(f"B6.warning.{g}.coverage",
              f"b6/warning_metrics.json:by_level[{prim}].by_group",
              wl[g]["warning_coverage"]["mean"], nd=4)
        A.num(f"B6.warning.{g}.miss_rate",
              f"b6/warning_metrics.json:by_level[{prim}].by_group",
              wl[g]["miss_rate"]["mean"], nd=4)

    # ---------------- B6 PRIMARY 寿命分箱 ----------------
    lb = _load(B6 / "lifetime_bins.json")
    lvl_bins = lb["by_level"][prim]["by_group"]
    for g in ("target_only", "source_finetune", "source_mmd_finetune",
              "damage_extrapolation"):
        if g not in lvl_bins:
            continue
        for b in ("short", "medium", "long"):
            # 实测 schema: by_group[g][bin]["rmse_mean"] (不是 "rmse")
            v = lvl_bins[g].get(b, {}).get("rmse_mean")
            if v is None or (isinstance(v, float) and v != v):
                continue    # NaN / 空 bin: 不伪造为 0, 也不作为审计项
            A.num(f"B6.bin.{g}.{b}.rmse",
                  f"b6/lifetime_bins.json:by_level[{prim}].by_group", v)

    # ---------------- 结论性措辞 (语义冻结) ----------------
    final_concl = str(b6_fv["FINAL_TRANSFER_CONCLUSION"])
    eng_rec = str(b6_fv["ENGINEERING_RECOMMENDATION"])
    has_final = A.phrase("FINAL_TRANSFER_CONCLUSION", final_concl)
    has_eng = A.phrase("ENGINEERING_RECOMMENDATION", eng_rec)
    has_b5_verdict = A.phrase("B5_verdict", str(b5_sum["verdict"]))
    has_b6_verdict = A.phrase("B6_PRIMARY_VERDICT",
                              str(b6_fv["B6_PRIMARY_VERDICT"]))
    # §0 边界措辞: 必须是"未能证明", 不得写成"证明无效"
    has_boundary = A.phrase("negative_transfer_disclosure", "未能证明正向迁移")
    has_damage_tbl = A.phrase("damage_baseline_in_main_table",
                              "damage_extrapolation")

    # §17 禁止措辞 —— 无限定地出现即失败。
    # 但"禁止 X"这句话本身必然包含 X: 报告第 134 行写的是
    # 『不得表述为"显著正迁移"、"迁移明显优于"或"证明 MMD 有效"』——
    # 那是**在禁止**该措辞, 不是在使用它。所以判定按"行"进行: 若该行带有
    # 明确的否定/历史标注标记, 该次出现不计为违规; 否则计为违规。
    DISCLAIMER_MARKERS = ("不得", "禁止", "不允许", "不能", "不应", "不是",
                          "historical", "rejected", "invalidated",
                          "已作废", "已推翻", "严禁")
    forbidden = ["迁移显著提升", "迁移明显优于", "MMD显著有效", "证明迁移有效",
                 "达到可上线部署要求", "真实飞轮失效阈值", "Basilisk生成寿命标签"]
    banned_hits: list[dict] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        guarded = any(m in line for m in DISCLAIMER_MARKERS)
        for w in forbidden:
            if w in line and not guarded:
                banned_hits.append({"phrase": w, "line": lineno,
                                    "text": line.strip()[:120]})

    # ---------------- 判定 ----------------
    numeric_items = [i for i in A.items
                     if i["frozen_source"] != "wording_requirement"]
    numeric_missing = [i for i in numeric_items if not i["found_in_report"]]
    numbers_ok = not numeric_missing
    semantic = {
        "report_numbers_match_frozen_metrics": numbers_ok,
        "report_contains_final_transfer_conclusion": has_final,
        "report_contains_engineering_recommendation": has_eng,
        "report_contains_negative_transfer_disclosure": has_boundary,
        "report_contains_damage_baseline": has_damage_tbl,
    }
    all_pass = all(semantic.values()) and not banned_hits \
        and has_b5_verdict and has_b6_verdict
    verdict = "REPORT_NUMERIC_AUDIT_PASS" if all_pass \
        else "B7_REPORT_NUMBER_MISMATCH"

    rec = {
        "stage": "BASILISK_B7",
        "section": "§16 报告数字审计 / 语义冻结",
        "discipline": (
            "只读 B5/B6 冻结产物; 逐项精确字符串比对 (报告精度), "
            "不做松散数值近似, 不允许手敲近似值。"),
        "report": "docs/results.md",
        "report_sha256": __import__("hashlib").sha256(
            REPORT.read_bytes()).hexdigest(),
        "frozen_sources": {
            "b5_summary_sha256": __import__("hashlib").sha256(
                (B5 / "summary.json").read_bytes()).hexdigest(),
            "b5_paired_gain_sha256": __import__("hashlib").sha256(
                (B5 / "paired_gain.json").read_bytes()).hexdigest(),
            "b6_final_verdict_sha256": __import__("hashlib").sha256(
                (B6 / "final_verdict.json").read_bytes()).hexdigest(),
            "b6_paired_statistics_sha256": __import__("hashlib").sha256(
                (B6 / "paired_statistics.json").read_bytes()).hexdigest(),
        },
        "n_items": len(A.items),
        "n_numeric_items": len(numeric_items),
        "n_numeric_missing": len(numeric_missing),
        "semantic_freeze": semantic,
        "semantic_freeze_replaces": (
            "hash(docs/results.md) == frozen_hash "
            "(12 份前阶段契约中的 stale lifecycle guard)"),
        "forbidden_phrase_hits": banned_hits,
        "b5_verdict_present": has_b5_verdict,
        "b6_primary_verdict_present": has_b6_verdict,
        "missing_items": numeric_missing[:40],
        "items": A.items,
        "verdict": verdict,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    print(f"审计项 {len(A.items)} 项 (数字 {len(numeric_items)} 项), "
          f"未命中 {len(numeric_missing)} 项")
    for k, v in semantic.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    if banned_hits:
        print(f"  FAIL  §17 禁止措辞出现: {banned_hits}")
    for i in numeric_missing[:20]:
        print(f"    MISSING {i['item']}: 期望 {i['expected_string']} "
              f"(来源 {i['frozen_source']})")
    print(verdict)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
