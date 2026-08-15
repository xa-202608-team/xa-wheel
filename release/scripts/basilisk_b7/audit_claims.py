"""§17 — 语义主张审计。

跨 docs/results.md 与 docs/技术方案报告/技术方案报告.md, 检查:
  1. 禁止无限定出现的主张 → 必须带限定词或标注 historical/rejected/invalidated
  2. 必要的限定主张必须出现
  3. 结果解读与 frozen final verdict 一致

输出 JSON 审计报告 → checkpoints/basilisk_b7/claim_audit.json
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# --------------------------------------------------------------------------
# §17 禁止的无限定主张
# --------------------------------------------------------------------------
FORBIDDEN_UNQUALIFIED = {
    "迁移显著提升": "宣称迁移效果统计显著, 需限定为 exploratory / 未通过正式检验",
    "迁移明显优于": "宣称迁移客观优效, 需限定条件且不得用于 PRIMARY 结论",
    "MMD显著有效": "宣称 MMD 适应本身有效, 未被正式 protocol 支持",
    "证明迁移有效": "宣称迁移已被证明, 实际结论是未证明有效",
    "达到可上线部署要求": "宣称方法满足部署标准, 实际 miss rate ~0.6, 仅作工程基线",
    "真实飞轮失效阈值": "宣称 D≥1 是硬件失效阈值, 实际是项目定义的仿真失效状态",
    "Basilisk生成寿命标签": "宣称 Basilisk 生成 RUL 标签, 实际仅提供工况",
}

# --------------------------------------------------------------------------
# §17 免责/限定标记 —— 命中任一即可放行前述主张
# --------------------------------------------------------------------------
DISCLAIMER_MARKERS = (
    "不得", "禁止", "不允许", "不能", "不应", "不是", "未获得",
    "historical", "rejected", "invalidated", "已作废", "已推翻",
    "exploratory", "探索性", "条件性", "未通过", "未获得足够",
    "仅作", "仅供", "不作为", "不代表", "未验证", "未满足",
    "项目定义", "仿真定义", "not established", "not supported",
)

# --------------------------------------------------------------------------
# 必须出现的限定主张 (minimum disclosure)
# --------------------------------------------------------------------------
REQUIRED_CLAIMS = [
    # 正式结论
    ("no_positive_transfer", "未能证明正向迁移", "正式结论必须明确"),
    ("no_positive_transfer2", "no positive transfer was supported", "英文正式结论必须明确"),
    ("engineering_recommendation", "damage_extrapolation", "工程推荐基线必须出现"),
    # 局限性披露 — 模式允许空格灵活变化
    ("d_is_simulated", r"D\s*≥\s*1.*project-defined simulated failure state|project-defined simulated failure state.*D", "必须明确 D≥1 是仿真失效状态"),
    ("basilisk_only_profile", "Basilisk.*mission.*load.*profile", "必须明确 Basilisk 仅提供工况"),
    ("basilisk_no_labels", r"(RUL|labels).*project.defined|project.defined.*RUL|Basilisk.*not.*generate", "必须明确 Basilisk 不生成寿命标签"),
    ("deployment_no_basilisk", "Deployment does not require Basilisk at runtime", "必须明确部署不需要 Basilisk"),
    ("warning_disclaimer", "miss rate", "必须披露预警指标 miss rate 的真实值"),
    ("no_trend_line", "no trend line is fitted", "必须声明 n=3/5/10/21 不得拟合趋势"),
    ("ci_not_recomputed", "no bootstrap was re-run in B7", "必须声明 CI 未在 B7 重新计算"),
    ("fig5_not_comparison", "NOT a fair algorithmic comparison", "必须声明 fig5 非算法公平比较"),
    # 转移层原则
    ("transfer_level", r"health.index.*degradation.*level|degradation.dynamics.*level", "必须声明迁移发生在 HI/退化动力学层"),
    ("no_waveform_transfer", "never on raw waveforms", "必须声明不得在原始波形层迁移"),
]

# --------------------------------------------------------------------------
# 可接受的限定性表述
# --------------------------------------------------------------------------
ALLOWED_QUALIFIED = {
    "未获得足够正迁移证据", "探索性稳定化信号", "条件性低标签信号",
    "exploratory stabilization signal", "conditional low-label signal",
    "insufficient evidence for positive transfer",
    "工程基线", "physics baseline", "engineering baseline",
    "项目定义的累计损伤失效状态", "project-defined simulated failure state",
    "Basilisk 生成任务工况", "Basilisk supplies the mission / load profile",
}


@dataclass
class Hit:
    phrase: str
    line: int
    column: int
    context: str
    has_disclaimer: bool
    file: str


def audit_file(rel: str) -> tuple[list[Hit], str]:
    p = ROOT / rel
    if not p.exists():
        return [], f"MISSING: {rel}"
    text = p.read_text(encoding="utf-8")
    hits: list[Hit] = []

    for lineno, line in enumerate(text.splitlines(), 1):
        for phrase in FORBIDDEN_UNQUALIFIED:
            idx = line.find(phrase)
            if idx == -1:
                continue
            has_disclaimer = any(m in line for m in DISCLAIMER_MARKERS)
            if not has_disclaimer:
                hits.append(Hit(
                    phrase=phrase,
                    line=lineno,
                    column=idx + 1,
                    context=line.strip()[:120],
                    has_disclaimer=False,
                    file=rel,
                ))

    return hits, ""


def check_required_claims() -> list[tuple[str, str, str, bool, list[str]]]:
    """检查必要主张。返回 (key, pattern, description, found, found_in_files)"""
    # 搜索范围: 结果文档 + 技术方案报告 + 最终图 SVG (图注是正式披露的一部分)
    search_files = [
        "docs/results.md",
        "docs/技术方案报告/技术方案报告.md",
        "docs/figures/basilisk_final/fig1_basilisk_health_trajectory.svg",
        "docs/figures/basilisk_final/fig2_transfer_gain_vs_labels.svg",
        "docs/figures/basilisk_final/fig3_primary_method_comparison.svg",
        "docs/figures/basilisk_final/fig4_health_management_flow.svg",
        "docs/figures/basilisk_final/fig5_split_coverage_diagnostic.svg",
    ]

    all_text = ""
    file_texts = {}
    for rel in search_files:
        p = ROOT / rel
        if p.exists():
            txt = p.read_text(encoding="utf-8")
            file_texts[rel] = txt
            all_text += txt + "\n"

    result = []
    for key, pattern, description in REQUIRED_CLAIMS:
        found_in = [rel for rel, txt in file_texts.items()
                    if re.search(pattern, txt, re.IGNORECASE)]
        found = len(found_in) > 0
        result.append((key, pattern, description, found, found_in))
    return result


def main() -> int:
    print("=== §17 语义主张审计 ===")

    forbidden_hits = []
    errors = []
    for rel in ("docs/results.md", "docs/技术方案报告/技术方案报告.md"):
        hits, err = audit_file(rel)
        if err:
            errors.append(err)
        forbidden_hits.extend(hits)
        for h in hits:
            print(f"  ❌ {h.file}:{h.line}:{h.column} | {h.phrase}")
            print(f"     上下文: {h.context}")
            print(f"     说明: {FORBIDDEN_UNQUALIFIED[h.phrase]}")

    required = check_required_claims()
    missing_required = [(k, p, d, fi) for k, p, d, f, fi in required if not f]
    for k, p, d, fi in missing_required:
        print(f"  ❌ 缺少必要主张: {d} | 模式: {p}")
    for k, p, d, f, fi in required:
        if f:
            print(f"  ✅ {d} | 出现在: {', '.join(fi)}")

    verdict = "CLAIM_AUDIT_PASS"
    if forbidden_hits or missing_required or errors:
        verdict = "B7_CLAIM_AUDIT_FAIL"
        print(f"\n⚠️  未通过: {len(forbidden_hits)} 无限定主张, {len(missing_required)} 必要主张缺失")
    else:
        print("  ✅ 全部无限定主张均已正确限定")
        print("  ✅ 全部必要披露主张均已出现")

    # 写 JSON 报告
    report = {
        "verdict": verdict,
        "forbidden_unqualified_hits": [
            {"file": h.file, "line": h.line, "column": h.column,
             "phrase": h.phrase, "context": h.context,
             "explanation": FORBIDDEN_UNQUALIFIED[h.phrase]}
            for h in forbidden_hits
        ],
        "missing_required_claims": [
            {"key": k, "pattern": p, "description": d}
            for k, p, d, fi in missing_required
        ],
        "required_claims": [
            {"key": k, "pattern": p, "description": d, "found": f, "found_in": fi}
            for k, p, d, f, fi in required
        ],
        "allowed_qualified": sorted(ALLOWED_QUALIFIED),
        "disclaimer_markers": DISCLAIMER_MARKERS,
    }

    out = ROOT / "checkpoints" / "basilisk_b7" / "claim_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告写入: {out.relative_to(ROOT)}")

    return 0 if verdict == "CLAIM_AUDIT_PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
