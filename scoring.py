# -*- coding: utf-8 -*-
"""风险评分（演示版）。

体检得分：100 - 命中加权分（高=3 / 中=2 / 低或提示=1），仅作参考量。
风险等级：由命中特征的严重程度直接判定，不只看总分——
  - 高风险：≥2 条高危特征；或 高危+中危 同时命中；或触发关键预警组合（R17+R19+R35 全中）
  - 中风险：1 条高危特征；或 ≥2 条中危特征；或 中危+提示 组合
  - 低风险：其余
权重与等级口径均为演示值，报告和页面需标注。
"""

LEVEL_WEIGHT = {"高": 3, "中": 2, "低": 1, "提示": 1}
LEVEL_RANK = {"高": 3, "中": 2, "低": 1, "提示": 1}


def risk_score(hits):
    """返回 (体检得分, 等级)。"""
    weighted = sum(LEVEL_WEIGHT.get(h.get("level", "低"), 1) for h in hits)
    score = max(0, 100 - weighted)
    return score, risk_level(hits)


def risk_level(hits):
    """按命中特征的严重程度判定等级（体检得分的补充规则）。"""
    high = [h for h in hits if h.get("level") == "高"]
    mid = [h for h in hits if h.get("level") == "中"]
    low = [h for h in hits if h.get("level") in ("低", "提示")]
    ids = {h.get("rule_id") for h in hits}
    combo_case1 = {"R17", "R19", "R35"}.issubset(ids)
    if len(high) >= 2 or (high and mid) or combo_case1:
        return "高风险"
    if high or len(mid) >= 2 or (mid and low):
        return "中风险"
    return "低风险"


def top_risks(hits, n=3):
    """监管视角 Top3：级别高优先，同级别按权重降序。"""
    ranked = sorted(
        hits,
        key=lambda h: (LEVEL_RANK.get(h.get("level", "低"), 1), LEVEL_WEIGHT.get(h.get("level", "低"), 1)),
        reverse=True,
    )
    return ranked[:n]


def risk_summary(hits):
    """汇总报告所需的统计信息。"""
    score, level = risk_score(hits)
    by_level = {k: 0 for k in ("高", "中", "低", "提示")}
    for h in hits:
        by_level[h.get("level", "低")] = by_level.get(h.get("level", "低"), 0) + 1
    return {
        "score": score,
        "level": level,
        "hit_count": len(hits),
        "by_level": by_level,
        "top3": top_risks(hits),
    }
