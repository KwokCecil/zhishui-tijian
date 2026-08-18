# -*- coding: utf-8 -*-
"""风险评分（演示版）。

规则：权重 高=3 / 中=2 / 低或提示=1；体检得分 = max(0, 100 - 命中加权分)；
等级：>=80 低风险；60-79 中风险；<60 高风险。
权重与分界均为演示口径，报告和页面需标注。
"""

LEVEL_WEIGHT = {"高": 3, "中": 2, "低": 1, "提示": 1}
LEVEL_RANK = {"高": 3, "中": 2, "低": 1, "提示": 1}


def risk_score(hits):
    """返回 (得分, 等级)。"""
    weighted = sum(LEVEL_WEIGHT.get(h.get("level", "低"), 1) for h in hits)
    score = max(0, 100 - weighted)
    if score >= 80:
        level = "低风险"
    elif score >= 60:
        level = "中风险"
    else:
        level = "高风险"
    return score, level


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
