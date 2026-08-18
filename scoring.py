# -*- coding: utf-8 -*-
"""风险指数（演示版，0-100，越高越危险）。

分数 = 命中规则点数之和，封顶 100。
点数：高危 60 分/条、中危 20 分/条、低危/提示 5 分/条。
等级由指数直接推导：≥60 高风险；20-59 中风险；<20 低风险。
组合升级由规则层完成（G 系列组合规则自带级别与点数），评分层不做任何加成。
口径均为演示值，报告和页面需标注。
"""

RISK_POINTS = {"高": 60, "中": 20, "低": 5, "提示": 5}
LEVEL_RANK = {"高": 3, "中": 2, "低": 1, "提示": 1}


def risk_index(hits):
    """返回风险指数：未并入其他规则的特征点数之和，封顶 100。"""
    counted = [h for h in hits if not h.get("merged_into")]
    return min(100, sum(RISK_POINTS.get(h.get("level", "低"), 5) for h in counted))


def risk_score(hits):
    """兼容接口：返回 (风险指数, 等级)。"""
    total = risk_index(hits)
    return total, risk_level(hits)


def risk_level(hits):
    """等级由风险指数直接推导，与指数完全一致。"""
    total = risk_index(hits)
    if total >= 60:
        return "高风险"
    if total >= 20:
        return "中风险"
    return "低风险"


def level_reason(hits):
    """指数构成说明（用于页面展示）。"""
    return f"命中点数合计 {risk_index(hits)}/100"


def level_reason_short(hits):
    """等级卡片下方的一句话原因，不带编号与括号。"""
    counted = [h for h in hits if not h.get("merged_into")]
    high = [h for h in counted if h.get("level") == "高"]
    mid = [h for h in counted if h.get("level") == "中"]
    if any(h.get("kind") == "combo" and h.get("level") == "高" for h in counted):
        return "组合预警链触发"
    if len(high) >= 2:
        return "多条高危特征"
    if high and mid:
        return "高危 + 中危"
    if high:
        return "高危特征"
    if len(mid) >= 2:
        return "多条中危特征"
    if mid:
        return "中危特征"
    return "无明显风险"


def score_breakdown(hits):
    """指数明细：每条计入规则的点数与最终指数。"""
    rows = []
    for h in hits:
        if h.get("merged_into"):
            continue
        level = h.get("level", "低")
        rows.append({
            "rule_id": h.get("rule_id", ""),
            "name": h.get("name", ""),
            "level": level,
            "points": RISK_POINTS.get(level, 3),
        })
    return {"rows": rows, "total": risk_index(hits)}


def top_risks(hits, n=3):
    """监管视角 Top3：级别高优先，同级别按权重降序。"""
    ranked = sorted(
        hits,
        key=lambda h: (LEVEL_RANK.get(h.get("level", "低"), 1), RISK_POINTS.get(h.get("level", "低"), 3)),
        reverse=True,
    )
    return ranked[:n]


def risk_summary(hits):
    """汇总报告所需的统计信息。"""
    score, level = risk_score(hits)
    counted = [h for h in hits if not h.get("merged_into")]
    by_level = {k: 0 for k in ("高", "中", "低", "提示")}
    for h in counted:
        by_level[h.get("level", "低")] = by_level.get(h.get("level", "低"), 0) + 1
    return {
        "score": score,
        "level": level,
        "hit_count": len(counted),
        "by_level": by_level,
        "top3": top_risks(counted),
        "level_reason": level_reason(hits),
        "breakdown": score_breakdown(hits),
    }
