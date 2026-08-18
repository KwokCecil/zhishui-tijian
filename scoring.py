# -*- coding: utf-8 -*-
"""风险指数（演示版，0-100，越高越危险）。

基础分：高危 45 分/条、中危 20 分/条、低危/提示 5 分/条。
组合加成仅一条：命中 R17+R19+R35 关键预警组合 +25（两条低/中危也能构成严重风险）。
等级由指数直接推导：≥60 高风险；20-59 中风险；<20 低风险。
口径均为演示值，报告和页面需标注。
"""

RISK_POINTS = {"高": 45, "中": 20, "低": 5, "提示": 5}
LEVEL_RANK = {"高": 3, "中": 2, "低": 1, "提示": 1}


def _bonuses(hits):
    """组合加成：小微临界家族（R17 或 R19）+ 收入利润不匹配（R35）。
    R19 合并 R17 后，组合按“家族+关联特征”判定，不重复计算已并入的 R17。"""
    ids = {h.get("rule_id") for h in hits}
    small_micro = (ids & {"R17", "R19"}) or ("C01" in ids)
    if small_micro and "R35" in ids:
        return [("小微临界×收入利润不匹配组合（R17/C01 + R35）", 25)]
    return []


def risk_index(hits):
    """返回 (基础分, 组合加成列表, 风险指数)。"""
    counted = [h for h in hits if not h.get("merged_into")]
    base = sum(RISK_POINTS.get(h.get("level", "低"), 3) for h in counted)
    bonuses = _bonuses(hits)
    total = min(100, base + sum(b for _, b in bonuses))
    return base, bonuses, total


def risk_score(hits):
    """兼容接口：返回 (风险指数, 等级)。"""
    base, bonuses, total = risk_index(hits)
    return total, risk_level(hits)


def risk_level(hits):
    """等级由风险指数直接推导，与指数完全一致。"""
    total = risk_index(hits)[2]
    if total >= 60:
        return "高风险"
    if total >= 20:
        return "中风险"
    return "低风险"


def level_reason(hits):
    """指数构成说明（用于页面展示）。"""
    base, bonuses, total = risk_index(hits)
    parts = [f"基础分 {base}"]
    parts += [f"{name}(+{b})" for name, b in bonuses]
    return "，".join(parts) + f" → {total}/100"


def score_breakdown(hits):
    """指数明细：每条命中的点数、基础分、加成与最终指数。"""
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
    base, bonuses, total = risk_index(hits)
    return {
        "rows": rows,
        "base": base,
        "bonuses": bonuses,
        "total": total,
    }


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
