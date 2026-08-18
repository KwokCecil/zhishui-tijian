# -*- coding: utf-8 -*-
"""政策卡片匹配（机会筛查模块）。

卡片结构见 config/policy_cards.json。可规则条件由代码判断，
需要资质/人工材料的输出"需人工确认"，不猜测。
"""

import json
import os

import pandas as pd

FIELD_ALIASES = {"从业人数": "个税申报人数"}
NEGATIVE_INDUSTRIES = ["烟草制造业", "住宿和餐饮业", "批发和零售业", "房地产业", "租赁和商务服务业", "娱乐业"]

# 政策 ↔ 风险规则映射：某项政策可享受时，若命中相关风险特征，
# 应提示"享受前提"（先自查、后享受）。这是通用机制，不针对具体案例。
POLICY_RISK_LINKS = {
    "P01": {
        "title": "小型微利企业低税率",
        "rules": ["R17", "R19", "C01", "R35"],
        "text": "命中与小微资格相关的风险特征（{}）。“可享受小微优惠”的前提是申报数据真实、符合小微条件；请先按备查清单自查申报真实性，确认无误后再享受。",
    },
    "P08": {
        "title": "六税两费减半征收",
        "rules": ["R17", "R19", "C01", "R35"],
        "text": "命中与小微资格相关的风险特征（{}）。六税两费减半以符合小微条件为前提，请先核实申报真实性后再确认享受。",
    },
    "P02": {
        "title": "高新技术企业减按15%",
        "rules": ["R30"],
        "text": "命中资格-优惠不匹配风险（{}）。高新优惠享受前提是高企资格真实有效；请先完成资格复核，未确认前不享受。",
    },
    "P03": {
        "title": "研发费用加计扣除100%",
        "rules": ["R20"],
        "text": "命中研发加计占比异常（{}）。享受研发加计前提是研发活动真实、费用归集合规；请先核对立项文件与辅助账。",
    },
}


def load_cards(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "policy_cards.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _small_micro_qualified(p):
    """小型微利企业四条件判定（供 P01/P08 与 small_micro_status 共用）。"""
    headcount = _num(p.get("个税申报人数"))
    assets = _num(p.get("资产总额(万元)"))
    taxable = _num(p.get("应纳税所得额(万元)"))
    industry = str(p.get("行业", ""))
    checks = [
        headcount is not None and headcount <= 300,
        assets is not None and assets <= 5000,
        taxable is not None and taxable <= 300,
        industry not in NEGATIVE_INDUSTRIES,
    ]
    return all(checks), headcount, assets, taxable, industry


def _eval_condition(cond, p):
    field = FIELD_ALIASES.get(cond["field"], cond["field"])
    op = cond.get("op", "==")
    value = cond.get("value")

    if field == "月销售额(万元)":
        revenue = _num(p.get("营业收入(万元)"))
        if revenue is None:
            return None
        raw = revenue / 12
    elif field == "小型微利企业":
        raw = "是" if _small_micro_qualified(p)[0] else "否"
    elif field == "六税两费适用":
        taxpayer = str(p.get("纳税人类型", "")).strip()
        raw = "是" if taxpayer in ("小规模纳税人", "个体工商户") or _small_micro_qualified(p)[0] else "否"
    else:
        raw = p.get(field)

    if op == "in":
        return str(raw).strip() in value
    if op == "not_in":
        return str(raw).strip() not in value
    if op in ("==", "!=", "<", "<=", ">", ">="):
        if isinstance(raw, str) and isinstance(value, str):
            if op == "==":
                return raw.strip() == value
            if op == "!=":
                return raw.strip() != value
            return None
        a = _num(raw)
        b = _num(value)
        if a is None or b is None:
            return None  # 数据缺失 → 不猜，返回"需人工确认"
        if op == "==":
            return a == b
        if op == "!=":
            return a != b
        if op == "<":
            return a < b
        if op == "<=":
            return a <= b
        if op == ">":
            return a > b
        if op == ">=":
            return a >= b
    return None


def _cond_value(cond, p):
    """条件当前值（用于展示）。"""
    field = FIELD_ALIASES.get(cond["field"], cond["field"])
    if field == "月销售额(万元)":
        revenue = _num(p.get("营业收入(万元)"))
        return f"{revenue/12:.1f}" if revenue is not None else "缺失"
    if field == "小型微利企业":
        return "是（符合演示口径）" if _small_micro_qualified(p)[0] else "否"
    if field == "六税两费适用":
        taxpayer = str(p.get("纳税人类型", "")).strip()
        if taxpayer in ("小规模纳税人", "个体工商户") or _small_micro_qualified(p)[0]:
            return "是（小规模纳税人/个体工商户，或符合小微条件）"
        return "否"
    raw = p.get(field)
    if raw is None or (isinstance(raw, str) and raw.strip() == ""):
        return "缺失"
    return str(raw)


def match_cards(profile, cards=None):
    """按企业指标匹配政策卡片。

    返回列表：{"policy_id", "title", "status", "doc_number", "benefit", "detail"}
    status: 可享受 / 需人工确认 / 不适用
    """
    cards = cards if cards is not None else load_cards()
    p = profile.iloc[0].to_dict()
    results = []
    for card in cards:
        conditions = card.get("conditions", [])
        if not conditions:
            results.append({
                "policy_id": card.get("policy_id"),
                "title": card.get("title"),
                "status": "需人工确认",
                "doc_number": card.get("doc_number", ""),
                "benefit": card.get("benefit", ""),
                "detail": "需人工核对资格/材料",
            })
            continue
        detail = []
        cond_results = []
        for cond in conditions:
            r = _eval_condition(cond, p)
            if r is True:
                detail.append(f"{cond['field']} ✓")
            elif r is False:
                detail.append(f"{cond['field']} ✗")
            else:
                detail.append(f"{cond['field']} ?（数据缺失）")
            cond_results.append({
                "field": cond["field"],
                "pass": r,
                "value": _cond_value(cond, p),
                "requirement": f"{cond.get('op')} {cond.get('value')}{cond.get('unit', '')}",
            })
        passes = [c["pass"] for c in cond_results]
        has_false = any(p is False for p in passes)
        unknown_fields = [c["field"] for c in cond_results if c["pass"] is None]
        kind = card.get("kind", "benefit")
        if kind == "restriction":
            if unknown_fields:
                status = "需人工确认"
                note = f"数据缺失：需补充【{'、'.join(unknown_fields)}】后再判定"
            elif passes and passes[0] is True:
                status = "命中限制"
                note = "行业在研发加计负面清单内，不得享受研发费用加计扣除"
            else:
                status = "未触发限制"
                note = "行业不在负面清单内，不影响享受研发费用加计扣除；本卡为限制规则，不产生优惠"
        else:
            if has_false:
                status = "不适用"
                note = "未满足条件，不享受该项"
            elif unknown_fields:
                status = "需人工确认"
                note = f"数据缺失：需补充【{'、'.join(unknown_fields)}】后再判定"
            elif all(p is True for p in passes):
                status = "可享受" if card.get("ruleable") == "可规则" else "需人工确认"
                if status == "需人工确认":
                    note = f"条件已满足，下一步：{card.get('manual_reason', '人工确认后享受')}"
                else:
                    note = "条件全部满足（演示口径），申报时以税务机关口径为准"
            else:
                status = "需人工确认"
                note = "条件部分满足，需人工确认实际情况"
        results.append({
            "policy_id": card.get("policy_id"),
            "title": card.get("title"),
            "status": status,
            "kind": kind,
            "doc_number": card.get("doc_number", ""),
            "benefit": card.get("benefit", ""),
            "detail": "；".join(detail),
            "conditions": cond_results,
            "note": note,
        })
    return _apply_exclusivity(results)


def _apply_exclusivity(results):
    """政策互斥：P01 小微低税率 与 P02 高新15% 同一所得只能享受一项，择优适用。"""
    by_id = {r["policy_id"]: r for r in results}
    p01 = by_id.get("P01")
    p02 = by_id.get("P02")
    if p01 and p02 and p01["status"] != "不适用" and p02["status"] != "不适用":
        p01["exclusive_with"] = "P02"
        p01["exclusive_note"] = (
            "与高新技术企业15%税率互斥，同一所得只能享受一项；"
            "小微实际税负5%更优，建议优先小微（前提：申报数据真实、小微条件成立）"
        )
        p02["exclusive_with"] = "P01"
        p02["exclusive_note"] = (
            "与小型微利企业5%税率互斥；若小微条件成立，小微更优，本项仅作备选"
        )
    return results


def linked_warnings(hits, matched):
    """风险-政策联动：返回应提示“享受前提”的政策列表（通用机制）。"""
    risk_ids = {h.get("rule_id") for h in hits}
    usable = {m.get("policy_id") for m in matched if m.get("status") in ("可享受", "需人工确认")}
    out = []
    for pid, link in POLICY_RISK_LINKS.items():
        if pid not in usable:
            continue
        hit_rules = sorted(risk_ids & set(link["rules"]))
        if hit_rules:
            out.append({
                "policy_id": pid,
                "title": link["title"],
                "rules": hit_rules,
                "text": link["text"].format("、".join(hit_rules)),
            })
    return out


def small_micro_status(profile):
    """小型微利企业条件逐项判定（配合 R17 及政策 P01）。"""
    p = profile.iloc[0].to_dict()
    qualified, headcount, assets, taxable, industry = _small_micro_qualified(p)
    checks = [
        ("从业人数≤300", headcount is not None and headcount <= 300, f"{headcount}/300"),
        ("资产总额≤5000万", assets is not None and assets <= 5000, f"{assets}/5000万"),
        ("应纳税所得额≤300万", taxable is not None and taxable <= 300, f"{taxable}/300万"),
        ("非限制禁止行业", industry not in NEGATIVE_INDUSTRIES, industry),
    ]
    return {
        "checks": [{"name": n, "pass": ok, "value": v} for n, ok, v in checks],
        "qualified": qualified,
        "missing": any(v is None for _, ok, v in checks),
    }


def policy_answer(query):
    """未收录政策问题的统一拒绝（范围受限，不猜测）。"""
    return (
        "该问题不在本 Demo 已收录的政策卡片范围内，无法给出判断。"
        "建议通过 12366 纳税服务热线或国家税务总局官网政策库查询；"
        "如需 Demo 支持，请把政策原文/文号补充进政策卡片。"
    )
