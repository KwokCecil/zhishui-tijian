# -*- coding: utf-8 -*-
"""政策卡片匹配（机会筛查模块）。

卡片结构见 config/policy_cards.json。可规则条件由代码判断，
需要资质/人工材料的输出"需人工确认"，不猜测。
"""

import json
import os

import pandas as pd


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


def _eval_condition(cond, p):
    field = cond["field"]
    op = cond.get("op", "==")
    value = cond.get("value")
    raw = p.get(field)

    if field == "月销售额(万元)":
        revenue = _num(p.get("营业收入(万元)"))
        if revenue is None:
            return None
        raw = revenue / 12

    if op == "in":
        return str(raw).strip() in value
    if op == "not_in":
        return str(raw).strip() not in value
    if op in ("==", "!=", "<", "<=", ">", ">="):
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
    field = cond["field"]
    if field == "月销售额(万元)":
        revenue = _num(p.get("营业收入(万元)"))
        return f"{revenue/12:.1f}" if revenue is not None else "缺失"
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
        matched = 0
        unknown = 0
        detail = []
        cond_results = []
        for cond in conditions:
            r = _eval_condition(cond, p)
            if r is True:
                matched += 1
                detail.append(f"{cond['field']} ✓")
            elif r is False:
                detail.append(f"{cond['field']} ✗")
            else:
                unknown += 1
                detail.append(f"{cond['field']} ?（数据缺失）")
            cond_results.append({
                "field": cond["field"],
                "pass": r,
                "value": _cond_value(cond, p),
            })
        if unknown and matched + 0 == 0:
            status = "需人工确认"
        elif matched == len(conditions):
            status = "可享受" if card.get("ruleable") == "可规则" else "需人工确认"
        elif matched > 0 or unknown > 0:
            status = "需人工确认"
        else:
            status = "不适用"
        results.append({
            "policy_id": card.get("policy_id"),
            "title": card.get("title"),
            "status": status,
            "doc_number": card.get("doc_number", ""),
            "benefit": card.get("benefit", ""),
            "detail": "；".join(detail),
            "conditions": cond_results,
        })
    return results


def small_micro_status(profile):
    """小型微利企业条件逐项判定（配合 R17 及政策 P01）。"""
    p = profile.iloc[0].to_dict()
    headcount = _num(p.get("个税申报人数"))
    assets = _num(p.get("资产总额(万元)"))
    taxable = _num(p.get("应纳税所得额(万元)"))
    industry = str(p.get("行业", ""))
    negative = industry in ["烟草制造业", "住宿和餐饮业", "批发和零售业", "房地产业", "租赁和商务服务业", "娱乐业"]
    checks = [
        ("从业人数≤300", headcount is not None and headcount <= 300, f"{headcount}/300"),
        ("资产总额≤5000万", assets is not None and assets <= 5000, f"{assets}/5000万"),
        ("应纳税所得额≤300万", taxable is not None and taxable <= 300, f"{taxable}/300万"),
        ("非限制禁止行业", not negative, industry),
    ]
    return {
        "checks": [{"name": n, "pass": ok, "value": v} for n, ok, v in checks],
        "qualified": all(ok for _, ok, _ in checks),
        "missing": any(v is None for _, ok, v in checks),
    }


def policy_answer(query):
    """未收录政策问题的统一拒绝（范围受限，不猜测）。"""
    return (
        "该问题不在本 Demo 已收录的政策卡片范围内，无法给出判断。"
        "建议通过 12366 纳税服务热线或国家税务总局官网政策库查询；"
        "如需 Demo 支持，请把政策原文/文号补充进政策卡片。"
    )
