# -*- coding: utf-8 -*-
"""Agent 对话入口：真实模式走 function calling，离线模式用关键词路由演示同一机制。"""

import json

import llm
import tools

SYSTEM_PROMPT = (
    "你是'智税体检'的税务数字化助手，服务对象是企业财务人员。\n"
    "规则：\n"
    "1. 判断必须来自工具返回的事实（run_tax_health_check / match_policy_cards / check_small_micro），"
    "禁止编造数据、规则或政策；\n"
    "2. 回答带溯源（特征ID、证据、文号）；\n"
    "3. 政策超出已收录范围时调用 answer_policy_question 或建议 12366，不要自行解答；\n"
    "4. 所有数据均为演示模拟数据，不是真实企业；\n"
    "5. 先调用工具拿事实，再组织回答，不要凭记忆回答。"
)


def run_agent(user_message, scenario="risk", profile=None, invoices=None, fund_flows=None, contracts=None):
    """返回 {"answer", "trace", "mode"}。"""
    if llm.available():
        return _run_llm_agent(user_message, scenario, profile, invoices, fund_flows, contracts)
    return _run_offline_agent(user_message, scenario, profile, invoices, fund_flows, contracts)


def _load_context(scenario, profile, invoices, fund_flows, contracts):
    """优先用页面传入的数据，否则用内置场景。"""
    if profile is not None:
        return {
            "company_profile": profile,
            "invoices": invoices or [],
            "fund_flows": fund_flows or [],
            "contracts": contracts or [],
        }
    return tools.get_demo_scenario(scenario)


def _run_llm_agent(user_message, scenario, profile, invoices, fund_flows, contracts):
    context = _load_context(scenario, profile, invoices, fund_flows, contracts)
    context_text = json.dumps({
        "当前企业数据": {
            "profile": context["company_profile"],
            "invoices": context["invoices"],
            "fund_flows": context["fund_flows"],
            "contracts": context["contracts"],
        }
    }, ensure_ascii=False)[:8000]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"当前演示数据如下（可调用 get_demo_scenario 重新获取）：\n{context_text}\n\n用户问题：{user_message}"},
    ]
    answer, trace = llm.chat_with_tools(messages, tools.tool_schemas())
    return {"answer": answer, "trace": trace, "mode": "llm"}


def _run_offline_agent(user_message, scenario, profile, invoices, fund_flows, contracts):
    """关键词路由：演示'意图→工具→事实→回答'的链路，不依赖网络。"""
    context = _load_context(scenario, profile, invoices, fund_flows, contracts)
    trace = []
    answer = ""

    if any(k in user_message for k in ("优惠", "政策", "享受", "减免")):
        r = tools.execute_tool("match_policy_cards", {"profile": context["company_profile"]})
        trace.append({"tool": "match_policy_cards", "arguments": "{}", "ok": True, "result": r})
        usable = [m for m in r if m["status"] in ("可享受", "需人工确认")]
        if usable:
            answer = "可关注的政策：\n" + "\n".join(
                f"- {m['title']}（{m['status']}，{m['doc_number']}）：{m['benefit']}" for m in usable
            )
        else:
            answer = "未匹配到可关注政策。"

    if any(k in user_message for k in ("小微", "小型微利")):
        r = tools.execute_tool("check_small_micro", {"profile": context["company_profile"]})
        trace.append({"tool": "check_small_micro", "arguments": "{}", "ok": True, "result": r})
        lines = ["小型微利企业条件逐项判定："]
        for c in r["checks"]:
            lines.append(f"- {'✅' if c['pass'] else '❌'} {c['name']}：当前 {c['value']}")
        lines.append("结论：" + ("符合" if r["qualified"] else "不符合或需补充数据"))
        answer += ("\n\n" if answer else "") + "\n".join(lines)

    if any(k in user_message for k in ("风险", "体检", "问题", "检查", "健康", "怎么样", "安全")):
        r = tools.execute_tool("run_tax_health_check", {
            "profile": context["company_profile"],
            "invoices": context["invoices"],
            "fund_flows": context["fund_flows"],
            "contracts": context["contracts"],
        })
        trace.append({"tool": "run_tax_health_check", "arguments": "{}", "ok": True, "result": r})
        s = r["summary"]
        lines = [f"风险指数 {s['score']}/100（越高越危险），等级：{s['level']}，命中 {s['hit_count']} 条特征。"]
        for t in s["top3"]:
            lines.append(f"- {t['rule_id']} [{t['level']}] {t['name']}：{t['evidence']}")
        answer += ("\n\n" if answer else "") + "\n".join(lines)

    if not answer:
        r = tools.execute_tool("generate_report", {
            "hits": [],
            "summary": {"score": 0, "level": "低风险", "hit_count": 0, "by_level": {}, "top3": []},
            "matched": [],
        })
        trace.append({"tool": "generate_report", "arguments": "{}", "ok": True, "result": r})
        answer = "我还没理解你的问题。可以问：'这家公司有什么风险？'、'能享受哪些优惠政策？'、'是否符合小微条件？'"

    return {"answer": answer, "trace": trace, "mode": "offline"}
