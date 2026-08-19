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
    "5. 先调用工具拿事实，再组织回答，不要凭记忆回答；\n"
    "6. 回答简短：结论先行，一般不超过150字，只给要点；详细过程不要复述，可在必要时用3-5条短列表；\n"
    "7. 不要重复调用已调用过的工具；拿到足够事实后直接回答，通常 1-3 次工具调用即可；\n"
    "8. 内置场景先用 load_scenario 加载，再传 scenario 给 run_tax_health_check，不要手工复制完整数据；"
    "上传场景直接传 scenario='upload'。\n"
    "9. 如果 run_tax_health_check 返回错误，修正参数重试；在拿到真实结果前不要下任何风险结论；\n"
    "10. 回答开头先给结论：风险指数、等级、命中条数，再列风险点；\n"
    "11. 能直接查的信息（如小微资格、可享优惠）用 check_small_micro / match_policy_cards 查完直接给出结论，"
    "不要反问用户是否需要；反问只在需要用户做选择时使用。"
)


def run_agent(user_message, scenario="risk", profile=None, invoices=None, fund_flows=None, contracts=None,
              external_docs=None, data_sources=None):
    """返回 {"answer", "trace", "mode"}。"""
    if llm.available():
        return _run_llm_agent(user_message, scenario, profile, invoices, fund_flows, contracts,
                              external_docs, data_sources)
    return _run_offline_agent(user_message, scenario, profile, invoices, fund_flows, contracts,
                              external_docs, data_sources)


def _load_context(scenario, profile, invoices, fund_flows, contracts, external_docs, data_sources):
    """优先用页面传入的数据，否则用内置场景。"""
    if profile is not None:
        return {
            "company_profile": profile,
            "invoices": invoices or [],
            "fund_flows": fund_flows or [],
            "contracts": contracts or [],
            "external_docs": external_docs or [],
            "data_sources": data_sources or [],
        }
    return tools.get_demo_scenario(scenario)


def _run_llm_agent(user_message, scenario, profile, invoices, fund_flows, contracts, external_docs, data_sources):
    if profile is not None:
        tools.set_data("upload", {
            "company_profile": profile,
            "invoices": invoices or [],
            "fund_flows": fund_flows or [],
            "contracts": contracts or [],
            "external_docs": external_docs or [],
            "data_sources": data_sources or [],
        })
        context_text = "当前为上传数据场景，句柄为 upload；请用 run_tax_health_check(scenario='upload') 体检。"
    else:
        context_text = (
            f"当前内置场景：{scenario}；请先 load_scenario('{scenario}')，"
            f"再用 run_tax_health_check(scenario='{scenario}') 体检。"
        )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{context_text}\n\n用户问题：{user_message}"},
    ]
    answer, trace = llm.chat_with_tools(messages, tools.tool_schemas())
    health_ok = any(
        t.get("ok") and t.get("tool") == "run_tax_health_check"
        and "error" not in str(t.get("result"))
        for t in trace
    )
    if not health_ok:
        offline = _run_offline_agent(
            user_message, scenario, profile, invoices, fund_flows, contracts,
            external_docs, data_sources,
        )
        offline["answer"] = offline["answer"] + "\n\n在线模式未拿到有效体检结果，以上为规则引擎结果。"
        return offline
    return {"answer": answer, "trace": trace, "mode": "llm"}


def _run_offline_agent(user_message, scenario, profile, invoices, fund_flows, contracts, external_docs, data_sources):
    """关键词路由：演示'意图→工具→事实→回答'的链路，不依赖网络。"""
    context = _load_context(scenario, profile, invoices, fund_flows, contracts, external_docs, data_sources)
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
            "external_docs": context["external_docs"],
            "data_sources": context["data_sources"],
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
