# -*- coding: utf-8 -*-
"""Agent 对话入口：真实模式走 function calling，离线模式用关键词路由演示同一机制。"""

import json
import re

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
    "10. 涉及风险判断的回答开头先给结论：风险指数、等级、命中条数，再列风险点；"
    "纯政策/资格类问题直接给政策结论，不要先报风险指数；\n"
    "11. 能直接查的信息（如小微资格、可享优惠）用 check_small_micro / match_policy_cards 查完直接给出结论，"
    "不要反问用户是否需要；反问只在需要用户做选择时使用；\n"
    "12. 回答中禁止出现任何工具/函数名称（如 load_scenario、run_tax_health_check、check_small_micro、"
    "match_policy_cards、generate_report 等），一律用自然语言表达，例如直接说'符合小微条件、可享受小微低税率'；\n"
    "13. 与税务体检无关的问题（如天气、数学、闲聊）直接说明'这不属于税务体检范围'即可，"
    "不要重复体检结果；只有用户明确要求时再提供体检摘要；\n"
    "14. 回答使用纯文本：禁止任何 Markdown 标记（加粗星号、井号标题、反引号、分隔线），"
    "需要列表时用短横线开头，段落之间不要留空行。"
)

TOOL_NAME_LABELS = {
    "check_small_micro": "小微资格判定",
    "match_policy_cards": "优惠政策匹配",
    "run_tax_health_check": "风险体检",
    "load_scenario": "数据加载",
    "get_demo_scenario": "场景数据",
    "generate_report": "报告生成",
    "answer_policy_question": "政策问答",
}

OFFTOPIC_KEYWORDS = (
    "天气", "几点", "时间", "1+1", "数学", "你是谁", "名字",
    "在吗", "吃饭", "股票", "新闻", "唱歌", "笑话",
)
REPORT_KEYWORDS = ("报告", "生成", "出报告", "下载")


def _is_offtopic(message):
    m = message.strip().lower()
    return any(k in m for k in OFFTOPIC_KEYWORDS)


def _report_intent(message):
    return any(k in message for k in REPORT_KEYWORDS)


def _build_report(scenario, profile, invoices, fund_flows, contracts, external_docs, data_sources):
    """用户要求生成报告时直接调用报告工具，不让模型自由发挥。"""
    trace = []
    if profile is not None:
        tools.set_data("upload", {
            "company_profile": profile,
            "invoices": invoices or [],
            "fund_flows": fund_flows or [],
            "contracts": contracts or [],
            "external_docs": external_docs or [],
            "data_sources": data_sources or [],
        })
        health = tools.execute_tool("run_tax_health_check", {"scenario": "upload"})
        p = profile
    else:
        tools.execute_tool("load_scenario", {"name": scenario})
        health = tools.execute_tool("run_tax_health_check", {"scenario": scenario})
        p = tools.DATA_CONTEXT[scenario]["company_profile"].iloc[0].to_dict()
    matched = tools.execute_tool("match_policy_cards", {"profile": p})
    report = tools.generate_report(health["hits"], health["summary"], matched)
    trace.append({"tool": "run_tax_health_check", "arguments": "{}", "ok": True, "result": health})
    trace.append({"tool": "match_policy_cards", "arguments": "{}", "ok": True, "result": matched})
    return {"answer": report["report"], "trace": trace, "mode": report.get("mode", "offline")}


def _sanitize_answer(text):
    """硬兜底：把答案里泄露的工具名替换成自然语言，禁止内部机制出现在用户面前。"""
    for name, label in TOOL_NAME_LABELS.items():
        text = re.sub(rf"\b{name}\b", label, text)
    return text


def _clean_answer(text):
    """剥掉 Markdown 标记、压缩空行与行尾空格，输出纯文本。"""
    text = _sanitize_answer(text)
    text = text.replace("&#x20;", "").replace("&nbsp;", " ")
    text = text.replace("**", "").replace("__", "").replace("`", "")
    text = re.sub(r"(?m)^#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^(\s*)[*•]\s+", r"\1- ", text)
    text = re.sub(r"(?m)^\s*([-*_]\s*){3,}$", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = text.strip().split("\n")
    while lines and re.search(r"(需要我|要我|是否(要|需)|要不要)[^。\n]{0,30}吗[？?]?$", lines[-1].strip()):
        lines.pop()
        if lines and not lines[-1].strip():
            lines.pop()
    return "\n".join(lines).strip()


def run_agent(user_message, scenario="risk", profile=None, invoices=None, fund_flows=None, contracts=None,
              external_docs=None, data_sources=None):
    """返回 {"answer", "trace", "mode"}。"""
    if _is_offtopic(user_message):
        return {
            "answer": "这个问题不在税务体检范围内，请聚焦企业税务风险、政策优惠或申报建议。",
            "trace": [],
            "mode": "rule",
        }
    if _report_intent(user_message):
        return _build_report(scenario, profile, invoices, fund_flows, contracts, external_docs, data_sources)
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
    return {"answer": _clean_answer(answer), "trace": trace, "mode": "llm"}


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

    if any(k in user_message for k in ("行动", "建议", "下一步", "整改", "怎么办", "处理", "措施")):
        r = tools.execute_tool("run_tax_health_check", {
            "profile": context["company_profile"],
            "invoices": context["invoices"],
            "fund_flows": context["fund_flows"],
            "contracts": context["contracts"],
            "external_docs": context["external_docs"],
            "data_sources": context["data_sources"],
        })
        trace.append({"tool": "run_tax_health_check", "arguments": "{}", "ok": True, "result": r})
        level_order = {"高": 0, "中": 1, "低": 2, "提示": 3}
        active = sorted(
            (h for h in r["hits"] if h.get("hit")),
            key=lambda h: level_order.get(h.get("level", "提示"), 9),
        )
        if active:
            lines = [f"下一步行动建议（按优先级，共命中 {len(active)} 条）："]
            for h in active[:3]:
                lines.append(f"- [{h['level']}] {h['rule_id']} {h['name']}：{h['suggestion']}（证据：{h['evidence']}）")
            if len(active) > 3:
                lines.append(f"其余 {len(active) - 3} 条详见体检报告。")
            answer += ("\n\n" if answer else "") + "\n".join(lines)
        elif not answer:
            answer = "未命中风险特征，暂无需整改；可进一步问：'能享受哪些优惠政策？'"

    if not answer:
        r = tools.execute_tool("generate_report", {
            "hits": [],
            "summary": {"score": 0, "level": "低风险", "hit_count": 0, "by_level": {}, "top3": []},
            "matched": [],
        })
        trace.append({"tool": "generate_report", "arguments": "{}", "ok": True, "result": r})
        answer = "我还没理解你的问题。可以问：'这家公司有什么风险？'、'能享受哪些优惠政策？'、'是否符合小微条件？'"

    return {"answer": answer, "trace": trace, "mode": "offline"}
