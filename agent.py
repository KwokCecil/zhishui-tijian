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
    "需要列表时用短横线开头，段落之间不要留空行；\n"
    "15. 对话回答必须通过 submit_answer 工具提交：conclusion 一句话结论，facts 事实发现"
    "（每条以规则编号开头，如 'R104 进销项不匹配'；事实内容由系统按规则库填充，你只给编号），"
    "actions 企业自查动作，need_info 只放确需用户补充的数据项（没有就空数组）；\n"
    "16. 不要重复：这是多轮对话，前几轮已经给过的事实、结论、建议不要再讲一遍，也不要换个说法重述；\n"
    "17. 追问按需裁剪：用户问'下一步/最优先/最重要/怎么办'时，conclusion 直接给优先级判断，"
    "actions 只给 3 条以内当务之急，facts 最多 2 条且只放本轮新增的支撑；只有首次概览才给 5 条以内事实；\n"
    "18. 不要在自由文本里重复结论，也不要写任何服务性引导或反问"
    "（如'如需完整报告可告知''我可整理成文档''需要我继续吗'）；"
    "完整报告由页面'体检报告'区域提供，无需在回答里说明；\n"
    "19. 不做法律后果定性：禁止出现'刑事''犯罪''违法定性''罚款倍数''补税金额'等表述与处罚测算；"
    "后果一律按规则库给定的演示口径复述，处置只写企业自查动作（核对、留证、补正、咨询）。"
)

TOOL_NAME_LABELS = {
    "check_small_micro": "小微资格判定",
    "match_policy_cards": "优惠政策匹配",
    "run_tax_health_check": "风险体检",
    "load_scenario": "数据加载",
    "get_demo_scenario": "场景数据",
    "generate_report": "报告生成",
    "answer_policy_question": "政策问答",
    "submit_answer": "结构化回答提交",
}

# 结构化回答提交工具：模型调用它即结束本轮对话，避免自由文本尾部出现客套话/反问
SUBMIT_TOOL = "submit_answer"

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
    s = health["summary"]
    short = (
        f"报告已生成：风险指数 {s['score']}/100，等级 {s['level']}，命中 {s['hit_count']} 条特征。"
        "完整报告见上方'体检报告'区域，可展开预览或下载 Markdown 文档。"
    )
    return {
        "answer": short,
        "trace": trace,
        "mode": report.get("mode", "offline"),
        "kind": "report",
        "report_payload": report,
    }


def _sanitize_answer(text):
    """硬兜底：把答案里泄露的工具名替换成自然语言，禁止内部机制出现在用户面前。"""
    for name, label in TOOL_NAME_LABELS.items():
        text = re.sub(rf"\b{name}\b", label, text)
    return text


SENTENCE_BOUND = "。！？；!?;\n"
SERVICE_INVITE = (
    "如需", "如要", "若需", "如果需要", "需要我", "要我", "我可以", "可为你",
    "要不要", "是否要", "是否需要", "随时", "欢迎", "可否", "能否",
)
SERVICE_ACTION = (
    "告知", "告诉", "联系", "咨询", "继续", "追问", "整理", "生成", "导出",
    "下载", "查看", "了解", "说明", "展开", "补充", "提供", "重跑", "再来", "试算",
)
SERVICE_PAYLOAD = re.compile(r"\d|R\d{3}|G\d{3}|P\d{3}|〔")
DATA_OBJECTS = ("发票", "资金", "流水", "合同", "明细", "数据", "凭证", "报表", "申报表", "资料")


def _is_service_tail(sentence):
    """判断一个句子是不是纯服务性引导（邀请/客套），而非业务内容。

    三个条件同时成立才算：有邀请词、有请求动作、且不含任何事实载荷。
    宁可漏删也不误删——带编号、金额或数据请求的句子一律保留。
    """
    t = str(sentence).strip().strip(SENTENCE_BOUND).strip()
    if not t or len(t) > 40:
        return False
    if not (any(k in t for k in SERVICE_INVITE) and any(k in t for k in SERVICE_ACTION)):
        return False
    if SERVICE_PAYLOAD.search(t):
        return False
    if any(k in t for k in DATA_OBJECTS) and any(k in t for k in ("提供", "补充", "上传")):
        return False
    return True


def _strip_service_tail(text, max_sentences=2):
    """按句子裁剪结尾的服务性引导，最多删 max_sentences 句。

    两条硬约束：只做减法（不注入任何替代文案）；不把非空回答清成空。
    报告指引改由页面常驻的"体检报告"区域承担，不再由清洗器补写。
    """
    s = str(text).rstrip()
    core = s.rstrip(SENTENCE_BOUND).rstrip()
    cut, removed = len(core), 0
    while removed < max_sentences:
        seg = core[:cut].rstrip(SENTENCE_BOUND).rstrip()
        if not seg:
            break
        start = max(seg.rfind(c) for c in SENTENCE_BOUND) + 1
        if start <= 0:  # 只剩一句，不再删
            break
        if not _is_service_tail(seg[start:]):
            break
        cut, removed = start, removed + 1
    if removed == 0:
        return s
    trimmed = core[:cut].strip()
    return trimmed or s


def _clean_answer(text):
    """剥掉 Markdown 标记、压缩空行与行尾空格，输出纯文本。"""
    text = _sanitize_answer(text)
    text = (text.replace("\r\n", "\n").replace("\r", "\n")
            .replace("\u2028", "\n").replace("\u2029", "\n").replace("\x85", "\n"))
    text = text.replace("&#x20;", "").replace("&nbsp;", " ")
    text = text.replace("**", "").replace("__", "").replace("`", "")
    text = re.sub(r"(?m)^#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^(\s*)[*•]\s+", r"\1- ", text)
    text = re.sub(r"(?m)^\s*([-*_]\s*){3,}$", "", text)
    text = re.sub(r"[ \t\u3000]+\n", "\n", text)
    text = re.sub(r"\n[ \t\u3000]+", "\n", text)
    lines = [ln for ln in text.split("\n") if ln.strip(" \t\u3000")]
    text = "\n".join(lines)
    text = re.sub(r"\n{2,}", "\n", text)
    return _strip_service_tail(text.strip())


def _as_list(value):
    """把模型给的字段规整成字符串列表（兼容字符串、null、列表）。"""
    if value is None:
        return []
    raw = value if isinstance(value, (list, tuple)) else [value]
    items = []
    for item in raw:
        t = _sanitize_answer(str(item)).strip().lstrip("-•* ").strip()
        if t:
            items.append(t)
    return items


def _hits_from_trace(trace):
    """从工具轨迹里取出命中规则，供 facts 按规则库重建。"""
    for t in reversed(trace or []):
        if t.get("tool") == "run_tax_health_check" and isinstance(t.get("result"), dict):
            hits = t["result"].get("hits")
            if isinstance(hits, list):
                return {h.get("rule_id"): h for h in hits if h.get("hit")}
    return {}


RULE_ID_AT_START = re.compile(r"^([RG]\d{3})")


def _build_fact(text, hits_by_id, already_said_ids):
    """facts 以规则库为准，返回渲染文本；None 表示本条应当丢弃。

    带编号：用规则库 evidence 重建（模型写不出建议，也编不出编号）；
    无编号：非规则类事实（如政策条件），原样保留。
    """
    m = RULE_ID_AT_START.match(text)
    if not m:
        return text
    rule_id = m.group(1)
    if rule_id in already_said_ids:
        return None  # 上一轮已经讲过，本轮不复述
    hit = hits_by_id.get(rule_id)
    if not hit:
        return None  # 规则库查无此编号：丢弃，避免编造
    return f"{rule_id} {hit.get('name', '')}（{hit.get('level', '')}）：{hit.get('evidence', '')}"


def _render_submitted(submitted, hits_by_id=None, already_said=""):
    """把结构化提交渲染成回答：结论 → 事实发现 → 下一步 → 需要你补充。

    渲染完全由本函数决定，模型没有自由文本的落点，因此不存在尾部客套话。
    事实按规则库重建，建议永远进不了事实层；需要用户补充数据走 need_info 字段。
    """
    if not isinstance(submitted, dict):
        return ""
    conclusion = _sanitize_answer(str(submitted.get("conclusion") or "")).strip()
    if not conclusion:
        return ""
    hits_by_id = hits_by_id or {}
    already_said_ids = set(re.findall(r"[RG]\d{3}", str(already_said)))
    lines = [conclusion]
    facts = []
    for raw in _as_list(submitted.get("facts")):
        fact = _build_fact(raw, hits_by_id, already_said_ids) if hits_by_id else raw
        if fact and fact not in facts:
            facts.append(fact)
    if facts:
        lines.append("事实发现：")
        lines.extend(f"- {f}" for f in facts)
    actions = _as_list(submitted.get("actions"))
    if actions:
        lines.append("下一步：")
        lines.extend(f"- {a}" for a in actions)
    need = _as_list(submitted.get("need_info"))
    if need:
        lines.append("需要你补充：" + "、".join(need))
    return "\n".join(lines).strip()


def run_agent(user_message, scenario="risk", profile=None, invoices=None, fund_flows=None, contracts=None,
              external_docs=None, data_sources=None, history=None):
    """返回 {"answer", "trace", "mode"}。history 为 [(role, content)]，用于多轮上下文。"""
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
                              external_docs, data_sources, history)
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


def _run_llm_agent(user_message, scenario, profile, invoices, fund_flows, contracts, external_docs,
                   data_sources, history=None):
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
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # 多轮上下文：不带上一轮，模型就会把已经讲过的风险点再讲一遍
    for role, content in (history or [])[-4:]:
        if role in ("user", "assistant") and str(content).strip():
            messages.append({"role": role, "content": str(content)[:800]})
    messages.append({"role": "user", "content": f"{context_text}\n\n用户问题：{user_message}"})
    answer, trace, submitted = llm.chat_with_tools(
        messages, tools.tool_schemas(), stop_tool=SUBMIT_TOOL
    )
    # 结构化提交必须由真实工具结果支撑（排除提交工具自身），否则退回规则引擎
    tool_ok = any(t.get("ok") and t.get("tool") != SUBMIT_TOOL for t in trace)
    already_said = ""
    for role, content in reversed(history or []):
        if role == "assistant":
            already_said = str(content)
            break
    rendered = _render_submitted(submitted, _hits_from_trace(trace), already_said) if tool_ok else ""
    if rendered:
        return {"answer": rendered, "trace": trace, "mode": "llm", "structured": submitted}
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
        offline["answer"] = offline["answer"] + "\n在线模式未拿到有效体检结果，以上为规则引擎结果。"
        return offline
    cleaned = _clean_answer(answer)
    if not cleaned:
        offline = _run_offline_agent(
            user_message, scenario, profile, invoices, fund_flows, contracts,
            external_docs, data_sources,
        )
        offline["answer"] = offline["answer"] + "\n在线模式未返回有效内容，以上为规则引擎结果。"
        return offline
    return {"answer": cleaned, "trace": trace, "mode": "llm"}


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
        # 计数口径与页面/报告一致：组合规则算一条，构成项不重复计数
        counted = [
            h for h in r["hits"]
            if h.get("hit") and not h.get("merged_into") and not h.get("informational")
        ]
        merged_count = len([h for h in r["hits"] if h.get("merged_into")])
        active = sorted(
            (h for h in r["hits"] if h.get("hit")),
            key=lambda h: level_order.get(h.get("level", "提示"), 9),
        )
        if active:
            head = f"下一步行动建议（按优先级，共命中 {len(counted)} 条特征"
            if merged_count:
                head += f"，另有 {merged_count} 条构成项并入组合规则、不重复计数"
            lines = [head + "）："]
            for h in active[:3]:
                lines.append(f"- [{h['level']}] {h['rule_id']} {h['name']}：{h['suggestion']}（证据：{h['evidence']}）")
            if len(active) > 3:
                lines.append("其余明细见体检报告。")
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

    return {"answer": _clean_answer(answer), "trace": trace, "mode": "offline"}
