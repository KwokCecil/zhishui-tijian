# -*- coding: utf-8 -*-
"""Agent 工具层：把规则引擎/政策/评分/演示数据包装成 function calling 标准工具。

每个工具 = {name, description, parameters, required, execute}
输入输出全部 JSON 可序列化。判断由工具内的规则完成，LLM 只负责调度与表达。
"""

import json

import pandas as pd

import generate_data
import policies
import rules
import scoring


def _profile_df(profile):
    return pd.DataFrame([{c: profile.get(c, "") for c in generate_data.PROFILE_COLS}],
                        columns=generate_data.PROFILE_COLS)


def _invoices_df(rows):
    return pd.DataFrame([{c: r.get(c, "") for c in generate_data.INVOICE_COLS} for r in rows],
                        columns=generate_data.INVOICE_COLS)


def _funds_df(rows):
    return pd.DataFrame([{c: r.get(c, "") for c in generate_data.FUND_COLS} for r in rows],
                        columns=generate_data.FUND_COLS)


def _contracts_df(rows):
    return pd.DataFrame([{c: r.get(c, "") for c in generate_data.CONTRACT_COLS} for r in rows],
                        columns=generate_data.CONTRACT_COLS)


def _to_records(df):
    return json.loads(df.to_json(orient="records", force_ascii=False))


TOOLS = []


def tool(name, description, parameters, required):
    def deco(fn):
        TOOLS.append({
            "name": name,
            "description": description,
            "parameters": parameters,
            "required": required,
            "execute": fn,
        })
        return fn
    return deco


@tool(
    "run_tax_health_check",
    "对企业进行税务健康检查：输入企业指标、发票、资金流水、合同（均为 JSON），"
    "运行 28 条风险规则，返回命中特征清单、体检得分、风险等级和监管视角 Top3。"
    "判定完全由规则完成，结果带特征ID与证据，可溯源。",
    {
        "type": "object",
        "properties": {
            "profile": {
                "type": "object",
                "description": "企业指标，字段见项目 README 的 PROFILE_COLS",
            },
            "invoices": {
                "type": "array",
                "items": {"type": "object"},
                "description": "发票明细列表，字段：发票号码/开票日期/发票类型/品名/税率(%)/金额(元)/税额(元)/对方名称/对方税号/对方状态/关联合同号/销售方类型",
            },
            "fund_flows": {
                "type": "array",
                "items": {"type": "object"},
                "description": "资金流水列表，字段：流水日期/收付方向/对方名称/金额(元)/备注/关联合同号/账户类型",
            },
            "contracts": {
                "type": "array",
                "items": {"type": "object"},
                "description": "合同列表，字段：合同号/对方名称/金额(元)/签订日期",
            },
        },
    },
    ["profile"],
)
def run_tax_health_check(profile, invoices=None, fund_flows=None, contracts=None):
    invoices = invoices or []
    fund_flows = fund_flows or []
    contracts = contracts or []
    hits = rules.run_all(
        _profile_df(profile),
        _invoices_df(invoices),
        _funds_df(fund_flows),
        _contracts_df(contracts),
    )
    summary = scoring.risk_summary(hits)
    return {"summary": summary, "hits": hits}


@tool(
    "match_policy_cards",
    "按企业指标匹配已收录的政策卡片，返回可享受/需人工确认/不适用的优惠清单，每条带文号。"
    "政策范围仅限已收录卡片，未收录的不要自行判断。",
    {
        "type": "object",
        "properties": {
            "profile": {"type": "object", "description": "企业指标"},
        },
    },
    ["profile"],
)
def match_policy_cards(profile):
    return policies.match_cards(_profile_df(profile))


@tool(
    "check_small_micro",
    "逐项判定企业是否符合小型微利企业条件（从业人数≤300、资产总额≤5000万、"
    "应纳税所得额≤300万、非限制禁止行业），返回每项通过与数值。",
    {
        "type": "object",
        "properties": {
            "profile": {"type": "object", "description": "企业指标"},
        },
    },
    ["profile"],
)
def check_small_micro(profile):
    return policies.small_micro_status(_profile_df(profile))


@tool(
    "answer_policy_question",
    "处理未收录的政策问答。本 Demo 只支持已收录的政策卡片，超范围问题统一拒绝并引导 12366，绝不猜测。",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "用户的政策问题"},
        },
    },
    ["query"],
)
def answer_policy_question(query):
    return {"answer": policies.policy_answer(query)}


@tool(
    "get_demo_scenario",
    "获取内置演示场景的四类模拟数据（JSON）。场景：clean 全正常、risk 混合风险、"
    "fuel 加油站模板、case1 小微临界预警链、case6 农产品收购发票。返回数据可直接传给 run_tax_health_check。",
    {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "enum": ["clean", "risk", "fuel", "case1", "case6"],
                "description": "场景名",
            },
        },
    },
    ["name"],
)
def get_demo_scenario(name):
    if name not in generate_data.SCENARIOS:
        return {"error": f"未知场景 {name}，可选 {list(generate_data.SCENARIOS)}"}
    data = generate_data.build_scenario(name)
    return {
        "scenario": name,
        "company_profile": data["company_profile"].iloc[0].to_dict(),
        "invoices": _to_records(data["invoices"]),
        "fund_flows": _to_records(data["fund_flows"]),
        "contracts": _to_records(data["contracts"]),
        "row_counts": {k: len(v) for k, v in data.items()},
    }


def _default_report(hits, summary, matched):
    lines = [
        f"# 智税体检报告（离线模板）\n",
        f"**风险指数：{summary['score']} / 100（越高越危险）｜风险等级：{summary['level']}｜命中特征：{summary['hit_count']} 条**\n",
    ]
    if summary["top3"]:
        lines.append("## 监管视角 Top3\n")
        for t in summary["top3"]:
            lines.append(f"- **{t['rule_id']} [{t['level']}] {t['name']}**：{t['evidence']}")
    if hits:
        lines.append("\n## 命中特征明细\n")
        for h in hits:
            lines.append(f"- **{h['rule_id']} [{h['level']}] {h['name']}**：{h['evidence']}")
            lines.append(f"  - 建议：{h['suggestion']}")
    lines.append("\n## 优惠政策\n")
    usable = [m for m in matched if m["status"] in ("可享受", "需人工确认")]
    if usable:
        for m in usable:
            lines.append(f"- {m['title']}（{m['status']}）：{m['doc_number']}")
    else:
        lines.append("- 未匹配到可关注政策。")
    lines.append("\n> 演示口径：判定由规则完成；LLM 仅做报告表达。生产环境需校准阈值。")
    return "\n".join(lines)


@tool(
    "generate_report",
    "把风险体检结果（hits+summary）和政策匹配结果翻译成一份完整的 Markdown 体检报告。"
    "只做表达与排版，不新增任何判断。",
    {
        "type": "object",
        "properties": {
            "hits": {"type": "array", "items": {"type": "object"}, "description": "run_tax_health_check 返回的 hits"},
            "summary": {"type": "object", "description": "run_tax_health_check 返回的 summary"},
            "matched": {"type": "array", "items": {"type": "object"}, "description": "match_policy_cards 返回的列表"},
        },
    },
    ["hits", "summary"],
)
def generate_report(hits, summary, matched=None):
    matched = matched or []
    from llm import complete

    payload = {
        "hits": hits,
        "summary": summary,
        "matched": matched,
    }
    try:
        text = complete(
            "你是税务数字化产品专家。把下面的结构化体检结果翻译成一份面向企业财务人员的"
            "中文 Markdown 体检报告（风险画像→命中明细→监管视角→优惠清单→行动建议→免责声明）。"
            "只整理和解释，禁止新增规则、数据或政策内容；每条结论保留特征ID/文号。"
            f"\n\n结构化结果：\n{json.dumps(payload, ensure_ascii=False, indent=2)}",
        )
        if text and text.strip():
            return {"report": text.strip()}
    except Exception:
        pass
    return {"report": _default_report(hits, summary, matched)}


TOOL_MAP = {t["name"]: t for t in TOOLS}


def tool_schemas():
    """OpenAI 风格 function calling schema。"""
    result = []
    for t in TOOLS:
        result.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        })
    return result


def execute_tool(name, arguments):
    """执行工具。arguments 为已解析的 dict，或 JSON 字符串。"""
    if name not in TOOL_MAP:
        raise KeyError(f"未知工具: {name}")
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    return TOOL_MAP[name]["execute"](**arguments)
