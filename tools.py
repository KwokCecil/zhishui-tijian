# -*- coding: utf-8 -*-
"""Agent 工具层：把规则引擎/政策/评分/演示数据包装成 function calling 标准工具。

每个工具 = {name, description, parameters, required, execute}
输入输出全部 JSON 可序列化。判断由工具内的规则完成，LLM 只负责调度与表达。
"""

import json
import re

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


def _external_docs_df(rows):
    return pd.DataFrame([{c: r.get(c, "") for c in generate_data.EXTERNAL_DOC_COLS} for r in rows],
                        columns=generate_data.EXTERNAL_DOC_COLS)


def _data_sources_df(rows):
    return pd.DataFrame([{c: r.get(c, "") for c in generate_data.DATA_SOURCE_COLS} for r in rows],
                        columns=generate_data.DATA_SOURCE_COLS)


def _to_records(df):
    return json.loads(df.to_json(orient="records", force_ascii=False))


TOOLS = []
DATA_CONTEXT = {}


def set_data(handle, data):
    DATA_CONTEXT[handle] = data


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
    "对企业进行税务健康检查。内置演示场景只需传 scenario=场景名（先调 load_scenario），"
    "上传数据场景传 scenario='upload'，不要手工复制完整数据。"
    "运行 33 条原子规则 + 5 条组合规则，返回命中特征清单、风险指数、风险等级和监管视角 Top3。",
    {
        "type": "object",
        "properties": {
            "profile": {
                "type": "object",
                "description": "企业指标，可选；与 scenario 二选一",
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
            "external_docs": {
                "type": "array",
                "items": {"type": "object"},
                "description": "外部单据列表，字段：单据号/供应商/品名/数量/单位/金额(元)/日期/来源方",
            },
            "data_sources": {
                "type": "array",
                "items": {"type": "object"},
                "description": "数据源清单，字段：数据源/来源方/可否篡改/版本/留存月数/可信度",
            },
            "scenario": {
                "type": "string",
                "enum": ["clean", "risk", "fuel", "solar", "lotus", "upload"],
                "description": "已加载的场景句柄；内置场景先调 load_scenario",
            },
        },
    },
    [],
)
def run_tax_health_check(profile=None, invoices=None, fund_flows=None, contracts=None,
                         external_docs=None, data_sources=None, scenario=None):
    if scenario:
        raw = DATA_CONTEXT.get(scenario)
        if raw is None:
            raw = generate_data.build_scenario(scenario)
        profile = raw["company_profile"]
        invoices = raw["invoices"]
        fund_flows = raw["fund_flows"]
        contracts = raw["contracts"]
        external_docs = raw["external_docs"]
        data_sources = raw["data_sources"]
    if profile is None:
        return {"error": "缺少 profile 或 scenario"}
    if not isinstance(invoices, pd.DataFrame):
        invoices = invoices or []
    if not isinstance(fund_flows, pd.DataFrame):
        fund_flows = fund_flows or []
    if not isinstance(contracts, pd.DataFrame):
        contracts = contracts or []
    if not isinstance(external_docs, pd.DataFrame):
        external_docs = external_docs or []
    if not isinstance(data_sources, pd.DataFrame):
        data_sources = data_sources or []
    if isinstance(profile, dict):
        profile = _profile_df(profile)
    hits = rules.run_all(
        profile,
        invoices if isinstance(invoices, pd.DataFrame) else _invoices_df(invoices),
        fund_flows if isinstance(fund_flows, pd.DataFrame) else _funds_df(fund_flows),
        contracts if isinstance(contracts, pd.DataFrame) else _contracts_df(contracts),
        external_docs if isinstance(external_docs, pd.DataFrame) else _external_docs_df(external_docs),
        data_sources if isinstance(data_sources, pd.DataFrame) else _data_sources_df(data_sources),
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
    "load_scenario",
    "加载内置演示场景，返回企业概况与数据摘要，并把数据注册到场景句柄。"
    "之后调用 run_tax_health_check 时传 scenario=场景名即可，不要复制完整数据。",
    {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "enum": ["clean", "risk", "fuel", "solar", "lotus"],
                "description": "场景名",
            },
        },
    },
    ["name"],
)
def load_scenario(name):
    if name not in generate_data.SCENARIOS:
        return {"error": f"未知场景 {name}，可选 {list(generate_data.SCENARIOS)}"}
    data = generate_data.build_scenario(name)
    set_data(name, data)
    p = data["company_profile"].iloc[0].to_dict()
    return {
        "scenario": name,
        "企业概况": {
            k: p.get(k) for k in (
                "企业ID", "行业", "纳税人类型", "营业收入(万元)",
                "个税申报人数", "应纳税所得额(万元)", "纳税信用等级",
            )
        },
        "row_counts": {k: len(v) for k, v in data.items()},
        "note": "数据已就绪，请用 run_tax_health_check(scenario=场景名) 体检，不要再传完整数据",
    }


@tool(
    "get_demo_scenario",
    "获取内置演示场景的四类模拟数据（JSON）。场景：clean 全正常、risk 混合风险、"
    "fuel 加油站模板、solar 小微临界预警链、lotus 农产品收购发票。返回数据可直接传给 run_tax_health_check。",
    {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "enum": ["clean", "risk", "fuel", "solar", "lotus"],
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
    set_data(name, data)
    return {
        "scenario": name,
        "company_profile": data["company_profile"].iloc[0].to_dict(),
        "invoices": _to_records(data["invoices"]),
        "fund_flows": _to_records(data["fund_flows"]),
        "contracts": _to_records(data["contracts"]),
        "external_docs": _to_records(data["external_docs"]),
        "data_sources": _to_records(data["data_sources"]),
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
    lines.append("\n## 优惠政策\n")
    usable = [m for m in matched if m["status"] in ("可享受", "需人工确认")]
    if usable:
        for m in usable:
            lines.append(f"- {m['title']}（{m['status']}）：{m['doc_number']}")
    else:
        lines.append("- 未匹配到可关注政策。")
    if hits:
        lines.append("\n## 行动建议\n")
        seen = set()
        idx = 0
        for h in sorted(hits, key=lambda x: {"高": 0, "中": 1, "低": 2, "提示": 3}.get(x["level"], 9)):
            for s in split_advice(h["suggestion"]):
                if s and s not in seen:
                    seen.add(s)
                    idx += 1
                    lines.append(f"{idx}. [{h['rule_id']}] {s}")
    lines.append("\n> 演示口径：判定由规则完成；LLM 仅做报告表达。生产环境需校准阈值。")
    return "\n".join(lines)


def split_advice(suggestion):
    """把建议文本按分号切分为独立行动项；括号内的分号不算分隔符。"""
    parts, buf, depth = [], "", 0
    for ch in str(suggestion):
        if ch in "（(":
            depth += 1
        elif ch in "）)":
            depth = max(0, depth - 1)
        if ch == "；" and depth == 0:
            parts.append(buf)
            buf = ""
        else:
            buf += ch
    parts.append(buf)
    return [p.strip() for p in parts if p.strip()]


REPORT_SYSTEM_PROMPT = (
    "你是税务数字化产品专家，负责把结构化体检结果写成面向企业财务人员的专业自查报告。\n"
    "排版规范：\n"
    "1. 输出标准 Markdown，结构严格按给定模板，标题层级不超过三级；\n"
    "2. 结论先行，每节先给结论再给明细；能用表格就用表格；\n"
    "3. 少用括号，不堆砌解释；句子短、信息密度高；\n"
    "4. 行动建议从企业自查视角写（核对、留证、补正、咨询），不要出现监管办案话术；\n"
    "5. 只整理和解释结构化结果，禁止新增数据、政策或判断；每条结论保留特征ID/文号；\n"
    "6. 全文 700-1000 字，专业、平实、可读。"
)

REPORT_TEMPLATE = """# 智税体检报告

## 一、结论
（一句话：风险指数、等级、命中条数、最高关注特征）

## 二、风险画像
（表格：指数 / 等级 / 高危·中危·低危命中数 / 可关注政策数）

## 三、命中特征
（表格：编号 / 名称 / 级别 / 证据；组合规则单列并注明构成。此节只写事实发现，建议统一放第五节）

## 四、政策机会
（可享受、需人工确认分别列出，带文号与下一步）

## 五、行动建议
（按优先级编号列表：后果量化与自查动作、核验清单、纠正路径；企业自查视角）

## 六、免责声明
（演示口径：阈值与权重为演示值，生产环境需校准；数据均为模拟）"""


@tool(
    "submit_answer",
    "提交最终的结构化回答。对话回答必须通过本工具提交，不要用自由文本作答。"
    "conclusion 一句话结论；facts 事实发现，每条以规则编号开头（如 'R104'），"
    "内容由系统按规则库填充、不要自己写证据与建议；actions 企业自查动作；"
    "need_info 只放确需用户补充的数据项。多轮对话中不要重复前几轮已给过的内容。",
    {
        "type": "object",
        "properties": {
            "conclusion": {
                "type": "string",
                "description": "一句话结论：风险指数、等级或关键判断",
            },
            "facts": {
                "type": "array",
                "items": {"type": "string"},
                "description": "事实发现，每条以规则编号开头（系统会按规则库填充证据）；"
                               "不建议自己展开证据与后果测算；非规则类事实（如政策条件）可直接描述",
            },
            "actions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "下一步动作，企业自查视角（核对、留证、补正、咨询）",
            },
            "need_info": {
                "type": "array",
                "items": {"type": "string"},
                "description": "确需用户补充的数据项，如 ['发票明细']；没有则空数组",
            },
        },
        "required": ["conclusion"],
    },
    ["conclusion"],
)
def submit_answer(conclusion, facts=None, actions=None, need_info=None):
    return {
        "conclusion": conclusion,
        "facts": facts or [],
        "actions": actions or [],
        "need_info": need_info or [],
    }


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
    error = ""
    try:
        text = complete(
            "请严格按以下模板生成报告，模板中的注释（括号内说明）不要出现在正文：\n\n"
            f"{REPORT_TEMPLATE}\n\n"
            f"结构化结果：\n{json.dumps(payload, ensure_ascii=False, indent=2)}",
            system=REPORT_SYSTEM_PROMPT,
        )
        if text and text.strip():
            return {"report": text.strip(), "mode": "llm"}
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
    return {"report": _default_report(hits, summary, matched), "mode": "offline", "error": error}


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
    if not isinstance(arguments, dict):
        arguments = _parse_arguments(arguments)
    return TOOL_MAP[name]["execute"](**arguments)


def _parse_arguments(arguments):
    """容错解析模型生成的 JSON 参数：剥代码围栏、截取大括号区间、清理尾逗号。"""
    text = str(arguments).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        cand = text[start:end + 1]
        cand = re.sub(r",\s*([}\]])", r"\1", cand)
        try:
            return json.loads(cand)
        except Exception:
            pass
    raise ValueError(f"参数 JSON 解析失败: {text[:120]}")
