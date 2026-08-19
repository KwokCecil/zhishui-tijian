# -*- coding: utf-8 -*-
"""智税体检 Streamlit 页面。"""

import os
import re

import pandas as pd
import streamlit as st

import agent
import generate_data
import llm
import policies
import rules
import scoring
import tools

st.set_page_config(page_title="智税体检 Demo", page_icon="🩺", layout="wide")

for secret_key in ("ZHI_SHUI_LLM_API_KEY", "ZHI_SHUI_LLM_BASE_URL", "ZHI_SHUI_LLM_MODEL"):
    if not os.environ.get(secret_key):
        try:
            if secret_key in st.secrets:
                os.environ[secret_key] = str(st.secrets[secret_key])
        except Exception:
            pass

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.8rem; max-width: 1080px;}
      h1 {font-size: 1.8rem;}
      h2 {font-size: 1.25rem; margin-top: 1.2rem;}
      .metric-row {display: flex; gap: .8rem; flex-wrap: wrap; margin: .4rem 0 1rem;}
      .metric-card {flex: 1; min-width: 150px; background: #fff;
                    border: 1px solid #cbd5e1; border-radius: 12px; padding: 12px 16px;}
      .metric-label {font-size: .9rem; color: #1f2937; font-weight: 600;}
      .metric-value {font-size: 1.7rem; font-weight: 700; color: #0f172a;}
      .metric-sub {font-size: .85rem; color: #4b5563; margin-top: 2px;}
      .lvl-low {color: #15803d;} .lvl-mid {color: #b45309;} .lvl-high {color: #b91c1c;}
      .pill {display: inline-block; padding: 1px 8px; border-radius: 999px;
             font-size: .78rem; font-weight: 700; margin-right: 6px;}
      .pill-high {background: #fee2e2; color: #b91c1c;}
      .pill-mid {background: #fef3c7; color: #b45309;}
      .pill-low {background: #dcfce7; color: #15803d;}
      .pill-ok {background: #dcfce7; color: #15803d;}
      .pill-warn {background: #fef3c7; color: #b45309;}
      .pill-no {background: #fee2e2; color: #b91c1c;}
      .row {background: #ffffff; border: 1px solid #cbd5e1; border-radius: 10px;
            padding: 10px 14px; margin-bottom: 8px; line-height: 1.6; color: #0f172a;}
      .muted {font-size: .82rem; color: #6b7280; margin-left: 6px;}
      .cond {font-size: .92rem; color: #1f2937;}
      .note {font-size: .88rem; color: #7c2d12;}
      .scenario-box {background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 10px;
                     padding: 10px 14px; margin: .4rem 0 1rem; font-size: .95rem;
                     line-height: 1.7; color: #0f172a;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🩺 智税体检")
st.caption("企业税务健康检查：上传模拟数据，输出风险画像、政策机会与行动建议")

SCENARIO_GUIDE = {
    "clean": {
        "企业画像": "长沙软件公司：收入3000万、40人、信用A级。",
        "风险种子": "无，对照组。",
        "预期结论": "低风险，0条命中，指数0。",
    },
    "risk": {
        "企业画像": "长沙软件公司：收入1200万、研发300万、个税30人/社保15人、信用B级。",
        "风险种子": "税负率低、进项全餐饮、个人账户收付款、上游走逃、顶额开票、红冲30%、小微临界、集群注册。",
        "预期结论": "高风险，约12条命中，指数封顶100。",
    },
    "fuel": {
        "企业画像": "某县国道旁加油站：年收入约3600万、资产4000万、应税所得350万，申报销量340吨，设备350吨、测算480吨，区域参考440吨。",
        "风险种子": "三源不一致、单价偏低、液位仪缺失12%、无票采购45%、旧芯片、单据与本地记录不一致。",
        "预期结论": "高风险，指数封顶100。",
    },
    "solar": {
        "企业画像": "光伏产业链设备企业：收入1.2亿、约120人、资产3200万、应税所得292.7万、研发加计600万。",
        "风险种子": "应税所得逼近300万临界，收入与利润严重不匹配。",
        "预期结论": "高风险，指数60，两条中危触发预警。",
    },
    "lotus": {
        "企业画像": "莲子加工企业：收购发票含贩子对象、单户超500万、付款与开票对象不一致。",
        "风险种子": "收购对象身份存疑、单户金额异常、业务流不匹配。",
        "预期结论": "高风险，指数60，特殊凭证对象联动触发。",
    },
}


def load_from_upload(uploaded_files):
    data = {}
    for key in ("company_profile", "invoices", "fund_flows", "contracts", "external_docs", "data_sources"):
        f = uploaded_files.get(key)
        if f is not None:
            data[key] = pd.read_csv(f, encoding="utf-8-sig")
    return data


def profile_summary(profile):
    p = profile.iloc[0].to_dict()
    fields = [
        ("企业ID", "企业ID"),
        ("行业", "行业"),
        ("纳税人类型", "纳税人类型"),
        ("营业收入(万元)", "收入"),
        ("个税申报人数", "人数"),
        ("应纳税所得额(万元)", "应税所得"),
        ("纳税信用等级", "信用"),
    ]
    items = []
    for col, label in fields:
        v = p.get(col)
        if v is not None and str(v).strip() not in ("", "nan", "None"):
            items.append(f"{label} {v}")
    return " ｜ ".join(items)


def level_pill(level):
    cls = {"高": "pill-high", "中": "pill-mid", "低": "pill-low", "提示": "pill-low"}.get(level, "pill-mid")
    return f'<span class="pill {cls}">{level}</span>'


def status_pill(status):
    cls = {
        "可享受": "pill-ok",
        "需人工确认": "pill-warn",
        "不适用": "pill-no",
        "命中限制": "pill-no",
        "未触发限制": "pill-low",
    }.get(status, "pill-no")
    return f'<span class="pill {cls}">{status}</span>'


def de_paren(text):
    text = re.sub(r"（[^（）]{1,60}）", "", str(text))
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip(" ，；、")


def cond_text(c):
    if c["pass"] is True:
        mark = '<span style="color:#16a34a;font-weight:700">✓</span>'
    elif c["pass"] is False:
        mark = '<span style="color:#dc2626;font-weight:700">✗</span>'
    else:
        mark = '<span style="color:#d97706;font-weight:700">?</span>'
    return f"{mark} {c['field']} {c['value']}"


with st.sidebar:
    st.header("数据源")
    mode = st.radio("选择方式", ["内置演示场景", "上传 CSV"], index=0)
    for key in ("data", "scenario", "report"):
        if key not in st.session_state:
            st.session_state[key] = None
    if mode == "内置演示场景":
        keys = list(generate_data.SCENARIOS)
        default_idx = keys.index(st.session_state["scenario"]) if st.session_state["scenario"] in keys else 0
        scenario = st.selectbox("场景", keys, index=default_idx)
        st.caption({
            "clean": "对照组 · 低风险",
            "risk": "软件企业 · 混合风险",
            "fuel": "加油站 · 三源比对",
            "solar": "光伏设备 · 小微临界",
            "lotus": "莲子加工 · 收购凭证",
        }[scenario])
        if st.button("生成并体检", type="primary"):
            st.session_state["data"] = generate_data.build_scenario(scenario)
            st.session_state["scenario"] = scenario
            st.session_state["report"] = None
    else:
        up_profile = st.file_uploader("企业指标 CSV", type="csv")
        up_invoices = st.file_uploader("发票明细 CSV", type="csv")
        up_funds = st.file_uploader("资金流水 CSV", type="csv")
        up_contracts = st.file_uploader("合同 CSV", type="csv")
        up_external = st.file_uploader("外部单据 CSV（可选）", type="csv")
        up_sources = st.file_uploader("数据源清单 CSV（可选）", type="csv")
        if st.button("开始体检", type="primary"):
            st.session_state["data"] = load_from_upload({
                "company_profile": up_profile,
                "invoices": up_invoices,
                "fund_flows": up_funds,
                "contracts": up_contracts,
                "external_docs": up_external,
                "data_sources": up_sources,
            })
            st.session_state["scenario"] = None
            st.session_state["report"] = None

data = st.session_state.get("data")
if data is not None and st.session_state.get("scenario"):
    scenario = st.session_state["scenario"]

if mode == "内置演示场景" and scenario:
    g = SCENARIO_GUIDE[scenario]
    st.markdown(
        f'<div class="scenario-box"><b>场景：{scenario}</b> · { {
            "clean": "对照组", "risk": "混合风险", "fuel": "行业模板",
            "solar": "光伏设备企业", "lotus": "莲子加工企业",
        }[scenario] }<br>'
        f'企业画像：{g["企业画像"]}<br>'
        f'风险种子：{g["风险种子"]}<br>'
        f'预期结论：{g["预期结论"]}</div>',
        unsafe_allow_html=True,
    )

if data is None:
    st.markdown(
        "**使用说明**：左侧选择场景或上传 CSV；页面输出风险画像、政策机会与行动建议。"
        "规则判断可溯源，模型只负责报告与对话。"
    )
    st.stop()

if "company_profile" not in data or data["company_profile"] is None:
    st.error("缺少企业指标数据。")
    st.stop()

profile = data["company_profile"]
invoices = data.get("invoices", pd.DataFrame())
fund_flows = data.get("fund_flows", pd.DataFrame())
contracts = data.get("contracts", pd.DataFrame())

hits = rules.run_all(
    profile, invoices, fund_flows, contracts,
    data.get("external_docs"), data.get("data_sources"),
)
summary = scoring.risk_summary(hits)
matched = policies.match_cards(profile)

st.markdown(f"**体检对象**：{profile_summary(profile)}")

if hits:
    top = summary["top3"][0]
    st.error(
        f"命中 {summary['hit_count']} 条风险特征，等级 {summary['level']}，"
        f"最高关注 {top['rule_id']} {top['name']}"
    )
else:
    st.success("未命中风险特征，等级低风险")

usable_count = sum(1 for m in matched if m["status"] in ("可享受", "需人工确认"))
lvl_cls = {"低风险": "lvl-low", "中风险": "lvl-mid", "高风险": "lvl-high"}[summary["level"]]
st.markdown(
    f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="metric-label">风险指数</div>
        <div class="metric-value">{summary['score']}/100</div>
        <div class="metric-sub">越高越危险</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">风险等级</div>
        <div class="metric-value {lvl_cls}">{summary['level']}</div>
        <div class="metric-sub">{scoring.level_reason_short(hits)}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">命中特征</div>
        <div class="metric-value">{summary['hit_count']}</div>
        <div class="metric-sub">高{summary['by_level'].get('高',0)} / 中{summary['by_level'].get('中',0)} / 低{summary['by_level'].get('低',0)}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">可关注政策</div>
        <div class="metric-value">{usable_count}</div>
        <div class="metric-sub">可享受 / 需确认</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("命中特征")


def hit_sort_key(h):
    rank = {"高": 3, "中": 2, "低": 1, "提示": 1}.get(h.get("level"), 1)
    return (-rank, h.get("rule_id", ""))


combo_hits = [h for h in hits if h.get("kind") == "combo"]
standalone = [h for h in hits if not h.get("merged_into") and h.get("kind") != "combo"]
combo_hits.sort(key=hit_sort_key)
standalone.sort(key=hit_sort_key)

for h in combo_hits:
    subs = [s for s in hits if s.get("merged_into") == h["rule_id"]]
    subs.sort(key=hit_sort_key)
    hit_deps = "、".join(s["rule_id"] for s in subs)
    sub_lines = "".join(
        f'<div style="margin:6px 0 0 14px;padding-top:6px;border-top:1px dashed #e2e8f0">'
        f'{level_pill(s["level"])}<b>{s["rule_id"]} {s["name"]}</b><br>'
        f'<span class="cond">证据：{de_paren(s["evidence"])}</span></div>'
        for s in subs
    )
    st.markdown(
        f'<div class="row">{level_pill(h["level"])}<b>{h["rule_id"]} {h["name"]}</b>'
        f'<span class="muted">组合规则 · 构成 {hit_deps}</span><br>'
        f'<span class="cond">建议：{de_paren(h["suggestion"])}</span>{sub_lines}</div>',
        unsafe_allow_html=True,
    )

for h in standalone:
    st.markdown(
        f'<div class="row">{level_pill(h["level"])}<b>{h["rule_id"]} {h["name"]}</b><br>'
        f'<span class="cond">证据：{de_paren(h["evidence"])}</span><br>'
        f'<span class="cond">建议：{de_paren(h["suggestion"])}</span></div>',
        unsafe_allow_html=True,
    )

st.subheader("优惠政策")
usable = [m for m in matched if m["status"] in ("可享受", "需人工确认", "命中限制")]
not_usable = [m for m in matched if m["status"] in ("不适用", "未触发限制")]
for m in usable:
    conds = " · ".join(cond_text(c) for c in m.get("conditions", []))
    notes = []
    if m.get("exclusive_note"):
        notes.append(f'<div class="note">与 {m["exclusive_with"]} 互斥，小微 5% 更优</div>')
    if m["status"] == "需人工确认":
        notes.append(f'<div class="note">{de_paren(m["note"])}</div>')
    if m["status"] == "命中限制":
        notes.append(f'<div class="note">{m["note"]}</div>')
    st.markdown(
        f'<div class="row">{status_pill(m["status"])}<b>{m["policy_id"]} {m["title"]}</b>'
        f'<span class="muted">{m["doc_number"]}</span><br>'
        f'<span class="cond">{de_paren(m["benefit"])}</span><br>'
        f'<span class="cond">{conds}</span>{"".join(notes)}</div>',
        unsafe_allow_html=True,
    )
if not_usable:
    with st.expander(f"不适用 / 未触发限制（{len(not_usable)} 条）"):
        for m in not_usable:
            st.markdown(f"- {m['policy_id']} {m['title']} · {m['status']}")

for w in policies.linked_warnings(hits, matched):
    st.warning(f"{w['title']}：享受前提是先核实申报真实性，已命中 {'、'.join(w['rules'])}")

st.subheader("行动建议")
grouped = [
    ("高风险应对", "高"),
    ("需关注事项", "中"),
    ("日常提示", "低"),
]
any_advice = False
for title, lvl in grouped:
    items = []
    seen = set()
    for h in hits:
        if h["level"] == lvl:
            for s in h["suggestion"].split("；"):
                s = de_paren(s).strip()
                if s and s not in seen:
                    seen.add(s)
                    items.append(s)
    if items:
        any_advice = True
        st.markdown(f"**{title}**")
        for i, s in enumerate(items, 1):
            st.markdown(f"{i}. {s}")
if not any_advice:
    st.markdown("数据表现正常，保持申报资料完整即可。")

with st.expander("指数怎么算"):
    bd = summary["breakdown"]
    st.markdown(
        f"分数 = 命中规则点数之和，点数：高 60 / 中 30 / 低 10。"
        f"一条高危或两条中危即触发高风险。"
        f"本次合计 {bd['total']}，等级阈值：≥60 高风险，30-59 中风险，<30 低风险。"
    )
    if bd["rows"]:
        st.dataframe(
            pd.DataFrame([{
                "编号": r["rule_id"],
                "特征": r["name"],
                "级别": r["level"],
                "点数": r["points"],
            } for r in bd["rows"]]),
            width="stretch",
            hide_index=True,
        )

st.subheader("AI 体检报告")
if st.button("生成报告", type="secondary"):
    with st.spinner("正在生成报告…"):
        st.session_state["report"] = tools.generate_report(hits, summary, matched)
rep = st.session_state.get("report")
if rep:
    st.download_button(
        "下载 Markdown 报告",
        rep["report"],
        file_name="智税体检报告.md",
        mime="text/markdown",
    )
    with st.expander("报告预览"):
        st.markdown(rep["report"])
    if rep.get("mode") == "llm":
        st.caption("本次报告由大模型生成")
    else:
        err = f"（{rep.get('error')}）" if rep.get("error") else ""
        st.caption(f"本次报告使用离线模板{err}")
elif llm.available():
    st.caption("已配置模型 API，点击生成报告")
else:
    st.caption("未配置模型 API，将使用离线模板")

st.subheader("Agent 对话")
if mode == "内置演示场景":
    agent_data = {
        "scenario": scenario,
        "profile": None,
        "invoices": None,
        "fund_flows": None,
        "contracts": None,
        "external_docs": None,
        "data_sources": None,
    }
else:
    agent_data = {
        "scenario": None,
        "profile": profile.iloc[0].to_dict(),
        "invoices": tools._to_records(invoices) if not invoices.empty else [],
        "fund_flows": tools._to_records(fund_flows) if not fund_flows.empty else [],
        "contracts": tools._to_records(contracts) if not contracts.empty else [],
        "external_docs": tools._to_records(data["external_docs"]) if "external_docs" in data and not data["external_docs"].empty else [],
        "data_sources": tools._to_records(data["data_sources"]) if "data_sources" in data and not data["data_sources"].empty else [],
    }
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
for role, content in st.session_state["chat_history"]:
    with st.chat_message(role):
        st.markdown(content)

default_q = "这家公司有什么风险？"
question = None
prompt = st.chat_input("问它，例如：这家公司有什么风险？")
if prompt:
    question = prompt
elif not st.session_state["chat_history"] and st.button(f"发送默认问题：{default_q}", key="send_default"):
    question = default_q

if question:
    if agent_data["scenario"] is None and agent_data["profile"] is None:
        with st.chat_message("assistant"):
            st.markdown("请先在左侧生成或上传数据。")
    else:
        st.session_state["chat_history"].append(("user", question))
        with st.chat_message("user"):
            st.markdown(question)
        with st.spinner("Agent 正在调用工具…"):
            try:
                result = agent.run_agent(
                    question,
                    scenario=agent_data["scenario"] or "risk",
                    profile=agent_data["profile"],
                    invoices=agent_data["invoices"],
                    fund_flows=agent_data["fund_flows"],
                    contracts=agent_data["contracts"],
                    external_docs=agent_data["external_docs"],
                    data_sources=agent_data["data_sources"],
                )
            except Exception as exc:  # noqa: BLE001
                st.warning(f"在线模式失败：{exc}，已切换离线演示")
                result = agent._run_offline_agent(
                    question,
                    scenario=agent_data["scenario"] or "risk",
                    profile=agent_data["profile"],
                    invoices=agent_data["invoices"],
                    fund_flows=agent_data["fund_flows"],
                    contracts=agent_data["contracts"],
                    external_docs=agent_data["external_docs"],
                    data_sources=agent_data["data_sources"],
                )
        st.session_state["chat_history"].append(("assistant", result["answer"]))
        with st.chat_message("assistant"):
            st.markdown(result["answer"])
        if result["trace"]:
            with st.expander(f"工具调用轨迹 {len(result['trace'])} 次 · {result['mode']}"):
                for t in result["trace"]:
                    st.markdown(f"**→ {t['tool']}**")
                    st.json(t["result"])

st.caption("演示口径：风险阈值与权重为演示值，生产环境需校准；数据均为模拟。")
