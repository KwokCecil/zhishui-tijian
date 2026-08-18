# -*- coding: utf-8 -*-
"""智税体检 Streamlit 页面（规则版 + Agent 对话）。

运行：streamlit run app.py
"""

import os

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

# Streamlit secrets 兜底：页面内配置的 key 也能生效
for secret_key in ("ZHI_SHUI_LLM_API_KEY", "ZHI_SHUI_LLM_BASE_URL", "ZHI_SHUI_LLM_MODEL"):
    if not os.environ.get(secret_key):
        try:
            if secret_key in st.secrets:
                os.environ[secret_key] = str(st.secrets[secret_key])
        except Exception:
            pass  # 没有 secrets.toml 时跳过，不影响运行

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1080px;}
      h1 {font-size: 1.9rem; letter-spacing: .5px;}
      h2 {font-size: 1.25rem; margin-top: 1.6rem;}
      h3 {font-size: 1.05rem;}
      .metric-row {display: flex; gap: .8rem; flex-wrap: wrap; margin: .4rem 0 1rem;}
      .metric-card {flex: 1; min-width: 150px; background: #f8fafc;
                    border: 1px solid #e2e8f0; border-radius: 12px; padding: 12px 16px;}
      .metric-label {font-size: .8rem; color: #64748b;}
      .metric-value {font-size: 1.65rem; font-weight: 700; margin-top: 2px; line-height: 1.2;}
      .metric-sub {font-size: .75rem; color: #94a3b8; margin-top: 2px;}
      .lvl-low {color: #15803d;} .lvl-mid {color: #b45309;} .lvl-high {color: #b91c1c;}
      .status-badge {display: inline-block; padding: 1px 9px; border-radius: 999px;
                     font-size: .76rem; font-weight: 600; margin-right: 6px;}
      .badge-ok {background: #dcfce7; color: #15803d;}
      .badge-warn {background: #fef3c7; color: #b45309;}
      .badge-no {background: #fee2e2; color: #b91c1c;}
      .scenario-box {background: #f0f7ff; border: 1px solid #dbeafe; border-radius: 12px;
                     padding: 12px 16px; margin: .4rem 0 1rem; font-size: .92rem;}
      .scenario-box b {color: #1d4ed8;}
      .policy-row {border: 1px solid #e2e8f0; border-radius: 10px; padding: 10px 14px; margin-bottom: 8px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🩺 智税体检（演示版）")
st.caption("模拟“金税四期视角”的企业税务健康检查：上传/生成模拟数据 → 风险画像 + 政策机会 + 行动建议")

SCENARIO_GUIDE = {
    "clean": {
        "企业画像": "长沙软件公司：收入3000万、40人、信用A级，数据规范。",
        "风险种子": "无（对照组）。",
        "预期结论": "低风险，0条命中，得分100。",
    },
    "risk": {
        "企业画像": "长沙软件公司：收入1200万、研发300万、个税30人/社保15人、信用B级。",
        "风险种子": "税负率低、进项全餐饮、个人账户收付款、上游走逃、顶额开票、红冲30%、小微临界、集群注册、资格-优惠不匹配。",
        "预期结论": "高风险，约12条命中（税负、三流、人资、发票异常为主）。",
    },
    "fuel": {
        "企业画像": "湘潭县加油站：申报销量100吨，设备160吨、测算150吨，区域参考240吨。",
        "风险种子": "三源不一致、横向偏离、单价偏低、液位仪缺失12%、无票采购45%、旧标准芯片、上游单据与本地记录不一致。",
        "预期结论": "高风险（数据可信度与少申报疑点）。",
    },
    "case1": {
        "企业画像": "光伏产业链设备企业（案例一）：收入1.2亿、应税所得292.7万、研发加计600万、280人/资产4800万。",
        "风险种子": "小微临界（三项均位于限额90%-100%）、临界点聚集、收入与应税所得严重不匹配。",
        "预期结论": "高风险预警链（R17+R19+R35 组合命中）。",
    },
    "case6": {
        "企业画像": "莲子加工企业（案例六）：收购发票对象含“贩子”、单户超500万、付款与开票对象不一致。",
        "风险种子": "收购对象身份存疑、单户金额异常、业务流不匹配。",
        "预期结论": "高风险（收购发票“自己开、自己抵”的对象身份是命门）。",
    },
}


def load_from_upload(uploaded_files):
    data = {}
    for key in ("company_profile", "invoices", "fund_flows", "contracts"):
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
        ("营业收入(万元)", "营业收入"),
        ("个税申报人数", "个税人数"),
        ("社保参保人数", "社保人数"),
        ("应纳税所得额(万元)", "应税所得"),
        ("纳税信用等级", "信用等级"),
        ("关联户数", "关联户数"),
    ]
    items = []
    for col, label in fields:
        v = p.get(col)
        if v is not None and str(v).strip() not in ("", "nan", "None"):
            items.append((label, str(v)))
    return items


def status_badge(status):
    cls = {"可享受": "badge-ok", "需人工确认": "badge-warn", "不适用": "badge-no"}.get(status, "badge-no")
    return f'<span class="status-badge {cls}">{status}</span>'


def level_class(level):
    return {"低风险": "lvl-low", "中风险": "lvl-mid", "高风险": "lvl-high"}.get(level, "lvl-mid")


with st.sidebar:
    st.header("数据源")
    mode = st.radio("选择方式", ["内置演示场景", "上传 CSV"], index=0)
    data = None
    scenario = None
    if mode == "内置演示场景":
        scenario = st.selectbox("场景", list(generate_data.SCENARIOS.keys()))
        st.caption({
            "clean": "对照组 · 低风险",
            "risk": "软件企业 · 混合风险",
            "fuel": "加油站 · 三源比对",
            "case1": "案例一 · 小微临界预警链",
            "case6": "案例六 · 农产品收购发票",
        }[scenario])
        if st.button("生成并体检", type="primary"):
            data = generate_data.build_scenario(scenario)
    else:
        up_profile = st.file_uploader("企业指标 CSV", type="csv")
        up_invoices = st.file_uploader("发票明细 CSV", type="csv")
        up_funds = st.file_uploader("资金流水 CSV", type="csv")
        up_contracts = st.file_uploader("合同 CSV", type="csv")
        if st.button("开始体检", type="primary"):
            data = load_from_upload({
                "company_profile": up_profile,
                "invoices": up_invoices,
                "fund_flows": up_funds,
                "contracts": up_contracts,
            })

st.info(
    "⚠️ 演示口径：金税四期具体模型与阈值不公开，本工具使用公开可见的风险逻辑 + 演示阈值，"
    "定位为企业自查工具；所有数据为模拟数据。生产环境需按地区、行业、主管税务机关口径校准。"
)

if mode == "内置演示场景" and scenario:
    g = SCENARIO_GUIDE[scenario]
    st.markdown(
        f"""
        <div class="scenario-box">
          <b>场景：{scenario}</b>（{ {
            "clean": "对照组", "risk": "混合风险", "fuel": "行业模板",
            "case1": "实习案例一", "case6": "实习案例六",
        }[scenario] }）<br>
          企业画像：{g["企业画像"]}<br>
          风险种子：{g["风险种子"]}<br>
          预期结论：{g["预期结论"]}
        </div>
        """,
        unsafe_allow_html=True,
    )

if data is None:
    st.markdown(
        "### 使用说明\n\n"
        "1. 左侧选择**内置演示场景**（推荐先跑 `risk`）或上传四类 CSV；\n"
        "2. 系统运行 28 条风险规则 → 风险评分 → 政策卡片匹配；\n"
        "3. 输出：体检得分 + 等级 + 命中特征 + 监管视角 Top3 + 政策机会 + 行动建议。\n\n"
        "判定全部由规则完成（可溯源），LLM 只负责报告与对话（下方 Agent 对话可演示工具调用链路）。"
    )
    st.stop()

if "company_profile" not in data or data["company_profile"] is None:
    st.error("缺少企业指标数据，无法体检。")
    st.stop()

profile = data["company_profile"]
invoices = data.get("invoices", pd.DataFrame())
fund_flows = data.get("fund_flows", pd.DataFrame())
contracts = data.get("contracts", pd.DataFrame())

hits = rules.run_all(profile, invoices, fund_flows, contracts)
summary = scoring.risk_summary(hits)
matched = policies.match_cards(profile)

# ---- 体检对象概况 ----
items = profile_summary(profile)
if items:
    st.markdown("**体检对象**：" + " ｜ ".join(f"{k} {v}" for k, v in items))

# ---- 关键结论横幅 ----
if hits:
    top = summary["top3"][0]
    st.error(
        f"⚠️ 关键结论：命中 {summary['hit_count']} 条风险特征，"
        f"最高关注 **{top['rule_id']} {top['name']}**（{top['level']}级）；"
        f"等级判定：**{summary['level']}**。"
    )
else:
    st.success("✅ 未命中风险特征，等级：低风险。仍建议保持申报资料完整、关注政策更新。")

# ---- 指标卡 ----
lvl_cls = level_class(summary["level"])
usable_count = sum(1 for m in matched if m["status"] != "不适用")
st.markdown(
    f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="metric-label">体检得分</div>
        <div class="metric-value">{summary['score']}<span style="font-size:1rem;color:#94a3b8"> /100</span></div>
        <div class="metric-sub">健康分（命中加权后）</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">风险等级</div>
        <div class="metric-value {lvl_cls}">{summary['level']}</div>
        <div class="metric-sub">高危特征直接升级等级</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">命中特征</div>
        <div class="metric-value">{summary['hit_count']} <span style="font-size:1rem;color:#94a3b8">条</span></div>
        <div class="metric-sub">高 {summary['by_level'].get('高',0)} / 中 {summary['by_level'].get('中',0)} / 低 {summary['by_level'].get('低',0)}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">可关注政策</div>
        <div class="metric-value">{usable_count} <span style="font-size:1rem;color:#94a3b8">条</span></div>
        <div class="metric-sub">可享受 / 需人工确认</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- 监管视角 Top3 ----
st.subheader("监管视角 Top3")
if summary["top3"]:
    for i, t in enumerate(summary["top3"], 1):
        st.markdown(f"{i}. **{t['rule_id']} [{t['level']}] {t['name']}**：{t['evidence']}")
else:
    st.markdown("- 未命中明显风险特征；建议保持申报资料完整，关注政策更新。")

# ---- 命中特征明细 ----
st.subheader("命中特征明细")
if hits:
    hits_df = pd.DataFrame([{
        "特征ID": h["rule_id"],
        "特征名称": h["name"],
        "级别": h["level"],
        "命中证据": h["evidence"],
        "行动建议": h["suggestion"],
    } for h in hits])
    st.dataframe(hits_df, width="stretch", hide_index=True)
else:
    st.success("未命中风险特征。")

# ---- 优惠政策清单 ----
st.subheader("优惠政策清单")
usable = [m for m in matched if m["status"] != "不适用"]
not_usable = [m for m in matched if m["status"] == "不适用"]
if usable:
    for m in usable:
        st.markdown(
            f'<div class="policy-row">{status_badge(m["status"])}'
            f'<b>{m["title"]}</b>　<code>{m["doc_number"]}</code><br>'
            f'<span style="font-size:.88rem;color:#475569">{m["benefit"]}</span><br>'
            f'<span style="font-size:.8rem;color:#94a3b8">判定：{m["detail"]}</span></div>',
            unsafe_allow_html=True,
        )
else:
    st.info("未匹配到可关注政策。")
if not_usable:
    with st.expander(f"不适用政策（{len(not_usable)} 条）"):
        for m in not_usable:
            st.markdown(f"- {m['title']}（{m['doc_number']}）：{m['detail']}")

# ---- 政策符合性逐项判定 ----
st.subheader("政策符合性判定（逐项）")
for m in matched:
    with st.expander(f"{m['policy_id']} {m['title']}　{status_badge(m['status'])}"):
        st.markdown(f"**文号**：{m['doc_number']}")
        st.markdown(f"**优惠内容**：{m['benefit']}")
        if m.get("conditions"):
            for c in m["conditions"]:
                if c["pass"] is True:
                    mark = "✅"
                elif c["pass"] is False:
                    mark = "❌"
                else:
                    mark = "❓ 数据缺失/需人工确认"
                st.markdown(f"{mark} **{c['field']}**：当前 {c['value']}")
        else:
            st.markdown("该政策需人工核对资格材料。")

# ---- 行动建议 ----
st.subheader("行动建议")
grouped = [
    ("🔴 高风险应对", "高"),
    ("🟠 需关注事项", "中"),
    ("🟡 日常提示", "低"),
]
any_advice = False
for title, lvl in grouped:
    items = []
    seen = set()
    for h in hits:
        if h["level"] == lvl:
            for s in h["suggestion"].split("；"):
                s = s.strip()
                if s and s not in seen:
                    seen.add(s)
                    items.append(s)
    if items:
        any_advice = True
        st.markdown(f"**{title}**")
        for i, s in enumerate(items, 1):
            st.markdown(f"{i}. {s}")
if not any_advice:
    st.success("数据表现正常：保持申报资料完整，关注政策更新即可。")

# ---- AI 报告 ----
st.subheader("AI 体检报告")
if st.button("生成 AI 体检报告", type="secondary"):
    report = tools.generate_report(hits, summary, matched)
    st.markdown(report["report"])
elif llm.available():
    st.caption("已配置大模型 API，将调用 LLM 生成报告；未配置则使用离线模板（同样可用）。")
else:
    st.caption("未配置大模型 API key，当前使用离线模板。配置方式见 README（环境变量 ZHI_SHUI_LLM_API_KEY）。")

# ---- Agent 对话 ----
st.subheader("Agent 对话（演示）")
if mode == "内置演示场景":
    agent_data = {
        "scenario": scenario,
        "profile": None,
        "invoices": None,
        "fund_flows": None,
        "contracts": None,
    }
else:
    agent_data = {
        "scenario": None,
        "profile": profile.iloc[0].to_dict(),
        "invoices": tools._to_records(invoices) if not invoices.empty else [],
        "fund_flows": tools._to_records(fund_flows) if not fund_flows.empty else [],
        "contracts": tools._to_records(contracts) if not contracts.empty else [],
    }
question = st.text_input(
    "问它（离线模式支持：风险/体检、优惠/政策、小微）：",
    placeholder="这家公司有什么风险？",
)
if st.button("发送给 Agent", type="primary"):
    if not question.strip():
        st.warning("先输入一个问题。")
    elif agent_data["scenario"] is None and agent_data["profile"] is None:
        st.warning("请先在左侧生成或上传数据，或切到'内置演示场景'。")
    else:
        with st.spinner("Agent 正在调用工具…"):
            try:
                result = agent.run_agent(
                    question.strip(),
                    scenario=agent_data["scenario"] or "risk",
                    profile=agent_data["profile"],
                    invoices=agent_data["invoices"],
                    fund_flows=agent_data["fund_flows"],
                    contracts=agent_data["contracts"],
                )
            except Exception as exc:  # noqa: BLE001
                st.warning(f"在线 Agent 调用失败（{exc}），已自动切换为离线演示模式。")
                result = agent._run_offline_agent(
                    question.strip(),
                    scenario=agent_data["scenario"] or "risk",
                    profile=agent_data["profile"],
                    invoices=agent_data["invoices"],
                    fund_flows=agent_data["fund_flows"],
                    contracts=agent_data["contracts"],
                )
        if result["trace"]:
            with st.expander(f"工具调用轨迹（{len(result['trace'])} 次，模式：{result['mode']}）"):
                for t in result["trace"]:
                    st.markdown(f"**→ {t['tool']}**")
                    st.code(t.get("arguments", "{}"), language="json")
                    st.markdown("结果：")
                    st.json(t["result"])
        st.markdown("### Agent 回答")
        st.markdown(result["answer"])
        if result["mode"] == "offline":
            st.caption("当前为离线演示模式（无 API key）。配置 key 后，同一问题将由 LLM 自主决定调用哪些工具。")
