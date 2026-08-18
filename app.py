# -*- coding: utf-8 -*-
"""智税体检 Streamlit 页面（规则版）。

运行：streamlit run app.py
"""

import os

import pandas as pd
import streamlit as st

import generate_data
import policies
import rules
import scoring

st.set_page_config(page_title="智税体检 Demo", page_icon="🩺", layout="wide")

st.title("🩺 智税体检（演示版）")
st.caption("模拟'金税四期视角'的企业税务健康检查：上传/生成模拟数据 → 风险画像 + 优惠政策清单 + 行动建议")


def load_from_upload(uploaded_files):
    data = {}
    for label, key in [
        ("企业指标 CSV", "company_profile"),
        ("发票明细 CSV", "invoices"),
        ("资金流水 CSV", "fund_flows"),
        ("合同 CSV", "contracts"),
    ]:
        f = uploaded_files.get(key)
        if f is not None:
            data[key] = pd.read_csv(f, encoding="utf-8-sig")
    return data


with st.sidebar:
    st.header("数据源")
    mode = st.radio("选择方式", ["内置演示场景", "上传 CSV"], index=0)
    data = None
    if mode == "内置演示场景":
        scenario = st.selectbox("场景", list(generate_data.SCENARIOS.keys()))
        st.caption({
            "clean": "全正常 → 低风险画像",
            "risk": "软件企业混合风险种子",
            "fuel": "加油站模板（三源比对）",
            "case1": "案例一：小微临界+研发加计预警链",
            "case6": "案例六：农产品收购发票",
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
    "⚠️ 演示口径声明：金税四期具体模型与阈值不公开，本工具使用公开可见的风险逻辑 + 演示阈值，"
    "定位为企业自查工具；所有数据为模拟数据，非真实企业。生产环境需按地区、行业、主管税务机关口径校准。"
)

if data is None:
    st.markdown(
        "### 使用说明\n\n"
        "1. 左侧选择**内置演示场景**（推荐先跑 `risk`）或上传四类 CSV；\n"
        "2. 系统运行风险特征库（当前 28 条规则）→ 风险评分 → 政策卡片匹配；\n"
        "3. 输出：体检得分 + 等级 + 命中特征清单 + 监管视角 Top3 + 优惠政策清单 + 行动建议。\n\n"
        "当前为**规则版**：判定全部由代码完成，LLM 报告为后续加分项。"
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
sm = policies.small_micro_status(profile)

st.header("体检结果")
c1, c2, c3, c4 = st.columns(4)
c1.metric("体检得分", f"{summary['score']} / 100")
c2.metric("风险等级", summary["level"])
c3.metric("命中特征", f"{summary['hit_count']} 条")
c4.metric("优惠卡片", f"{sum(1 for m in matched if m['status'] in ('可享受', '需人工确认'))} 条可关注")

st.subheader("监管视角 Top3")
if summary["top3"]:
    for t in summary["top3"]:
        st.markdown(f"- **{t['rule_id']} [{t['level']}] {t['name']}**：{t['evidence']}")
else:
    st.markdown("- 未命中明显风险特征；建议保持申报资料完整，关注政策更新。")

st.subheader("命中特征明细")
if hits:
    hits_df = pd.DataFrame([{
        "特征ID": h["rule_id"],
        "特征名称": h["name"],
        "级别": h["level"],
        "命中证据": h["evidence"],
        "行动建议": h["suggestion"],
    } for h in hits])
    st.dataframe(hits_df, use_container_width=True, hide_index=True)
else:
    st.success("未命中风险特征。")

st.subheader("优惠政策清单（政策卡片匹配）")
matched_df = pd.DataFrame([{
    "卡片": m["policy_id"],
    "政策名称": m["title"],
    "状态": m["status"],
    "文号": m["doc_number"],
    "优惠内容": m["benefit"],
    "判定明细": m["detail"],
} for m in matched])
st.dataframe(matched_df, use_container_width=True, hide_index=True)

with st.expander("小型微利企业条件逐项判定（P01）"):
    for c in sm["checks"]:
        mark = "✅" if c["pass"] else "❌"
        st.markdown(f"{mark} {c['name']}：当前 {c['value']}")
    st.markdown("**结论：** " + ("符合小型微利企业条件（演示口径）" if sm["qualified"] else "不满足或需补充数据"))

with st.expander("行动建议汇总（去重）"):
    seen = set()
    for h in hits:
        for s in h["suggestion"].split("；"):
            s = s.strip()
            if s and s not in seen:
                seen.add(s)
                st.markdown(f"- {s}")
    if not seen:
        st.markdown("- 数据表现正常，按日常申报节奏维护即可。")

st.caption("LLM 报告模块为加分项（未接入）：后续可把上述结构化结果交给大模型翻译成完整体检报告，"
           "约束'只翻译、不新增判断'。")
