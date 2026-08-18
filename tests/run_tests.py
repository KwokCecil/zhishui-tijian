# -*- coding: utf-8 -*-
"""智税体检验收测试（对应 Demo 方案第 10 节 25 条用例）。

运行：python tests/run_tests.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import generate_data  # noqa: E402
import agent  # noqa: E402
import policies  # noqa: E402
import rules  # noqa: E402
import scoring  # noqa: E402
import tools  # noqa: E402

PROFILE_COLS = generate_data.PROFILE_COLS
INVOICE_COLS = generate_data.INVOICE_COLS
FUND_COLS = generate_data.FUND_COLS
CONTRACT_COLS = generate_data.CONTRACT_COLS


def profile(data=None, **kw):
    row = {c: "" for c in PROFILE_COLS}
    if data:
        row.update(data)
    row.update(kw)
    return pd.DataFrame([row], columns=PROFILE_COLS)


def invoices(rows):
    return pd.DataFrame(rows, columns=INVOICE_COLS)


def funds(rows):
    return pd.DataFrame(rows, columns=FUND_COLS)


def contracts(rows):
    return pd.DataFrame(rows, columns=CONTRACT_COLS)


RESULTS = []


def case(n, name, fn):
    try:
        fn()
        RESULTS.append((n, name, "PASS", ""))
    except AssertionError as exc:
        RESULTS.append((n, name, "FAIL", str(exc)))
    except Exception as exc:  # noqa: BLE001
        RESULTS.append((n, name, "ERROR", f"{type(exc).__name__}: {exc}"))


def hits_of(data):
    return rules.run_all(data["company_profile"], data["invoices"], data["fund_flows"], data["contracts"])


def rule(hits, rid):
    found = [h for h in hits if h["rule_id"] == rid]
    assert found, f"未命中 {rid}"
    return found[0]


def test_01_tax_burden_low():
    p = profile({
        "行业": "软件和信息技术服务业", "纳税人类型": "一般纳税人",
        "营业收入(万元)": 1000, "资产总额(万元)": 500,
    })
    inv = invoices([
        ("XS1", "2026-07-01", "销项发票", "软件服务", 6, 10000000, 600000, "甲", "", "正常", "", ""),
        ("JJ1", "2026-07-02", "进项发票", "技术服务", 6, 9500000, 570000, "乙", "", "正常", "", ""),
    ])
    r = rule(rules.run_all(p, inv, funds([]), contracts([])), "R12")
    assert r["hit"] and r["level"] == "高", f"预期高，实际 {r}"


def test_02_input_output_mismatch():
    inv = invoices([
        ("XS1", "2026-07-01", "销项发票", "软件服务", 6, 1000000, 60000, "甲", "", "正常", "", ""),
        ("JJ1", "2026-07-02", "进项发票", "餐饮服务", 6, 800000, 48000, "乙", "", "正常", "", ""),
    ])
    r = rule(rules.run_all(profile({"资产总额(万元)": 100}), inv, funds([]), contracts([])), "R04")
    assert r["hit"], r


def test_03_top_up_invoices():
    inv = invoices([
        ("XS1", "2026-07-28", "销项发票", "软件服务", 6, 99990, 5999.4, "甲", "", "正常", "", ""),
        ("XS2", "2026-07-28", "销项发票", "软件服务", 6, 99990, 5999.4, "乙", "", "正常", "", ""),
        ("XS3", "2026-07-28", "销项发票", "软件服务", 6, 99990, 5999.4, "丙", "", "正常", "", ""),
    ])
    r = rule(rules.run_all(profile({"资产总额(万元)": 100}), inv, funds([]), contracts([])), "R01")
    assert r["hit"] and r["level"] == "高", r


def test_04_three_flows_mismatch():
    inv = invoices([
        ("XS1", "2026-07-01", "销项发票", "软件服务", 6, 1000000, 60000, "甲方", "", "正常", "HT-001", ""),
    ])
    f = funds([
        ("2026-07-05", "收入", "乙方", 1000000, "软件服务费", "HT-001", "对公账户"),
    ])
    c = contracts([("HT-001", "甲方", 1000000, "2026-06-01")])
    r = rule(rules.run_all(profile({"资产总额(万元)": 100}), inv, f, c), "R06")
    assert r["hit"] and r["level"] == "高", r


def test_05_headcount_social_mismatch():
    p = profile({"个税申报人数": 30, "社保参保人数": 15, "资产总额(万元)": 100})
    r = rule(rules.run_all(p, invoices([]), funds([]), contracts([])), "R15")
    assert r["hit"], r


def test_06_upstream_risk():
    inv = invoices([
        ("JJ1", "2026-07-02", "进项发票", "技术服务", 6, 1000000, 60000, "走逃公司", "", "走逃失联", "", ""),
    ])
    r = rule(rules.run_all(profile({"资产总额(万元)": 100}), inv, funds([]), contracts([])), "R09")
    assert r["hit"] and "异常凭证" in r["suggestion"] and "A级" in r["suggestion"], r


def test_07_red_void_ratio():
    rows = [("XS1", "2026-07-01", "销项发票", "软件服务", 6, 1000000, 60000, "甲", "", "正常", "", "")]
    for i in range(2):
        rows.append((f"HC{i}", "2026-07-10", "红字发票", "软件服务", 6, 200000, 12000, "甲", "", "正常", "", ""))
    r = rule(rules.run_all(profile({"资产总额(万元)": 100}), invoices(rows), funds([]), contracts([])), "R03")
    assert r["hit"], r


def test_08_small_micro_near_limit():
    p = profile({"个税申报人数": 280, "资产总额(万元)": 4800, "应纳税所得额(万元)": 292.7})
    hits = rules.run_all(p, invoices([]), funds([]), contracts([]))
    r = rule(hits, "R17")
    assert r["hit"], r
    sm = policies.small_micro_status(p)
    assert sm["qualified"], sm


def test_09_small_micro_exceeded():
    p = profile({"个税申报人数": 350, "资产总额(万元)": 4800, "应纳税所得额(万元)": 292.7})
    sm = policies.small_micro_status(p)
    assert not sm["qualified"], sm
    head = [c for c in sm["checks"] if "从业人数" in c["name"]][0]
    assert not head["pass"] and "350" in head["value"], sm


def test_10_policy_software():
    p = profile({
        "行业": "软件和信息技术服务业", "纳税人类型": "一般纳税人",
        "营业收入(万元)": 500, "研发费用(万元)": 80, "是否销售自研软件": "是",
    })
    matched = policies.match_cards(p)
    titles = " ".join(m["title"] for m in matched if m["status"] in ("可享受", "需人工确认"))
    assert "研发费用加计扣除" in titles, titles
    assert "软件产品增值税即征即退" in titles, titles


def test_11_clean_low_risk():
    data = generate_data.build_scenario("clean")
    hits = hits_of(data)
    summary = scoring.risk_summary(hits)
    assert summary["level"] == "低风险" and summary["score"] == 100, summary


def test_12_unknown_policy_refused():
    text = policies.policy_answer("某个未收录的政策问题")
    assert "12366" in text and "不" in text, text


def test_13_missing_data_no_guess():
    p = profile({"行业": "软件和信息技术服务业", "资产总额(万元)": ""})
    r = rule(rules.run_all(p, invoices([]), funds([]), contracts([])), "R31")
    assert r["hit"] and "缺失" in r["evidence"] and "补充" in r["suggestion"], r


def test_14_score_boundaries():
    score, level = scoring.risk_score([])
    assert score == 100 and level == "低风险"
    # 1 条高危 → 中风险（得分只反映加权，等级按严重程度升级）
    s1, l1 = scoring.risk_score([{"level": "高"}])
    assert l1 == "中风险" and s1 == 97
    # 高危 + 中危 → 高风险
    assert scoring.risk_level([{"level": "高"}, {"level": "中"}]) == "高风险"
    # ≥2 条中危 → 中风险
    assert scoring.risk_level([{"level": "中"}, {"level": "中"}]) == "中风险"
    # 关键预警组合 R17+R19+R35 → 高风险
    combo = [
        {"level": "低", "rule_id": "R17"},
        {"level": "中", "rule_id": "R19"},
        {"level": "中", "rule_id": "R35"},
    ]
    assert scoring.risk_level(combo) == "高风险"


def test_15_report_traceable():
    data = generate_data.build_scenario("risk")
    hits = hits_of(data)
    assert len(hits) >= 5, f"风险场景命中过少: {len(hits)}"
    for h in hits:
        assert h["rule_id"] and h["evidence"] and h["suggestion"], h


def test_16_case1_similar_case_and_docs():
    data = generate_data.build_scenario("case1")
    hits = hits_of(data)
    r17 = rule(hits, "R17")
    r35 = rule(hits, "R35")
    assert "备查" in r17["suggestion"] or "备查" in r35["suggestion"], (r17, r35)


def test_17_fuel_three_source():
    data = generate_data.build_scenario("fuel")
    hits = hits_of(data)
    r22 = rule(hits, "R22")
    assert r22["hit"] and "以进控销" in r22["suggestion"], r22
    assert any(h["rule_id"] in ("R21", "R23", "R31", "R32", "R33", "R34") for h in hits), hits


def test_18_gang_cluster_features():
    p = profile({"关联户数": 6, "资产总额(万元)": 100})
    rows = [("XS1", "2026-07-01", "销项发票", "咨询费", 6, 100000, 6000, "甲", "", "正常", "", "")]
    rows += [("XS2", "2026-07-02", "销项发票", "咨询费", 6, 100000, 6000, "乙", "", "正常", "", "")]
    hits = rules.run_all(p, invoices(rows), funds([]), contracts([]))
    r24 = rule(hits, "R24")
    r27 = rule(hits, "R27")
    assert r24["hit"] and r27["hit"], (r24, r27)


def test_19_qualification_deduction_mismatch():
    p = profile({"是否高新技术企业": "否", "是否享受加计抵减": "是", "资产总额(万元)": 100})
    r = rule(rules.run_all(p, invoices([]), funds([]), contracts([])), "R30")
    assert r["hit"] and "高企资格复核清单" in r["suggestion"], r


def test_20_fuel_data_gap():
    p = profile({"行业": "成品油零售", "资产总额(万元)": 100, "液位仪月缺失率(%)": 12})
    r = rule(rules.run_all(p, invoices([]), funds([]), contracts([])), "R31")
    assert r["hit"] and "缺失率" in r["evidence"], r


def test_21_upstream_docs_vs_local():
    p = profile({"行业": "成品油零售", "资产总额(万元)": 100, "上游进油单存在": "是", "本地入库记录缺失": "是"})
    r = rule(rules.run_all(p, invoices([]), funds([]), contracts([])), "R33")
    assert r["hit"] and "锚点" in r["suggestion"], r


def test_22_device_chip_risk():
    p = profile({"行业": "成品油零售", "资产总额(万元)": 100, "设备芯片标准": "旧标准"})
    r = rule(rules.run_all(p, invoices([]), funds([]), contracts([])), "R34")
    assert r["hit"] and "降级" in r["evidence"], r


def test_23_case1_full_warning_chain():
    data = generate_data.build_scenario("case1")
    hits = hits_of(data)
    for rid in ("R17", "R19", "R35"):
        rule(hits, rid)
    r17 = rule(hits, "R17")
    assert "六税两费" in r17["suggestion"] or "政策包" in r17["suggestion"], r17


def test_24_agricultural_purchase_risk():
    data = generate_data.build_scenario("case6")
    hits = hits_of(data)
    r36 = rule(hits, "R36")
    r37 = rule(hits, "R37")
    assert "核验清单" in r36["suggestion"] and "代开" in r36["suggestion"], r36
    assert r37["hit"], r37


def test_25_abnormal_invoice_path():
    inv = invoices([
        ("JJ1", "2026-07-02", "进项发票", "技术服务", 6, 1000000, 60000, "失联公司", "", "非正常户", "", ""),
    ])
    r = rule(rules.run_all(profile({"资产总额(万元)": 100}), inv, funds([]), contracts([])), "R09")
    assert "暂不允许抵扣" in r["suggestion"] and "进项转出" in r["suggestion"], r


def test_26_tool_schemas_valid():
    assert len(tools.TOOLS) >= 6, "工具数量不足"
    for t in tools.TOOLS:
        assert t["name"] and t["description"], t
        assert t["parameters"]["type"] == "object", t
        props = t["parameters"]["properties"]
        assert isinstance(props, dict) and props, t
        assert set(t["required"]).issubset(props.keys()), t


def test_27_health_check_tool():
    scenario = tools.execute_tool("get_demo_scenario", {"name": "risk"})
    r = tools.execute_tool("run_tax_health_check", {
        "profile": scenario["company_profile"],
        "invoices": scenario["invoices"],
        "fund_flows": scenario["fund_flows"],
        "contracts": scenario["contracts"],
    })
    assert len(r["hits"]) >= 5, r
    assert r["summary"]["level"] == "高风险", r["summary"]
    assert r["summary"]["top3"], r["summary"]


def test_28_policy_tool():
    scenario = tools.execute_tool("get_demo_scenario", {"name": "case1"})
    r = tools.execute_tool("match_policy_cards", {"profile": scenario["company_profile"]})
    assert len(r) == 8, r
    assert all(m["doc_number"] for m in r), r
    assert all("conditions" in m and m["conditions"] for m in r), r


def test_29_small_micro_tool():
    scenario = tools.execute_tool("get_demo_scenario", {"name": "case1"})
    r = tools.execute_tool("check_small_micro", {"profile": scenario["company_profile"]})
    assert r["qualified"] is True, r


def test_30_policy_question_tool():
    r = tools.execute_tool("answer_policy_question", {"query": "某个没收录的政策"})
    assert "12366" in r["answer"], r


def test_31_scenario_tool_unknown():
    r = tools.execute_tool("get_demo_scenario", {"name": "nope"})
    assert "error" in r, r


def test_32_report_tool_offline():
    scenario = tools.execute_tool("get_demo_scenario", {"name": "case1"})
    health = tools.execute_tool("run_tax_health_check", {
        "profile": scenario["company_profile"],
        "invoices": scenario["invoices"],
        "fund_flows": scenario["fund_flows"],
        "contracts": scenario["contracts"],
    })
    matched = tools.execute_tool("match_policy_cards", {"profile": scenario["company_profile"]})
    r = tools.execute_tool("generate_report", {
        "hits": health["hits"],
        "summary": health["summary"],
        "matched": matched,
    })
    assert "体检得分" in r["report"], r
    assert "R17" in r["report"], r


def test_33_offline_agent_loop():
    r1 = agent.run_agent("这家公司有什么风险？", scenario="risk")
    assert r1["mode"] == "offline" and r1["trace"], r1
    assert "体检得分" in r1["answer"] or "等级" in r1["answer"], r1

    r2 = agent.run_agent("能享受哪些优惠政策？", scenario="case1")
    assert any(t["tool"] == "match_policy_cards" for t in r2["trace"]), r2
    assert "政策" in r2["answer"], r2

    r3 = agent.run_agent("这家公司符合小型微利企业条件吗？", scenario="case1")
    assert any(t["tool"] == "check_small_micro" for t in r3["trace"]), r3


def test_34_case1_level_high():
    data = generate_data.build_scenario("case1")
    hits = rules.run_all(data["company_profile"], data["invoices"], data["fund_flows"], data["contracts"])
    summary = scoring.risk_summary(hits)
    assert summary["level"] == "高风险", summary


def test_35_case6_level_high():
    data = generate_data.build_scenario("case6")
    hits = rules.run_all(data["company_profile"], data["invoices"], data["fund_flows"], data["contracts"])
    summary = scoring.risk_summary(hits)
    assert summary["level"] == "高风险", summary


def main():
    case(1, "税负率明显低于行业参考区间 → R12 高", test_01_tax_burden_low)
    case(2, "销项软件、进项全餐饮 → R04", test_02_input_output_mismatch)
    case(3, "连续3张接近顶额 → R01", test_03_top_up_invoices)
    case(4, "发票与资金付款方不一致 → R06", test_04_three_flows_mismatch)
    case(5, "个税30人、社保15人 → R15", test_05_headcount_social_mismatch)
    case(6, "上游走逃失联 → R09+异常凭证路径", test_06_upstream_risk)
    case(7, "红冲/作废率30% → R03", test_07_red_void_ratio)
    case(8, "小微临界（280/4800/292.7）→ R17+仍符合", test_08_small_micro_near_limit)
    case(9, "人数350 → 小微不满足并指出超项", test_09_small_micro_exceeded)
    case(10, "软件公司+研发80万 → 研发加计+即征即退机会", test_10_policy_software)
    case(11, "全部正常 → 低风险画像", test_11_clean_low_risk)
    case(12, "未收录政策问题 → 拒绝并引导12366", test_12_unknown_policy_refused)
    case(13, "资产总额缺失 → 不猜，标注需补充", test_13_missing_data_no_guess)
    case(14, "评分边界（100/79/58）→ 等级稳定", test_14_score_boundaries)
    case(15, "报告逐条可溯源（ID+证据+建议）", test_15_report_traceable)
    case(16, "案例一命中 → 展示相似案例+备查资料", test_16_case1_similar_case_and_docs)
    case(17, "加油站模板 → R22 三源不一致+以进控销", test_17_fuel_three_source)
    case(18, "团伙特征（同址+咨询费集中）→ R24+R27", test_18_gang_cluster_features)
    case(19, "非高企享受加计抵减 → R30 复核清单", test_19_qualification_deduction_mismatch)
    case(20, "液位仪缺失率12% → R31", test_20_fuel_data_gap)
    case(21, "上游进油单存在但本地记录缺失 → R33", test_21_upstream_docs_vs_local)
    case(22, "旧标准芯片 → R34 数据降级", test_22_device_chip_risk)
    case(23, "案例一预警链（R17/R19/R35+降负政策包）", test_23_case1_full_warning_chain)
    case(24, "收购对象身份存疑+单户巨大 → R36+R37+代开路径", test_24_agricultural_purchase_risk)
    case(25, "上游走逃 → 异常凭证处理路径提示", test_25_abnormal_invoice_path)
    case(26, "工具层 schema 完整（名称/描述/参数）", test_26_tool_schemas_valid)
    case(27, "run_tax_health_check 工具（risk→高风险）", test_27_health_check_tool)
    case(28, "match_policy_cards 工具（8张卡带文号）", test_28_policy_tool)
    case(29, "check_small_micro 工具（案例一符合）", test_29_small_micro_tool)
    case(30, "answer_policy_question 工具（引导12366）", test_30_policy_question_tool)
    case(31, "get_demo_scenario 未知场景返回错误", test_31_scenario_tool_unknown)
    case(32, "generate_report 离线报告含体检得分", test_32_report_tool_offline)
    case(33, "离线 Agent 对话循环（风险/政策/小微）", test_33_offline_agent_loop)
    case(34, "案例一等级：高风险（组合升级）", test_34_case1_level_high)
    case(35, "案例六等级：高风险（高危+中危）", test_35_case6_level_high)

    print(f"\n{'#':<3}{'用例':<52}{'结果':<6}说明")
    print("-" * 100)
    failed = 0
    for n, name, status, detail in RESULTS:
        print(f"{n:<3}{name:<52}{status:<6}{detail}")
        if status != "PASS":
            failed += 1
    print("-" * 100)
    print(f"总计 {len(RESULTS)} 条，通过 {len(RESULTS)-failed} 条，失败 {failed} 条")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
