# -*- coding: utf-8 -*-
"""风险特征库：每个特征一个函数，统一输出结构。

输出结构:
    {"rule_id", "name", "hit", "level", "evidence", "suggestion"}

说明：金税四期具体模型与阈值不公开，以下为公开风险逻辑 + 演示阈值，
生产环境需按地区、行业、主管税务机关口径校准。
"""

import math

import pandas as pd

INDUSTRY_VAT_LOWER = {
    "软件和信息技术服务业": 3.0,
    "制造业": 2.0,
    "批发和零售业": 0.5,
    "成品油零售": 1.0,
    "农副产品加工": 1.0,
}

INDUSTRY_GROSS_MARGIN_LOWER = {
    "软件和信息技术服务业": 30.0,
    "制造业": 15.0,
    "批发和零售业": 5.0,
    "成品油零售": 5.0,
    "农副产品加工": 5.0,
}

ABNORMAL_STATUS = {"非正常户", "走逃失联"}
SERVICE_ITEMS = ("咨询费", "服务费", "管理费")
UNRELATED_INPUT_KEYWORDS = ("餐饮", "住宿", "娱乐", "礼品", "广告", "旅游")
TECH_SALE_KEYWORDS = ("软件", "信息技术", "技术开发", "系统")
NON_DEDUCTIBLE_KEYWORDS = ("餐饮", "娱乐", "居民日常", "旅游", "贷款服务")

EXTERNAL_DOC_COLS = ["单据号", "供应商", "品名", "数量", "单位", "金额(元)", "日期", "来源方"]
DATA_SOURCE_COLS = ["数据源", "来源方", "可否篡改", "版本", "留存月数", "可信度"]


def _num(value):
    """安全转 float，缺失/空返回 None。"""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None
    try:
        result = float(value)
        if math.isnan(result):
            return None
        return result
    except (TypeError, ValueError):
        return None


def _pct(part, total):
    if part is None or not total:
        return None
    return part / total * 100


def _cno(value):
    """合同号归一化：NaN/None/空 一律返回空字符串。"""
    if value is None:
        return ""
    s = str(value).strip()
    return "" if s.lower() in ("nan", "none", "null") else s


def _sales_invoices(invoices):
    return invoices[invoices["发票类型"] == "销项发票"]


def _input_invoices(invoices):
    return invoices[invoices["发票类型"] == "进项发票"]


def _rule(rule_id, name, hit, level, evidence, suggestion):
    return {
        "rule_id": rule_id,
        "name": name,
        "hit": bool(hit),
        "level": level,
        "evidence": evidence,
        "suggestion": suggestion,
    }


def check_r01(invoices):
    """R101 顶额开票：单张接近限额（演示：>=90000 元）且同月>=3张。"""
    sales = _sales_invoices(invoices)
    if sales.empty:
        return _rule("R101", "顶额开票", False, "低", "无销项发票数据", "")
    sales = sales.copy()
    sales["金额(元)"] = sales["金额(元)"].apply(_num)
    # 顶额指接近开票限额（演示：万元版发票限额 99999 元，取 90000-101000 区间）
    top = sales[sales["金额(元)"].notna() & (sales["金额(元)"] >= 90000) & (sales["金额(元)"] <= 101000)]
    if top.empty:
        return _rule("R101", "顶额开票", False, "低", f"无顶额发票（共{len(sales)}张）", "")
    by_month = top["开票日期"].str[:7].value_counts()
    hit_month = by_month[by_month >= 3]
    if hit_month.empty:
        return _rule("R101", "顶额开票", False, "低", f"顶额票{len(top)}张，但单月不足3张", "")
    m = hit_month.index[0]
    return _rule(
        "R101", "顶额开票", True, "低",
        f"{m} 月顶额发票 {int(hit_month.iloc[0])} 张（金额≥90000元，接近开票限额99999元）",
        "核对开票业务实质：是否拆票规避限额/拆分客户；查验合同、物流、付款对应关系。",
    )


def check_r02(invoices):
    """R102 月末集中开票：当月发票集中在最后5个自然日，占比>60%。"""
    sales = _sales_invoices(invoices)
    if sales.empty:
        return _rule("R102", "月末集中开票", False, "低", "无销项发票数据", "")
    sales = sales.copy()
    sales["金额(元)"] = sales["金额(元)"].apply(_num).fillna(0)
    for month, group in sales.groupby(sales["开票日期"].str[:7]):
        days = group["开票日期"].str[-2:].astype(int)
        # 演示按 31 天月估算月末窗口
        last5 = group[days >= 27]["金额(元)"].sum()
        total = group["金额(元)"].sum()
        if total > 0 and last5 / total > 0.6:
            return _rule(
                "R102", "月末集中开票", True, "低",
                f"{month} 月最后5个自然日开票占比 {last5/total*100:.1f}%（演示阈值>60%）",
                "核实集中开票的业务原因（合同结算节奏/业绩压力），关注是否存在跨期调节收入。",
            )
    return _rule("R102", "月末集中开票", False, "低", "月末集中度未超演示阈值", "")


def check_r03(invoices):
    """R103 红冲/作废率过高：红冲+作废金额或张数占当期开票比例>20%。"""
    sales = _sales_invoices(invoices)
    red = invoices[invoices["发票类型"].isin(["红字发票", "作废发票"])]
    if sales.empty:
        return _rule("R103", "红冲/作废率过高", False, "中", "无销项发票数据", "")
    sales_amount = sales["金额(元)"].apply(_num).fillna(0).sum()
    red_amount = red["金额(元)"].apply(_num).fillna(0).sum()
    count_ratio = len(red) / len(sales) * 100
    amount_ratio = _pct(red_amount, sales_amount) or 0
    if count_ratio > 20 or amount_ratio > 20:
        return _rule(
            "R103", "红冲/作废率过高", True, "中",
            f"红冲/作废 {len(red)} 张 vs 销项 {len(sales)} 张（张数占比{count_ratio:.1f}%，金额占比{amount_ratio:.1f}%）",
            "逐张核对红冲/作废原因（开票错误/退货/走逃作废），防止利用红冲调节销项、隐匿收入。",
        )
    return _rule("R103", "红冲/作废率过高", False, "中", f"红冲/作废占比 {count_ratio:.1f}%（阈值20%）", "")


def check_r04(invoices):
    """R104 进销项品名不匹配：销项为技术类，进项却全是无关消费类品名。"""
    sales = _sales_invoices(invoices)
    inputs = _input_invoices(invoices)
    if sales.empty or inputs.empty:
        return _rule("R104", "进销项品名不匹配", False, "高", "进项或销项数据缺失", "")
    sale_items = {str(x).strip() for x in sales["品名"].dropna()}
    input_items = {str(x).strip() for x in inputs["品名"].dropna()}
    sale_is_tech = any(k in "".join(sale_items) for k in TECH_SALE_KEYWORDS)
    inputs_unrelated = all(any(k in item for k in UNRELATED_INPUT_KEYWORDS) for item in input_items)
    if sale_is_tech and inputs_unrelated and not (sale_items & input_items):
        non_deduct = inputs[
            inputs["品名"].astype(str).apply(lambda x: any(k in str(x) for k in NON_DEDUCTIBLE_KEYWORDS))
        ]
        vat = float(non_deduct["税额(元)"].apply(_num).sum()) if not non_deduct.empty else 0.0
        if vat > 0:
            return _rule(
                "R104", "进销项品名不匹配", True, "高",
                f"销项品名 {sorted(sale_items)}，进项品名 {sorted(input_items)}，语义无关；"
                f"其中餐饮/娱乐等法定不得抵扣品目的进项税额合计约 {vat/10000:.0f} 万元",
                f"后果测算：财税〔2016〕36号第二十七条规定餐饮、娱乐等服务进项不得抵扣，"
                f"若已申报抵扣应做进项税额转出约 {vat/10000:.0f} 万元；"
                "自查进项业务实质：核实是否用于生产经营，有无集体福利/个人消费情形。",
            )
        return _rule(
            "R104", "进销项品名不匹配", True, "高",
            f"销项品名 {sorted(sale_items)}，进项品名 {sorted(input_items)}，语义无关",
            "自查进项发票业务实质：与经营无关的餐饮/娱乐进项应及时转出，避免影响抵扣与申报。",
        )
    return _rule("R104", "进销项品名不匹配", False, "高", "进销项品名存在业务关联", "")


def check_r06(invoices, fund_flows, contracts):
    """R201 三流不一致：发票、资金、合同按合同号/对方名称比对不一致。"""
    if invoices.empty or fund_flows.empty:
        return _rule("R201", "三流不一致", False, "中", "发票或资金数据缺失", "")
    if "收付方向" not in fund_flows.columns:
        return _rule("R201", "三流不一致", False, "中", "资金流水缺少收付方向", "")
    income_by_contract = {}
    expense_by_contract = {}
    for _, f in fund_flows.iterrows():
        cno = _cno(f.get("关联合同号"))
        if not cno:
            continue
        name = str(f.get("对方名称", "")).strip()
        if str(f.get("收付方向", "")).strip() in ("收入", "收款", "收"):
            income_by_contract.setdefault(cno, []).append(name)
        elif str(f.get("收付方向", "")).strip() in ("支出", "付款", "付"):
            expense_by_contract.setdefault(cno, []).append(name)
    contract_party = {}
    for _, c in contracts.iterrows():
        contract_party[_cno(c.get("合同号"))] = str(c.get("对方名称", "")).strip()

    mismatches = []
    personal = []
    for _, inv in invoices.iterrows():
        cno = _cno(inv.get("关联合同号"))
        inv_party = str(inv.get("对方名称", "")).strip()
        if not cno:
            continue
        inv_type = str(inv.get("发票类型", ""))
        if inv_type == "销项发票":
            peers = income_by_contract.get(cno, [])
            label = "付款方"
        elif inv_type == "进项发票":
            peers = expense_by_contract.get(cno, [])
            label = "收款方"
        else:
            peers = []
            label = "对方"
        for fp in peers:
            if inv_party and fp and fp != inv_party:
                mismatches.append(f"发票方[{inv_party}]≠{label}[{fp}]({cno})")
        if "账户类型" in fund_flows.columns and (
            fund_flows[fund_flows["关联合同号"].astype(str).str.strip() == cno]["账户类型"] == "个人账户"
        ).any():
            personal.append(cno)
        if cno in contract_party and inv_party and contract_party[cno] != inv_party:
            mismatches.append(f"发票方[{inv_party}]≠合同方[{contract_party[cno]}]({cno})")
    if mismatches:
        return _rule(
            "R201", "三流不一致", True, "中",
            "；".join(mismatches[:5]) + (f"；个人账户收款 {len(personal)} 笔" if personal else ""),
            "逐笔核对合同、发票、资金流对应关系，补齐不一致事项的凭证与书面说明，降低认定风险。",
        )
    if personal:
        return _rule(
            "R201", "三流不一致", True, "中",
            f"对公业务通过个人账户收付（合同号：{'、'.join(personal[:5])}）",
            "逐笔核对合同、发票、资金流三流对应关系；个人账户收付对公货款是隐匿收入/资金回流常见通道。",
        )
    return _rule("R201", "三流不一致", False, "中", "发票、资金、合同三流一致", "")


def check_r07(fund_flows):
    """R202 个人账户收付款占比高：对公业务通过个人账户收付款金额占比>10%。"""
    if fund_flows.empty or "账户类型" not in fund_flows.columns:
        return _rule("R202", "个人账户收付款占比高", False, "中", "无资金流水数据", "")
    flows = fund_flows.copy()
    flows["金额(元)"] = flows["金额(元)"].apply(_num)
    total = flows["金额(元)"].sum()
    personal = flows[flows["账户类型"] == "个人账户"]["金额(元)"].sum()
    if total and personal / total > 0.1:
        return _rule(
            "R202", "个人账户收付款占比高", True, "中",
            f"个人账户收付款 {personal:,.0f} 元，占总流水 {personal/total*100:.1f}%（阈值10%）",
            "梳理个人账户收付款对应的业务，规范公私账户使用，保留完整凭证。",
        )
    return _rule("R202", "个人账户收付款占比高", False, "中", f"个人账户占比 {_pct(personal, total) or 0:.1f}%（阈值10%）", "")


def check_r203(fund_flows):
    """R203 资金回流：收入后短期内向同一对方支出，或备注含回流/转回。"""
    if fund_flows.empty or "收付方向" not in fund_flows.columns:
        return _rule("R203", "资金回流", False, "中", "无资金流水数据", "")
    flows = fund_flows.copy()
    flows["金额(元)"] = flows["金额(元)"].apply(_num)
    flows["日期"] = pd.to_datetime(flows["流水日期"], errors="coerce")
    incomes = flows[flows["收付方向"].isin(["收入", "收款", "收"])]
    expenses = flows[flows["收付方向"].isin(["支出", "付款", "付"])]
    for _, inc in incomes.iterrows():
        name = str(inc.get("对方名称", "")).strip()
        if not name:
            continue
        window = expenses[
            (expenses["对方名称"].astype(str).str.strip() == name)
            & (expenses["日期"] >= inc["日期"])
            & ((expenses["日期"] - inc["日期"]).dt.days <= 7)
        ]
        if not window.empty:
            e = window.iloc[0]
            return _rule(
                "R203", "资金回流", True, "中",
                f"收入 {inc['金额(元)']:,.0f} 元后 {int((e['日期'] - inc['日期']).days)} 天内"
                f"向同一对方支出 {e['金额(元)']:,.0f} 元，疑似资金回流",
                "梳理资金回流原因（借款、代收代付、退还款等），保留合同与凭证，避免形成回流闭环。",
            )
    if flows["备注"].astype(str).str.contains("回流|转回", na=False).any():
        return _rule(
            "R203", "资金回流", True, "中",
            "资金流水备注含'回流/转回'字样",
            "核查回流资金对应的业务实质，是否构成资金回流闭环。",
        )
    return _rule("R203", "资金回流", False, "中", "未发现资金回流", "")


def check_r204(invoices, fund_flows):
    """R204 收款无销项覆盖：收款对象在销项发票中查无对应开票（按对方名称匹配）。"""
    if fund_flows.empty:
        return _rule("R204", "收款无销项覆盖", False, "中", "无资金流水数据", "")
    sale_names = set()
    if not invoices.empty and "发票类型" in invoices.columns:
        sales = invoices[invoices["发票类型"] == "销项发票"]
        sale_names = {str(x).strip() for x in sales["对方名称"].dropna()}
    flows = fund_flows.copy()
    flows["金额(元)"] = flows["金额(元)"].apply(_num)
    incomes = flows[flows["收付方向"].astype(str).str.contains("收入|收款|收", na=False)]
    agg = {}
    for _, row in incomes.iterrows():
        name = str(row.get("对方名称", "")).strip()
        if name and name not in sale_names:
            agg[name] = agg.get(name, 0.0) + float(row["金额(元)"])
    if agg:
        detail = "；".join(f"{n} 收款{a/10000:.0f}万" for n, a in list(agg.items())[:3])
        total = sum(agg.values())
        evidence = f"存在收款但查无对应销项发票的对象：{detail}，合计 {total/10000:.1f} 万元"
        suggestion = (
            f"后果测算：无销项覆盖的收款合计约 {total/10000:.1f} 万元，若属已实现销售存在未申报风险；"
            "核验清单：合同履行与交付凭证、该对象全部开票情况（含未开票收入申报）；"
            "纠正路径：属已实现销售的应补开补报，属往来款项的留存证明备查。"
        )
        return _rule("R204", "收款无销项覆盖", True, "中", evidence, suggestion)
    return _rule("R204", "收款无销项覆盖", False, "中", "收款对象均有销项发票覆盖", "")


def check_r105(invoices):
    """R105 红冲金额与原票倒挂：同一对方红字发票合计超过销项发票合计。"""
    if invoices.empty or "发票类型" not in invoices.columns:
        return _rule("R105", "红冲金额超过原票", False, "中", "无发票数据", "")
    inv = invoices.copy()
    inv["金额(元)"] = inv["金额(元)"].apply(_num)
    reds = inv[inv["发票类型"] == "红字发票"]
    if reds.empty:
        return _rule("R105", "红冲金额超过原票", False, "中", "无红字发票", "")
    sales = inv[inv["发票类型"] == "销项发票"]
    sales_sum = sales.groupby(sales["对方名称"].astype(str).str.strip())["金额(元)"].sum()
    red_sum = reds.groupby(reds["对方名称"].astype(str).str.strip())["金额(元)"].sum()
    offenders = []
    for name, rsum in red_sum.items():
        bsum = float(sales_sum.get(name, 0.0))
        if rsum > bsum:
            offenders.append((name, float(rsum), bsum))
    if offenders:
        detail = "；".join(f"{n} 红字{r/10000:.0f}万 > 销项{b/10000:.0f}万" for n, r, b in offenders[:3])
        total_over = sum(r - b for _, r, b in offenders)
        evidence = f"红字发票合计超过对应销项合计：{detail}（红冲超原票差额约 {total_over/10000:.1f} 万元）"
        suggestion = (
            f"后果测算：异常冲销差额约 {total_over/10000:.1f} 万元，若已冲减销项需核实红冲依据；"
            "核验清单：红冲对应原票、业务终止凭证、资金退回记录；"
            "纠正路径：无真实红冲依据的不得冲减，已冲减的应更正申报。"
        )
        return _rule("R105", "红冲金额超过原票", True, "中", evidence, suggestion)
    return _rule("R105", "红冲金额超过原票", False, "中", "红字金额均未超过对应销项", "")


def check_r09(invoices):
    """R301 风险企业传导：上游/下游存在非正常户、走逃失联。"""
    if invoices.empty or "对方状态" not in invoices.columns:
        return _rule("R301", "风险企业传导", False, "中", "无发票数据", "")
    bad = invoices[invoices["对方状态"].isin(ABNORMAL_STATUS)]
    if bad.empty:
        return _rule("R301", "风险企业传导", False, "中", "无异常状态上下游企业", "")
    names = "、".join(str(x) for x in bad["对方名称"].dropna().unique()[:5])
    return _rule(
        "R301", "风险企业传导", True, "中",
        f"存在{len(bad)}张异常状态发票，对方：{names}",
        "异常凭证处理路径：未抵扣的暂不允许抵扣，已抵扣的先作进项转出；"
        "A级信用纳税人可在10个工作日内申请核实，核实通过可不转出。",
    )


def check_r12(invoices, profile):
    """R401 增值税税负率异常：税负率低于行业参考区间下限。"""
    sales = _sales_invoices(invoices)
    inputs = _input_invoices(invoices)
    if sales.empty or inputs.empty:
        return _rule("R401", "增值税税负率异常", False, "中", "发票数据不足，无法计算税负率", "")
    output_tax = sales["税额(元)"].apply(_num).fillna(0).sum()
    input_tax = inputs["税额(元)"].apply(_num).fillna(0).sum()
    sales_amt = sales["金额(元)"].apply(_num).fillna(0).sum()
    if not sales_amt:
        return _rule("R401", "增值税税负率异常", False, "中", "销项金额为0", "")
    burden = (output_tax - input_tax) / sales_amt * 100
    industry = str(profile.get("行业", ""))
    lower = INDUSTRY_VAT_LOWER.get(industry, 1.0)
    if burden < lower:
        level = "中"
        return _rule(
            "R401", "增值税税负率异常", True, level,
            f"增值税税负率 {burden:.1f}%，行业[{industry}]参考下限 {lower}%（演示值）",
            "核对进项抵扣与申报口径，检查应转出未转出的进项，准备税负率偏低的合理解释与资料。",
        )
    return _rule("R401", "增值税税负率异常", False, "中", f"税负率 {burden:.1f}%（行业参考下限 {lower}%）", "")


def check_r15(profile):
    """R501 个税与社保人数不一致：差异>20%。"""
    declare = _num(profile.get("个税申报人数"))
    social = _num(profile.get("社保参保人数"))
    if declare is None or social is None:
        return _rule("R501", "个税与社保人数不一致", False, "低", "个税或社保人数缺失", "")
    if declare <= 0:
        return _rule("R501", "个税与社保人数不一致", False, "低", "个税申报人数为0", "")
    diff = abs(declare - social) / declare * 100
    if diff > 20:
        level = "低"
        return _rule(
            "R501", "个税与社保人数不一致", True, level,
            f"个税申报 {declare:.0f} 人 vs 社保参保 {social:.0f} 人，差异 {diff:.0f}%",
            "核实用工形式（临时工/劳务外包/未参保人员），防范未足额参保、隐匿用工。",
        )
    return _rule("R501", "个税与社保人数不一致", False, "低", f"人数差异 {diff:.0f}%（阈值20%）", "")


def check_r16(profile):
    """R502 纳税信用等级低：D 级预警。"""
    level = str(profile.get("纳税信用等级", "")).strip().upper()
    if level == "D":
        return _rule(
            "R502", "纳税信用等级低", True, "中",
            "纳税信用等级为 D 级",
            "D级纳税人将受到发票领用、出口退税、融资授信等限制；核查失信原因并整改。",
        )
    return _rule("R502", "纳税信用等级低", False, "中", f"纳税信用等级 {level or '未知'}", "")


def check_r17(profile):
    """R601 小微临界：人数/资产/所得额位于限额 90%-100%。"""
    headcount = _num(profile.get("个税申报人数"))
    assets = _num(profile.get("资产总额(万元)"))
    income = _num(profile.get("应纳税所得额(万元)"))
    near = []
    if headcount is not None and 270 <= headcount <= 300:
        near.append(f"人数{headcount:.0f}/300")
    if assets is not None and 4500 <= assets <= 5000:
        near.append(f"资产{assets:,.0f}万/5000万")
    if income is not None and 270 <= income <= 300:
        near.append(f"所得{income:.1f}万/300万")
    if near:
        return _rule(
            "R601", "小微临界", True, "低",
            "；".join(near) + " 均位于小型微利企业限额的90%-100%区间",
            "临界企业是常见预警对象：核实申报准确性；若同时存在大额调减项，需准备备查资料。"
            "若符合小型微利企业条件，可叠加六税两费减半等优惠。",
        )
    return _rule("R601", "小微临界", False, "低", "未处于小微限额临界区间", "")


def check_r18(profile):
    """R402 利润与申报应纳税所得额差异过大：差异率>50%。"""
    profit = _num(profile.get("利润总额(万元)"))
    taxable = _num(profile.get("应纳税所得额(万元)"))
    if profit is None or taxable is None or taxable == 0:
        return _rule("R402", "利润与申报应纳税所得额差异过大", False, "中", "利润总额或应纳税所得额缺失", "")
    diff = abs(profit - taxable) / abs(taxable) * 100
    if diff > 50:
        level = "中"
        return _rule(
            "R402", "利润与申报应纳税所得额差异过大", True, level,
            f"会计利润 {profit:.1f}万 vs 申报应纳税所得额 {taxable:.1f}万，差异率 {diff:.0f}%",
            "核对纳税调增调减项目及备查资料：差异过大需解释税会差异来源。",
        )
    return _rule("R402", "利润与申报应纳税所得额差异过大", False, "中", f"税会差异率 {diff:.0f}%（阈值50%）", "")


def _big_deduction(profile):
    """大额调减项（演示界定）：研发费用>0 且研发费用 ≥ 应纳税所得额 × 50%。
    实际场景中大额调减还可能来自其他调减项目，生产环境需按调减明细校准。"""
    rd = _num(profile.get("研发费用(万元)"))
    taxable = _num(profile.get("应纳税所得额(万元)"))
    return rd is not None and taxable is not None and rd > 0 and rd >= taxable * 0.5


def check_r39(profile):
    """R602 大额调减项（低危提示）：单独出现不判定违规，只提示备查。"""
    rd = _num(profile.get("研发费用(万元)"))
    taxable = _num(profile.get("应纳税所得额(万元)"))
    if rd is None or taxable is None:
        return _rule("R602", "大额调减项", False, "低", "研发费用或应纳税所得额缺失", "")
    if _big_deduction(profile):
        return _rule(
            "R602", "大额调减项", True, "低",
            f"研发费用 {rd:.0f}万 ≥ 应纳税所得额 {taxable:.1f}万的50%（{taxable*0.5:.1f}万），构成大额调减（演示口径）",
            "大额调减本身不违规：备查调减依据（研发立项、费用归集、辅助账、其他调减项目凭证），"
            "与临界组合时风险升级。",
        )
    return _rule("R602", "大额调减项", False, "低", f"未达大额调减演示阈值（研发≥所得×50%）", "")


def check_r19(profile, hits=None):
    """G001 临界调减联动（组合规则）：所得额落于小微限额85%-100% 且 大额调减项命中。"""
    taxable = _num(profile.get("应纳税所得额(万元)"))
    rd = _num(profile.get("研发费用(万元)"))
    if taxable is None or rd is None:
        return _rule("G001", "临界调减联动", False, "中", "应纳税所得额或研发费用缺失", "")
    if 255 <= taxable <= 300 and _big_deduction(profile):
        return _rule(
            "G001", "临界调减联动", True, "中",
            f"应纳税所得额 {taxable:.1f}万（位于300万限额的85%-100%）+ 大额调减项（研发费用 {rd:.0f}万，"
            f"≥ 所得额50%）",
            "自查调减项依据与申报口径，备好研发立项、费用归集、辅助账等资料。",
        )
    return _rule("G001", "临界调减联动", False, "中", "未同时命中所得临界区间与大额调减", "")


def check_r20(profile):
    """R603 研发加计扣除占比异常：研发费用占收入比例>15%。"""
    rd = _num(profile.get("研发费用(万元)"))
    revenue = _num(profile.get("营业收入(万元)"))
    if rd is None or revenue is None or revenue == 0:
        return _rule("R603", "研发加计扣除占比异常", False, "中", "研发费用或营业收入缺失", "")
    ratio = rd / revenue * 100
    if ratio > 15:
        return _rule(
            "R603", "研发加计扣除占比异常", True, "中",
            f"研发费用占收入 {ratio:.1f}%（演示阈值15%）",
            "研发费用畸高需核查研发真实性：立项、人员、工时、材料领用、辅助账。",
        )
    return _rule("R603", "研发加计扣除占比异常", False, "中", f"研发占收入 {ratio:.1f}%（阈值15%）", "")


def _fuel_num(profile, field):
    return _num(profile.get(field))


def _fuel_exposure(reported, reference, price, base):
    """加油站差异的推定口径量化：行业参考推算，仅作为核查线索，不构成应补税金额。"""
    if reported is None or reference is None or price is None or price <= 0:
        return base
    ton_gap = abs(reference - reported)
    if ton_gap <= 0:
        return base
    income_gap = ton_gap * 1176 * price / 10000
    return (
        f"{base}"
        f"后果测算（推定口径）：按 1吨≈1176升、单价 {price:.2f}元/升 推定，"
        f"差额销量 {ton_gap:.0f} 吨涉及收入约 {income_gap:.0f} 万元"
        "——系行业参考推算，仅作为进一步核实的线索，不构成应补税金额，以实际核查结果为准。"
    )


def check_r21(profile):
    """R801 单站销售横向偏离（加油站模板）：申报销量显著偏离区域参考。"""
    reported = _fuel_num(profile, "申报销量(吨)")
    reference = _fuel_num(profile, "区域参考销量(吨)")
    if reported is None or reference is None:
        return _rule("R801", "单站销售横向偏离", False, "中", "加油站模板字段缺失", "")
    if reference == 0:
        return _rule("R801", "单站销售横向偏离", False, "中", "区域参考销量为0", "")
    diff = (reported - reference) / reference * 100
    if abs(diff) > 50:
        level = "中"
        return _rule(
            "R801", "单站销售横向偏离", True, level,
            f"申报销量 {reported:.0f}吨 vs 区域参考 {reference:.0f}吨，偏离 {diff:.0f}%",
            _fuel_exposure(reported, reference, _fuel_num(profile, "申报单价(元/升)"),
                           "横向比对同地段/车流条件站点；偏离过大需核实是否存在少申报，建议多源交叉验证。"),
        )
    return _rule("R801", "单站销售横向偏离", False, "中", f"横向偏离 {diff:.0f}%（阈值±50%）", "")


def check_r22(profile):
    """R802 多源数据不一致（加油站模板）：申报/设备/测算三源两两差异>20%。"""
    reported = _fuel_num(profile, "申报销量(吨)")
    device = _fuel_num(profile, "设备销量(吨)")
    estimated = _fuel_num(profile, "测算销量(吨)")
    if reported is None or device is None or estimated is None:
        return _rule("R802", "多源数据不一致", False, "高", "加油站三源数据缺失", "")
    pairs = [
        ("申报/设备", reported, device),
        ("申报/测算", reported, estimated),
        ("设备/测算", device, estimated),
    ]
    diffs = []
    for label, a, b in pairs:
        base = max(abs(a), abs(b))
        if base and abs(a - b) / base > 0.2:
            diffs.append(f"{label}差异{abs(a-b)/base*100:.0f}%")
    if diffs:
        price = _fuel_num(profile, "申报单价(元/升)")
        return _rule(
            "R802", "多源数据不一致", True, "高",
            f"申报{reported:.0f}吨/设备{device:.0f}吨/测算{estimated:.0f}吨；{'；'.join(diffs)}",
            _fuel_exposure(reported, max(device, estimated), price,
                           "用供应商进油单、运单等外部来源核对三套销量数据，找出差异原因并规范记录。"),
        )
    return _rule("R802", "多源数据不一致", False, "高", "三源数据两两差异≤20%", "")


def check_r23(profile):
    """R803 申报单价偏离区域均价（加油站模板）：低>10%。"""
    price = _fuel_num(profile, "申报单价(元/升)")
    avg = _fuel_num(profile, "区域均价(元/升)")
    if price is None or avg is None:
        return _rule("R803", "申报单价偏离区域均价", False, "中", "申报单价或区域均价缺失", "")
    if avg == 0:
        return _rule("R803", "申报单价偏离区域均价", False, "中", "区域均价为0", "")
    diff = (avg - price) / avg * 100
    if diff > 10:
        volume = _fuel_num(profile, "申报销量(吨)")
        income_gap = (avg - price) * volume * 1176 / 10000 if volume else 0
        return _rule(
            "R803", "申报单价偏离区域均价", True, "中",
            f"申报单价 {price:.2f}元/升 vs 区域均价 {avg:.2f}元/升，低 {diff:.0f}%",
            "单价明显偏低需核实让利真实性；结合会员折扣、促销政策判断，防止低价开票隐匿收入。"
            f"后果测算（推定口径）：按区域均价推定，申报销量对应差额收入约 {income_gap:.0f} 万元"
            "——推算仅为核查线索，不构成应补税金额，以实际核查结果为准。",
        )
    return _rule("R803", "申报单价偏离区域均价", False, "中", f"单价偏离 {diff:.0f}%（阈值10%）", "")


def check_r24(profile):
    """R901 集群注册同址：同一地址关联企业/个体户>=5户。"""
    count = _num(profile.get("关联户数"))
    if count is None:
        return _rule("R901", "集群注册同址", False, "低", "关联户数缺失", "")
    if count >= 5:
        return _rule(
            "R901", "集群注册同址", True, "低",
            f"同一注册地址关联 {count:.0f} 户企业/个体户",
            "单户看都合法、跨户汇总才暴露拆分：按关联户/关联人组织核查。",
        )
    return _rule("R901", "集群注册同址", False, "低", f"关联户数 {count:.0f}（阈值5户）", "")


def check_r903(profile):
    """R903 关联户贴线：同一控制人关联户中多户指标位于优惠门槛临界区间。"""
    related = _num(profile.get("关联户数"))
    near = _num(profile.get("关联户贴线数"))
    if near is None or near == 0:
        return _rule("R903", "关联户贴线", False, "中", "关联户贴线数缺失或为0", "")
    if near >= 2:
        return _rule(
            "R903", "关联户贴线", True, "中",
            f"同一控制人关联 {related:.0f} 户，其中 {near:.0f} 户指标位于优惠门槛临界区间",
            "单户看都合法、跨户汇总才暴露拆分：核查各户业务实质、人员、场地与资金是否独立。",
        )
    return _rule("R903", "关联户贴线", False, "中", f"关联户贴线数 {near:.0f}（阈值2户）", "")


def check_r27(invoices):
    """R902 单一服务类品目集中：咨询费/服务费/管理费占销项金额>80%。"""
    sales = _sales_invoices(invoices)
    if sales.empty:
        return _rule("R902", "单一服务类品目集中", False, "中", "无销项发票数据", "")
    total = sales["金额(元)"].apply(_num).fillna(0).sum()
    service = sales[sales["品名"].astype(str).str.contains("|".join(SERVICE_ITEMS), na=False)]["金额(元)"].apply(_num).fillna(0).sum()
    if total and service / total > 0.8:
        return _rule(
            "R902", "单一服务类品目集中", True, "中",
            f"无实物品目（咨询/服务/管理费）占销项 {service/total*100:.0f}%",
            "服务费/咨询费虚开只能靠实质资料验证：人员考勤、服务记录、交付物、合同、付款凭证、物流。",
        )
    return _rule("R902", "单一服务类品目集中", False, "中", "服务类品目占比未超80%", "")


def check_r30(profile):
    """R605 资格-优惠不匹配：非高企却享受增值税加计抵减。"""
    htech = str(profile.get("是否高新技术企业", ""))
    deduction = str(profile.get("是否享受加计抵减", ""))
    if htech == "否" and deduction == "是":
        return _rule(
            "R605", "资格-优惠不匹配", True, "高",
            "非高新技术企业（或资格存疑）却享受增值税加计抵减",
            "自查高企资格与加计抵减适用名单是否匹配；若不满足，主动停止享受并更正申报，避免被追缴。",
        )
    return _rule("R605", "资格-优惠不匹配", False, "高", "资格与优惠享受状态匹配", "")


def check_r606(profile):
    """R606 高新研发费用率贴线提示（提示级）：税务机关不作资格判定，仅就优惠享受前提提示问询。"""
    htech = str(profile.get("是否高新技术企业", ""))
    if htech != "是":
        return _rule("R606", "高新研发费用率贴线", False, "低", "非高新技术企业", "")
    rd = _num(profile.get("研发费用(万元)"))
    revenue = _num(profile.get("营业收入(万元)"))
    if rd is None or revenue is None or revenue <= 0:
        return _rule("R606", "高新研发费用率贴线", False, "低", "研发费用或收入数据缺失", "")
    ratio = rd / revenue * 100
    if revenue < 5000:
        threshold, tier = 5.0, "收入5,000万元以下"
    elif revenue <= 20000:
        threshold, tier = 4.0, "收入5,000万~2亿元"
    else:
        threshold, tier = 3.0, "收入2亿元以上"
    if ratio < threshold:
        status = f"低于认定档位要求（{tier}档要求≥{threshold:.0f}%）"
    elif ratio <= threshold * 1.25:
        status = f"贴近认定档位门槛（{tier}档要求≥{threshold:.0f}%）"
    else:
        return _rule("R606", "高新研发费用率贴线", False, "低", f"研发费用率 {ratio:.1f}%，高于档位要求", "")
    evidence = f"研发费用 {rd:.0f} 万元占收入 {ratio:.1f}%，{status}"
    suggestion = (
        "提示：高新技术企业资格由科技部门认定，税务机关不掌握认定过程、不作资格判定；"
        "本条仅为确认优惠享受前提的问询线索——建议核实研发费用归集口径与留存备查资料，"
        "资格有效性与比例口径以认定机构为准。"
    )
    result = _rule("R606", "高新研发费用率贴线", True, "低", evidence, suggestion)
    result["informational"] = True  # 提示级问询线索：展示但不进风险评分与统计
    return result


def check_r31(profile, invoices):
    """R701 数据完整性异常：关键字段缺失，或加油站液位仪缺失率>5%。"""
    assets = _num(profile.get("资产总额(万元)"))
    missing_rate = _fuel_num(profile, "液位仪月缺失率(%)")
    if assets is None:
        return _rule(
            "R701", "数据完整性异常", True, "中",
            "企业指标中资产总额缺失",
            "不猜测、不补数：标注'需补充后判断'；数据缺失本身就是风险信号。",
        )
    if missing_rate is not None and missing_rate > 5:
        return _rule(
            "R701", "数据完整性异常", True, "中",
            f"液位仪月记录缺失率 {missing_rate:.0f}%（阈值5%）",
            "检查记录空缺断点/删除痕迹/留存周期；用供应商进油单等外部单据交叉验证。",
        )
    return _rule("R701", "数据完整性异常", False, "中", "关键数据完整", "")


def check_r32(profile):
    """R702 上游发票缺失（加油站模板）：无票采购占比>30%。"""
    ratio = _fuel_num(profile, "无票采购占比(%)")
    if ratio is None:
        return _rule("R702", "上游发票缺失", False, "中", "无票采购占比缺失", "")
    if ratio > 30:
        level = "中"
        return _rule(
            "R702", "上游发票缺失", True, level,
            f"无票采购占比 {ratio:.0f}%（演示阈值30%）",
            "无票采购（尤其私人炼油厂）导致'以进控销'的进端不完整；生产环境需罐车/油罐硬件计量与多源数据补位。",
        )
    return _rule("R702", "上游发票缺失", False, "中", f"无票采购占比 {ratio:.0f}%（阈值30%）", "")


def check_r33(profile, external_docs=None):
    """R703 外部单据与本地记录不一致：第三方单据存在但本地记录缺失或对不上。"""
    doc = str(profile.get("上游进油单存在", ""))
    missing = str(profile.get("本地入库记录缺失", ""))
    ext_count = 0
    suppliers = ""
    if external_docs is not None and not external_docs.empty:
        ext_count = len(external_docs)
        suppliers = "、".join(str(x) for x in external_docs["供应商"].dropna().unique()[:3])
    if ext_count and missing == "是":
        return _rule(
            "R703", "外部单据与本地记录不一致", True, "高",
            f"外部单据 {ext_count} 笔（{suppliers}）存在，本地入库/液位仪记录缺失",
            "用供应商单据反向核对本地入库记录，找出缺失或对不上的原因，及时补录并留存说明。",
        )
    if doc == "是" and missing == "是":
        return _rule(
            "R703", "外部单据与本地记录不一致", True, "高",
            "供应商进油单存在，但本地入库/液位仪记录缺失",
            "用供应商单据反向核对本地入库记录，找出缺失或对不上的原因，及时补录并留存说明。",
        )
    return _rule("R703", "外部单据与本地记录不一致", False, "高", "外部单据与本地记录匹配", "")


def check_r34(profile, data_sources=None):
    """R704 数据源可信度（两步判定）：
    第一步比对一致性由 R802 等规则完成，不一致即提示风险；
    第二步评估"一致"的可信度：必须结合来源方与是否可篡改——
    不可篡改的独立来源（第三方/监管机构且链路难篡改）才算可信；
    "内部自洽"仅指全部数据来自企业内部；含外部来源但可篡改，属低可信而非内部自洽。"""
    chip = str(profile.get("加油机税控芯片标准", ""))
    rows = []
    if data_sources is not None and not data_sources.empty:
        rows = data_sources.to_dict("records")
    internal = [r for r in rows if str(r.get("来源方", "")) in ("企业自报", "企业内部")]
    external = [r for r in rows if str(r.get("来源方", "")) in ("第三方", "监管机构")]
    tamperable = [r for r in rows if str(r.get("可否篡改", "")) in ("可", "可破解", "是")]
    trusted = [
        r for r in rows
        if str(r.get("来源方", "")) in ("第三方", "监管机构")
        and str(r.get("可否篡改", "")) not in ("可", "可破解", "是")
    ]
    if rows and internal and not external and tamperable:
        return _rule(
            "R704", "数据源可信度", True, "中",
            f"一致数据全部来自企业内部（{len(internal)} 个数据源）且均可被篡改，内部自洽不等于真实",
            "主动用供应商单据、运单等外部来源核对自身数据，确保申报口径可验证。",
        )
    if rows and tamperable and not trusted:
        return _rule(
            "R704", "数据源可信度", True, "中",
            f"一致数据含监管/第三方来源（{len(external)} 个）但均可被篡改，可信度低",
            "梳理各数据源可信度，对可篡改来源加强留存与校验，关键数据用不可篡改的外部来源佐证。",
        )
    if chip == "旧标准":
        return _rule(
            "R704", "数据源可信度", True, "中",
            "加油机税控芯片为旧标准，强制安装但可被破解，上报商务局联网系统前即可被修改，设备数据可信度降级",
            "评估设备数据可信度，保留原始记录；关键数据用供应商单据、运单等外部来源核对。",
        )
    return _rule("R704", "数据源可信度", False, "中", "一致数据中包含不可篡改的独立来源，可信度达标", "")


def check_r35(profile):
    """R604 收入与应税所得严重不匹配：收入>5000万且收入利润率<3%。"""
    revenue = _num(profile.get("营业收入(万元)"))
    taxable = _num(profile.get("应纳税所得额(万元)"))
    if revenue is None or taxable is None or revenue == 0:
        return _rule("R604", "收入与应税所得严重不匹配", False, "中", "收入或应税所得缺失", "")
    ratio = taxable / revenue * 100
    if revenue > 5000 and ratio < 3:
        level = "中"
        return _rule(
            "R604", "收入与应税所得严重不匹配", True, level,
            f"营业收入 {revenue:,.0f}万，应税所得 {taxable:,.1f}万，收入利润率 {ratio:.1f}%（阈值3%）",
            "收入规模大但利润极薄：核查成本归集、研发加计、税前扣除是否合规；小微临界+大额调减组合是常见预警场景，需准备备查资料。",
        )
    return _rule("R604", "收入与应税所得严重不匹配", False, "中", f"收入利润率 {ratio:.1f}%（阈值3%）", "")


def check_r36(invoices):
    """R804 收购发票对象身份存疑：收购发票销售方非自产农业生产者。"""
    agri = invoices[invoices["发票类型"] == "农产品收购发票"]
    if agri.empty:
        return _rule("R804", "收购发票对象身份存疑", False, "高", "无农产品收购发票", "")
    suspects = agri[agri.get("销售方类型", pd.Series(dtype=str)).fillna("") != "农业生产者"]
    if not suspects.empty:
        names = "、".join(str(x) for x in suspects["对方名称"].dropna().unique()[:5])
        total = float(suspects["金额(元)"].apply(_num).sum())
        evidence = (
            f"{len(suspects)} 张收购发票销售方身份存疑（非自产农业生产者），对象：{names}，"
            f"涉及金额 {total/10000:.0f} 万元。"
        )
        suggestion = (
            f"后果测算：若已申报抵扣，进项税额转出约 {total*0.09/10000:.0f} 万元"
            "（按9%计算抵扣口径；如用于生产13%税率货物按10%扣除率约 "
            f"{total*0.10/10000:.0f} 万元）；"
            "核验清单：自产证明、收购现场记录、物流、付款到农户本人；"
            "纠正路径：可要求贩子到办税服务大厅代开增值税发票补正。"
        )
        return _rule(
            "R804", "收购发票对象身份存疑", True, "高",
            evidence,
            suggestion,
        )
    return _rule("R804", "收购发票对象身份存疑", False, "高", "收购对象均为农业生产者", "")


def check_r37(invoices):
    """R805 单户收购金额异常：单户>500万或Top3占>70%。"""
    agri = invoices[invoices["发票类型"] == "农产品收购发票"]
    if agri.empty:
        return _rule("R805", "单户收购金额异常", False, "中", "无农产品收购发票", "")
    amounts = agri.copy()
    amounts["金额(元)"] = amounts["金额(元)"].apply(_num)
    by_name = amounts.groupby("对方名称")["金额(元)"].sum()
    big = by_name[by_name > 5000000]
    if not big.empty:
        detail = "；".join(f"{k} {v/10000:.0f}万" for k, v in big.items())
        return _rule(
            "R805", "单户收购金额异常", True, "中",
            f"单户收购金额超500万：{detail}",
            "收购大户核实生产能力与自产证据：单户金额巨大且身份存疑时风险升级。",
        )
    return _rule("R805", "单户收购金额异常", False, "中", "单户收购金额≤500万", "")


def check_r38(invoices, fund_flows):
    """R806 收购发票与业务流不匹配：发票对象与资金付款对象不一致。"""
    agri = invoices[invoices["发票类型"] == "农产品收购发票"]
    if agri.empty or fund_flows.empty:
        return _rule("R806", "收购发票与业务流不匹配", False, "高", "无收购发票或资金流水", "")
    flow_names = {str(x).strip() for x in fund_flows["对方名称"].dropna()}
    mismatches = []
    for _, inv in agri.iterrows():
        name = str(inv.get("对方名称", "")).strip()
        if name and name not in flow_names:
            mismatches.append(name)
    if mismatches:
        return _rule(
            "R806", "收购发票与业务流不匹配", True, "高",
            f"收购发票对象与付款对象不一致：{'、'.join(mismatches[:5])}",
            "发票流写企业←农户，资金流却对应贩子：存在中间商，应取得贩子开具的增值税发票（可要求到办税大厅代开补正）——三流不一致的农产品版。",
        )
    return _rule("R806", "收购发票与业务流不匹配", False, "高", "收购发票对象与付款对象一致", "")


# 规则分类与编号：R + 三位编码 = 第一位业务大类 + 后两位该类序号（R=Rule，单点规则）。
# 只登记已实现的规则；新增规则按所属大类顺延序号，不设预留位。
RULE_CATEGORIES = {
    "发票与开票": ["R101", "R102", "R103", "R104", "R105"],
    "资金与结算": ["R201", "R202", "R203", "R204"],
    "上下游与供应链": ["R301"],
    "申报与财务": ["R401", "R402"],
    "人资与信用": ["R501", "R502"],
    "资格与优惠": ["R601", "R602", "R603", "R604", "R605", "R606"],
    "数据质量与完整性": ["R701", "R702", "R703", "R704"],
    "行业模板": ["R801", "R802", "R803", "R804", "R805", "R806"],
    "关联与团伙": ["R901", "R902", "R903"],
}
CATEGORY_ORDER = list(RULE_CATEGORIES)
CATEGORY_OF = {rid: cat for cat, ids in RULE_CATEGORIES.items() for rid in ids}

# 组合规则（G 系列）：由原子规则推导，与原子规则分开编号、分开执行（G=Group，规则组）。
# G001 起编号，新组合按 G002、G003… 追加。
def _make_combo_check(combo_id, name, level, depends, min_hits, suggestion):
    """组合规则检查工厂：构成规则命中达到 min_hits 即命中。"""
    def check(p, hits):
        ids = {h.get("rule_id") for h in hits}
        hit = [d for d in depends if d in ids]
        if len(hit) >= min_hits:
            return _rule(
                combo_id, name, True, level,
                f"构成规则 {'、'.join(hit)} 同时命中",
                suggestion,
            )
        return _rule(
            combo_id, name, False, level,
            f"构成规则命中 {len(hit)}/{len(depends)}，不足 {min_hits} 条",
            "",
        )
    return check


COMBO_RULES = {
    "G001": {
        "name": "临界调减联动",
        "level": "中",
        "depends_on": ["R601", "R602"],
        "label": "R601 小微临界 + R602 大额调减",
        "check": check_r19,
    },
    "G002": {
        "name": "拆分享受优惠联动",
        "level": "中",
        "depends_on": ["R901", "R903"],
        "label": "R901 集群注册同址 + R903 关联户贴线",
        "check": _make_combo_check(
            "G002", "拆分享受优惠联动", "中", ["R901", "R903"], 2,
            "梳理关联户业务实质与独立性，确保各户独立经营、资料完整，避免被认定拆分。",
        ),
    },
    "G003": {
        "name": "虚开风险联动",
        "level": "高",
        "depends_on": ["R104", "R201", "R203"],
        "label": "R104 进销项不匹配 + R201 三流不一致 + R203 资金回流",
        "check": _make_combo_check(
            "G003", "虚开风险联动", "高", ["R104", "R201", "R203"], 2,
            "自查业务真实性：补齐合同、物流、交付物与资金流向资料，避免被认定为虚开。",
        ),
    },
    "G004": {
        "name": "特殊凭证对象联动",
        "level": "高",
        "depends_on": ["R804", "R805", "R806"],
        "label": "R804 凭证对象身份存疑 + R805 单户金额异常 + R806 业务流不匹配",
        "check": _make_combo_check(
            "G004", "特殊凭证对象联动", "高", ["R804", "R805", "R806"], 2,
            "特殊凭证对象与限额合规风险：输出身份与业务流核验清单，必要时要求代开补正。",
        ),
    },
    "G005": {
        "name": "申报数据可信度联动",
        "level": "高",
        "depends_on": ["R701", "R702", "R703", "R704"],
        "label": "R701 数据完整性 + R702 上游发票缺失 + R703 外部单据比对 + R704 数据源可信度",
        "check": _make_combo_check(
            "G005", "申报数据可信度联动", "高", ["R701", "R702", "R703", "R704"], 2,
            "主动用外部独立来源核对申报数据，完善数据留存，确保申报口径可验证。",
        ),
    },
}

# 规则注册表（未排序），执行顺序按分类分组、组内按编号
_RULES = [
    ("R101", "顶额开票", "低", check_r01),
    ("R102", "月末集中开票", "低", check_r02),
    ("R103", "红冲/作废率过高", "中", check_r03),
    ("R104", "进销项品名不匹配", "高", check_r04),
    ("R105", "红冲金额超过原票", "中", check_r105),
    ("R201", "三流不一致", "中", check_r06),
    ("R202", "个人账户收付款占比高", "中", check_r07),
    ("R203", "资金回流", "中", check_r203),
    ("R204", "收款无销项覆盖", "中", check_r204),
    ("R301", "风险企业传导", "中", check_r09),
    ("R401", "增值税税负率异常", "中", check_r12),
    ("R501", "个税与社保人数不一致", "低", check_r15),
    ("R502", "纳税信用等级低", "中", check_r16),
    ("R601", "小微临界", "低", check_r17),
    ("R402", "利润与申报应纳税所得额差异过大", "中", check_r18),
    ("R603", "研发加计扣除占比异常", "中", check_r20),
    ("R602", "大额调减项", "低", check_r39),
    ("R801", "单站销售横向偏离", "中", check_r21),
    ("R802", "多源数据不一致", "高", check_r22),
    ("R803", "申报单价偏离区域均价", "中", check_r23),
    ("R901", "集群注册同址", "低", check_r24),
    ("R902", "单一服务类品目集中", "中", check_r27),
    ("R903", "关联户贴线", "中", check_r903),
    ("R605", "资格-优惠不匹配", "高", check_r30),
    ("R606", "高新研发费用率贴线", "低", check_r606),
    ("R701", "数据完整性异常", "中", check_r31),
    ("R702", "上游发票缺失", "中", check_r32),
    ("R703", "外部单据与本地记录不一致", "高", check_r33),
    ("R704", "数据源可信度", "中", check_r34),
    ("R604", "收入与应税所得严重不匹配", "中", check_r35),
    ("R804", "收购发票对象身份存疑", "高", check_r36),
    ("R805", "单户收购金额异常", "中", check_r37),
    ("R806", "收购发票与业务流不匹配", "高", check_r38),
]

RULE_CHECKS = sorted(
    _RULES,
    key=lambda r: (CATEGORY_ORDER.index(CATEGORY_OF.get(r[0], "其他")), int(r[0][1:])),
)


def run_all(profile, invoices, fund_flows=None, contracts=None, external_docs=None, data_sources=None):
    """执行全部规则，返回命中列表。"""
    fund_flows = fund_flows if fund_flows is not None else pd.DataFrame(
        columns=["流水日期", "收付方向", "对方名称", "金额(元)", "备注", "关联合同号", "账户类型"]
    )
    contracts = contracts if contracts is not None else pd.DataFrame(columns=["合同号", "对方名称", "金额(元)", "签订日期"])
    external_docs = external_docs if external_docs is not None else pd.DataFrame(columns=EXTERNAL_DOC_COLS)
    data_sources = data_sources if data_sources is not None else pd.DataFrame(columns=DATA_SOURCE_COLS)
    p = profile.iloc[0].to_dict()
    hits = []
    for rule_id, name, default_level, fn in RULE_CHECKS:
        try:
            if rule_id == "R201":
                result = fn(invoices, fund_flows, contracts)
            elif rule_id == "R806":
                result = fn(invoices, fund_flows)
            elif rule_id == "R701":
                result = fn(p, invoices)
            elif rule_id == "R703":
                result = fn(p, external_docs)
            elif rule_id == "R704":
                result = fn(p, data_sources)
            elif rule_id == "R401":
                result = fn(invoices, p)
            elif rule_id == "R202":
                result = fn(fund_flows)
            elif rule_id == "R203":
                result = fn(fund_flows)
            elif rule_id == "R204":
                result = fn(invoices, fund_flows)
            elif rule_id in ("R101", "R102", "R103", "R104", "R105", "R301", "R902", "R804", "R805"):
                result = fn(invoices)
            else:
                result = fn(p)
        except Exception as exc:  # 单条规则失败不阻断整体
            result = _rule(rule_id, name, False, default_level, f"规则执行异常：{exc}", "")
        if result.get("hit"):
            result["category"] = CATEGORY_OF.get(rule_id, "其他")
            hits.append(result)
    # 组合规则：独立于原子规则执行，单独编号（G001…）
    for combo_id, meta in COMBO_RULES.items():
        try:
            result = meta["check"](p, hits)
        except Exception as exc:  # noqa: BLE001
            result = _rule(combo_id, meta["name"], False, meta["level"], f"规则执行异常：{exc}", "")
        if result.get("hit"):
            result["rule_id"] = combo_id
            result["combo_id"] = combo_id
            result["name"] = meta["name"]
            result["kind"] = "combo"
            result["depends_on"] = meta["depends_on"]
            result["category"] = "组合规则"
            hits.append(result)
    return _merge_dependent_rules(hits)


def _merge_dependent_rules(hits):
    """组合规则合并：组合规则命中时，把其原子依赖并入组合结果，避免重复计分。"""
    by_id = {h["rule_id"]: h for h in hits}
    for rid, meta in COMBO_RULES.items():
        combo = by_id.get(rid)
        if not combo:
            continue
        merged = []
        for dep in meta["depends_on"]:
            sub = by_id.get(dep)
            if sub:
                sub["merged_into"] = rid
                combo["evidence"] = f"{combo['evidence']}；{dep}项：{sub['evidence']}"
                merged.append(dep)
        if merged:
            combo["suggestion"] = (
                f"{combo['suggestion']}（已合并 {'、'.join(merged)} 提示，不重复计分）"
            )
    return hits


if __name__ == "__main__":
    import generate_data

    for name in generate_data.SCENARIOS:
        data = generate_data.build_scenario(name)
        hits = run_all(data["company_profile"], data["invoices"], data["fund_flows"], data["contracts"])
        print(f"\n[{name}] 命中 {len(hits)} 条")
        for h in hits:
            print(f"  {h['rule_id']} [{h['level']}] {h['name']}: {h['evidence']}")
