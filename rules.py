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
    """R01 顶额开票：单张接近限额（演示：>=90000 元）且同月>=3张。"""
    sales = _sales_invoices(invoices)
    if sales.empty:
        return _rule("R01", "顶额开票", False, "高", "无销项发票数据", "")
    sales = sales.copy()
    sales["金额(元)"] = sales["金额(元)"].apply(_num)
    # 顶额指接近开票限额（演示：万元版发票限额 99999 元，取 90000-101000 区间）
    top = sales[sales["金额(元)"].notna() & (sales["金额(元)"] >= 90000) & (sales["金额(元)"] <= 101000)]
    if top.empty:
        return _rule("R01", "顶额开票", False, "高", f"无顶额发票（共{len(sales)}张）", "")
    by_month = top["开票日期"].str[:7].value_counts()
    hit_month = by_month[by_month >= 3]
    if hit_month.empty:
        return _rule("R01", "顶额开票", False, "高", f"顶额票{len(top)}张，但单月不足3张", "")
    m = hit_month.index[0]
    return _rule(
        "R01", "顶额开票", True, "高",
        f"{m} 月顶额发票 {int(hit_month.iloc[0])} 张（金额≥90000元，接近开票限额99999元）",
        "核对开票业务实质：是否拆票规避限额/拆分客户；查验合同、物流、付款对应关系。",
    )


def check_r02(invoices):
    """R02 月末集中开票：当月发票集中在最后5个自然日，占比>60%。"""
    sales = _sales_invoices(invoices)
    if sales.empty:
        return _rule("R02", "月末集中开票", False, "中", "无销项发票数据", "")
    sales = sales.copy()
    sales["金额(元)"] = sales["金额(元)"].apply(_num).fillna(0)
    for month, group in sales.groupby(sales["开票日期"].str[:7]):
        days = group["开票日期"].str[-2:].astype(int)
        # 演示按 31 天月估算月末窗口
        last5 = group[days >= 27]["金额(元)"].sum()
        total = group["金额(元)"].sum()
        if total > 0 and last5 / total > 0.6:
            return _rule(
                "R02", "月末集中开票", True, "中",
                f"{month} 月最后5个自然日开票占比 {last5/total*100:.1f}%（演示阈值>60%）",
                "核实集中开票的业务原因（合同结算节奏/业绩压力），关注是否存在跨期调节收入。",
            )
    return _rule("R02", "月末集中开票", False, "中", "月末集中度未超演示阈值", "")


def check_r03(invoices):
    """R03 红冲/作废率过高：红冲+作废金额或张数占当期开票比例>20%。"""
    sales = _sales_invoices(invoices)
    red = invoices[invoices["发票类型"].isin(["红字发票", "作废发票"])]
    if sales.empty:
        return _rule("R03", "红冲/作废率过高", False, "中", "无销项发票数据", "")
    sales_amount = sales["金额(元)"].apply(_num).fillna(0).sum()
    red_amount = red["金额(元)"].apply(_num).fillna(0).sum()
    count_ratio = len(red) / len(sales) * 100
    amount_ratio = _pct(red_amount, sales_amount) or 0
    if count_ratio > 20 or amount_ratio > 20:
        return _rule(
            "R03", "红冲/作废率过高", True, "中",
            f"红冲/作废 {len(red)} 张 vs 销项 {len(sales)} 张（张数占比{count_ratio:.1f}%，金额占比{amount_ratio:.1f}%）",
            "逐张核对红冲/作废原因（开票错误/退货/走逃作废），防止利用红冲调节销项、隐匿收入。",
        )
    return _rule("R03", "红冲/作废率过高", False, "中", f"红冲/作废占比 {count_ratio:.1f}%（阈值20%）", "")


def check_r04(invoices):
    """R04 进销项品名不匹配：销项为技术类，进项却全是无关消费类品名。"""
    sales = _sales_invoices(invoices)
    inputs = _input_invoices(invoices)
    if sales.empty or inputs.empty:
        return _rule("R04", "进销项品名不匹配", False, "高", "进项或销项数据缺失", "")
    sale_items = {str(x).strip() for x in sales["品名"].dropna()}
    input_items = {str(x).strip() for x in inputs["品名"].dropna()}
    sale_is_tech = any(k in "".join(sale_items) for k in TECH_SALE_KEYWORDS)
    inputs_unrelated = all(any(k in item for k in UNRELATED_INPUT_KEYWORDS) for item in input_items)
    if sale_is_tech and inputs_unrelated and not (sale_items & input_items):
        return _rule(
            "R04", "进销项品名不匹配", True, "高",
            f"销项品名 {sorted(sale_items)}，进项品名 {sorted(input_items)}，语义无关",
            "核对进项业务真实性：技术类企业大量餐饮/娱乐进项，涉嫌虚增进项、套取进项抵扣。",
        )
    return _rule("R04", "进销项品名不匹配", False, "高", "进销项品名存在业务关联", "")


def check_r06(invoices, fund_flows, contracts):
    """R06 三流不一致：发票、资金、合同按合同号/对方名称比对不一致。"""
    if invoices.empty or fund_flows.empty:
        return _rule("R06", "三流不一致", False, "高", "发票或资金数据缺失", "")
    if "收付方向" not in fund_flows.columns:
        return _rule("R06", "三流不一致", False, "高", "资金流水缺少收付方向", "")
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
            "R06", "三流不一致", True, "高",
            "；".join(mismatches[:5]) + (f"；个人账户收款 {len(personal)} 笔" if personal else ""),
            "逐笔核对合同、发票、资金流三流对应关系；三流不一致是虚开/偷税核查重点。",
        )
    if personal:
        return _rule(
            "R06", "三流不一致", True, "高",
            f"对公业务通过个人账户收付（合同号：{'、'.join(personal[:5])}）",
            "逐笔核对合同、发票、资金流三流对应关系；个人账户收付对公货款是隐匿收入/资金回流常见通道。",
        )
    return _rule("R06", "三流不一致", False, "高", "发票、资金、合同三流一致", "")


def check_r07(fund_flows):
    """R07 个人账户收付款占比高：对公业务通过个人账户收付款金额占比>10%。"""
    if fund_flows.empty or "账户类型" not in fund_flows.columns:
        return _rule("R07", "个人账户收付款占比高", False, "高", "无资金流水数据", "")
    flows = fund_flows.copy()
    flows["金额(元)"] = flows["金额(元)"].apply(_num)
    total = flows["金额(元)"].sum()
    personal = flows[flows["账户类型"] == "个人账户"]["金额(元)"].sum()
    if total and personal / total > 0.1:
        return _rule(
            "R07", "个人账户收付款占比高", True, "高",
            f"个人账户收付款 {personal:,.0f} 元，占总流水 {personal/total*100:.1f}%（阈值10%）",
            "核实个人账户资金对应的真实业务：公私混用、隐匿收入或账外资金回流。",
        )
    return _rule("R07", "个人账户收付款占比高", False, "高", f"个人账户占比 {_pct(personal, total) or 0:.1f}%（阈值10%）", "")


def check_r09(invoices):
    """R09 风险企业传导：上游/下游存在非正常户、走逃失联。"""
    if invoices.empty or "对方状态" not in invoices.columns:
        return _rule("R09", "风险企业传导", False, "高", "无发票数据", "")
    bad = invoices[invoices["对方状态"].isin(ABNORMAL_STATUS)]
    if bad.empty:
        return _rule("R09", "风险企业传导", False, "高", "无异常状态上下游企业", "")
    names = "、".join(str(x) for x in bad["对方名称"].dropna().unique()[:5])
    return _rule(
        "R09", "风险企业传导", True, "高",
        f"存在{len(bad)}张异常状态发票，对方：{names}",
        "异常凭证处理路径：未抵扣的暂不允许抵扣，已抵扣的先作进项转出；"
        "A级信用纳税人可在10个工作日内申请核实，核实通过可不转出。",
    )


def check_r12(invoices, profile):
    """R12 增值税税负率异常：税负率低于行业参考区间下限。"""
    sales = _sales_invoices(invoices)
    inputs = _input_invoices(invoices)
    if sales.empty or inputs.empty:
        return _rule("R12", "增值税税负率异常", False, "中", "发票数据不足，无法计算税负率", "")
    output_tax = sales["税额(元)"].apply(_num).fillna(0).sum()
    input_tax = inputs["税额(元)"].apply(_num).fillna(0).sum()
    sales_amt = sales["金额(元)"].apply(_num).fillna(0).sum()
    if not sales_amt:
        return _rule("R12", "增值税税负率异常", False, "中", "销项金额为0", "")
    burden = (output_tax - input_tax) / sales_amt * 100
    industry = str(profile.get("行业", ""))
    lower = INDUSTRY_VAT_LOWER.get(industry, 1.0)
    if burden < lower:
        level = "高" if burden < lower / 2 else "中"
        return _rule(
            "R12", "增值税税负率异常", True, level,
            f"增值税税负率 {burden:.1f}%，行业[{industry}]参考下限 {lower}%（演示值）",
            "核对进项抵扣与申报口径：是否存在虚增进项、应转出未转出；税负率明显偏低是预警常见特征。",
        )
    return _rule("R12", "增值税税负率异常", False, "中", f"税负率 {burden:.1f}%（行业参考下限 {lower}%）", "")


def check_r15(profile):
    """R15 个税与社保人数不一致：差异>20%。"""
    declare = _num(profile.get("个税申报人数"))
    social = _num(profile.get("社保参保人数"))
    if declare is None or social is None:
        return _rule("R15", "个税与社保人数不一致", False, "中", "个税或社保人数缺失", "")
    if declare <= 0:
        return _rule("R15", "个税与社保人数不一致", False, "中", "个税申报人数为0", "")
    diff = abs(declare - social) / declare * 100
    if diff > 20:
        level = "高" if diff > 50 else "中"
        return _rule(
            "R15", "个税与社保人数不一致", True, level,
            f"个税申报 {declare:.0f} 人 vs 社保参保 {social:.0f} 人，差异 {diff:.0f}%",
            "核实用工形式（临时工/劳务外包/未参保人员），防范未足额参保、隐匿用工。",
        )
    return _rule("R15", "个税与社保人数不一致", False, "中", f"人数差异 {diff:.0f}%（阈值20%）", "")


def check_r16(profile):
    """R16 纳税信用等级低：D 级预警。"""
    level = str(profile.get("纳税信用等级", "")).strip().upper()
    if level == "D":
        return _rule(
            "R16", "纳税信用等级低", True, "高",
            "纳税信用等级为 D 级",
            "D级纳税人将受到发票领用、出口退税、融资授信等限制；核查失信原因并整改。",
        )
    return _rule("R16", "纳税信用等级低", False, "高", f"纳税信用等级 {level or '未知'}", "")


def check_r17(profile):
    """R17 小微临界：人数/资产/所得额位于限额 90%-100%。"""
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
            "R17", "小微临界", True, "低",
            "；".join(near) + " 均位于小型微利企业限额的90%-100%区间",
            "临界企业是常见预警对象：核实申报准确性；若同时存在大额调减项，需准备备查资料。"
            "若符合小型微利企业条件，可叠加六税两费减半等优惠。",
        )
    return _rule("R17", "小微临界", False, "低", "未处于小微限额临界区间", "")


def check_r18(profile):
    """R18 利润与申报应纳税所得额差异过大：差异率>50%。"""
    profit = _num(profile.get("利润总额(万元)"))
    taxable = _num(profile.get("应纳税所得额(万元)"))
    if profit is None or taxable is None or taxable == 0:
        return _rule("R18", "利润与申报应纳税所得额差异过大", False, "中", "利润总额或应纳税所得额缺失", "")
    diff = abs(profit - taxable) / abs(taxable) * 100
    if diff > 50:
        level = "高" if diff > 100 else "中"
        return _rule(
            "R18", "利润与申报应纳税所得额差异过大", True, level,
            f"会计利润 {profit:.1f}万 vs 申报应纳税所得额 {taxable:.1f}万，差异率 {diff:.0f}%",
            "核对纳税调增调减项目及备查资料：差异过大需解释税会差异来源。",
        )
    return _rule("R18", "利润与申报应纳税所得额差异过大", False, "中", f"税会差异率 {diff:.0f}%（阈值50%）", "")


def check_r19(profile):
    """R19 临界点聚集：所得额落于小微限额85%-100%且存在大额调减项（演示：研发费用>0）。"""
    taxable = _num(profile.get("应纳税所得额(万元)"))
    rd = _num(profile.get("研发费用(万元)"))
    if taxable is None or rd is None:
        return _rule("R19", "临界点聚集", False, "中", "应纳税所得额或研发费用缺失", "")
    if 255 <= taxable <= 300 and rd > 0:
        return _rule(
            "R19", "临界点聚集", True, "中",
            f"应纳税所得额 {taxable:.1f}万（位于300万限额的85%-100%），且研发费用 {rd:.0f}万（大额调减项）",
            "临界点聚集（拆户/调减规避）是团伙虚开与偷逃税核查重点；备查研发立项、费用归集、辅助账。",
        )
    return _rule("R19", "临界点聚集", False, "中", "未同时命中临界区间与大额调减", "")


def check_r20(profile):
    """R20 研发加计扣除占比异常：研发费用占收入比例>15%。"""
    rd = _num(profile.get("研发费用(万元)"))
    revenue = _num(profile.get("营业收入(万元)"))
    if rd is None or revenue is None or revenue == 0:
        return _rule("R20", "研发加计扣除占比异常", False, "中", "研发费用或营业收入缺失", "")
    ratio = rd / revenue * 100
    if ratio > 15:
        return _rule(
            "R20", "研发加计扣除占比异常", True, "中",
            f"研发费用占收入 {ratio:.1f}%（演示阈值15%）",
            "研发费用畸高需核查研发真实性：立项、人员、工时、材料领用、辅助账。",
        )
    return _rule("R20", "研发加计扣除占比异常", False, "中", f"研发占收入 {ratio:.1f}%（阈值15%）", "")


def _fuel_num(profile, field):
    return _num(profile.get(field))


def check_r21(profile):
    """R21 单站销售横向偏离（加油站模板）：申报销量显著偏离区域参考。"""
    reported = _fuel_num(profile, "申报销量(吨)")
    reference = _fuel_num(profile, "区域参考销量(吨)")
    if reported is None or reference is None:
        return _rule("R21", "单站销售横向偏离", False, "中", "加油站模板字段缺失", "")
    if reference == 0:
        return _rule("R21", "单站销售横向偏离", False, "中", "区域参考销量为0", "")
    diff = (reported - reference) / reference * 100
    if abs(diff) > 50:
        level = "高" if abs(diff) > 100 else "中"
        return _rule(
            "R21", "单站销售横向偏离", True, level,
            f"申报销量 {reported:.0f}吨 vs 区域参考 {reference:.0f}吨，偏离 {diff:.0f}%",
            "横向比对同地段/车流条件站点；偏离过大需核实是否存在少申报，建议多源交叉验证。",
        )
    return _rule("R21", "单站销售横向偏离", False, "中", f"横向偏离 {diff:.0f}%（阈值±50%）", "")


def check_r22(profile):
    """R22 多源数据不一致（加油站模板）：申报/设备/测算三源两两差异>20%。"""
    reported = _fuel_num(profile, "申报销量(吨)")
    device = _fuel_num(profile, "设备销量(吨)")
    estimated = _fuel_num(profile, "测算销量(吨)")
    if reported is None or device is None or estimated is None:
        return _rule("R22", "多源数据不一致", False, "高", "加油站三源数据缺失", "")
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
        return _rule(
            "R22", "多源数据不一致", True, "高",
            f"申报{reported:.0f}吨/设备{device:.0f}吨/测算{estimated:.0f}吨；{'；'.join(diffs)}",
            "以进控销三源交叉验证：优先核对供应商进油单与外部锚点（运单/轨迹），"
            "内部自洽不等于真实，数据缺失本身是风险信号。",
        )
    return _rule("R22", "多源数据不一致", False, "高", "三源数据两两差异≤20%", "")


def check_r23(profile):
    """R23 申报单价偏离区域均价（加油站模板）：低>10%。"""
    price = _fuel_num(profile, "申报单价(元/升)")
    avg = _fuel_num(profile, "区域均价(元/升)")
    if price is None or avg is None:
        return _rule("R23", "申报单价偏离区域均价", False, "中", "申报单价或区域均价缺失", "")
    if avg == 0:
        return _rule("R23", "申报单价偏离区域均价", False, "中", "区域均价为0", "")
    diff = (avg - price) / avg * 100
    if diff > 10:
        return _rule(
            "R23", "申报单价偏离区域均价", True, "中",
            f"申报单价 {price:.2f}元/升 vs 区域均价 {avg:.2f}元/升，低 {diff:.0f}%",
            "单价明显偏低需核实让利真实性；结合会员折扣、促销政策判断，防止低价开票隐匿收入。",
        )
    return _rule("R23", "申报单价偏离区域均价", False, "中", f"单价偏离 {diff:.0f}%（阈值10%）", "")


def check_r24(profile):
    """R24 集群注册同址：同一地址关联企业/个体户>=5户。"""
    count = _num(profile.get("关联户数"))
    if count is None:
        return _rule("R24", "集群注册同址", False, "高", "关联户数缺失", "")
    if count >= 5:
        return _rule(
            "R24", "集群注册同址", True, "高",
            f"同一注册地址关联 {count:.0f} 户企业/个体户",
            "单户看都合法、跨户汇总才暴露拆分：按关联户/关联人组织核查。",
        )
    return _rule("R24", "集群注册同址", False, "高", f"关联户数 {count:.0f}（阈值5户）", "")


def check_r27(invoices):
    """R27 单一服务类品目集中：咨询费/服务费/管理费占销项金额>80%。"""
    sales = _sales_invoices(invoices)
    if sales.empty:
        return _rule("R27", "单一服务类品目集中", False, "中", "无销项发票数据", "")
    total = sales["金额(元)"].apply(_num).fillna(0).sum()
    service = sales[sales["品名"].astype(str).str.contains("|".join(SERVICE_ITEMS), na=False)]["金额(元)"].apply(_num).fillna(0).sum()
    if total and service / total > 0.8:
        return _rule(
            "R27", "单一服务类品目集中", True, "中",
            f"无实物品目（咨询/服务/管理费）占销项 {service/total*100:.0f}%",
            "服务费/咨询费虚开只能靠实质资料验证：人员考勤、服务记录、交付物、合同、付款凭证、物流。",
        )
    return _rule("R27", "单一服务类品目集中", False, "中", "服务类品目占比未超80%", "")


def check_r30(profile):
    """R30 资格-优惠不匹配：非高企却享受增值税加计抵减。"""
    htech = str(profile.get("是否高新技术企业", ""))
    deduction = str(profile.get("是否享受加计抵减", ""))
    if htech == "否" and deduction == "是":
        return _rule(
            "R30", "资格-优惠不匹配", True, "高",
            "非高新技术企业（或资格存疑）却享受增值税加计抵减",
            "资格认定权（科技部门）与优惠享受权（税务）分离：输出高企资格复核清单（立项记录、高新收入占比、研发人员占比、知识产权、适用名单）；按提示提醒→约谈劝退→停止享受→提请复核→追缴处理。",
        )
    return _rule("R30", "资格-优惠不匹配", False, "高", "资格与优惠享受状态匹配", "")


def check_r31(profile, invoices):
    """R31 数据完整性异常：关键字段缺失，或加油站液位仪缺失率>5%。"""
    assets = _num(profile.get("资产总额(万元)"))
    missing_rate = _fuel_num(profile, "液位仪月缺失率(%)")
    if assets is None:
        return _rule(
            "R31", "数据完整性异常", True, "高",
            "企业指标中资产总额缺失",
            "不猜测、不补数：标注'需补充后判断'；数据缺失本身就是风险信号。",
        )
    if missing_rate is not None and missing_rate > 5:
        return _rule(
            "R31", "数据完整性异常", True, "高",
            f"液位仪月记录缺失率 {missing_rate:.0f}%（阈值5%）",
            "检查记录空缺断点/删除痕迹/留存周期；用供应商进油单等外部单据交叉验证。",
        )
    return _rule("R31", "数据完整性异常", False, "高", "关键数据完整", "")


def check_r32(profile):
    """R32 上游发票缺失（加油站模板）：无票采购占比>30%。"""
    ratio = _fuel_num(profile, "无票采购占比(%)")
    if ratio is None:
        return _rule("R32", "上游发票缺失", False, "中", "无票采购占比缺失", "")
    if ratio > 30:
        level = "高" if ratio > 60 else "中"
        return _rule(
            "R32", "上游发票缺失", True, level,
            f"无票采购占比 {ratio:.0f}%（演示阈值30%）",
            "无票采购（尤其私人炼油厂）导致'以进控销'的进端不完整；生产环境需罐车/油罐硬件计量与多源数据补位。",
        )
    return _rule("R32", "上游发票缺失", False, "中", f"无票采购占比 {ratio:.0f}%（阈值30%）", "")


def check_r33(profile):
    """R33 上游单据与本地记录不一致（加油站模板）。"""
    doc = str(profile.get("上游进油单存在", ""))
    missing = str(profile.get("本地入库记录缺失", ""))
    if doc == "是" and missing == "是":
        return _rule(
            "R33", "上游单据与本地记录不一致", True, "高",
            "供应商进油单存在，但本地入库/液位仪记录缺失",
            "外部单据是戳穿数据删改的锚点：以进油单反推入库量，定性先行、定量后算。",
        )
    return _rule("R33", "上游单据与本地记录不一致", False, "高", "上游单据与本地记录匹配", "")


def check_r34(profile):
    """R34 设备版本风险（加油站模板）：旧标准芯片数据降级。"""
    chip = str(profile.get("设备芯片标准", ""))
    if chip == "旧标准":
        return _rule(
            "R34", "设备版本风险", True, "中",
            "设备仍为可破解的旧标准芯片（无防篡改能力），数据可信度降级",
            "旧芯片存在硬件漏洞（公安部2024-09典型案例、央视2025-01曝光）；新国标GB/T 9081-2023防篡改但未铺开——关键结论用外部数据交叉验证。",
        )
    return _rule("R34", "设备版本风险", False, "中", f"设备芯片标准：{chip or '未知'}", "")


def check_r35(profile):
    """R35 收入与应税所得严重不匹配：收入>5000万且收入利润率<3%。"""
    revenue = _num(profile.get("营业收入(万元)"))
    taxable = _num(profile.get("应纳税所得额(万元)"))
    if revenue is None or taxable is None or revenue == 0:
        return _rule("R35", "收入与应税所得严重不匹配", False, "中", "收入或应税所得缺失", "")
    ratio = taxable / revenue * 100
    if revenue > 5000 and ratio < 3:
        level = "高" if ratio < 1.5 else "中"
        return _rule(
            "R35", "收入与应税所得严重不匹配", True, level,
            f"营业收入 {revenue:,.0f}万，应税所得 {taxable:,.1f}万，收入利润率 {ratio:.1f}%（阈值3%）",
            "收入规模大但利润极薄：核查成本归集、研发加计、税前扣除是否合规；小微临界+大额调减组合是常见预警场景，需准备备查资料。",
        )
    return _rule("R35", "收入与应税所得严重不匹配", False, "中", f"收入利润率 {ratio:.1f}%（阈值3%）", "")


def check_r36(invoices):
    """R36 收购发票对象身份存疑：收购发票销售方非自产农业生产者。"""
    agri = invoices[invoices["发票类型"] == "农产品收购发票"]
    if agri.empty:
        return _rule("R36", "收购发票对象身份存疑", False, "高", "无农产品收购发票", "")
    suspects = agri[agri.get("销售方类型", pd.Series(dtype=str)).fillna("") != "农业生产者"]
    if not suspects.empty:
        names = "、".join(str(x) for x in suspects["对方名称"].dropna().unique()[:5])
        return _rule(
            "R36", "收购发票对象身份存疑", True, "高",
            f"{len(suspects)} 张收购发票销售方身份存疑（非自产农业生产者），对象：{names}",
            "农产品收购发票'自己开、自己抵'，对象必须是自产农业生产者；输出身份与业务流核验清单：自产证明、收购现场记录、物流、付款到农户本人。"
            "纠正路径：可要求贩子到办税服务大厅代开增值税发票补正。",
        )
    return _rule("R36", "收购发票对象身份存疑", False, "高", "收购对象均为农业生产者", "")


def check_r37(invoices):
    """R37 单户收购金额异常：单户>500万或Top3占>70%。"""
    agri = invoices[invoices["发票类型"] == "农产品收购发票"]
    if agri.empty:
        return _rule("R37", "单户收购金额异常", False, "中", "无农产品收购发票", "")
    amounts = agri.copy()
    amounts["金额(元)"] = amounts["金额(元)"].apply(_num)
    by_name = amounts.groupby("对方名称")["金额(元)"].sum()
    big = by_name[by_name > 5000000]
    if not big.empty:
        detail = "；".join(f"{k} {v/10000:.0f}万" for k, v in big.items())
        return _rule(
            "R37", "单户收购金额异常", True, "中",
            f"单户收购金额超500万：{detail}",
            "收购大户核实生产能力与自产证据：单户金额巨大且身份存疑时风险升级。",
        )
    return _rule("R37", "单户收购金额异常", False, "中", "单户收购金额≤500万", "")


def check_r38(invoices, fund_flows):
    """R38 收购发票与业务流不匹配：发票对象与资金付款对象不一致。"""
    agri = invoices[invoices["发票类型"] == "农产品收购发票"]
    if agri.empty or fund_flows.empty:
        return _rule("R38", "收购发票与业务流不匹配", False, "高", "无收购发票或资金流水", "")
    flow_names = {str(x).strip() for x in fund_flows["对方名称"].dropna()}
    mismatches = []
    for _, inv in agri.iterrows():
        name = str(inv.get("对方名称", "")).strip()
        if name and name not in flow_names:
            mismatches.append(name)
    if mismatches:
        return _rule(
            "R38", "收购发票与业务流不匹配", True, "高",
            f"收购发票对象与付款对象不一致：{'、'.join(mismatches[:5])}",
            "发票流写企业←农户，资金流却对应贩子：存在中间商，应取得贩子开具的增值税发票（可要求到办税大厅代开补正）——三流不一致的农产品版。",
        )
    return _rule("R38", "收购发票与业务流不匹配", False, "高", "收购发票对象与付款对象一致", "")


# 规则注册表：依次执行，保持稳定顺序
RULE_CHECKS = [
    ("R01", "顶额开票", "高", check_r01),
    ("R02", "月末集中开票", "中", check_r02),
    ("R03", "红冲/作废率过高", "中", check_r03),
    ("R04", "进销项品名不匹配", "高", check_r04),
    ("R06", "三流不一致", "高", check_r06),
    ("R07", "个人账户收付款占比高", "高", check_r07),
    ("R09", "风险企业传导", "高", check_r09),
    ("R12", "增值税税负率异常", "中", check_r12),
    ("R15", "个税与社保人数不一致", "中", check_r15),
    ("R16", "纳税信用等级低", "高", check_r16),
    ("R17", "小微临界", "低", check_r17),
    ("R18", "利润与申报应纳税所得额差异过大", "中", check_r18),
    ("R19", "临界点聚集", "中", check_r19),
    ("R20", "研发加计扣除占比异常", "中", check_r20),
    ("R21", "单站销售横向偏离", "中", check_r21),
    ("R22", "多源数据不一致", "高", check_r22),
    ("R23", "申报单价偏离区域均价", "中", check_r23),
    ("R24", "集群注册同址", "高", check_r24),
    ("R27", "单一服务类品目集中", "中", check_r27),
    ("R30", "资格-优惠不匹配", "高", check_r30),
    ("R31", "数据完整性异常", "高", check_r31),
    ("R32", "上游发票缺失", "中", check_r32),
    ("R33", "上游单据与本地记录不一致", "高", check_r33),
    ("R34", "设备版本风险", "中", check_r34),
    ("R35", "收入与应税所得严重不匹配", "中", check_r35),
    ("R36", "收购发票对象身份存疑", "高", check_r36),
    ("R37", "单户收购金额异常", "中", check_r37),
    ("R38", "收购发票与业务流不匹配", "高", check_r38),
]


def run_all(profile, invoices, fund_flows=None, contracts=None):
    """执行全部规则，返回命中列表。"""
    fund_flows = fund_flows if fund_flows is not None else pd.DataFrame(
        columns=["流水日期", "收付方向", "对方名称", "金额(元)", "备注", "关联合同号", "账户类型"]
    )
    contracts = contracts if contracts is not None else pd.DataFrame(columns=["合同号", "对方名称", "金额(元)", "签订日期"])
    p = profile.iloc[0].to_dict()
    hits = []
    for rule_id, name, default_level, fn in RULE_CHECKS:
        try:
            if rule_id == "R06":
                result = fn(invoices, fund_flows, contracts)
            elif rule_id == "R38":
                result = fn(invoices, fund_flows)
            elif rule_id == "R31":
                result = fn(p, invoices)
            elif rule_id == "R12":
                result = fn(invoices, p)
            elif rule_id == "R07":
                result = fn(fund_flows)
            elif rule_id in ("R01", "R02", "R03", "R04", "R09", "R27", "R36", "R37"):
                result = fn(invoices)
            else:
                result = fn(p)
        except Exception as exc:  # 单条规则失败不阻断整体
            result = _rule(rule_id, name, False, default_level, f"规则执行异常：{exc}", "")
        if result.get("hit"):
            hits.append(result)
    return _merge_dependent_rules(hits)


def _merge_dependent_rules(hits):
    """规则层级合并：R19 临界点聚集是 R17 小微临界的升级，
    同时命中时把 R17 并入 R19，避免重复计分。"""
    by_id = {h["rule_id"]: h for h in hits}
    if "R19" in by_id and "R17" in by_id:
        r17 = by_id["R17"]
        r19 = by_id["R19"]
        r17["merged_into"] = "R19"
        r19["evidence"] = f"{r19['evidence']}；临界项：{r17['evidence']}"
        r19["suggestion"] = (
            f"{r19['suggestion']}（本特征已合并 R17 小微临界提示，不重复计分）"
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
