# -*- coding: utf-8 -*-
"""数据契约校验：上传数据在进入规则引擎前的 schema 关卡。

设计原则：坏数据不能悄悄流入规则层——缺列、类型错、空表都必须在入口处
给出明确清单。这是数据质量规则（R7xx）思想在输入端的延伸。
"""

import pandas as pd

from generate_data import (
    CONTRACT_COLS,
    DATA_SOURCE_COLS,
    EXTERNAL_DOC_COLS,
    FUND_COLS,
    INVOICE_COLS,
    PROFILE_COLS,
)

# 表名 -> (中文名, 契约列, 是否必传)
TABLE_CONTRACTS = {
    "company_profile": ("企业指标", PROFILE_COLS, True),
    "invoices": ("发票明细", INVOICE_COLS, True),
    "fund_flows": ("资金流水", FUND_COLS, True),
    "contracts": ("合同", CONTRACT_COLS, True),
    "external_docs": ("外部单据", EXTERNAL_DOC_COLS, False),
    "data_sources": ("数据源清单", DATA_SOURCE_COLS, False),
}

# 需要数值类型的关键列（出现非数值内容即拦截）
NUMERIC_COLUMNS = {
    "company_profile": [
        "个税申报人数", "社保参保人数", "资产总额(万元)", "营业收入(万元)",
        "营业成本(万元)", "期间费用(万元)", "研发费用(万元)", "利润总额(万元)",
        "应纳税所得额(万元)",
    ],
    "invoices": ["税率(%)", "金额(元)", "税额(元)"],
    "fund_flows": ["金额(元)"],
    "contracts": ["金额(元)"],
    "external_docs": ["数量", "金额(元)"],
    "data_sources": ["留存月数"],
}


def validate_tables(data):
    """校验上传数据是否符合契约。

    返回 {"ok": bool, "errors": [str], "warnings": [str], "checked": int}。
    ok=False 时规则引擎不应运行。
    """
    errors, warnings, checked = [], [], 0
    for key, (label, required_cols, is_required) in TABLE_CONTRACTS.items():
        df = data.get(key)
        if df is None:
            if is_required:
                errors.append(f"[{label}] 缺少必传表。")
            continue
        checked += 1
        if not isinstance(df, pd.DataFrame) or df.empty:
            if is_required:
                errors.append(f"[{label}] 表为空（0 行）。")
            else:
                warnings.append(f"[{label}] 可选表为空，将按无数据处理。")
            continue
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            errors.append(f"[{label}] 缺少契约列：{'、'.join(missing)}。")
        extra = [c for c in df.columns if c not in required_cols]
        if extra:
            warnings.append(f"[{label}] 存在契约之外的列（将忽略）：{'、'.join(extra)}。")
        for col in NUMERIC_COLUMNS.get(key, []):
            if col not in df.columns:
                continue
            coerced = pd.to_numeric(df[col], errors="coerce")
            bad = df[coerced.isna() & df[col].notna()]
            if not bad.empty:
                rows = "、".join(str(i + 2) for i in bad.index[:3])
                more = f" 等 {len(bad)} 行" if len(bad) > 3 else ""
                errors.append(f"[{label}] 列「{col}」存在非数值内容（CSV 第 {rows} 行{more}）。")
    return {"ok": not errors, "errors": errors, "warnings": warnings, "checked": checked}
