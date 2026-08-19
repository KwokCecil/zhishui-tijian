# -*- coding: utf-8 -*-
"""极简 OpenAI 兼容 function calling 客户端（requests 实现，无额外依赖）。

环境变量（也可在 Streamlit secrets 中配置，app.py 会读取）：
    ZHI_SHUI_LLM_API_KEY
    ZHI_SHUI_LLM_BASE_URL  默认 https://api.deepseek.com/v1
    ZHI_SHUI_LLM_MODEL     默认 deepseek-chat
"""

import json
import os

import requests


class LLMError(RuntimeError):
    pass


def config():
    return {
        "api_key": os.environ.get("ZHI_SHUI_LLM_API_KEY", ""),
        "base_url": os.environ.get("ZHI_SHUI_LLM_BASE_URL", "https://api.deepseek.com/v1").rstrip("/"),
        "model": os.environ.get("ZHI_SHUI_LLM_MODEL", "deepseek-chat"),
    }


def available():
    key = config()["api_key"]
    return bool(key) and "在这里填入" not in key and "your_key" not in key.lower()


def complete(prompt, system="你是严谨的税务数字化产品专家。", temperature=0.3, max_tokens=2000):
    """单轮补全（报告生成用）。"""
    cfg = config()
    if not cfg["api_key"]:
        raise LLMError("未配置 ZHI_SHUI_LLM_API_KEY")
    resp = requests.post(
        f"{cfg['base_url']}/chat/completions",
        headers={"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"},
        json={
            "model": cfg["model"],
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=90,
    )
    if resp.status_code != 200:
        raise LLMError(f"LLM 接口返回 {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def chat_with_tools(messages, tools, max_turns=8):
    """多轮 function calling 循环：模型决定调用工具→执行→回填→继续，直到给出最终回答。"""
    cfg = config()
    if not cfg["api_key"]:
        raise LLMError("未配置 ZHI_SHUI_LLM_API_KEY")
    trace = []
    payload_messages = list(messages)
    executed = {}
    for _ in range(max_turns):
        body = {
            "model": cfg["model"],
            "messages": payload_messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.2,
            "max_tokens": 800,
        }
        resp = requests.post(
            f"{cfg['base_url']}/chat/completions",
            headers={"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"},
            json=body,
            timeout=120,
        )
        if resp.status_code != 200:
            raise LLMError(f"LLM 接口返回 {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        choice = data["choices"][0]
        message = choice["message"]
        payload_messages.append(message)
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return message.get("content") or "", trace
        for call in tool_calls:
            fn = call["function"]
            key = (fn["name"], fn.get("arguments", "{}"))
            if key in executed:
                result = {"note": "该工具已调用过，结果不变", **executed[key]}
                ok = True
            else:
                try:
                    result = _run_tool(fn["name"], fn.get("arguments", "{}"))
                    executed[key] = result
                    ok = True
                except Exception as exc:  # noqa: BLE001
                    result = {"error": f"{type(exc).__name__}: {exc}"}
                    ok = False
            trace.append({
                "tool": fn["name"],
                "arguments": fn.get("arguments", "{}"),
                "ok": ok,
                "result": result,
            })
            payload_messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps(result, ensure_ascii=False),
            })
    # 轮数用尽：禁止再调工具，强制模型基于已有结果直接作答
    final_body = {
        "model": cfg["model"],
        "messages": payload_messages + [
            {"role": "user", "content": "请直接根据已有工具结果给出最终回答，不要再调用任何工具。"}
        ],
        "tool_choice": "none",
        "temperature": 0.2,
        "max_tokens": 800,
    }
    resp = requests.post(
        f"{cfg['base_url']}/chat/completions",
        headers={"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"},
        json=final_body,
        timeout=120,
    )
    if resp.status_code != 200:
        raise LLMError(f"LLM 接口返回 {resp.status_code}: {resp.text[:300]}")
    content = resp.json()["choices"][0]["message"].get("content") or ""
    if not content.strip():
        raise LLMError(f"工具调用超过 {max_turns} 轮，且强制收尾未返回内容")
    return content.strip(), trace


def _run_tool(name, arguments):
    from tools import execute_tool

    return execute_tool(name, arguments)
