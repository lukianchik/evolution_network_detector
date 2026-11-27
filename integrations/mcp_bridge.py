"""MCP-клиент для контрольных проверок по завершении поколения.

Обращается к серверу метрик по stdio или Streamable HTTP внутри Compose. Не участвует в
массовой оценке кандидатов: она выполняется напрямую через NumPy.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_tool_result(result: object) -> dict:
    """Извлекает структурированный ответ MCP и отклоняет ошибки сервера."""
    if getattr(result, "is_error", False):
        blocks = getattr(result, "content", [])
        message = " ".join(str(getattr(item, "text", "")) for item in blocks)
        raise RuntimeError(f"MCP-инструмент вернул ошибку: {message[:240]}")
    data = getattr(result, "structured_content", None)
    if data is None:
        blocks = getattr(result, "content", [])
        for block in blocks:
            if getattr(block, "type", "") == "text":
                data = json.loads(block.text)
                break
    if not isinstance(data, dict):
        raise RuntimeError("MCP не вернул JSON-объект")
    return data


async def _invoke(calls: list[tuple[str, dict]], timeout: float) -> list[dict]:
    """Открывает MCP-сеанс и вызывает нужные инструменты последовательно."""
    try:
        from mcp import Client, StdioServerParameters
    except ImportError as error:
        raise RuntimeError("Нет MCP SDK. Установи requirements-mcp.txt") from error
    url = os.getenv("EVO_MCP_URL", "").strip()
    settings = url or StdioServerParameters(
        command=sys.executable,
        args=["-m", "integrations.mcp_metrics"],
        cwd=str(ROOT),
        env={"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"},
    )
    async with Client(settings) as client:
        output = []
        for name, args in calls:
            response = await asyncio.wait_for(client.call_tool(name, args), timeout=timeout)
            output.append(parse_tool_result(response))
        return output


def call_tools(calls: list[tuple[str, dict]], timeout: float = 15.0) -> list[dict]:
    """Выполняет реальные MCP-вызовы с общим ограничением ожидания."""
    if not calls or len(calls) > 4:
        raise ValueError("За один сеанс разрешено 1–4 MCP-инструмента")
    if not 1 <= timeout <= 30:
        raise ValueError("Таймаут MCP должен быть от 1 до 30 секунд")
    try:
        return asyncio.run(asyncio.wait_for(_invoke(calls, timeout), timeout=5 + timeout * len(calls)))
    except TimeoutError as error:
        raise RuntimeError("MCP-сервер не ответил вовремя") from error
