"""Tools bound to a single specialist.

``web_search`` is the research agent's only tool. ``score_sentiment`` and
``calculate`` are the analyst's only tools. Failures return JSON with an
``error`` field so the agent can report the gap instead of raising.
"""

import ast
import json
import logging
import math
import operator
import os
import re

from langchain_core.tools import tool

from settings import settings

logger = logging.getLogger(__name__)

_SECRET = re.compile(r"tvly-[A-Za-z0-9_\-]+|sk-[A-Za-z0-9_\-]+")
_MAX_EXPRESSION_LENGTH = 200
_MAX_ABS_EXPONENT = 8
_MAX_ABS_BASE_FOR_POW = 1_000_000

_POSITIVE = {
    "beat",
    "beats",
    "boost",
    "boosted",
    "breakthrough",
    "bullish",
    "gain",
    "gains",
    "grew",
    "growth",
    "improve",
    "improved",
    "improvement",
    "optimistic",
    "outperform",
    "positive",
    "profit",
    "profits",
    "rally",
    "rallied",
    "record",
    "rise",
    "rose",
    "strong",
    "stronger",
    "success",
    "successful",
    "surge",
    "surged",
    "upgrade",
    "upgraded",
}
_NEGATIVE = {
    "bearish",
    "concern",
    "concerns",
    "decline",
    "declined",
    "delay",
    "delayed",
    "downgrade",
    "downgraded",
    "drop",
    "dropped",
    "fall",
    "falling",
    "fell",
    "investigation",
    "lawsuit",
    "layoff",
    "layoffs",
    "loss",
    "losses",
    "miss",
    "missed",
    "negative",
    "pessimistic",
    "probe",
    "risk",
    "risks",
    "slump",
    "slumped",
    "weak",
    "weaker",
    "worse",
    "worst",
}

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _public_error(detail: object) -> str:
    text = str(detail)
    text = _SECRET.sub("[redacted]", text)
    text = " ".join(text.split())
    return text[:300]


def _dumps(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _trim_results(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict):
        return {"error": "Web search returned an unexpected payload."}
    if raw.get("error"):
        return {"error": "Web search failed.", "detail": _public_error(raw["error"])}
    results: list[dict[str, str]] = []
    rows = raw.get("results") or []
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            results.append(
                {
                    "title": str(row.get("title") or ""),
                    "url": str(row.get("url") or ""),
                    "content": str(row.get("content") or "")[:800],
                }
            )
    if not results:
        return {"error": "Web search returned no results."}
    return {"query": str(raw.get("query") or ""), "results": results}


@tool
def web_search(query: str) -> str:
    """Search the public web and return titles, URLs, and snippets.

    query: a focused search string. The response is JSON. A successful payload
    has results with title, url, and content. A failed search has an error field
    and is not a fact.
    """
    cleaned = " ".join(query.split())
    if not cleaned:
        return _dumps({"error": "The search query is empty."})
    if len(cleaned) > 400:
        cleaned = cleaned[:400]

    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        return _dumps({"error": "TAVILY_API_KEY is not configured."})

    try:
        from langchain_tavily import TavilySearch

        search = TavilySearch(
            max_results=settings.tavily_max_results,
            search_depth="basic",
            include_raw_content=False,
            tavily_api_key=api_key,
        )
        raw = search.invoke({"query": cleaned})
    except Exception as exc:
        reason = _public_error(exc)
        logger.warning("Tavily search failed query=%r reason=%s", cleaned, reason)
        return _dumps({"error": "Web search failed.", "detail": reason})
    if isinstance(raw, dict) and raw.get("error"):
        reason = _public_error(raw["error"])
        logger.warning("Tavily search failed query=%r reason=%s", cleaned, reason)
    return _dumps(_trim_results(raw))


@tool
def score_sentiment(text: str) -> str:
    """Score the sentiment of a passage with a fixed word list.

    text: the passage to score, usually research snippets. Returns JSON with
    label (positive, negative, mixed, or neutral) and score in [-1, 1].
    An empty passage returns an error field.
    """
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    if not tokens:
        return _dumps({"error": "No text to score."})
    positive_hits = sum(token in _POSITIVE for token in tokens)
    negative_hits = sum(token in _NEGATIVE for token in tokens)
    total = positive_hits + negative_hits
    if total == 0:
        label = "neutral"
        score = 0.0
    else:
        score = (positive_hits - negative_hits) / total
        if score > 0.15:
            label = "positive"
        elif score < -0.15:
            label = "negative"
        else:
            label = "mixed"
    return _dumps(
        {
            "label": label,
            "score": round(score, 3),
            "positive_hits": positive_hits,
            "negative_hits": negative_hits,
        }
    )


def _eval_arithmetic(node: ast.AST, depth: int = 0) -> float:
    if depth > 32:
        raise ValueError("Expression is too deep.")
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return float(_UNARY[type(node.op)](_eval_arithmetic(node.operand, depth + 1)))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left = _eval_arithmetic(node.left, depth + 1)
        right = _eval_arithmetic(node.right, depth + 1)
        if isinstance(node.op, ast.Pow):
            if abs(right) > _MAX_ABS_EXPONENT or abs(left) > _MAX_ABS_BASE_FOR_POW:
                raise ValueError("Exponent is out of range.")
        try:
            value = _BINOPS[type(node.op)](left, right)
        except ZeroDivisionError as exc:
            raise ValueError("Division by zero.") from exc
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Result is not a finite number.")
        return value
    raise ValueError("Only +, -, *, /, **, parentheses, and numbers are allowed.")


@tool
def calculate(expression: str) -> str:
    """Evaluate one arithmetic expression.

    expression: digits with +, -, *, /, **, and parentheses. Returns JSON with
    the expression and its value. Words, names, and function calls return an
    error field. Do not estimate a value when this tool returns an error.
    """
    cleaned = expression.strip()
    if not cleaned:
        return _dumps({"error": "The expression is empty."})
    if len(cleaned) > _MAX_EXPRESSION_LENGTH:
        return _dumps({"error": "The expression is too long."})
    try:
        tree = ast.parse(cleaned, mode="eval")
        value = _eval_arithmetic(tree.body)
    except (SyntaxError, ValueError) as exc:
        return _dumps({"error": "Could not evaluate the expression.", "detail": _public_error(exc)})
    if value.is_integer():
        rendered: int | float = int(value)
    else:
        rendered = round(value, 6)
    return _dumps({"expression": cleaned, "value": rendered})
