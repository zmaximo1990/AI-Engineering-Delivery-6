"""Scoped inputs for specialists. Tool transcripts are not copied upward."""

import re
import warnings

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.warnings import LangGraphDeprecatedSinceV10

from settings import settings
from state import OrchestratorState

_SECRET = re.compile(r"tvly-[A-Za-z0-9_\-]+|sk-[A-Za-z0-9_\-]+")


def silence_react_agent_deprecation() -> None:
    """The assignment asks for create_react_agent, which LangGraph 1.1 warns about."""
    warnings.filterwarnings("ignore", category=LangGraphDeprecatedSinceV10)


def public_error(exc: BaseException) -> str:
    """Short error text with API keys removed."""
    text = f"{type(exc).__name__}: {exc}"
    text = _SECRET.sub("[redacted]", text)
    text = " ".join(text.split())
    return text[:300]


def message_text(message: BaseMessage) -> str:
    """Flatten a message body to plain text."""
    content = message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                value = block.get("text", "")
                if value:
                    parts.append(str(value))
        return "\n".join(part.strip() for part in parts if part and part.strip())
    return str(content).strip()


def user_question(state: OrchestratorState) -> str:
    """The question that opened this run."""
    for message in state.get("messages") or []:
        if isinstance(message, HumanMessage) and not getattr(message, "name", None):
            text = message_text(message)
            if text:
                return text
    return ""


def excerpt(text: str, limit: int | None = None) -> str:
    """Trim an artifact before it is handed to another model call."""
    cap = settings.artifact_excerpt_chars if limit is None else limit
    cleaned = text.strip()
    if len(cleaned) <= cap:
        return cleaned
    return cleaned[:cap].rstrip() + "\n[truncated]"


def called_tools(messages: list[BaseMessage]) -> set[str]:
    """Tool names requested by the specialist during its private run."""
    names: set[str] = set()
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls or []:
                name = call.get("name")
                if name:
                    names.add(str(name))
    return names


def final_text(messages: list[BaseMessage]) -> str:
    """Last assistant message that is not a pending tool call."""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            text = message_text(message)
            if text:
                return text
    return ""


async def run_specialist(agent: CompiledStateGraph, task: str) -> list[BaseMessage]:
    """Run one specialist on a single task message."""
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=task)]},
        {"recursion_limit": settings.specialist_recursion_limit},
    )
    return list(result["messages"])
