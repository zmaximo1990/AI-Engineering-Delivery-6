"""Hierarchical StateGraph: supervisor, researcher, analyst, validator, synthesizer."""

import logging
from datetime import datetime, timezone
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.analyst_agent import make_analyst_node
from agents.context import public_error
from agents.research_agent import make_research_node
from agents.supervisor import make_supervisor_node
from agents.synthesizer import make_synthesizer_node
from agents.validator import validator_node
from factory import LLMFactory
from schemas import LLMConfig
from settings import settings
from state import OrchestratorState, initial_state

logger = logging.getLogger(__name__)

SupervisorRoute = Literal["researcher", "analyst", "validator", "synthesizer"]
ValidatorRoute = Literal["supervisor", "synthesizer"]


def route_from_supervisor(state: OrchestratorState) -> SupervisorRoute:
    """Conditional edge. The step cap leaves the graph through the synthesizer."""
    if int(state.get("step_count") or 0) >= settings.max_supervisor_steps:
        return "synthesizer"
    next_agent = state.get("next_agent") or "researcher"
    if next_agent in {"researcher", "analyst", "validator"}:
        return next_agent
    return "researcher"


def route_from_validator(state: OrchestratorState) -> ValidatorRoute:
    """Approved work is synthesized. A rejection returns to the supervisor while a retry is still possible."""
    if state.get("validation_passed"):
        return "synthesizer"
    if int(state.get("step_count") or 0) >= settings.max_supervisor_steps:
        return "synthesizer"

    feedback = (state.get("validation_feedback") or "").lower()
    research_runs = int(state.get("research_runs") or 0)
    analyst_runs = int(state.get("analyst_runs") or 0)
    limit = settings.max_specialist_runs
    has_research = bool((state.get("research_notes") or "").strip())
    has_analysis = bool((state.get("analysis_notes") or "").strip())
    retry_research = research_runs < limit and (not has_research or "researcher" in feedback)
    retry_analyst = analyst_runs < limit and (not has_analysis or "analyst" in feedback)
    if retry_research or retry_analyst:
        return "supervisor"
    return "synthesizer"


def compile_graph(config: LLMConfig) -> CompiledStateGraph:
    """Compile the orchestrator. The chat model is built on the first node call."""
    cache: dict[str, BaseChatModel] = {}

    def get_model() -> BaseChatModel:
        if "model" not in cache:
            cache["model"] = LLMFactory.create_client(config).get_llm_model()
        return cache["model"]

    workflow = StateGraph(OrchestratorState)
    workflow.add_node("supervisor", make_supervisor_node(get_model))
    workflow.add_node("researcher", make_research_node(get_model))
    workflow.add_node("analyst", make_analyst_node(get_model))
    workflow.add_node("validator", validator_node)
    workflow.add_node("synthesizer", make_synthesizer_node(get_model))

    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "researcher": "researcher",
            "analyst": "analyst",
            "validator": "validator",
            "synthesizer": "synthesizer",
        },
    )
    workflow.add_edge("researcher", "supervisor")
    workflow.add_edge("analyst", "supervisor")
    workflow.add_conditional_edges(
        "validator",
        route_from_validator,
        {"supervisor": "supervisor", "synthesizer": "synthesizer"},
    )
    workflow.add_edge("synthesizer", END)
    return workflow.compile()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_trace(text: str) -> None:
    """Append one formatted turn to ``trace.log``."""
    path = settings.trace_log_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")
        handle.write("\n")


def format_turn(question: str, state: OrchestratorState) -> str:
    """Render supervisor decisions, specialist artifacts, and the final answer."""
    lines = [
        "=" * 80,
        (
            f"{_now()}  steps={int(state.get('step_count') or 0)}"
            f"  research_runs={int(state.get('research_runs') or 0)}"
            f"  analyst_runs={int(state.get('analyst_runs') or 0)}"
            f"  validation_passed={bool(state.get('validation_passed'))}"
            f"  task_completed={bool(state.get('task_completed'))}"
        ),
        "-" * 80,
        "USER",
        question,
        "",
    ]
    for item in state.get("contributions") or []:
        lines.append(f"{item['agent'].upper()} [{item['kind']}]")
        lines.append(item["content"].strip())
        lines.append("")
    last_error = (state.get("last_error") or "").strip()
    if last_error:
        lines.append("LAST_ERROR")
        lines.append(last_error)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def format_failure(question: str, exc: BaseException) -> str:
    """Render a run that stopped before a final answer."""
    lines = [
        "=" * 80,
        f"{_now()}  status=error",
        "-" * 80,
        "USER",
        question,
        "",
        "ERROR",
        public_error(exc),
        "",
    ]
    return "\n".join(lines)


async def ask(graph: CompiledStateGraph, question: str) -> str:
    """Run one question from a fresh state and append the trace."""
    try:
        result = await graph.ainvoke(
            initial_state(question),
            {"recursion_limit": settings.recursion_limit},
        )
    except Exception as exc:
        logger.exception("Orchestrator invocation failed")
        text = format_failure(question, exc)
        append_trace(text)
        return text

    text = format_turn(question, result)
    append_trace(text)
    return text


def draw_mermaid(graph: CompiledStateGraph) -> str:
    """Mermaid source for the compiled graph."""
    return graph.get_graph().draw_mermaid()
