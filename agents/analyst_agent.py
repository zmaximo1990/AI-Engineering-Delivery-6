"""Analyst specialist. It scores sentiment and evaluates arithmetic on research notes."""

import logging
from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langgraph.prebuilt import create_react_agent

from agents.context import (
    called_tools,
    excerpt,
    final_text,
    public_error,
    run_specialist,
    silence_react_agent_deprecation,
    user_question,
)
from agents.markers import judgment_kind
from state import OrchestratorState
from tools import calculate, score_sentiment

logger = logging.getLogger(__name__)

ANALYST_PROMPT = """\
You judge research notes. Your only tools are score_sentiment and calculate. You cannot browse.
Use the research notes as your only evidence. Write the summary in the user's language.
For a sentiment question, pass the research notes to score_sentiment and report that tool's label and score.
For a numeric question, copy the figures that appear in the research notes and pass one arithmetic expression to calculate. The expression uses digits, +, -, *, /, **, and parentheses.
Do not compute a number yourself and do not estimate a missing figure.
End with exactly one of these lines:
JUDGMENT: sentiment=<label> score=<score>
JUDGMENT: value=<number>
JUDGMENT: insufficient
The label is positive, negative, mixed, or neutral, copied from score_sentiment.
The number is the value field returned by calculate, unchanged.
Use JUDGMENT: insufficient when a required figure or passage is missing, or the research notes are marked NO_RELIABLE_SOURCES.
"""


def _task(state: OrchestratorState) -> str:
    parts = [
        f"User question:\n{user_question(state)}",
        "Supervisor instruction:\n"
        + ((state.get("instruction") or "").strip() or "Judge the research notes."),
        "Research notes:\n" + (excerpt(state.get("research_notes") or "") or "(none)"),
    ]
    feedback = (state.get("validation_feedback") or "").strip()
    if feedback:
        parts.append(f"Validation feedback:\n{feedback}")
    previous = (state.get("analysis_notes") or "").strip()
    if previous:
        parts.append(f"Previous analysis notes:\n{excerpt(previous)}")
    return "\n\n".join(parts)


def _tool_error(summary: str, tool_names: set[str]) -> str:
    kind = judgment_kind(summary)
    if kind is None:
        return "Analyst summary is missing a JUDGMENT line."
    if kind == "sentiment" and "score_sentiment" not in tool_names:
        return "Analyst reported a sentiment judgment without calling score_sentiment."
    if kind == "value" and "calculate" not in tool_names:
        return "Analyst reported a numeric judgment without calling calculate."
    return ""


def make_analyst_node(get_model: Callable[[], BaseChatModel]):
    """Return the analyst node. The react agent is built on the first visit."""
    agent_box: dict[str, object] = {}

    def agent():
        if "agent" not in agent_box:
            silence_react_agent_deprecation()
            agent_box["agent"] = create_react_agent(
                get_model(),
                tools=[score_sentiment, calculate],
                prompt=ANALYST_PROMPT,
            )
        return agent_box["agent"]

    async def analyst_node(state: OrchestratorState) -> dict[str, object]:
        runs = int(state.get("analyst_runs") or 0) + 1
        base: dict[str, object] = {"analyst_runs": runs}
        try:
            messages = await run_specialist(agent(), _task(state))
            summary = final_text(messages)
        except Exception as exc:
            logger.exception("Analyst agent failed")
            error = public_error(exc)
            return {
                **base,
                "last_error": error,
                "messages": [AIMessage(content=f"Analysis failed: {error}", name="analyst")],
                "contributions": [{"agent": "analyst", "kind": "error", "content": error}],
            }

        error = _tool_error(summary, called_tools(messages)) if summary else "Analyst returned an empty summary."
        if error:
            logger.info("Analyst run %s rejected: %s", runs, error)
            return {
                **base,
                "last_error": error,
                "messages": [AIMessage(content=error, name="analyst")],
                "contributions": [{"agent": "analyst", "kind": "error", "content": error}],
            }

        logger.info("Analysis notes stored (%s chars, judgment=%s)", len(summary), judgment_kind(summary))
        return {
            **base,
            "analysis_notes": summary,
            "last_error": "",
            "validation_feedback": "",
            "validation_passed": False,
            "messages": [AIMessage(content=summary, name="analyst")],
            "contributions": [{"agent": "analyst", "kind": "analysis", "content": summary}],
        }

    return analyst_node
