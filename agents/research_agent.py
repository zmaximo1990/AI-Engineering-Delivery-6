"""Research specialist. Its only tool is web search."""

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
from agents.markers import NO_RELIABLE_SOURCES, research_status
from state import OrchestratorState
from tools import web_search

logger = logging.getLogger(__name__)

RESEARCH_PROMPT = f"""\
You search the public web with the web_search tool. Facts in your summary come from that tool.
Call web_search at least once.
Write the summary in the user's language.
Copy every number the question needs, with the URL of the source that stated it.
Include the source URL for every factual claim.
When the tool returns an error, or the results do not contain the requested fact, say so and end with a line that is exactly:
{NO_RELIABLE_SOURCES}
If you include a source URL, do not emit that marker.
Do not invent numbers, dates, quotes, or URLs.
Do not score sentiment, and do not compute a difference or a percentage. Copy the figures and leave the judgment to the analyst.
"""


def _task(state: OrchestratorState) -> str:
    parts = [
        f"User question:\n{user_question(state)}",
        "Supervisor instruction:\n"
        + ((state.get("instruction") or "").strip() or "Search for the facts this question needs."),
    ]
    feedback = (state.get("validation_feedback") or "").strip()
    if feedback:
        parts.append(f"Validation feedback:\n{feedback}")
    previous = (state.get("research_notes") or "").strip()
    if previous:
        parts.append(f"Previous research notes:\n{excerpt(previous)}")
    return "\n\n".join(parts)


def make_research_node(get_model: Callable[[], BaseChatModel]):
    """Return the research node. The react agent is built on the first visit."""
    agent_box: dict[str, object] = {}

    def agent():
        if "agent" not in agent_box:
            silence_react_agent_deprecation()
            agent_box["agent"] = create_react_agent(
                get_model(),
                tools=[web_search],
                prompt=RESEARCH_PROMPT,
            )
        return agent_box["agent"]

    async def research_node(state: OrchestratorState) -> dict[str, object]:
        runs = int(state.get("research_runs") or 0) + 1
        base: dict[str, object] = {"research_runs": runs}
        try:
            messages = await run_specialist(agent(), _task(state))
            summary = final_text(messages)
        except Exception as exc:
            logger.exception("Research agent failed")
            error = public_error(exc)
            return {
                **base,
                "last_error": error,
                "messages": [AIMessage(content=f"Research failed: {error}", name="researcher")],
                "contributions": [{"agent": "researcher", "kind": "error", "content": error}],
            }

        if "web_search" not in called_tools(messages):
            error = "Researcher did not call web_search."
        elif not summary:
            error = "Researcher returned an empty summary."
        elif research_status(summary) == "invalid":
            error = (
                "Research summary has no source URL and does not mark "
                f"{NO_RELIABLE_SOURCES}."
            )
        else:
            error = ""

        if error:
            logger.info("Research run %s rejected: %s", runs, error)
            return {
                **base,
                "last_error": error,
                "messages": [AIMessage(content=error, name="researcher")],
                "contributions": [{"agent": "researcher", "kind": "error", "content": error}],
            }

        logger.info("Research notes stored (%s chars)", len(summary))
        return {
            **base,
            "research_notes": summary,
            "last_error": "",
            "validation_feedback": "",
            "validation_passed": False,
            "messages": [AIMessage(content=summary, name="researcher")],
            "contributions": [{"agent": "researcher", "kind": "research", "content": summary}],
        }

    return research_node
