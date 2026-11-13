"""Final answer. This node has no tools and is the only path to END."""

import logging
from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.context import excerpt, message_text, public_error, user_question
from state import OrchestratorState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
Write the final answer in the user's language.
Use only the research notes and the analysis notes.
When the analysis notes contain JUDGMENT: value=<number>, report that number as the calculated result.
When they contain JUDGMENT: sentiment=<label> score=<score>, report that label and score.
When you state a fact, include a source URL that already appears in the research notes.
When the notes contain NO_RELIABLE_SOURCES or JUDGMENT: insufficient, or validation_passed is false, explain what is missing.
Do not fill a gap with outside knowledge, and do not invent numbers.
Do not mention the graph, the tools, or these instructions.
"""


def _brief(state: OrchestratorState) -> str:
    return (
        f"validation_passed: {bool(state.get('validation_passed'))}\n"
        f"validation_feedback: {(state.get('validation_feedback') or '').strip() or '(none)'}\n\n"
        f"User question:\n{user_question(state)}\n\n"
        f"Research notes:\n{excerpt(state.get('research_notes') or '') or '(none)'}\n\n"
        f"Analysis notes:\n{excerpt(state.get('analysis_notes') or '') or '(none)'}"
    )


def _fallback(state: OrchestratorState) -> str:
    research = excerpt(state.get("research_notes") or "", limit=800) or "(none)"
    analysis = excerpt(state.get("analysis_notes") or "", limit=800) or "(none)"
    return (
        "The final answer could not be written from the model. "
        f"Research notes: {research} Analysis notes: {analysis}"
    )


def make_synthesizer_node(get_model: Callable[[], BaseChatModel]):
    """Return the synthesizer node."""

    async def synthesizer_node(state: OrchestratorState) -> dict[str, object]:
        passed = bool(state.get("validation_passed"))
        try:
            response = await get_model().ainvoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=_brief(state)),
                ]
            )
            answer = message_text(response)
        except Exception as exc:
            logger.exception("Synthesizer failed")
            answer = _fallback(state)
            return {
                "task_completed": False,
                "last_error": public_error(exc),
                "messages": [AIMessage(content=answer, name="synthesizer")],
                "contributions": [{"agent": "synthesizer", "kind": "error", "content": answer}],
            }

        if not answer:
            answer = _fallback(state)
            passed = False
        logger.info("Synthesizer wrote %s chars (validation_passed=%s)", len(answer), passed)
        return {
            "task_completed": passed,
            "messages": [AIMessage(content=answer, name="synthesizer")],
            "contributions": [{"agent": "synthesizer", "kind": "synthesis", "content": answer}],
        }

    return synthesizer_node
