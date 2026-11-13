"""Supervisor node. It routes. It does not search, score, or write the answer."""

import logging
from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agents.context import excerpt, public_error, user_question
from schemas import SupervisorDecision
from settings import settings
from state import OrchestratorState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You route a research-and-analysis task. You do not search the web, score sentiment, calculate, or write the final answer.

Specialists:
- researcher: searches the public web. Give it one focused search instruction.
- analyst: scores sentiment or evaluates arithmetic on the research notes. It cannot browse.
- validator: checks that the research notes and the analysis notes are sufficient together. Choose it when both artifacts are already present and you are not requesting the single allowed retry.

Rules:
- With no research notes, choose researcher.
- With research notes and no analysis notes, choose analyst.
- A specialist may run at most twice. Do not choose a specialist that has already used both runs.
- When validation feedback names a gap, send that feedback's specialist back once. After that specialist's second run, choose validator.
- Do not invent figures, sources, or sentiment.
- instruction is one short task for the chosen specialist, in the user's language. It is not the final answer.
"""


def normalize_route(choice: str, state: OrchestratorState) -> str:
    """Apply the rubric in code so a bad model choice cannot loop a specialist."""
    research_runs = int(state.get("research_runs") or 0)
    analyst_runs = int(state.get("analyst_runs") or 0)
    has_research = bool((state.get("research_notes") or "").strip())
    has_analysis = bool((state.get("analysis_notes") or "").strip())
    limit = settings.max_specialist_runs

    if choice not in {"researcher", "analyst", "validator"}:
        choice = "researcher" if not has_research else "analyst" if not has_analysis else "validator"
    if choice == "researcher" and research_runs >= limit:
        choice = "analyst" if analyst_runs < limit and not has_analysis else "validator"
    if choice == "analyst" and (analyst_runs >= limit or (not has_research and research_runs < limit)):
        choice = "researcher" if not has_research and research_runs < limit else "validator"
    if choice == "validator" and not has_research and research_runs < limit:
        choice = "researcher"
    if choice == "validator" and has_research and not has_analysis and analyst_runs < limit:
        choice = "analyst"
    return choice


def _summary(state: OrchestratorState, step: int) -> str:
    feedback = (state.get("validation_feedback") or "").strip() or "(none)"
    last_error = (state.get("last_error") or "").strip() or "(none)"
    research = excerpt(state.get("research_notes") or "") or "(none)"
    analysis = excerpt(state.get("analysis_notes") or "") or "(none)"
    return (
        f"Step {step} of {settings.max_supervisor_steps}\n"
        f"research_runs: {int(state.get('research_runs') or 0)} of {settings.max_specialist_runs}\n"
        f"analyst_runs: {int(state.get('analyst_runs') or 0)} of {settings.max_specialist_runs}\n"
        f"last_error: {last_error}\n"
        f"validation_feedback: {feedback}\n\n"
        f"User question:\n{user_question(state)}\n\n"
        f"Research notes:\n{research}\n\n"
        f"Analysis notes:\n{analysis}"
    )


def make_supervisor_node(get_model: Callable[[], BaseChatModel]):
    """Return the supervisor node. The chat model is built on the first visit."""

    async def supervisor_node(state: OrchestratorState) -> dict[str, object]:
        step = int(state.get("step_count") or 0) + 1
        if step >= settings.max_supervisor_steps:
            logger.info("Supervisor step cap reached at step %s", step)
            return {
                "step_count": step,
                "next_agent": "validator",
                "instruction": "Step cap reached. Synthesize with the evidence collected so far.",
                "contributions": [
                    {
                        "agent": "supervisor",
                        "kind": "route",
                        "content": "synthesizer: step cap reached",
                    }
                ],
            }

        try:
            decision = await get_model().with_structured_output(SupervisorDecision).ainvoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=_summary(state, step)),
                ]
            )
            parsed = decision if isinstance(decision, SupervisorDecision) else SupervisorDecision.model_validate(decision)
            choice = parsed.next_agent
            instruction = parsed.instruction.strip()
            reason = parsed.reason.strip()
        except Exception as exc:
            logger.exception("Supervisor decision failed")
            choice = "researcher" if not (state.get("research_notes") or "").strip() else "analyst"
            instruction = "Continue the task from the artifacts already collected."
            reason = public_error(exc)

        choice = normalize_route(choice, state)
        instruction = instruction[:500] or "Continue from the current artifacts."
        logger.info("Supervisor routed to %s (step %s)", choice, step)
        return {
            "step_count": step,
            "next_agent": choice,
            "instruction": instruction,
            "contributions": [
                {
                    "agent": "supervisor",
                    "kind": "route",
                    "content": f"{choice}: {instruction} ({reason})",
                }
            ],
        }

    return supervisor_node
