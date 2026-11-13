"""Deterministic check that both specialist artifacts are usable together."""

import logging

from agents.markers import judgment_kind, research_status
from state import OrchestratorState

logger = logging.getLogger(__name__)


def assess(state: OrchestratorState) -> tuple[bool, str]:
    """Return whether the artifacts pass, and feedback the supervisor can route on."""
    last_error = (state.get("last_error") or "").strip()
    if last_error:
        return False, f"The last specialist reported an error: {last_error}"

    research = (state.get("research_notes") or "").strip()
    analysis = (state.get("analysis_notes") or "").strip()
    if not research:
        return False, "Research notes are empty. Send the question back to the researcher."
    if not analysis:
        return False, "Analysis notes are empty. Send the research notes to the analyst."

    sources = research_status(research)
    kind = judgment_kind(analysis)
    if sources == "invalid":
        return False, "Research notes have no source URL and no NO_RELIABLE_SOURCES marker. Retry the researcher."
    if kind is None:
        return False, "Analysis notes are missing a JUDGMENT line. Retry the analyst."

    if kind == "insufficient":
        if sources == "gap":
            return True, "No reliable public sources. The final answer should explain that gap."
        return (
            False,
            "Research notes cite sources, but the analysis declined to judge them. Retry the analyst.",
        )

    if sources != "sourced":
        return (
            False,
            "The analysis reports a judgment without source URLs. Retry the researcher.",
        )
    if kind == "sentiment":
        return True, "Research cites sources and the analysis includes a sentiment judgment."
    return True, "Research cites sources and the analysis includes a calculated value."


def validator_node(state: OrchestratorState) -> dict[str, object]:
    """Record the rubric result. Routing is a conditional edge on this state."""
    passed, feedback = assess(state)
    logger.info("Validation passed=%s", passed)
    return {
        "validation_passed": passed,
        "validation_feedback": feedback,
        "task_completed": False,
        "contributions": [
            {
                "agent": "validator",
                "kind": "validation",
                "content": f"passed={passed}: {feedback}",
            }
        ],
    }
