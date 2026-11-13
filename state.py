"""Shared orchestrator state.

Specialist nodes append one short message and one contribution. Tool transcripts
stay inside the specialist and are not copied into this state.
"""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph.message import MessagesState


class Contribution(TypedDict):
    """One specialist or supervisor update, in the order it was produced."""

    agent: str
    kind: Literal["route", "research", "analysis", "validation", "synthesis", "error"]
    content: str


def add_contributions(
    left: list[Contribution] | None,
    right: list[Contribution] | None,
) -> list[Contribution]:
    """Append contributions in arrival order."""
    return [*(left or ()), *(right or ())]


class OrchestratorState(MessagesState):
    """Conversation plus the artifacts each agent is allowed to read."""

    next_agent: str
    instruction: str
    step_count: int
    task_completed: bool
    research_notes: str
    analysis_notes: str
    validation_feedback: str
    validation_passed: bool
    last_error: str
    research_runs: int
    analyst_runs: int
    contributions: Annotated[list[Contribution], add_contributions]


def initial_state(question: str) -> OrchestratorState:
    """State for one user question. Earlier questions are not reloaded."""
    return {
        "messages": [HumanMessage(content=question)],
        "next_agent": "",
        "instruction": "",
        "step_count": 0,
        "task_completed": False,
        "research_notes": "",
        "analysis_notes": "",
        "validation_feedback": "",
        "validation_passed": False,
        "last_error": "",
        "research_runs": 0,
        "analyst_runs": 0,
        "contributions": [],
    }
