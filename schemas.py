from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from settings import settings


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class LLMConfig(BaseModel):
    """Provider, model, and sampling settings for one orchestrator run."""

    provider: LLMProvider
    model: str | None = None
    max_tokens: int = Field(default=settings.default_max_tokens, gt=0)
    temperature: float = Field(default=settings.default_temperature, ge=0.0, le=2.0)


class SupervisorDecision(BaseModel):
    """Routing decision. The literal is the conditional edge out of the supervisor."""

    next_agent: Literal["researcher", "analyst", "validator"]
    instruction: str = Field(
        ...,
        min_length=1,
        description="One short task for the chosen specialist. Not the final answer.",
    )
    reason: str = Field(
        ...,
        min_length=1,
        description="Why this specialist is the next step.",
    )
