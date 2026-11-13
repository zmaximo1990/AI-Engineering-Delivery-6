"""Runtime settings loaded from the environment and the project ``.env`` file."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_DIR = Path(__file__).resolve().parent

_PROVIDERS = {"openai", "anthropic", "gemini"}


class Settings(BaseSettings):
    """Process settings. Empty environment values fall back to the field default."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
    )

    default_max_tokens: int = 2048
    default_temperature: float = 0.0
    gemini_max_tokens: int = 2048
    default_provider: str = Field(default="openai", validation_alias="LLM_PROVIDER")

    trace_log_path: Path = PROJECT_DIR / "trace.log"
    recursion_limit: int = 25
    specialist_recursion_limit: int = 12
    max_supervisor_steps: int = 6
    max_specialist_runs: int = 2
    tavily_max_results: int = 5
    artifact_excerpt_chars: int = 4000

    @model_validator(mode="before")
    @classmethod
    def _blank_as_unset(cls, data: object) -> object:
        """Treat blank strings as missing so the field default is used."""
        if not isinstance(data, dict):
            return data
        cleaned: dict[object, object] = {}
        for key, value in data.items():
            if isinstance(value, str):
                value = value.strip()
                if value == "":
                    continue
            cleaned[key] = value
        return cleaned

    @field_validator("default_provider", mode="before")
    @classmethod
    def _provider(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        lowered = value.strip().lower()
        if lowered not in _PROVIDERS:
            allowed = ", ".join(sorted(_PROVIDERS))
            raise ValueError(f"LLM_PROVIDER must be one of: {allowed}")
        return lowered

    @field_validator("trace_log_path", mode="before")
    @classmethod
    def _resolve_project_path(cls, value: object) -> Path:
        path = value if isinstance(value, Path) else Path(str(value).strip())
        if not path.is_absolute():
            path = PROJECT_DIR / path
        return path

    @field_validator("recursion_limit")
    @classmethod
    def _recursion_floor(cls, value: int) -> int:
        return max(8, value)

    @field_validator("specialist_recursion_limit")
    @classmethod
    def _specialist_recursion_floor(cls, value: int) -> int:
        return max(4, value)

    @field_validator("max_supervisor_steps")
    @classmethod
    def _supervisor_steps_floor(cls, value: int) -> int:
        return max(4, value)

    @field_validator("max_specialist_runs")
    @classmethod
    def _specialist_runs_floor(cls, value: int) -> int:
        return max(1, value)

    @field_validator("tavily_max_results")
    @classmethod
    def _tavily_results_bounds(cls, value: int) -> int:
        return min(10, max(1, value))

    @field_validator("artifact_excerpt_chars")
    @classmethod
    def _excerpt_floor(cls, value: int) -> int:
        return max(200, value)


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings, loaded once."""
    return Settings()


settings = get_settings()
