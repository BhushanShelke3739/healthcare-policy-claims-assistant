"""
Application configuration.

We use `pydantic-settings` so all configuration comes from environment
variables (or a `.env` file in development). This keeps secrets out of code
and makes the same image deployable across local / staging / production by
swapping the environment.

Why a single `Settings` object?
    * One place to see every knob the app exposes.
    * Type-checked at startup — typos in env vars fail loudly instead of
      surfacing as mysterious runtime bugs.
    * Easy to inject into FastAPI dependencies (`Depends(get_settings)`).
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve `.env` paths from the source file location rather than the current
# working directory. This way `alembic upgrade head` works whether it's run
# from the repo root, from `backend/`, or from inside a Docker container.
#
#   __file__                       -> .../backend/app/core/config.py
#   parents[0] core, [1] app, [2] backend, [3] repo root
_BACKEND_DIR = Path(__file__).resolve().parents[2]
_REPO_ROOT = _BACKEND_DIR.parent


class Settings(BaseSettings):
    """Strongly-typed application settings, loaded from env / .env file."""

    model_config = SettingsConfigDict(
        # Items later in the tuple take precedence — a backend-local `.env`
        # (rare) overrides the project-level one.
        env_file=(_REPO_ROOT / ".env", _BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- App ---
    app_env: Literal["development", "staging", "production"] = "development"
    app_name: str = "healthcare-policy-claims-assistant"
    app_version: str = "0.1.0"
    log_level: str = "INFO"

    # --- Database ---
    # Either provide DATABASE_URL directly, or set the POSTGRES_* parts and
    # let the computed field assemble a URL from them.
    database_url: str | None = None
    postgres_user: str = "hpca"
    postgres_password: str = "hpca_password"
    postgres_db: str = "hpca"
    postgres_host: str = "db"
    postgres_port: int = 5432

    # --- Embeddings (used from Phase 3) ---
    embedding_provider: Literal["mock", "openai", "sentence-transformers"] = "mock"
    embedding_model: str = "text-embedding-3-small"
    # 1536 matches OpenAI's text-embedding-3-small. Keep it configurable so we
    # can swap to a different provider/dimension without touching code.
    embedding_dimensions: int = 1536
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"

    # --- LLM (Phase 4) ---
    # "mock"   : deterministic non-LLM responses (default — works without a key).
    # "openai" : any OpenAI-compatible chat endpoint. Works with OpenAI proper,
    #            Ollama, vLLM, LM Studio, llamafile — point OPENAI_BASE_URL at
    #            the right URL (Ollama default: http://localhost:11434/v1).
    llm_provider: Literal["mock", "openai"] = "mock"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    # Structured output strategy. OpenAI proper supports strict json_schema;
    # most self-hosted "OpenAI-compatible" servers (Ollama, vLLM <0.6) only
    # support the looser `json_object` mode. Default works against both —
    # set to "strict" when targeting OpenAI for the safer schema guarantee.
    llm_structured_output: Literal["json_object", "strict"] = "json_object"
    # If retrieval returns no chunks (or only chunks below this similarity
    # floor) we short-circuit to the refusal phrase instead of calling the LLM.
    # 0.0 = never refuse based on score; rely on the LLM to refuse.
    refusal_score_floor: float = Field(default=0.0, ge=0.0, le=1.0)
    # The exact phrase the generator returns when it cannot answer. Spec'd
    # this way because the user/operations team should be able to grep for it.
    refusal_phrase: str = "I could not find this in the available policy documents."

    # --- RAG defaults ---
    chunk_size: int = Field(default=800, ge=100, le=4000)
    chunk_overlap: int = Field(default=120, ge=0, le=1000)
    default_top_k: int = Field(default=5, ge=1, le=50)

    # --- Retrieval (Phase 3) ---
    # Weight on the vector score in hybrid search (1 - alpha is the keyword
    # weight). 0.6 leans on semantics while still letting an exact identifier
    # match (e.g. "HF-022") win.
    hybrid_alpha: float = Field(default=0.6, ge=0.0, le=1.0)
    # PostgreSQL text search configuration (regconfig). "english" is the
    # built-in default; switch to "simple" if you want literal token matches
    # without stemming.
    fts_language: str = "english"

    # --- Document ingestion (Phase 2) ---
    # Hard cap to keep a single bad upload from OOMing the process. PDFs
    # holding scanned images can be 50+ MB — set higher if you need that.
    max_upload_size_mb: int = Field(default=10, ge=1, le=100)
    # Allowlist of file extensions accepted by /documents/upload.
    allowed_upload_extensions: tuple[str, ...] = (".txt", ".md", ".pdf")

    # --- Agents (Phase 5) ---
    # Per-workflow safety cap on retrieval calls (top_k per search).
    agent_default_top_k: int = Field(default=5, ge=1, le=20)
    # Max distinct tools / nodes an agent may execute. Keeps a runaway
    # graph from melting the LLM bill.
    agent_max_steps: int = Field(default=10, ge=1, le=50)

    # --- CORS (Phase 7) ---
    # Allowed origins for the demo frontend. In production this would be
    # a list of trusted domains; for local dev we permit the Next.js dev
    # server and the Docker Compose alias.
    cors_origins: tuple[str, ...] = (
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    )

    # --- Auth (Phase 9, placeholder) ---
    # OFF by default: every request is the anonymous viewer, so the demo runs
    # with no tokens. Flip AUTH_ENABLED=true to require a valid bearer token on
    # endpoints that depend on `get_current_user` / `require_role`.
    auth_enabled: bool = False
    # HS256 symmetric secret. The default is intentionally obvious so it can
    # never be mistaken for production-safe; a real deployment injects this via
    # Secret Manager / env and would also consider asymmetric (RS256) keys so
    # verifiers don't hold signing material.
    jwt_secret: str = "dev-insecure-secret-change-me"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = Field(default=60, ge=1, le=1440)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sqlalchemy_url(self) -> str:
        """Final URL handed to SQLAlchemy. Honors DATABASE_URL if set."""
        if self.database_url:
            return self.database_url
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached Settings instance.

    `lru_cache` makes this effectively a singleton — settings are read from
    the environment exactly once per process. Tests that need to override
    values can call `get_settings.cache_clear()` and patch env vars.
    """
    return Settings()
