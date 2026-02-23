"""
Central config for the entire voice pipeline.

Every env var used across LLM_service.py, STT_service.py, TTS_service.py, voice_graph.py,
main.py, controller.py, and shared.py is declared and validated here.
Validation runs at import time so a misconfigured container fails
immediately on startup rather than crashing mid-traffic when the first
request hits a bad value.

Usage in any module:
    from app.common.settings import settings

    api_key = settings.openai_api_key.get_secret_value()
    redis_url = settings.redis_url

Field sections
──────────────
  OpenAI                — shared credential used by LLM, STT, and TTS nodes
  LLM node              — model selection, concurrency, rate limiting, caching
  Redis                 — LLM response cache and stampede lock
  STT node              — model, concurrency, rate limiting, S3 transcript storage
  TTS node              — model, voice, format, output dirs, rate limiting, S3 storage
  AWS                   — credentials and region for S3 integration
  Graph / orchestration — timeouts, retry limits, queue depths, inflight cap, version
  FastAPI gateway       — upload limits, CORS, pipeline timeout, server bind
  Controller            — desktop PTT keybindings and timing parameters
  OpenTelemetry         — tracing endpoint, service name, toggle
"""

from __future__ import annotations

import sys
import warnings
from functools import (
    lru_cache,
)  # noqa — kept for callers that use cached_settings pattern
from pathlib import Path
from typing import Literal

from pydantic import (
    AnyUrl,  # noqa
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

# ── valid enum values ──────────────────────────────────────────────────────────
# Defined at module level so they can be imported independently for use in
# other validators or documentation generators without instantiating Settings.

# OpenAI TTS API valid values (as of current API version)
VALID_TTS_VOICES = {"alloy", "echo", "fable", "onyx", "nova", "shimmer"}
VALID_TTS_FORMATS = {"mp3", "opus", "aac", "flac", "wav", "pcm"}
VALID_TTS_MODELS = {"tts-1", "tts-1-hd"}
VALID_STT_MODELS = {"whisper-1"}

# gpt-4o and gpt-3.5 families — not exhaustive, but catches obvious typos.
# New model names from OpenAI trigger a warning rather than a hard error so a
# model release doesn't break deployments that need to move fast.
KNOWN_CHAT_MODELS = {
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    # GPT-5 family
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
}

# AWS region names — not exhaustive, but catches the most common mistake of
# writing a region with incorrect formatting (e.g. "us_east_1").
KNOWN_AWS_REGIONS = {
    "us-east-1",
    "us-east-2",
    "us-west-1",
    "us-west-2",
    "eu-west-1",
    "eu-west-2",
    "eu-west-3",
    "eu-central-1",
    "ap-southeast-1",
    "ap-southeast-2",
    "ap-northeast-1",
    "ap-south-1",
    "ca-central-1",
    "sa-east-1",
    "af-south-1",
    "me-south-1",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # Extra env vars in the environment are silently ignored so we don't
        # crash when running alongside unrelated services that export their own
        # vars into the shared environment.
        extra="ignore",
    )

    # ── OpenAI ────────────────────────────────────────────────────────────────
    # SecretStr prevents the key from appearing in logs or repr() output.
    # All three nodes (LLM, STT, TTS) share this single credential.
    openai_api_key: SecretStr = Field(
        ...,
        alias="OPENAI_API_KEY",
        description="OpenAI API key. Must start with 'sk-'.",
    )

    # ── LLM node ──────────────────────────────────────────────────────────────
    llm_model: str = Field("gpt-4o-mini", alias="LLM_MODEL")
    llm_fallback_model: str = Field("gpt-3.5-turbo", alias="LLM_FALLBACK_MODEL")
    llm_temperature: float = Field(0.7, alias="LLM_TEMPERATURE", ge=0.0, le=2.0)
    llm_max_concurrent: int = Field(50, alias="LLM_MAX_CONCURRENT", ge=1, le=500)
    llm_rate_per_sec: float = Field(20.0, alias="LLM_RATE_PER_SEC", gt=0)
    llm_rate_burst: float = Field(40.0, alias="LLM_RATE_BURST", gt=0)
    llm_cache_ttl: int = Field(
        3600,
        alias="LLM_CACHE_TTL",
        ge=0,
        description="LLM response cache TTL in seconds. 0 = never expire.",
    )

    # ── Redis ─────────────────────────────────────────────────────────────────
    # Used by the LLM node for response caching and stampede locks.
    redis_enabled: bool = Field(
        True,
        alias="REDIS_ENABLED",
        description="Set to false to disable Redis caching and stampede locks entirely. "
        "When disabled, REDIS_URL and REDIS_MAX_CONN are still validated "
        "if present but check_redis() and pool-size cross-checks are skipped.",
    )
    redis_url: str = Field("redis://localhost:6379", alias="REDIS_URL")
    redis_max_connections: int = Field(200, alias="REDIS_MAX_CONN", ge=1, le=5000)

    # ── STT node ──────────────────────────────────────────────────────────────
    stt_model: str = Field("whisper-1", alias="STT_MODEL")
    stt_max_file_mb: float = Field(
        25.0,
        alias="STT_MAX_FILE_MB",
        gt=0,
        le=25.0,
        description="Max upload size for audio files. Cannot exceed 25 MB (Whisper API hard ceiling).",
    )
    stt_max_concurrent: int = Field(20, alias="STT_MAX_CONCURRENT", ge=1, le=200)
    stt_rate_per_sec: float = Field(10.0, alias="STT_RATE_PER_SEC", gt=0)
    stt_rate_burst: float = Field(20.0, alias="STT_RATE_BURST", gt=0)
    stt_s3_bucket: str | None = Field(None, alias="STT_S3_BUCKET")
    stt_s3_transcript_prefix: str = Field(
        "transcripts/", alias="STT_S3_TRANSCRIPT_PREFIX"
    )

    # ── TTS node ──────────────────────────────────────────────────────────────
    tts_model: str = Field("tts-1-hd", alias="TTS_MODEL")
    tts_voice: str = Field("nova", alias="TTS_VOICE")
    tts_format: str = Field("mp3", alias="TTS_FORMAT")
    tts_output_dir: Path = Field(Path("audio/audio_OUTPUT"), alias="TTS_OUTPUT_DIR")
    tts_max_concurrent: int = Field(20, alias="TTS_MAX_CONCURRENT", ge=1, le=200)
    tts_rate_per_sec: float = Field(10.0, alias="TTS_RATE_PER_SEC", gt=0)
    tts_rate_burst: float = Field(20.0, alias="TTS_RATE_BURST", gt=0)
    tts_local_file_ttl: float = Field(
        3600.0,
        alias="TTS_LOCAL_FILE_TTL",
        ge=0,
        description="Seconds before local TTS output files are cleaned up. 0 = never.",
    )
    tts_s3_bucket: str | None = Field(None, alias="TTS_S3_BUCKET")
    tts_s3_prefix: str = Field("tts/", alias="TTS_S3_PREFIX")

    # ── AWS ───────────────────────────────────────────────────────────────────
    # Credentials are optional — IAM instance roles are supported.
    # If either S3 bucket is configured, aws_region must be set.
    aws_region: str = Field("us-east-1", alias="AWS_REGION")
    aws_access_key_id: SecretStr | None = Field(None, alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: SecretStr | None = Field(None, alias="AWS_SECRET_ACCESS_KEY")

    # ── Graph / orchestration ─────────────────────────────────────────────────
    #
    # These are the authoritative values that VoiceGraphConfig reads via
    # getattr(settings, ...). Declaring them here means fat-fingered env vars
    # are caught at startup rather than silently defaulted at first request.
    #
    # Per-instance variant overrides (e.g. the tighter timeouts on
    # voice_graph_realtime) are expressed as constructor arguments in
    # voice_graph.py and are intentionally not surfaced here — they are design
    # decisions, not operator tuning knobs.

    voice_graph_version: str = Field(
        "v2",
        alias="VOICE_GRAPH_VERSION",
        description=(
            "Version label embedded in metrics and result payloads. "
            "Change this when deploying a graph topology change to track "
            "old vs new graph behaviour in dashboards during a rollout."
        ),
    )

    # ── Stage timeouts ────────────────────────────────────────────────────────
    # Applied to the standard voice_graph singleton. The realtime and
    # low_latency singletons use tighter values hardcoded in voice_graph.py
    # because those variants are design constraints rather than operator knobs.
    graph_stt_timeout: float = Field(
        30.0,
        alias="GRAPH_STT_TIMEOUT",
        gt=0,
        description="Hard wall-clock limit for STT transcription per request (seconds).",
    )
    graph_llm_timeout: float = Field(
        45.0,
        alias="GRAPH_LLM_TIMEOUT",
        gt=0,
        description="Hard wall-clock limit for LLM generation per request (seconds).",
    )
    graph_tts_timeout: float = Field(
        30.0,
        alias="GRAPH_TTS_TIMEOUT",
        gt=0,
        description="Hard wall-clock limit for TTS synthesis per request (seconds).",
    )

    # ── Text size guards ──────────────────────────────────────────────────────
    graph_min_prompt_chars: int = Field(
        25,
        alias="GRAPH_MIN_PROMPT_CHARS",
        ge=1,
        description=(
            "Minimum transcript length (characters) before the LLM is fired in "
            "stream_full(). Prevents sending filler words ('uh', 'so') to the LLM "
            "before the user has expressed enough intent to generate a useful response."
        ),
    )
    graph_max_transcript_chars: int = Field(
        2000,
        alias="GRAPH_MAX_TRANSCRIPT_CHARS",
        ge=1,
        description="STT transcript is truncated to this length before being passed to the LLM.",
    )
    graph_max_llm_response_chars: int = Field(
        4096,
        alias="GRAPH_MAX_LLM_RESPONSE_CHARS",
        ge=1,
        description="LLM response is capped to this length before sanitisation and TTS.",
    )
    graph_max_tts_chars: int = Field(
        4000,
        alias="GRAPH_MAX_TTS_CHARS",
        ge=1,
        le=4096,
        description=(
            "Characters sent to TTS per request. Cannot exceed 4096 (OpenAI TTS API limit). "
            "Setting this below graph_max_llm_response_chars gives the sanitize node room "
            "to strip markdown and special characters without hitting the API ceiling."
        ),
    )
    # Retained for recorder/player logic that enforces a maximum audio clip duration.
    # No longer read by voice_graph.py itself (the graph delegates duration checking
    # to the STT node), but still useful for pre-flight validation in the recorder.
    graph_max_audio_duration_s: float = Field(
        300.0,
        alias="GRAPH_MAX_AUDIO_DURATION_S",
        gt=0,
        description="Max accepted audio clip duration in seconds. Enforced by the recorder.",
    )

    # ── Concurrency and load shedding ─────────────────────────────────────────
    # The standard voice_graph singleton's LoadSheddingGuard is initialised with
    # this value. The realtime (30) and low_latency (50) singletons use lower
    # hardcoded caps to keep those pools reserved. This field governs only the
    # standard pool.
    graph_max_inflight: int = Field(
        100,
        alias="GRAPH_MAX_INFLIGHT",
        ge=1,
        le=10_000,
        description=(
            "Maximum simultaneous in-flight requests admitted to the standard "
            "voice_graph singleton. Requests beyond this cap are load-shed immediately "
            "rather than queued, so this should reflect real node capacity."
        ),
    )

    # ── Streaming queue depths ────────────────────────────────────────────────
    # These bound the inter-stage queues inside stream_full(). Each request gets
    # its own pair of queues — these are per-call limits, not global pool sizes.
    # A deeper queue absorbs short bursts from a fast upstream stage but uses
    # more memory per active stream. The LLM→TTS queue should be at least as
    # deep as the STT→LLM queue because TTS is typically slower than LLM.
    graph_stt_llm_queue_depth: int = Field(
        8,
        alias="GRAPH_STT_LLM_QUEUE_DEPTH",
        ge=1,
        le=256,
        description="Max STT segments buffered between the STT and LLM stages in stream_full().",
    )
    graph_llm_tts_queue_depth: int = Field(
        16,
        alias="GRAPH_LLM_TTS_QUEUE_DEPTH",
        ge=1,
        le=256,
        description=(
            "Max LLM tokens buffered between the LLM and TTS stages in stream_full(). "
            "Should be >= GRAPH_STT_LLM_QUEUE_DEPTH because TTS synthesis is typically "
            "slower than LLM token generation, so the downstream queue fills faster."
        ),
    )

    # ── FastAPI gateway ───────────────────────────────────────────────────────
    audio_input_dir: Path = Field(Path("audio/audio_INPUT"), alias="AUDIO_INPUT_DIR")
    max_upload_mb: float = Field(
        25.0,
        alias="MAX_UPLOAD_MB",
        gt=0,
        le=25.0,
        description="Max audio upload size at the API layer. Must not exceed STT_MAX_FILE_MB.",
    )
    # Outer wall-clock deadline for the full pipeline under voice_graph (standard tier).
    # Must be larger than the sum of stage timeouts so the pipeline has a chance
    # to finish all three stages before the timeout fires.
    pipeline_timeout: float = Field(
        120.0,
        alias="PIPELINE_TIMEOUT",
        gt=0,
        description=(
            "Wall-clock deadline (seconds) for a full voice_graph.run() call. "
            "Must be >= GRAPH_STT_TIMEOUT + GRAPH_LLM_TIMEOUT + GRAPH_TTS_TIMEOUT."
        ),
    )
    # Outer deadline for the streaming path (STT then LLM stream). Must cover
    # at least STT + LLM since both run in sequence before yielding audio chunks.
    stream_timeout: float = Field(
        90.0,
        alias="STREAM_TIMEOUT",
        gt=0,
        description=(
            "Wall-clock deadline (seconds) for a voice_graph.stream() or WebSocket "
            "streaming session. Must be >= GRAPH_STT_TIMEOUT + GRAPH_LLM_TIMEOUT "
            "because both stages run in sequence before the first token is yielded."
        ),
    )
    cors_origins_raw: str = Field("*", alias="CORS_ORIGINS")
    host: str = Field("0.0.0.0", alias="HOST")
    port: int = Field(8000, alias="PORT", ge=1, le=65535)
    env: Literal["production", "development", "test"] = Field("production", alias="ENV")

    # ── Controller (desktop PTT) ──────────────────────────────────────────────
    # Timing parameters for the hold-to-talk desktop controller. All values are
    # in seconds unless the name ends with _key. Declaring them here gives the
    # same startup-time validation as the rest of the pipeline instead of
    # silently defaulting to hardcoded values in controller.py.

    # How often the keyboard poll loop checks for key state. Lower values feel
    # more responsive but burn more CPU on the main thread.
    controller_poll_interval_s: float = Field(
        0.03,
        alias="CONTROLLER_POLL_INTERVAL_S",
        gt=0,
        description="Keyboard poll interval in seconds (default ~33 Hz).",
    )
    # How long to wait after a PTT release before allowing the next PTT press.
    # Prevents accidental double-triggers from key bounce.
    controller_debounce_s: float = Field(
        0.35,
        alias="CONTROLLER_DEBOUNCE_S",
        gt=0,
        description="Debounce delay after PTT key release (seconds).",
    )
    # Short sleep inserted between sending the cancel signal and clearing the
    # interrupt flag so the event loop has time to process the cancellation.
    controller_interrupt_drain_s: float = Field(
        0.05,
        alias="CONTROLLER_INTERRUPT_DRAIN_S",
        gt=0,
        description=(
            "Delay between sending cancel signal and clearing the interrupt flag "
            "so the event loop can process the cancellation (seconds)."
        ),
    )
    # Sleep duration before calling loop.stop() during shutdown, giving in-flight
    # coroutines a window to finish or at least notice the cancellation.
    controller_loop_drain_s: float = Field(
        0.10,
        alias="CONTROLLER_LOOP_DRAIN_S",
        gt=0,
        description="Drain window before stopping the async event loop on shutdown (seconds).",
    )
    # How long to wait for the event loop thread to exit after loop.stop().
    controller_loop_join_timeout: float = Field(
        3.0,
        alias="CONTROLLER_LOOP_JOIN_TIMEOUT",
        gt=0,
        description="Thread join timeout waiting for the async loop to stop (seconds).",
    )
    # Key bindings — these are keyboard library key names, not character codes.
    controller_ptt_key: str = Field(
        "h",
        alias="CONTROLLER_PTT_KEY",
        description="Hold-to-talk key. Any keyboard library key name (e.g. 'h', 'space', 'f1').",
    )
    controller_exit_key: str = Field(
        "esc",
        alias="CONTROLLER_EXIT_KEY",
        description="Clean-exit key. Any keyboard library key name.",
    )

    # ── OpenTelemetry ─────────────────────────────────────────────────────────
    otel_exporter_otlp_endpoint: str = Field(
        "http://localhost:4317",
        alias="OTEL_EXPORTER_OTLP_ENDPOINT",
    )
    otel_service_name: str = Field("voice-pipeline", alias="OTEL_SERVICE_NAME")
    otel_enabled: bool = Field(True, alias="OTEL_ENABLED")

    # ══════════════════════════════════════════════════════════════════════════
    # FIELD-LEVEL VALIDATORS
    # Run per-field before model_validators. A failure here prevents the field
    # from being set at all, so model_validators can trust their inputs.
    # ══════════════════════════════════════════════════════════════════════════

    @field_validator("openai_api_key")
    @classmethod
    def _validate_openai_key(cls, v: SecretStr) -> SecretStr:
        raw = v.get_secret_value()
        if not raw:
            raise ValueError("OPENAI_API_KEY is empty.")
        if not raw.startswith("sk-"):
            raise ValueError(
                "OPENAI_API_KEY doesn't look right — it should start with 'sk-'. "
                "Check your .env or environment."
            )
        return v

    @field_validator("llm_model")
    @classmethod
    def _validate_llm_model(cls, v: str) -> str:
        if v not in KNOWN_CHAT_MODELS:
            # Warn but don't hard-fail — OpenAI releases new model names regularly
            # and a warning is far better than a deployment failure on model launch day.
            warnings.warn(
                f"LLM_MODEL '{v}' isn't in the known model list. "
                "If this is a new model name, you can ignore this. "
                f"Known models: {sorted(KNOWN_CHAT_MODELS)}",
                stacklevel=2,
            )
        return v

    @field_validator("llm_fallback_model")
    @classmethod
    def _validate_fallback_model(cls, v: str) -> str:
        if v not in KNOWN_CHAT_MODELS:
            warnings.warn(
                f"LLM_FALLBACK_MODEL '{v}' isn't in the known model list.",
                stacklevel=2,
            )
        return v

    @field_validator("stt_model")
    @classmethod
    def _validate_stt_model(cls, v: str) -> str:
        if v not in VALID_STT_MODELS:
            raise ValueError(
                f"STT_MODEL must be one of {sorted(VALID_STT_MODELS)}, got '{v}'."
            )
        return v

    @field_validator("tts_model")
    @classmethod
    def _validate_tts_model(cls, v: str) -> str:
        if v not in VALID_TTS_MODELS:
            raise ValueError(
                f"TTS_MODEL must be one of {sorted(VALID_TTS_MODELS)}, got '{v}'."
            )
        return v

    @field_validator("tts_voice")
    @classmethod
    def _validate_tts_voice(cls, v: str) -> str:
        if v not in VALID_TTS_VOICES:
            raise ValueError(
                f"TTS_VOICE must be one of {sorted(VALID_TTS_VOICES)}, got '{v}'."
            )
        return v

    @field_validator("tts_format")
    @classmethod
    def _validate_tts_format(cls, v: str) -> str:
        if v not in VALID_TTS_FORMATS:
            raise ValueError(
                f"TTS_FORMAT must be one of {sorted(VALID_TTS_FORMATS)}, got '{v}'."
            )
        return v

    @field_validator("aws_region")
    @classmethod
    def _validate_aws_region(cls, v: str) -> str:
        if v not in KNOWN_AWS_REGIONS:
            warnings.warn(
                f"AWS_REGION '{v}' isn't in the known region list — "
                "if this is a new or private region, ignore this warning.",
                stacklevel=2,
            )
        return v

    @field_validator("redis_url")
    @classmethod
    def _validate_redis_url(cls, v: str) -> str:
        if not v.startswith(("redis://", "rediss://", "unix://")):
            raise ValueError(
                f"REDIS_URL must start with redis://, rediss://, or unix://. Got: '{v}'"
            )
        return v

    @field_validator("voice_graph_version")
    @classmethod
    def _validate_graph_version(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("VOICE_GRAPH_VERSION must not be empty.")
        return v.strip()

    @field_validator("stt_s3_transcript_prefix", "tts_s3_prefix")
    @classmethod
    def _validate_s3_prefix(cls, v: str) -> str:
        # S3 prefixes must end with "/" so object keys like "transcripts/abc.txt"
        # are formed correctly. Auto-correct rather than rejecting.
        if v and not v.endswith("/"):
            return v + "/"
        return v

    @field_validator("otel_exporter_otlp_endpoint")
    @classmethod
    def _validate_otel_endpoint(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):  # noqa
            raise ValueError(
                f"OTEL_EXPORTER_OTLP_ENDPOINT must be http:// or https://. Got: '{v}'"  # noqa
            )
        return v

    @field_validator("controller_ptt_key", "controller_exit_key")
    @classmethod
    def _validate_controller_keys(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Controller key bindings must not be empty.")
        cleaned = v.strip().lower()
        if cleaned == cleaned[::-1] and len(cleaned) > 1:
            # Trivially palindromic long strings are suspicious (likely a copy-paste error).
            # This is intentionally loose — the keyboard library accepts free-form key names.
            pass
        return cleaned

    @field_validator("controller_poll_interval_s")
    @classmethod
    def _validate_poll_interval(cls, v: float) -> float:
        # Below 10 ms starts burning noticeable CPU on the polling thread with
        # no perceptible responsiveness improvement — warn but don't block.
        if v < 0.01:
            warnings.warn(
                f"CONTROLLER_POLL_INTERVAL_S={v} is very low (< 10 ms). "
                "This will increase CPU usage on the main keyboard polling thread "
                "without meaningful responsiveness improvement.",
                stacklevel=2,
            )
        return v

    # ══════════════════════════════════════════════════════════════════════════
    # CROSS-FIELD (MODEL) VALIDATORS
    # Run after all fields are validated. All inputs can be assumed valid types.
    # ══════════════════════════════════════════════════════════════════════════

    @model_validator(mode="after")
    def _validate_rate_limiters(self) -> Settings:
        """
        Burst capacity smaller than the per-second rate makes the token bucket
        limiter useless — you'd be rate-limited below the configured rate from
        the very first request.
        """
        pairs = [
            (
                "LLM_RATE_BURST",
                self.llm_rate_burst,
                "LLM_RATE_PER_SEC",
                self.llm_rate_per_sec,
            ),
            (
                "STT_RATE_BURST",
                self.stt_rate_burst,
                "STT_RATE_PER_SEC",
                self.stt_rate_per_sec,
            ),
            (
                "TTS_RATE_BURST",
                self.tts_rate_burst,
                "TTS_RATE_PER_SEC",
                self.tts_rate_per_sec,
            ),
        ]
        for burst_name, burst, rate_name, rate in pairs:
            if burst < rate:
                raise ValueError(
                    f"{burst_name} ({burst}) must be >= {rate_name} ({rate}). "
                    "Burst is the bucket capacity; it must be at least as large as the fill rate "
                    "or you'll be rate-limited below the target throughput immediately."
                )
        return self

    @model_validator(mode="after")
    def _validate_redis_pool_vs_concurrency(self) -> Settings:
        """
        If the Redis connection pool is smaller than LLM concurrency, requests
        beyond the pool limit block waiting for a free connection — effectively
        serialising LLM calls through the Redis bottleneck.
        Skipped when redis_enabled=False since no pool is created.
        """
        if not self.redis_enabled:
            return self
        if self.redis_max_connections < self.llm_max_concurrent:
            raise ValueError(
                f"REDIS_MAX_CONN ({self.redis_max_connections}) is smaller than "
                f"LLM_MAX_CONCURRENT ({self.llm_max_concurrent}). "
                "The connection pool will be exhausted under load. "
                "Set REDIS_MAX_CONN >= LLM_MAX_CONCURRENT."
            )
        return self

    @model_validator(mode="after")
    def _validate_primary_fallback_differ(self) -> Settings:
        """
        Primary and fallback LLM models must differ. If they're the same, a model
        outage takes down both at the same time — the fallback provides no resilience.
        """
        if self.llm_model == self.llm_fallback_model:
            raise ValueError(
                f"LLM_MODEL and LLM_FALLBACK_MODEL are both '{self.llm_model}'. "
                "The fallback should be a different model — otherwise a model outage "
                "takes down both the primary and fallback simultaneously."
            )
        return self

    @model_validator(mode="after")
    def _validate_pipeline_timeout_vs_stages(self) -> Settings:
        """
        The outer pipeline timeout must exceed the sum of all three stage timeouts.
        If it's smaller, the pipeline will always time out before all stages finish —
        which makes the stage timeouts meaningless as actual limits.
        """
        min_reasonable = (
            self.graph_stt_timeout + self.graph_llm_timeout + self.graph_tts_timeout
        )
        if self.pipeline_timeout < min_reasonable:
            raise ValueError(
                f"PIPELINE_TIMEOUT ({self.pipeline_timeout}s) is less than the sum of "
                f"stage timeouts ({min_reasonable}s = "
                f"{self.graph_stt_timeout} + {self.graph_llm_timeout} + {self.graph_tts_timeout}). "
                "The pipeline will always time out before all stages can complete. "
                "Increase PIPELINE_TIMEOUT or reduce stage timeouts."
            )
        return self

    @model_validator(mode="after")
    def _validate_stream_timeout_vs_stages(self) -> Settings:
        """
        The streaming timeout must cover at least the STT + LLM stages because
        both run in sequence in voice_graph.stream() before any tokens are yielded.
        A timeout smaller than their sum guarantees every streaming request fails
        before producing output.
        """
        min_stream = self.graph_stt_timeout + self.graph_llm_timeout
        if self.stream_timeout < min_stream:
            raise ValueError(
                f"STREAM_TIMEOUT ({self.stream_timeout}s) is less than "
                f"GRAPH_STT_TIMEOUT + GRAPH_LLM_TIMEOUT "
                f"({self.graph_stt_timeout} + {self.graph_llm_timeout} = {min_stream}s). "
                "The stream will always time out before STT + LLM can complete. "
                "Increase STREAM_TIMEOUT or reduce stage timeouts."
            )
        return self

    @model_validator(mode="after")
    def _validate_inflight_vs_node_concurrency(self) -> Settings:
        """
        If graph_max_inflight exceeds the smallest node concurrency limit,
        pipelines beyond that limit will queue-stall waiting on the bottleneck node
        rather than being admitted or shed cleanly. This is a warning rather than
        an error because there are valid reasons to allow a small overage (e.g. to
        absorb short bursts while queued requests drain).
        """
        bottleneck_cap = min(
            self.stt_max_concurrent,
            self.llm_max_concurrent,
            self.tts_max_concurrent,
        )
        if self.graph_max_inflight > bottleneck_cap * 2:
            warnings.warn(
                f"GRAPH_MAX_INFLIGHT ({self.graph_max_inflight}) is more than 2× the "
                f"bottleneck node concurrency cap ({bottleneck_cap}). Requests beyond "
                f"{bottleneck_cap} in-flight will queue-stall at the bottleneck node. "
                "Consider lowering GRAPH_MAX_INFLIGHT or raising the node concurrency caps.",
                stacklevel=2,
            )
        return self

    @model_validator(mode="after")
    def _validate_queue_depth_ordering(self) -> Settings:
        """
        The LLM→TTS queue should be at least as deep as the STT→LLM queue.

        TTS synthesis is typically slower than LLM token generation, so the
        downstream queue fills faster. If it's shallower than the upstream queue,
        the LLM worker will frequently hit backpressure sleeps unnecessarily.
        This is a warning because it doesn't break anything — it just means the
        stream pipeline will be less efficient than it could be.
        """
        if self.graph_llm_tts_queue_depth < self.graph_stt_llm_queue_depth:
            warnings.warn(
                f"GRAPH_LLM_TTS_QUEUE_DEPTH ({self.graph_llm_tts_queue_depth}) < "
                f"GRAPH_STT_LLM_QUEUE_DEPTH ({self.graph_stt_llm_queue_depth}). "
                "The LLM→TTS queue is shallower than STT→LLM even though TTS is "
                "typically slower. The LLM stage will hit backpressure more often than "
                "necessary. Consider setting GRAPH_LLM_TTS_QUEUE_DEPTH >= GRAPH_STT_LLM_QUEUE_DEPTH.",
                stacklevel=2,
            )
        return self

    @model_validator(mode="after")
    def _validate_tts_chars_vs_response_chars(self) -> Settings:
        """
        If the TTS char cap exceeds the LLM response cap, the sanitize node will
        never truncate for TTS — the LLM cap is always the binding constraint first.
        This isn't an error, but it means GRAPH_MAX_TTS_CHARS has no practical effect.
        """
        if self.graph_max_tts_chars > self.graph_max_llm_response_chars:
            warnings.warn(
                f"GRAPH_MAX_TTS_CHARS ({self.graph_max_tts_chars}) > "
                f"GRAPH_MAX_LLM_RESPONSE_CHARS ({self.graph_max_llm_response_chars}). "
                "The LLM response cap will always trigger before the TTS cap — "
                "GRAPH_MAX_TTS_CHARS has no practical effect at current settings.",
                stacklevel=2,
            )
        return self

    @model_validator(mode="after")
    def _validate_s3_credentials(self) -> Settings:
        """
        S3 integration requires at least an AWS region. Explicit key/secret
        are optional since credentials can come from instance roles, but the
        region must always be set explicitly.
        """
        if (self.stt_s3_bucket or self.tts_s3_bucket) and not self.aws_region:
            raise ValueError(
                "S3 buckets are configured but AWS_REGION is not set. "
                "Set AWS_REGION to the region where your buckets live."
            )
        return self

    @model_validator(mode="after")
    def _validate_upload_vs_stt_limit(self) -> Settings:
        """
        The API upload limit and the STT node's file size limit must agree.
        A mismatch means the API accepts files that the STT node immediately rejects —
        wasting the upload bandwidth and returning a confusing downstream error.
        """
        if self.max_upload_mb > self.stt_max_file_mb:
            raise ValueError(
                f"MAX_UPLOAD_MB ({self.max_upload_mb}) > "
                f"STT_MAX_FILE_MB ({self.stt_max_file_mb}). "
                "The API accepts files that the STT node will then reject. "
                "Set MAX_UPLOAD_MB <= STT_MAX_FILE_MB."
            )
        return self

    @model_validator(mode="after")
    def _validate_controller_timings(self) -> Settings:
        """
        Controller timing values must be internally coherent:

        - debounce_s > poll_interval_s: the debounce window must last longer than
          a single poll tick or it fires again on the very next check.
        - interrupt_drain_s < debounce_s: the interrupt drain pause happens before
          the debounce window. If it were longer, the drain would eat the debounce
          budget entirely and the next PTT press could be missed.
        - loop_join_timeout should comfortably exceed loop_drain_s so the thread
          has time to actually finish draining before we give up waiting for it.
        """
        if self.controller_debounce_s <= self.controller_poll_interval_s:
            raise ValueError(
                f"CONTROLLER_DEBOUNCE_S ({self.controller_debounce_s}s) must be greater than "
                f"CONTROLLER_POLL_INTERVAL_S ({self.controller_poll_interval_s}s). "
                "The debounce window must span at least one full poll interval."
            )
        if self.controller_interrupt_drain_s >= self.controller_debounce_s:
            raise ValueError(
                f"CONTROLLER_INTERRUPT_DRAIN_S ({self.controller_interrupt_drain_s}s) must be "
                f"less than CONTROLLER_DEBOUNCE_S ({self.controller_debounce_s}s). "
                "The interrupt drain happens inside the debounce window — "
                "if it's longer, the debounce provides no protection."
            )
        if self.controller_loop_join_timeout < self.controller_loop_drain_s * 3:
            warnings.warn(
                f"CONTROLLER_LOOP_JOIN_TIMEOUT ({self.controller_loop_join_timeout}s) is less than "
                f"3× CONTROLLER_LOOP_DRAIN_S ({self.controller_loop_drain_s}s). "
                "The thread may not have enough time to drain before the join times out.",
                stacklevel=2,
            )
        return self

    @model_validator(mode="after")
    def _validate_dev_flags(self) -> Settings:
        """
        Catch common production misconfiguration at startup:
        - Wildcard CORS in production is a security concern worth surfacing explicitly.
        - localhost bind address in production means the service is unreachable.
        """
        if self.env == "production":
            if self.cors_origins_raw.strip() == "*":
                warnings.warn(
                    "CORS_ORIGINS is set to '*' in production. "
                    "Consider restricting this to your actual frontend origin(s).",
                    stacklevel=2,
                )
            if self.host in ("127.0.0.1", "localhost"):
                warnings.warn(
                    f"HOST is set to '{self.host}' in production. "
                    "This only accepts connections from localhost. "
                    "Set HOST=0.0.0.0 for the service to be reachable inside a container.",
                    stacklevel=2,
                )
        return self

    # ══════════════════════════════════════════════════════════════════════════
    # DERIVED PROPERTIES
    # Computed from raw validated fields — no additional validation needed.
    # ══════════════════════════════════════════════════════════════════════════

    @property
    def cors_origins(self) -> list[str]:
        """Parsed list of allowed CORS origins from the comma-separated raw string."""
        return [o.strip() for o in self.cors_origins_raw.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.env == "production"

    @property
    def is_development(self) -> bool:
        return self.env == "development"

    @property
    def s3_enabled(self) -> bool:
        """True when at least one S3 bucket is configured."""
        return bool(self.stt_s3_bucket or self.tts_s3_bucket)

    @property
    def uvicorn_reload(self) -> bool:
        """Whether uvicorn should hot-reload on file changes — development only."""
        return self.env == "development"

    @property
    def graph_stage_timeout_sum(self) -> float:
        """
        Sum of the three stage timeouts for the standard graph instance.
        Useful as a reference value when computing effective pipeline deadlines.
        Equal to the minimum valid PIPELINE_TIMEOUT.
        """
        return self.graph_stt_timeout + self.graph_llm_timeout + self.graph_tts_timeout

    @property
    def graph_stream_timeout_min(self) -> float:
        """
        Minimum valid STREAM_TIMEOUT: the sum of STT and LLM timeouts, since
        both run in sequence before the first token is yielded to the caller.
        """
        return self.graph_stt_timeout + self.graph_llm_timeout

    @property
    def node_bottleneck_concurrency(self) -> int:
        """
        The smallest concurrency cap across all three nodes — the effective
        throughput ceiling of the pipeline regardless of how much in-flight
        capacity the load shedder admits.
        """
        return min(
            self.stt_max_concurrent,
            self.llm_max_concurrent,
            self.tts_max_concurrent,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # STARTUP CHECKS
    # Call these from the FastAPI lifespan or __main__ to verify external
    # dependencies are reachable before accepting traffic.
    # ══════════════════════════════════════════════════════════════════════════

    async def check_redis(self) -> None:
        """
        Ping Redis and raise RuntimeError if unreachable.
        Call from the FastAPI lifespan startup hook before accepting requests.
        No-op when redis_enabled=False.
        """
        if not self.redis_enabled:
            return
        import redis.asyncio as aioredis

        try:
            r = await aioredis.from_url(
                self.redis_url,
                socket_connect_timeout=3,
                socket_timeout=3,
            )
            await r.ping()
            await r.aclose()
        except Exception as exc:
            raise RuntimeError(
                f"Redis is not reachable at {self.redis_url}: {exc}. "
                "Check REDIS_URL and that the Redis server is running."
            ) from exc

    async def check_openai(self) -> None:
        """
        Make a cheap OpenAI API call (models list) to verify the key is valid
        and the service is reachable. Call from the lifespan startup hook.
        """
        from openai import AsyncOpenAI

        try:
            client = AsyncOpenAI(api_key=self.openai_api_key.get_secret_value())
            await client.models.list()
            await client.close()
        except Exception as exc:
            raise RuntimeError(
                f"OpenAI API check failed: {exc}. "
                "Verify OPENAI_API_KEY and that you have network access to api.openai.com."
            ) from exc

    def ensure_directories(self) -> None:
        """
        Create required local directories if they don't exist.
        Call once at startup before any file I/O is attempted.
        """
        self.audio_input_dir.mkdir(parents=True, exist_ok=True)
        self.tts_output_dir.mkdir(parents=True, exist_ok=True)


# ── singleton ──────────────────────────────────────────────────────────────────
# One shared instance for the entire process lifetime.
# Import this everywhere instead of reading os.environ directly.
#
# If any required field is missing or any validation fails, this raises a
# ValidationError immediately and the process exits — which is exactly the
# intended behaviour: a misconfigured service should refuse to start rather
# than start and fail on the first request with a cryptic error.

try:
    settings = Settings()
except Exception as exc:
    # Print a clean human-readable error instead of a raw traceback so the
    # misconfiguration is immediately obvious in container startup logs.
    print(
        f"\n[settings] Configuration error — service cannot start:\n{exc}\n",
        file=sys.stderr,
    )
    sys.exit(1)

# ── smoke / debug entry point ──────────────────────────────────────────────────

if __name__ == "__main__":
    # Running this file directly prints a redacted summary of the resolved
    # configuration — useful for verifying what the service actually sees at
    # runtime without needing to attach a debugger.
    import json

    summary = settings.model_dump(
        exclude={"openai_api_key", "aws_access_key_id", "aws_secret_access_key"}
    )

    # Show secrets as redacted tokens rather than omitting them entirely
    # so the operator can confirm the fields are set without seeing the values.
    summary["openai_api_key"] = "sk-***"
    if settings.aws_access_key_id:
        summary["aws_access_key_id"] = "***"
    if settings.aws_secret_access_key:
        summary["aws_secret_access_key"] = "***"

    # Append derived property values so the summary captures everything
    # the pipeline actually operates with, not just the raw field values.
    summary["_derived"] = {
        "cors_origins": settings.cors_origins,
        "s3_enabled": settings.s3_enabled,
        "is_production": settings.is_production,
        "is_development": settings.is_development,
        "graph_stage_timeout_sum": settings.graph_stage_timeout_sum,
        "graph_stream_timeout_min": settings.graph_stream_timeout_min,
        "node_bottleneck_concurrency": settings.node_bottleneck_concurrency,
    }

    print(json.dumps(summary, indent=2, default=str))
