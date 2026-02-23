"""
STT node — OpenAI Whisper API with all the production layers.

Base version features:
  - Uses Faster-Whisper (base model) for offline transcription
  - CPU execution with int8 quantization for lower memory usage
  - Single-shot file-based transcription
  - Beam search decoding (beam_size=5) for balanced accuracy/speed
  - Concatenates segment outputs into a single string
  - Basic file existence validation
  - Graceful error handling (returns empty string on failure)

Beyond the base version:
  - asyncio.Semaphore to cap concurrent transcription jobs so the
    event loop doesn't get crushed under burst traffic
  - per-request asyncio.Task cancellation support (cancel mid-flight)
  - optional S3 upload: if configured, the source audio is pulled from
    S3 rather than local disk, and the transcript is stored back
  - circuit breaker around Whisper API calls
  - exponential backoff retries with jitter
  - language confidence and segment metadata logged per-request
  - full OTel spans + Prometheus counters/histograms
  - structured logging with request_id, file size, duration, language

Partial-streaming additions:
  - transcribe_stream(): splits .wav audio into overlapping N-second
    chunks and yields STTSegment objects as each chunk completes
  - transcribe_fast(): response_format="text" for lower overhead

Distributed-service additions:
  - STTNodeProtocol: the structural interface that both STTNode and
    RemoteSTTClient must satisfy. VoiceGraph imports only the protocol.
  - RemoteSTTClient: calls a remote STT microservice over HTTPS.
    Binary audio is posted as multipart/form-data; streaming responses
    are delivered as SSE. OTel trace headers and LatencyBudget are
    injected on every outbound request.
  - Local Whisper fallback: if the primary API is down and a local
    Whisper endpoint (OPENAI_BASE_URL override) is configured, the node
    falls back to it automatically before the circuit breaker opens.
  - Separate batch/stream bulkheads so a flood of chunked streaming
    requests cannot starve short batch transcriptions.
  - LatencyBudget enforcement at entry so the node self-aborts when
    the pipeline SLA is already blown.
  - get_stt_node() factory selects local vs remote based on env config.
"""

from __future__ import annotations

import asyncio  # noqa
import io
import os
import time
import wave
from io import BytesIO
from pathlib import Path
from typing import Any, AsyncIterator, Protocol, TypedDict, runtime_checkable
from contextlib import asynccontextmanager  # noqa
from opentelemetry import trace

import httpx
from openai import AsyncOpenAI
from opentelemetry.trace import StatusCode

from app.common.shared import (
    CircuitBreaker,
    # noqa
    LatencyBudget,
    LatencyBudgetExceeded,
    RateLimiter,
    ServiceHealthState,
    backoff_retry,
    bulkheads,
    current_request_id,
    get_logger,
    get_tracer,
    inject_trace_headers,
    make_counter,
    make_gauge,
    make_histogram,
    new_request_id,
)

log = get_logger(__name__)
tracer = get_tracer(__name__)

# ── config ────────────────────────────────────────────────────────────────────

OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

STT_MODEL: str = os.getenv("STT_MODEL", "whisper-1")

# If set, used as a secondary local Whisper endpoint (e.g. whisper.cpp server)
STT_LOCAL_FALLBACK_URL: str = os.getenv("STT_LOCAL_FALLBACK_URL", "")

MAX_FILE_MB: float = float(os.getenv("STT_MAX_FILE_MB", "25.0"))
RATE_PER_SEC: float = float(os.getenv("STT_RATE_PER_SEC", "10.0"))
RATE_BURST: float = float(os.getenv("STT_RATE_BURST", "20.0"))

STREAM_CHUNK_S: float = float(os.getenv("STT_STREAM_CHUNK_S", "10.0"))
STREAM_OVERLAP_S: float = float(os.getenv("STT_STREAM_OVERLAP_S", "0.5"))
STREAM_SINGLE_PASS_THRESHOLD_S: float = float(
    os.getenv("STT_STREAM_SINGLE_PASS_THRESHOLD_S", "12.0")
)
STREAM_MAX_PARALLEL: int = int(os.getenv("STT_STREAM_MAX_PARALLEL", "4"))

# S3 — all optional
S3_BUCKET: str | None = os.getenv("STT_S3_BUCKET")
S3_REGION: str = os.getenv("AWS_REGION", "us-east-1")
S3_TRANSCRIPT_PREFIX: str = os.getenv("STT_S3_TRANSCRIPT_PREFIX", "transcripts/")

# Remote service config
STT_SERVICE_URL: str = os.getenv("STT_SERVICE_URL", "")
STT_SERVICE_API_KEY: str = os.getenv("STT_SERVICE_API_KEY", "")
STT_SERVICE_TIMEOUT: float = float(os.getenv("STT_SERVICE_TIMEOUT", "120.0"))

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}
)

# ── Prometheus metrics ────────────────────────────────────────────────────────

_req_total = make_counter(
    "stt_requests_total", "Total STT requests", ["status", "mode", "provider"]
)
_latency = make_histogram(
    "stt_latency_seconds",
    "End-to-end STT latency",
    buckets=(0.5, 1, 2, 3, 5, 8, 15, 30, 60),
)
_ttfs = make_histogram(
    "stt_time_to_first_segment_seconds",
    "Latency from call start to first segment yielded (stream mode)",
    buckets=(0.2, 0.5, 1, 1.5, 2, 3, 5, 8),
)
_file_size_mb = make_histogram(
    "stt_file_size_mb",
    "Audio file size distribution",
    buckets=(0.1, 0.5, 1, 2, 5, 10, 25),
)
_chunks_per_stream = make_histogram(
    "stt_chunks_per_stream",
    "Wav chunks per streaming transcription",
    buckets=(1, 2, 3, 5, 8, 13, 21),
)
_active = make_gauge("stt_active_requests", "STT jobs currently in flight", ["mode"])
_circuit_open = make_gauge(
    "stt_circuit_breaker_open", "1 when the STT breaker is OPEN", ["provider"]
)
_budget_exceeded = make_counter(
    "stt_latency_budget_exceeded_total", "STT requests aborted due to blown SLA"
)


# ── result types ──────────────────────────────────────────────────────────────


class STTResult(TypedDict):
    text: str
    language: str
    duration_s: float
    processing_s: float
    source: str
    s3_transcript_key: str


class STTSegment(TypedDict):
    text: str
    language: str
    start: float
    end: float
    avg_logprob: float
    chunk_index: int
    is_final: bool


# ── STTNodeProtocol ───────────────────────────────────────────────────────────


@runtime_checkable
class STTNodeProtocol(Protocol):
    """
    The contract for local STTNode and RemoteSTTClient.
    VoiceGraph depends only on this protocol.
    """

    async def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        prompt: str | None = None,
        request_id: str | None = None,
    ) -> STTResult: ...

    def transcribe_stream(
        self,
        audio_path: str,
        language: str | None = None,
        prompt: str | None = None,
        request_id: str | None = None,
    ) -> AsyncIterator[STTSegment]: ...

    async def health(self) -> ServiceHealthState: ...

    async def close(self) -> None: ...


# ── S3 helpers ────────────────────────────────────────────────────────────────


async def _s3_download(bucket: str, key: str) -> bytes:
    try:
        import aioboto3  # type: ignore
    except ImportError as exc:
        raise ImportError("Install aioboto3 to use S3 integration.") from exc
    session = aioboto3.Session()
    async with session.client("s3", region_name=S3_REGION) as s3:
        resp = await s3.get_object(Bucket=bucket, Key=key)
        return await resp["Body"].read()


async def _s3_upload_text(bucket: str, key: str, text: str) -> None:
    try:
        import aioboto3  # type: ignore
    except ImportError:
        return
    session = aioboto3.Session()
    async with session.client("s3", region_name=S3_REGION) as s3:
        await s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=text.encode("utf-8"),
            ContentType="text/plain",
        )


# ── wav chunk splitter ────────────────────────────────────────────────────────


def _wav_duration_s(data: bytes) -> float:
    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:  # noqa
        return 0.0


def _split_wav_chunks(
    data: bytes,
    chunk_s: float = STREAM_CHUNK_S,
    overlap_s: float = STREAM_OVERLAP_S,
) -> list[bytes]:
    """
    Split a wav byte blob into overlapping N-second sub-blobs.
    Falls back to the original blob on any parse error.
    """
    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            rate = wf.getframerate()
            nch = wf.getnchannels()
            sw = wf.getsampwidth()
            total_frames = wf.getnframes()
            all_frames = wf.readframes(total_frames)
    except Exception as exc:
        log.warning("wav_split_parse_error", error=str(exc))
        return [data]

    frames_per_chunk = int(rate * chunk_s)
    frames_per_overlap = int(rate * overlap_s)
    bytes_per_frame = nch * sw
    chunks: list[bytes] = []
    offset = 0

    while offset < total_frames:
        end = min(offset + frames_per_chunk + frames_per_overlap, total_frames)
        raw = all_frames[offset * bytes_per_frame : end * bytes_per_frame]
        buf = io.BytesIO()
        with wave.open(buf, "wb") as out:
            out.setnchannels(nch)
            out.setsampwidth(sw)
            out.setframerate(rate)
            out.writeframes(raw)
        chunks.append(buf.getvalue())
        offset += frames_per_chunk

    return chunks or [data]


# ── local STT node ────────────────────────────────────────────────────────────


class STTNode:
    """
    LangGraph-compatible async STT node (local / in-process implementation).

    State contract
    ──────────────
    reads:  state["audio_path"]     (str — local path or s3://bucket/key)
            state.get("language")   (str | None)
            state.get("stt_prompt") (str | None)
    writes: state["user_input"]     (str — transcript)
            state["stt_result"]     (STTResult dict)
            state["request_id"]     (str)

    Distributed usage
    ─────────────────
    Use get_stt_node() instead of instantiating directly. The factory
    returns a RemoteSTTClient when STT_SERVICE_URL is configured.
    """

    def __init__(
        self,
        model: str = STT_MODEL,
        max_file_mb: float = MAX_FILE_MB,
        rate_per_sec: float = RATE_PER_SEC,
        rate_burst: float = RATE_BURST,
    ) -> None:
        self._model = model
        self._max_file_mb = max_file_mb
        self._inflight_batch = 0
        self._inflight_stream = 0

        self._client = AsyncOpenAI(
            api_key=OPENAI_API_KEY,
            max_retries=0,
            timeout=120.0,
        )

        # Local Whisper fallback client (whisper.cpp or compatible server)
        self._local_client: AsyncOpenAI | None = None
        if STT_LOCAL_FALLBACK_URL:
            self._local_client = AsyncOpenAI(
                api_key="local",
                base_url=STT_LOCAL_FALLBACK_URL,
                max_retries=0,
                timeout=120.0,
            )

        self._rate_limiter = RateLimiter(rate_per_sec, rate_burst)

        # Separate breakers for primary API and local fallback
        self._breaker_primary = CircuitBreaker(name="stt:whisper:primary")
        self._breaker_local = CircuitBreaker(name="stt:whisper:local")
        self._stream_chunk_semaphore = asyncio.Semaphore(STREAM_MAX_PARALLEL)

        self._tasks: dict[str, asyncio.Task[Any]] = {}

    # ── validation ────────────────────────────────────────────────────────────

    def _validate_local(self, path: Path) -> float:
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported format '{path.suffix}'. "
                f"Accepted: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
        size_mb = path.stat().st_size / (1024**2)
        if size_mb > self._max_file_mb:
            raise ValueError(
                f"File is {size_mb:.1f} MB — limit is {self._max_file_mb} MB."
            )
        return size_mb

    # ── S3 resolution ─────────────────────────────────────────────────────────

    async def _resolve_audio(
        self, audio_path: str
    ) -> tuple[bytes | None, Path | None, str, float]:
        if audio_path.startswith("s3://"):
            parts = audio_path[5:].split("/", 1)
            if len(parts) != 2:
                raise ValueError(f"Malformed S3 URI: {audio_path}")
            bucket, key = parts
            data = await _s3_download(bucket, key)
            size_mb = len(data) / (1024**2)
            if size_mb > self._max_file_mb:
                raise ValueError(
                    f"S3 object is {size_mb:.1f} MB — limit is {self._max_file_mb} MB."
                )
            return data, None, "s3", size_mb

        path = Path(audio_path)
        size_mb = self._validate_local(path)
        return None, path, "local", size_mb

    # ── Whisper API call (primary + local fallback) ───────────────────────────

    async def _call_whisper(
        self,
        audio_data: bytes | None,
        audio_path: Path | None,
        language: str | None,
        prompt: str | None,
        use_local: bool = False,
    ) -> tuple[str, str, float]:
        """
        Returns (text, detected_language, audio_duration_s).

        If use_local=False (default) we call the primary OpenAI Whisper API.
        If use_local=True we call the local fallback server (e.g. whisper.cpp).
        The local fallback is tried automatically by transcribe() when the
        primary circuit breaker trips.
        """
        client = self._local_client if use_local else self._client
        breaker = self._breaker_local if use_local else self._breaker_primary

        if client is None:
            raise RuntimeError(
                "Local Whisper client not configured (STT_LOCAL_FALLBACK_URL not set)."
            )

        kwargs: dict[str, Any] = {
            "model": self._model,
            "response_format": "verbose_json",
            "timestamp_granularities": ["segment"],
        }
        if language:
            kwargs["language"] = language
        if prompt:
            kwargs["prompt"] = prompt

        async def _do_call() -> Any:
            if audio_data is not None:
                kwargs["file"] = ("audio.wav", BytesIO(audio_data))
                return await client.audio.transcriptions.create(**kwargs)
            with open(audio_path, "rb") as f:  # type: ignore[arg-type]
                kwargs["file"] = f
                return await client.audio.transcriptions.create(**kwargs)

        response = await breaker.call(
            backoff_retry, _do_call, attempts=3, base_delay=1.5, exceptions=(Exception,)
        )

        text = (response.text or "").strip()
        detected_lang = getattr(response, "language", language or "unknown")
        duration = getattr(response, "duration", 0.0) or 0.0

        segments = getattr(response, "segments", []) or []
        if segments:
            avg_logprob = sum(getattr(s, "avg_logprob", 0) for s in segments) / len(
                segments
            )
            log.info(
                "stt_segment_confidence",
                segments=len(segments),
                avg_logprob=round(avg_logprob, 3),
                detected_language=detected_lang,
                provider="local" if use_local else "openai",
            )

        return text, detected_lang, duration

    # ── chunk-level Whisper call (streaming path) ─────────────────────────────

    async def _call_whisper_chunk(
        self,
        wav_bytes: bytes,
        language: str | None,
        prompt: str | None,
        chunk_index: int,
        time_offset_s: float,
    ) -> list[dict[str, Any]]:
        """Transcribe one wav chunk; returns shifted segment dicts."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "response_format": "verbose_json",
            "timestamp_granularities": ["segment"],
        }
        if language:
            kwargs["language"] = language
        if prompt:
            kwargs["prompt"] = prompt

        async def _do_call() -> Any:
            kwargs["file"] = (f"chunk_{chunk_index}.wav", BytesIO(wav_bytes))
            return await self._client.audio.transcriptions.create(**kwargs)

        async with self._stream_chunk_semaphore:
            response = await self._breaker_primary.call(
                backoff_retry,
                _do_call,
                attempts=3,
                base_delay=1.5,
                exceptions=(Exception,),
            )

        detected_lang = getattr(response, "language", language or "unknown")
        raw_segments = getattr(response, "segments", []) or []

        result: list[dict[str, Any]] = []
        for seg in raw_segments:
            result.append(
                {
                    "text": (getattr(seg, "text", "") or "").strip(),
                    "language": detected_lang,
                    "start": round(
                        time_offset_s + (getattr(seg, "start", 0.0) or 0.0), 3
                    ),
                    "end": round(time_offset_s + (getattr(seg, "end", 0.0) or 0.0), 3),
                    "avg_logprob": round(getattr(seg, "avg_logprob", 0.0) or 0.0, 4),
                    "chunk_index": chunk_index,
                }
            )
        return result

    # ── core transcribe ───────────────────────────────────────────────────────

    async def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        prompt: str | None = None,
        request_id: str | None = None,
    ) -> STTResult:
        """
        Transcribe audio using the primary Whisper provider with production safeguards.

        Flow:
        1. Validate input and establish request context.
        2. Enforce latency budget if one is active.
        3. Apply rate limiting and bulkhead isolation.
        4. Resolve audio from local path or remote source.
        5. Attempt transcription using primary provider.
        6. Fall back to local Whisper if configured and primary fails.
        7. Emit metrics, tracing, and optional S3 transcript storage.
        8. Handle cancellation and errors cleanly.
        """

        # Validate input path
        if not audio_path or not audio_path.strip():
            raise ValueError("audio_path must not be empty.")

        # Generate or reuse request identifier
        rid = request_id or new_request_id()

        # Check latency budget if present
        budget = LatencyBudget.current()
        if budget:
            try:
                budget.check(stage="stt.transcribe")
            except LatencyBudgetExceeded:
                _budget_exceeded.inc()
                raise

        # Start tracing span for observability
        with tracer.start_as_current_span("stt.transcribe") as span:
            span.set_attribute("request_id", rid)
            span.set_attribute("audio_path", audio_path)

            # Apply rate limiting before processing
            await self._rate_limiter.acquire()

            # Use bulkhead isolation to protect system resources
            async with bulkheads.acquire("stt.batch"):
                _active.labels(mode="batch").inc()

                # Track circuit breaker state for monitoring
                _circuit_open.labels(provider="primary").set(
                    1 if self._breaker_primary.state == "OPEN" else 0
                )

                self._inflight_batch += 1
                t0 = time.monotonic()
                use_local = False

                try:
                    # Resolve audio source (local file, S3, etc.)
                    audio_data, path, source, size_mb = await self._resolve_audio(
                        audio_path
                    )
                    _file_size_mb.observe(size_mb)

                    span.set_attribute("source", source)
                    span.set_attribute("size_mb", round(size_mb, 2))

                    # Register current task for cooperative cancellation
                    task = asyncio.current_task()
                    if task:
                        self._tasks[rid] = task

                    # Decide provider based on circuit breaker state
                    if self._breaker_primary.state == "OPEN" and self._local_client:
                        log.info("stt_primary_breaker_open_using_local", request_id=rid)
                        use_local = True

                    # Perform transcription with fallback strategy
                    try:
                        text, lang, duration = await self._call_whisper(
                            audio_data,
                            path,
                            language,
                            prompt,
                            use_local=use_local,
                        )
                    except Exception as primary_exc:
                        # Attempt local fallback if primary fails and fallback exists
                        if not use_local and self._local_client:
                            log.warning(
                                "stt_primary_failed_using_local_fallback",
                                request_id=rid,
                                error=str(primary_exc),
                            )
                            text, lang, duration = await self._call_whisper(
                                audio_data,
                                path,
                                language,
                                prompt,
                                use_local=True,
                            )
                            use_local = True
                        else:
                            raise

                    # Successful transcription path
                    processing_s = time.monotonic() - t0
                    provider_label = "local" if use_local else "openai"

                    _req_total.labels(
                        status="ok",
                        mode="batch",
                        provider=provider_label,
                    ).inc()

                    _latency.observe(processing_s)

                    # Optionally upload transcript to S3
                    s3_key: str | None = None
                    if S3_BUCKET:
                        safe_name = (
                            Path(audio_path).stem
                            if not audio_path.startswith("s3://")
                            else audio_path.split("/")[-1]
                        )

                        s3_key = f"{S3_TRANSCRIPT_PREFIX}{rid}_{safe_name}.txt"

                        try:
                            await _s3_upload_text(S3_BUCKET, s3_key, text)
                        except Exception as exc:
                            log.warning("s3_transcript_upload_failed", error=str(exc))
                            s3_key = None

                    # Structured success logging
                    log.info(
                        "stt_ok",
                        request_id=rid,
                        language=lang,
                        audio_duration_s=round(duration, 2),
                        processing_s=round(processing_s, 3),
                        size_mb=round(size_mb, 2),
                        source=source,
                        provider=provider_label,
                    )

                    span.set_attribute("language", lang)
                    span.set_attribute("audio_duration_s", round(duration, 2))

                    return STTResult(
                        text=text,
                        language=lang,
                        duration_s=duration,
                        processing_s=processing_s,
                        source=source,
                        s3_transcript_key=s3_key,
                    )

                except asyncio.CancelledError:
                    # Handle cooperative cancellation
                    log.warning("stt_cancelled", request_id=rid)

                    _req_total.labels(
                        status="cancelled",
                        mode="batch",
                        provider="local" if use_local else "openai",
                    ).inc()

                    raise

                except Exception as exc:
                    # Record failure metrics and tracing details
                    _req_total.labels(
                        status="error",
                        mode="batch",
                        provider="local" if use_local else "openai",
                    ).inc()

                    span.set_status(StatusCode.ERROR, str(exc))
                    log.error("stt_error", request_id=rid, error=str(exc))
                    raise

                finally:
                    # Cleanup resources on all exit paths
                    self._tasks.pop(rid, None)  # noqa
                    _active.labels(mode="batch").dec()
                    self._inflight_batch -= 1

    # ── streaming transcription ───────────────────────────────────────────────

    async def transcribe_stream(
        self,
        audio_path: str,
        language: str | None = None,
        prompt: str | None = None,
        request_id: str | None = None,
        chunk_s: float = STREAM_CHUNK_S,
        overlap_s: float = STREAM_OVERLAP_S,
    ) -> AsyncIterator[STTSegment]:
        """
        Yield STTSegment objects as each wav chunk is transcribed.
        Non-wav or short files fall back to a single pass.
        """
        if not audio_path or not audio_path.strip():
            raise ValueError("audio_path must not be empty.")

        rid = request_id or new_request_id()

        budget = LatencyBudget.current()
        if budget:
            try:
                budget.check(stage="stt.transcribe_stream")
            except LatencyBudgetExceeded:
                _budget_exceeded.inc()
                raise

        span = tracer.start_span("stt.transcribe_stream")
        try:
            with trace.use_span(span, end_on_exit=False):

                span.set_attribute("request_id", rid)
                span.set_attribute("audio_path", audio_path)

                await self._rate_limiter.acquire()

                async with bulkheads.acquire("stt.stream"):
                    # --- Telemetry: track active streaming requests and circuit state ---
                    _active.labels(mode="stream").inc()
                    _circuit_open.labels(provider="primary").set(
                        1 if self._breaker_primary.state == "OPEN" else 0
                    )

                    # Track in-flight streams for backpressure / health checks
                    self._inflight_stream += 1

                    # Wall-clock anchor used for TTFS and total processing latency
                    t0 = time.monotonic()
                    first_segment_emitted = False
                    tasks: list[asyncio.Task[Any]] = []
                    completed_chunks = 0

                try:
                    #    Resolve audio source (URL, file path, or raw bytes) and record
                    #    file size so we can correlate latency with input size later.
                    audio_data, path, source, size_mb = await self._resolve_audio(
                        audio_path
                    )
                    _file_size_mb.observe(size_mb)
                    span.set_attribute("source", source)

                    # Detect WAV format — chunked streaming is only supported for WAV
                    is_wav = (
                        path is not None and path.suffix.lower() == ".wav"
                    ) or audio_path.endswith(".wav")

                    # Materialise raw bytes once; used for duration probing and splitting
                    wav_bytes = (
                        audio_data
                        if audio_data is not None
                        else (path.read_bytes() if path is not None else None)
                    )

                    # Only chunk if the file is WAV and exceeds the single-pass threshold;
                    # short files are transcribed in one shot to avoid unnecessary overhead.
                    use_chunks = (
                        is_wav
                        and wav_bytes is not None
                        and _wav_duration_s(wav_bytes) > STREAM_SINGLE_PASS_THRESHOLD_S
                    )

                    # PATH A — Chunked streaming (long WAV files)
                    # Split the audio into overlapping windows and transcribe them
                    # concurrently. Segments are yielded as soon as any chunk finishes,
                    # minimising time-to-first-segment (TTFS).
                    if use_chunks and wav_bytes is not None:
                        wav_chunks = _split_wav_chunks(wav_bytes, chunk_s, overlap_s)
                        _chunks_per_stream.observe(len(wav_chunks))
                        span.set_attribute("chunks", len(wav_chunks))
                        log.info(
                            "stt_stream_chunked",
                            request_id=rid,
                            chunks=len(wav_chunks),
                            chunk_s=chunk_s,
                        )

                        # Chunk runner — returns (chunk_index, segments) so callers
                        # don't need a side-channel task→index map. This keeps typing
                        # clean when consuming results via asyncio.as_completed().
                        async def run_chunk(
                            idx: int,
                            chunk_wav: bytes,
                            offset: float,
                        ) -> tuple[int, list[dict[str, Any]]]:
                            """
                            Wrapper around _call_whisper_chunk so the task itself
                            returns its chunk_index along with segment results.

                            This avoids needing an external task_map and keeps
                            typing clean when using asyncio.as_completed().
                            """
                            budget = LatencyBudget.current()
                            if budget:
                                budget.check(stage="stt.chunk.before_call")

                            segs = await self._call_whisper_chunk(
                                chunk_wav,
                                language,
                                prompt,
                                idx,
                                offset,
                            )

                            return idx, segs

                        # Launch all chunk tasks concurrently; internal semaphore inside
                        # _call_whisper_chunk prevents us from overwhelming the Whisper API.
                        chunk_tasks: list[
                            asyncio.Task[tuple[int, list[dict[str, Any]]]]
                        ] = []
                        time_offset = 0.0

                        for idx, chunk_wav in enumerate(wav_chunks):
                            task = asyncio.create_task(
                                run_chunk(idx, chunk_wav, time_offset)
                            )
                            chunk_tasks.append(task)
                            # Each chunk starts `chunk_s` seconds later in the timeline
                            time_offset += chunk_s

                        try:
                            total_chunks = len(chunk_tasks)
                            completed_chunks = 0
                            last_chunk_index = total_chunks - 1

                            # Consume tasks in *completion* order rather than submission
                            # order so the fastest chunk can unblock the caller immediately
                            # instead of stalling behind a slow chunk 0.
                            for future in asyncio.as_completed(chunk_tasks):
                                idx, segs = await future
                                completed_chunks += 1

                                for seg_idx, seg in enumerate(segs):
                                    # A segment is logically "final" only when it belongs
                                    # to the last chunk AND is the last segment in that
                                    # chunk AND all concurrent chunks have finished.
                                    is_last_logical_chunk = idx == last_chunk_index
                                    is_last_segment_in_chunk = seg_idx == len(segs) - 1
                                    all_chunks_finished = (
                                        completed_chunks == total_chunks
                                    )

                                    is_final = (
                                        is_last_logical_chunk
                                        and is_last_segment_in_chunk
                                        and all_chunks_finished
                                    )

                                    # Gate on SLA before paying the cost of a yield
                                    budget = LatencyBudget.current()
                                    if budget:
                                        budget.check(stage="stt.chunk.before_yield")

                                    # Record TTFS on the very first segment we emit
                                    if not first_segment_emitted:
                                        _ttfs.observe(time.monotonic() - t0)
                                        first_segment_emitted = True

                                    try:
                                        yield STTSegment(
                                            text=seg["text"],
                                            language=seg["language"],
                                            start=seg["start"],
                                            end=seg["end"],
                                            avg_logprob=seg["avg_logprob"],
                                            chunk_index=idx,
                                            is_final=is_final,
                                        )
                                    except (GeneratorExit, asyncio.CancelledError):
                                        return
                        except asyncio.CancelledError:
                            # Client disconnected or the generator was closed — cancel
                            # every in-flight chunk immediately so we stop burning
                            # Whisper API quota for a request nobody is waiting on.
                            for t in chunk_tasks:
                                if not t.done():
                                    t.cancel()

                            # Await cancelled tasks to prevent "Task destroyed but pending"
                            # warnings from the event loop.
                            if tasks:
                                await asyncio.gather(*tasks, return_exceptions=True)

                            log.warning(
                                "stt_stream_cancelled_chunks",
                                request_id=rid,
                                chunks_total=len(chunk_tasks),
                                chunks_completed=completed_chunks,
                            )

                            raise

                    # PATH B — Single-pass transcription (short files or non-WAV)
                    # Send the entire audio to Whisper in one request and emit a single
                    # terminal segment covering the full duration.
                    else:
                        span.set_attribute("chunks", 1)
                        text, detected_lang, duration = await self._call_whisper(
                            audio_data, path, language, prompt
                        )

                        # TTFS is effectively the full processing time in single-pass mode
                        if not first_segment_emitted:
                            _ttfs.observe(time.monotonic() - t0)

                        try:
                            yield STTSegment(
                                text=text,
                                language=detected_lang,
                                start=0.0,
                                end=duration,
                                avg_logprob=0.0,
                                chunk_index=0,
                                is_final=True,  # Only Segment
                            )
                        except (GeneratorExit, asyncio.CancelledError):
                            # Normal shutdown path (interrupt, SLA abort, etc.)
                            return

                    # Emit success telemetry now that all segments have been yielded.
                    processing_s = time.monotonic() - t0

                    span.set_attribute("processing_s", round(processing_s, 3))
                    chunks_total = len(chunk_tasks) if use_chunks else 1
                    span.set_attribute("chunks_total", chunks_total)

                    _req_total.labels(
                        status="ok", mode="stream", provider="openai"
                    ).inc()
                    _latency.observe(processing_s)

                    log.info(
                        "stt_stream_ok",
                        request_id=rid,
                        processing_s=round(processing_s, 3),
                    )

                except (asyncio.CancelledError, LatencyBudgetExceeded):
                    # Request was cancelled mid-stream or blew its end-to-end SLA.
                    # Cancel any remaining chunk tasks to release Whisper quota promptly.
                    for t in chunk_tasks:
                        if not t.done():
                            t.cancel()

                    # Drain cancelled tasks so the event loop doesn't leak pending futures
                    await asyncio.gather(*chunk_tasks, return_exceptions=True)

                    log.warning(
                        "stt_stream_aborted",
                        request_id=rid,
                        chunks_total=len(chunk_tasks),
                        chunks_completed=completed_chunks,
                    )

                    raise  # Propagate so the caller / framework handles the cancellation

                except Exception as exc:
                    # Unexpected error — record it and re-raise; do not swallow
                    _req_total.labels(
                        status="error", mode="stream", provider="openai"
                    ).inc()
                    span.set_status(StatusCode.ERROR, str(exc))
                    log.error("stt_stream_error", request_id=rid, error=str(exc))
                    raise

                finally:
                    # Always decrement counters regardless of how the request exits so
                    # that metrics and backpressure signals stay accurate.
                    _active.labels(mode="stream").dec()
                    self._inflight_stream -= 1
                    # The bulkhead semaphore is released here as the `async with` exits

        except asyncio.CancelledError:
            return

        finally:
            span.end()

    # ── cancellation support ──────────────────────────────────────────────────

    def cancel(self, request_id: str) -> bool:
        task = self._tasks.get(request_id)
        if task and not task.done():
            task.cancel()
            log.info("stt_cancel_requested", request_id=request_id)
            return True
        return False

    # ── health ────────────────────────────────────────────────────────────────

    async def health(self) -> ServiceHealthState:
        return ServiceHealthState(
            service="stt.local",
            healthy=self._breaker_primary.state != "OPEN",
            circuit_state=self._breaker_primary.state,
            inflight=self._inflight_batch + self._inflight_stream,
        )

    # ── LangGraph node ────────────────────────────────────────────────────────

    async def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        rid = state.get("request_id") or new_request_id()
        result = await self.transcribe(
            audio_path=state["audio_path"],
            language=state.get("language"),
            prompt=state.get("stt_prompt"),
            request_id=rid,
        )
        return {
            **state,
            "request_id": rid,
            "user_input": result["text"],
            "stt_result": dict(result),
        }

    # ── cleanup ───────────────────────────────────────────────────────────────

    async def close(self) -> None:
        await self._client.close()
        if self._local_client:
            await self._local_client.close()


# ── remote STT client ─────────────────────────────────────────────────────────


class RemoteSTTClient:
    """
    Calls a remote STT microservice over HTTPS. Satisfies STTNodeProtocol.

    The remote service is expected to expose:
      POST /transcribe            → STTResult JSON
      POST /transcribe/stream     → SSE of STTSegment JSON objects
      GET  /health                → {"healthy": bool, ...}

    Audio files are posted as multipart/form-data (field name: "audio").
    S3 URIs are posted as a JSON body instead of uploading the bytes.
    """

    def __init__(
        self,
        base_url: str = STT_SERVICE_URL,
        api_key: str = STT_SERVICE_API_KEY,
        timeout: float = STT_SERVICE_TIMEOUT,
    ) -> None:
        if not base_url:
            raise ValueError("STT_SERVICE_URL must be set to use RemoteSTTClient.")
        self._base_url = base_url.rstrip("/")
        self._breaker = CircuitBreaker(name="stt:remote")
        self._inflight = 0

        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        )

    def _request_headers(self) -> dict[str, str]:  # noqa
        headers: dict[str, str] = {}
        inject_trace_headers(headers)
        budget = LatencyBudget.current()
        if budget:
            headers["X-Latency-Budget-Ms"] = budget.as_header_value()
            headers["X-Request-Id"] = current_request_id()
        return headers

    async def _load_audio(self, audio_path: str) -> bytes | None:  # noqa
        """Return raw bytes for local files; None for s3:// URIs (sent as JSON field)."""
        if audio_path.startswith("s3://"):
            return None
        return Path(audio_path).read_bytes()

    async def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        prompt: str | None = None,
        request_id: str | None = None,
    ) -> STTResult:
        rid = request_id or new_request_id()
        budget = LatencyBudget.current()
        if budget:
            budget.check(stage="remote_stt.transcribe")

        headers = self._request_headers()
        headers["X-Request-Id"] = rid

        with tracer.start_as_current_span("stt.remote.transcribe") as span:
            span.set_attribute("request_id", rid)
            self._inflight += 1
            t0 = time.monotonic()
            try:
                audio_bytes = await self._load_audio(audio_path)

                async def _call() -> dict[str, Any]:
                    if audio_bytes is not None:
                        files = {"audio": (Path(audio_path).name, audio_bytes)}
                        data = {}
                        if language:
                            data["language"] = language
                        if prompt:
                            data["prompt"] = prompt
                        resp = await self._http.post(
                            "/transcribe", files=files, data=data, headers=headers
                        )
                    else:
                        # S3 URI path — send as JSON
                        resp = await self._http.post(
                            "/transcribe",
                            json={
                                "audio_path": audio_path,
                                "language": language,
                                "prompt": prompt,
                            },
                            headers=headers,
                        )
                    resp.raise_for_status()
                    return resp.json()

                result = await self._breaker.call(
                    backoff_retry,
                    _call,
                    attempts=3,
                    base_delay=1.5,
                    exceptions=(Exception,),
                )
                span.set_attribute("latency_s", round(time.monotonic() - t0, 3))
                return STTResult(**result)

            except Exception as exc:
                span.set_status(StatusCode.ERROR, str(exc))
                log.error("remote_stt_transcribe_error", request_id=rid, error=str(exc))
                raise
            finally:
                self._inflight -= 1

    async def transcribe_stream(
        self,
        audio_path: str,
        language: str | None = None,
        prompt: str | None = None,
        request_id: str | None = None,
    ) -> AsyncIterator[STTSegment]:
        """
        Stream STTSegment objects from the remote service via SSE.
        Each event is a JSON-serialised STTSegment.
        """
        import json as _json

        rid = request_id or new_request_id()
        budget = LatencyBudget.current()
        if budget:
            budget.check(stage="remote_stt.stream")

        headers = self._request_headers()
        headers["X-Request-Id"] = rid
        headers["Accept"] = "text/event-stream"

        with tracer.start_as_current_span("stt.remote.stream") as span:
            span.set_attribute("request_id", rid)
            self._inflight += 1
            try:
                audio_bytes = await self._load_audio(audio_path)
                if audio_bytes is not None:
                    files = {"audio": (Path(audio_path).name, audio_bytes)}
                    data = {}
                    if language:
                        data["language"] = language
                    if prompt:
                        data["prompt"] = prompt
                    req_kwargs = {"files": files, "data": data}
                else:
                    req_kwargs = {
                        "json": {
                            "audio_path": audio_path,
                            "language": language,
                            "prompt": prompt,
                        }
                    }

                async with self._http.stream(
                    "POST", "/transcribe/stream", headers=headers, **req_kwargs
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if raw == "[DONE]":
                            break

                        # Enforce SLA before emitting remote segment
                        budget = LatencyBudget.current()
                        if budget:
                            budget.check(stage="remote_stt.stream.before_yield")

                        try:
                            segment = STTSegment(**_json.loads(raw))
                        except Exception:  # noqa
                            log.warning("remote_stt_segment_parse_error", raw=raw[:100])
                        else:
                            try:
                                yield segment
                            except (GeneratorExit, asyncio.CancelledError):
                                return

            except asyncio.CancelledError:
                log.warning("remote_stt_stream_cancelled", request_id=rid)
                raise
            except Exception as exc:
                span.set_status(StatusCode.ERROR, str(exc))
                log.error("remote_stt_stream_error", request_id=rid, error=str(exc))
                raise
            finally:
                self._inflight -= 1

    async def health(self) -> ServiceHealthState:
        try:
            resp = await self._http.get("/health", timeout=5.0)
            data = resp.json()
            return ServiceHealthState(
                service="stt.remote",
                healthy=data.get("healthy", False),
                circuit_state=self._breaker.state,
                inflight=self._inflight,
            )
        except Exception as exc:
            return ServiceHealthState(
                service="stt.remote",
                healthy=False,
                circuit_state=self._breaker.state,
                inflight=self._inflight,
                error=str(exc),
            )

    async def close(self) -> None:
        await self._http.aclose()


# ── node factory ──────────────────────────────────────────────────────────────


def get_stt_node() -> STTNodeProtocol:
    """
    Return RemoteSTTClient if STT_SERVICE_URL is set, else local STTNode.
    VoiceGraph imports only this factory.
    """
    if STT_SERVICE_URL:
        log.info("stt_using_remote_client", url=STT_SERVICE_URL)
        return RemoteSTTClient()
    log.info("stt_using_local_node", model=STT_MODEL)
    return STTNode()


# ── module-level singleton (backward-compatible) ──────────────────────────────

stt_node: STTNodeProtocol = get_stt_node()

if __name__ == "__main__":
    import asyncio
    import sys

    async def _smoke():
        path = sys.argv[1] if len(sys.argv) > 1 else "audio/audio_IN/test.wav"
        node = get_stt_node()

        print("── batch transcribe ──")
        result = await node.transcribe(path)
        print("Transcript:", result["text"])

        print("\n── streaming transcribe ──")
        async for seg in node.transcribe_stream(path):
            flag = "[FINAL]" if seg["is_final"] else f"[chunk {seg['chunk_index']}]"
            print(f"{flag} {seg['start']:.1f}s–{seg['end']:.1f}s: {seg['text']}")

        print("\nHealth:", await node.health())
        await node.close()

    asyncio.run(_smoke())
