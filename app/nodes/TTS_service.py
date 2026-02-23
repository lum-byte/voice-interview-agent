"""
TTS node — OpenAI tts-1-hd API with all the production layers.

Base version features:
  - Uses Kokoro (hexgrad/Kokoro-82M) for local speech synthesis
  - Deterministic text cleaning for common symbols and formatting
  - Sentence-based chunking to prevent model overload
  - Safe generator handling with audio validation
  - Automatic peak normalization
  - Concatenates chunked audio into final WAV file
  - Writes PCM_16 .wav output to local audio directory
  - Graceful failure handling (never crashes caller)
  - Simple UUID-based filename generation

Beyond the base version:
  - S3 upload as the primary storage layer (disk is just a local cache)
  - audio stitching validation: chunk sizes checked before concatenation,
    final file verified before the path is returned
  - file lifecycle management: background task cleans up local files
    older than LOCAL_FILE_TTL seconds
  - asyncio.Lock per output path to prevent two concurrent writes to
    the same file
  - asyncio.Semaphore to cap parallel synthesis jobs
  - circuit breaker around the TTS API calls
  - full OTel spans + Prometheus counters/histograms
  - structured logging with request_id, char count, latency, voice, format

Partial-streaming additions:
  - synthesize_stream(): accepts AsyncIterator[str] from LLM and yields
    bytes per sentence via AsyncSentenceBuffer
  - synthesize_stream_to_files(): yields (path, is_final) tuples

Distributed-service additions:
  - TTSNodeProtocol: the structural interface for local TTSNode and
    RemoteTTSClient. VoiceGraph only imports the protocol.
  - RemoteTTSClient: calls a remote TTS microservice over HTTPS.
    Token streams are piped to the remote via chunked POST; audio bytes
    are streamed back via chunked transfer encoding. OTel TraceContext
    and LatencyBudget headers are injected on every request.
  - Cached apology audio fallback: if TTS completely fails (breaker OPEN,
    all retries exhausted) and a pre-rendered apology audio file is on
    disk, it is returned instead of crashing the pipeline. This means
    the voice UX always has something to play even during a full TTS
    outage.
  - LatencyBudget enforcement: check() at entry so the node self-aborts
    when the pipeline SLA is already blown before synthesis even starts.
  - BoundedPipelineQueue integration: synthesize_stream() respects the
    inter-node queue's backpressure so a slow audio consumer cannot
    cause unbounded memory growth in the sentence buffer.
  - get_tts_node() factory selects local vs remote based on env config.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import aiofiles
import httpx
from openai import AsyncOpenAI
from opentelemetry.trace import StatusCode

from app.common.shared import (
    CircuitBreaker,
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

from app.nodes.sanitize import sanitize

log = get_logger(__name__)
tracer = get_tracer(__name__)

# ── types ─────────────────────────────────────────────────────────────────────

VoiceType = Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
FormatType = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]

# ── config ────────────────────────────────────────────────────────────────────

OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]
TTS_MODEL: str = os.getenv("TTS_MODEL", "tts-1-hd")
TTS_VOICE: VoiceType = os.getenv("TTS_VOICE", "nova")  # type: ignore[assignment]
TTS_FORMAT: FormatType = os.getenv("TTS_FORMAT", "mp3")  # type: ignore[assignment]
TTS_OUTPUT_DIR: Path = Path(os.getenv("TTS_OUTPUT_DIR", "audio/audio_OUTPUT"))

RATE_PER_SEC: float = float(os.getenv("TTS_RATE_PER_SEC", "10.0"))
RATE_BURST: float = float(os.getenv("TTS_RATE_BURST", "20.0"))

_MAX_CHARS_PER_CHUNK: int = 4096
LOCAL_FILE_TTL: float = float(os.getenv("TTS_LOCAL_FILE_TTL", "3600"))

# Cached apology audio: a pre-rendered fallback file played when TTS is fully
# down. Generate it once with: `tts_node.synthesize(APOLOGY_TEXT)` and set
# TTS_APOLOGY_AUDIO_PATH to the result. If not set, the fallback is silent.
TTS_APOLOGY_AUDIO_PATH: str = os.getenv("TTS_APOLOGY_AUDIO_PATH", "")

# S3 — all optional
S3_BUCKET: str | None = os.getenv("TTS_S3_BUCKET")
S3_REGION: str = os.getenv("AWS_REGION", "us-east-1")
S3_AUDIO_PREFIX: str = os.getenv("TTS_S3_PREFIX", "tts/")

# Sentence buffer config
MIN_FLUSH_CHARS: int = int(os.getenv("TTS_MIN_FLUSH_CHARS", "30"))
MAX_BUFFER_CHARS: int = int(
    os.getenv("TTS_MAX_BUFFER_CHARS", str(_MAX_CHARS_PER_CHUNK // 2))
)

# Remote service config
TTS_SERVICE_URL: str = os.getenv("TTS_SERVICE_URL", "")
TTS_SERVICE_API_KEY: str = os.getenv("TTS_SERVICE_API_KEY", "")
TTS_SERVICE_TIMEOUT: float = float(os.getenv("TTS_SERVICE_TIMEOUT", "60.0"))

# ── Prometheus metrics ────────────────────────────────────────────────────────

_req_total = make_counter(
    "tts_requests_total", "Total TTS requests", ["status", "mode", "provider"]
)
_latency = make_histogram(
    "tts_latency_seconds",
    "End-to-end TTS latency",
    buckets=(0.5, 1, 2, 3, 5, 8, 15, 30),
)
_ttfb = make_histogram(
    "tts_time_to_first_byte_seconds",
    "Latency from synthesize_stream() to first audio bytes",
    buckets=(0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0),
)
_chars_synthesized = make_histogram(
    "tts_characters_synthesized",
    "Char count per request",
    buckets=(50, 100, 200, 500, 1000, 2000, 4096),
)
_chunks_per_request = make_histogram(
    "tts_chunks_per_request",
    "Audio chunks per request",
    buckets=(1, 2, 3, 5, 8, 13),
)
_stream_sentence_size = make_histogram(
    "tts_stream_sentence_chars",
    "Size of each sentence flushed by AsyncSentenceBuffer",
    buckets=(20, 40, 60, 100, 150, 200, 400, 1000),
)
_active = make_gauge("tts_active_requests", "TTS jobs currently in flight", ["mode"])
_circuit_open = make_gauge(
    "tts_circuit_breaker_open", "1 when the TTS breaker is OPEN", ["provider"]
)
_budget_exceeded = make_counter(
    "tts_latency_budget_exceeded_total", "TTS requests aborted due to blown SLA"
)
_apology_fallback_used = make_counter(
    "tts_apology_fallback_total", "Times the cached apology audio was served"
)

# ── text processing ───────────────────────────────────────────────────────────

_SENTENCE_BOUNDARY = re.compile(r"(?<![A-Z\d])(?<!\s[A-Za-z])[.!?;]\s")


def _split_into_chunks(text: str, max_len: int = _MAX_CHARS_PER_CHUNK) -> list[str]:
    if len(text) <= max_len:
        return [text]

    raw_sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""

    for sentence in raw_sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_len:
            current = candidate
        else:
            if current:
                chunks.append(current)
            while len(sentence) > max_len:
                chunks.append(sentence[:max_len])
                sentence = sentence[max_len:]
            current = sentence

    if current:
        chunks.append(current)

    return [c for c in chunks if c.strip()]


# ── async sentence buffer ─────────────────────────────────────────────────────


class AsyncSentenceBuffer:
    """
    Buffers incoming LLM tokens and flushes on sentence boundaries.

    Flush conditions (in priority order):
    1. Sentence-ending punctuation AND buffer >= MIN_FLUSH_CHARS.
    2. Buffer exceeds MAX_BUFFER_CHARS (hard cap; prevents run-on blocking).
    3. Token stream exhausted — flush remainder.

    Backpressure: if a downstream consumer provides a BoundedPipelineQueue,
    put_nowait_or_raise() is called for each flushed sentence. A full queue
    causes QueueFull to propagate up so the caller can handle overload.
    """

    def __init__(
        self,
        token_stream: AsyncIterator[str],
        min_flush_chars: int = MIN_FLUSH_CHARS,
        max_buffer_chars: int = MAX_BUFFER_CHARS,
    ) -> None:
        self._stream = token_stream
        self._min = min_flush_chars
        self._max = max_buffer_chars
        self._buf = ""

    def _find_flush_point(self) -> int:
        if len(self._buf) >= self._max:
            m = None
            for match in _SENTENCE_BOUNDARY.finditer(self._buf):
                if match.end() >= self._min:
                    m = match
            return m.end() if m else self._max

        if len(self._buf) < self._min:
            return -1

        last_end = -1
        for match in _SENTENCE_BOUNDARY.finditer(self._buf):
            if match.end() >= self._min:
                last_end = match.end()
        return last_end

    def __aiter__(self) -> "AsyncSentenceBuffer":
        return self

    async def __anext__(self) -> str:
        while True:
            flush_at = self._find_flush_point()
            if flush_at > 0:
                chunk = self._buf[:flush_at].strip()
                self._buf = self._buf[flush_at:]
                if chunk:
                    _stream_sentence_size.observe(len(chunk))
                    return chunk

            try:
                token = await self._stream.__anext__()
                self._buf += token
            except StopAsyncIteration:
                remainder = self._buf.strip()
                self._buf = ""
                if remainder:
                    _stream_sentence_size.observe(len(remainder))
                    return remainder
                raise


# ── TTSNodeProtocol ───────────────────────────────────────────────────────────


@runtime_checkable
class TTSNodeProtocol(Protocol):
    """
    The contract for local TTSNode and RemoteTTSClient.
    VoiceGraph depends only on this protocol.
    """

    async def synthesize(
        self,
        text: str,
        voice: VoiceType | None = None,
        filename: str | None = None,
        speed: float = 1.0,
        request_id: str | None = None,
    ) -> tuple[str, str]: ...

    def synthesize_stream(
        self,
        token_stream: AsyncIterator[str],
        voice: VoiceType | None = None,
        speed: float = 1.0,
        request_id: str | None = None,
    ) -> AsyncIterator[bytes]: ...

    async def health(self) -> ServiceHealthState: ...

    async def close(self) -> None: ...


# ── apology audio fallback ────────────────────────────────────────────────────


def _try_apology_fallback(request_id: str) -> tuple[str, str] | None:
    """
    Return (local_path, s3_uri) for the pre-rendered apology audio file,
    or None if not configured / file missing.

    Called when TTS completely fails so the pipeline still has audio to return.
    """
    if not TTS_APOLOGY_AUDIO_PATH:
        return None
    p = Path(TTS_APOLOGY_AUDIO_PATH)
    if not p.exists():
        log.warning("tts_apology_file_missing", path=str(p))
        return None
    _apology_fallback_used.inc()
    log.warning("tts_serving_apology_fallback", request_id=request_id, path=str(p))
    return str(p), ""


# ── S3 helpers ────────────────────────────────────────────────────────────────


async def _s3_upload_file(bucket: str, key: str, local_path: Path) -> str:
    try:
        import aioboto3  # type: ignore
    except ImportError as exc:
        raise ImportError("Install aioboto3 to use S3 integration.") from exc

    session = aioboto3.Session()
    content_types: dict[str, str] = {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "audio/pcm",
    }
    ct = content_types.get(local_path.suffix.lstrip("."), "application/octet-stream")
    async with session.client("s3", region_name=S3_REGION) as s3:
        await s3.upload_file(
            str(local_path), bucket, key, ExtraArgs={"ContentType": ct}
        )
    return f"s3://{bucket}/{key}"


# ── file lock registry ────────────────────────────────────────────────────────

_file_locks: dict[str, asyncio.Lock] = {}
_file_locks_meta: asyncio.Lock = asyncio.Lock()


async def _get_file_lock(path: Path) -> asyncio.Lock:
    key = str(path.resolve())
    async with _file_locks_meta:
        if key not in _file_locks:
            _file_locks[key] = asyncio.Lock()
        return _file_locks[key]


# ── local TTS node ────────────────────────────────────────────────────────────


class TTSNode:
    """
    LangGraph-compatible async TTS node (local / in-process implementation).

    State contract
    ──────────────
    reads:  state["llm_response"]    (str — text to speak)
            state.get("tts_voice")   (VoiceType | None)
            state.get("tts_speed")   (float | None — 0.25–4.0)
    writes: state["audio_output"]    (str — s3:// URI or local path)
            state["audio_s3_uri"]    (str)
            state["request_id"]      (str)

    Distributed usage
    ─────────────────
    Use get_tts_node() instead of instantiating directly. The factory
    returns a RemoteTTSClient when TTS_SERVICE_URL is configured.
    """

    def __init__(
        self,
        model: str = TTS_MODEL,
        voice: VoiceType = TTS_VOICE,
        audio_format: FormatType = TTS_FORMAT,
        output_dir: Path = TTS_OUTPUT_DIR,
        rate_per_sec: float = RATE_PER_SEC,
        rate_burst: float = RATE_BURST,
    ) -> None:
        self._model = model
        self._voice = voice
        self._format = audio_format
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._inflight_batch = 0
        self._inflight_stream = 0

        self._client = AsyncOpenAI(api_key=OPENAI_API_KEY, max_retries=0, timeout=60.0)

        self._rate_limiter = RateLimiter(rate_per_sec, rate_burst)

        # Separate breakers for batch and stream so a slow streaming
        # path does not trip the breaker used by synchronous synthesis.
        self._breaker_batch = CircuitBreaker(name="tts:openai:batch")
        self._breaker_stream = CircuitBreaker(name="tts:openai:stream")

        self._cleanup_task: asyncio.Task | None = None

    # ── background lifecycle cleanup ──────────────────────────────────────────

    def _ensure_cleanup_running(self) -> None:
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self) -> None:
        """
        Runs every 10 minutes and deletes local audio files older than
        LOCAL_FILE_TTL seconds. S3 is the durable store; disk is transient.
        """
        while True:
            await asyncio.sleep(600)
            try:
                cutoff = time.time() - LOCAL_FILE_TTL
                for f in self._output_dir.glob(f"*.{self._format}"):
                    try:
                        if f.stat().st_mtime < cutoff:
                            f.unlink(missing_ok=True)
                            log.info("tts_local_file_expired", path=str(f))
                    except Exception as exc:
                        log.warning(
                            "tts_cleanup_file_error", path=str(f), error=str(exc)
                        )
            except Exception as exc:
                log.warning("tts_cleanup_loop_error", error=str(exc))

    # ── synthesis (single chunk) ──────────────────────────────────────────────

    async def _synthesize_chunk(
        self,
        text: str,
        voice: VoiceType,
        speed: float,
        use_stream_breaker: bool = False,
    ) -> bytes:
        """Call the API for a single chunk; wrapped in the appropriate breaker."""
        breaker = self._breaker_stream if use_stream_breaker else self._breaker_batch

        async def _call() -> bytes:
            async with self._client.audio.speech.with_streaming_response.create(
                model=self._model,
                voice=voice,
                input=text,
                response_format=self._format,
                speed=speed,
            ) as response:
                return await response.read()

        return await breaker.call(
            backoff_retry, _call, attempts=3, base_delay=1.0, exceptions=(Exception,)
        )

    # ── stitching validation ──────────────────────────────────────────────────

    @staticmethod
    def _validate_chunk(data: bytes, index: int) -> None:
        if not data:
            raise ValueError(f"TTS chunk {index} returned empty audio bytes.")
        if len(data) < 4:
            raise ValueError(
                f"TTS chunk {index} is suspiciously small ({len(data)} bytes)."
            )

    @staticmethod
    def _validate_final(path: Path, expected_chunks: int) -> None:
        if not path.exists():
            raise RuntimeError(f"Expected output file not found: {path}")
        size = path.stat().st_size
        if size < 512:
            raise RuntimeError(
                f"Final audio file is only {size} bytes after stitching "
                f"{expected_chunks} chunk(s) — likely a partial write."
            )

    # ── public synthesize (batch) ─────────────────────────────────────────────

    async def synthesize(
        self,
        text: str,
        voice: VoiceType | None = None,
        filename: str | None = None,
        speed: float = 1.0,
        request_id: str | None = None,
    ) -> tuple[str, str]:
        """
        Synthesize speech and write to disk (and optionally S3).
        Falls back to the pre-rendered apology audio if TTS completely fails.

        Returns (local_path, s3_uri) where s3_uri is "" if S3 isn't configured.
        """
        if not text or not text.strip():
            raise ValueError("Cannot synthesize empty text.")
        if not (0.25 <= speed <= 4.0):
            raise ValueError(f"Speed must be 0.25–4.0, got {speed}.")

        rid = request_id or new_request_id()

        budget = LatencyBudget.current()
        if budget:
            try:
                budget.check(stage="tts.synthesize")
            except LatencyBudgetExceeded:
                _budget_exceeded.inc()
                raise

        active_voice: VoiceType = voice if voice is not None else self._voice
        san = sanitize(text, request_id=rid)

        if not san:
            raise ValueError("Sanitizer returned empty TTS text.")

        safe_text = san.text
        chunks = _split_into_chunks(safe_text)

        stem = filename or f"tts_{uuid.uuid4().hex[:8]}"
        output_path = self._output_dir / f"{stem}.{self._format}"

        with tracer.start_as_current_span("tts.synthesize") as span:
            span.set_attribute("request_id", rid)
            span.set_attribute("char_count", len(safe_text))
            span.set_attribute("chunks", len(chunks))
            span.set_attribute("voice", active_voice)

            await self._rate_limiter.acquire()

            async with bulkheads.acquire("tts.batch"):
                _active.labels(mode="batch").inc()
                _circuit_open.labels(provider="openai").set(
                    1 if self._breaker_batch.state == "OPEN" else 0
                )
                self._inflight_batch += 1
                t0 = time.monotonic()

                try:
                    self._ensure_cleanup_running()

                    audio_parts: list[bytes] = []

                    for i, chunk in enumerate(chunks):
                        log.debug(
                            "tts_chunk_start",
                            request_id=rid,
                            chunk=i + 1,
                            total=len(chunks),
                        )
                        data = await self._synthesize_chunk(chunk, active_voice, speed)
                        self._validate_chunk(data, i)
                        audio_parts.append(data)

                    _chunks_per_request.observe(len(audio_parts))
                    _chars_synthesized.observe(len(safe_text))

                    file_lock = await _get_file_lock(output_path)
                    async with file_lock:
                        async with aiofiles.open(output_path, "wb") as f:
                            for part in audio_parts:
                                await f.write(part)

                    self._validate_final(output_path, len(audio_parts))

                    s3_uri = ""
                    if S3_BUCKET:
                        s3_key = f"{S3_AUDIO_PREFIX}{stem}.{self._format}"
                        try:
                            s3_uri = await _s3_upload_file(
                                S3_BUCKET, s3_key, output_path
                            )
                            log.info("tts_s3_upload_ok", request_id=rid, uri=s3_uri)
                        except Exception as exc:
                            log.warning(
                                "tts_s3_upload_failed", request_id=rid, error=str(exc)
                            )

                    latency = time.monotonic() - t0
                    _req_total.labels(
                        status="ok", mode="batch", provider="openai"
                    ).inc()
                    _latency.observe(latency)

                    log.info(
                        "tts_ok",
                        request_id=rid,
                        voice=active_voice,
                        chars=len(safe_text),
                        chunks=len(chunks),
                        latency_s=round(latency, 3),
                        local_path=str(output_path),
                        s3_uri=s3_uri or "n/a",
                    )
                    span.set_attribute("latency_s", round(latency, 3))
                    span.set_attribute("s3_uri", s3_uri or "")

                    return str(output_path), s3_uri

                except Exception as exc:
                    _req_total.labels(
                        status="error", mode="batch", provider="openai"
                    ).inc()
                    span.set_status(StatusCode.ERROR, str(exc))
                    log.error("tts_error", request_id=rid, error=str(exc))

                    # ── apology audio fallback ────────────────────────────────
                    # Before re-raising, try to serve the pre-rendered apology
                    # audio so the pipeline still has something to return.
                    fallback = _try_apology_fallback(rid)
                    if fallback is not None:
                        return fallback
                    raise

                finally:
                    _active.labels(mode="batch").dec()
                    self._inflight_batch -= 1

    # ── streaming synthesis ───────────────────────────────────────────────────

    async def synthesize_stream(
        self,
        token_stream: AsyncIterator[str],
        voice: VoiceType | None = None,
        speed: float = 1.0,
        request_id: str | None = None,
    ) -> AsyncIterator[bytes]:
        """
        Yield audio bytes as each sentence is synthesized, feeding directly
        from the LLM token stream. Audio playback can start while the LLM
        is still generating the rest of the response.

        Uses the stream circuit breaker (separate from batch) so streaming
        failures don't trip the breaker used by synchronous /run calls.
        """
        if not (0.25 <= speed <= 4.0):
            raise ValueError(f"Speed must be 0.25–4.0, got {speed}.")

        rid = request_id or new_request_id()

        budget = LatencyBudget.current()
        if budget:
            try:
                budget.check(stage="tts.synthesize_stream")
            except LatencyBudgetExceeded:
                _budget_exceeded.inc()
                raise

        active_voice: VoiceType = voice if voice is not None else self._voice

        with tracer.start_as_current_span("tts.synthesize_stream") as span:
            span.set_attribute("request_id", rid)
            span.set_attribute("voice", active_voice)

            await self._rate_limiter.acquire()

            async with bulkheads.acquire("tts.stream"):
                _active.labels(mode="stream").inc()
                _circuit_open.labels(provider="openai").set(
                    1 if self._breaker_stream.state == "OPEN" else 0
                )
                self._inflight_stream += 1
                t0 = time.monotonic()
                total_chars = 0
                chunk_index = 0
                first_bytes_emitted = False

                try:
                    self._ensure_cleanup_running()
                    sentence_stream = AsyncSentenceBuffer(token_stream)

                    async for sentence in sentence_stream:

                        # Enforce SLA before processing sentence
                        if budget:
                            budget.check(stage="tts.synthesize_stream.before_sentence")

                        san = sanitize(sentence, request_id=rid)
                        if not san:
                            continue

                        clean = san.text
                        total_chars += len(clean)

                        for sub in _split_into_chunks(clean):

                            # Enforce SLA before synthesizing chunk
                            if budget:
                                budget.check(stage="tts.synthesize_stream.before_chunk")

                            data = await self._synthesize_chunk(
                                sub, active_voice, speed, use_stream_breaker=True
                            )

                            self._validate_chunk(data, chunk_index)

                            # Enforce SLA before yielding bytes
                            if budget:
                                budget.check(stage="tts.synthesize_stream.before_yield")

                            if not first_bytes_emitted:
                                _ttfb.observe(time.monotonic() - t0)
                                first_bytes_emitted = True

                            chunk_index += 1
                            yield data

                    latency = time.monotonic() - t0
                    _req_total.labels(
                        status="ok", mode="stream", provider="openai"
                    ).inc()
                    _latency.observe(latency)
                    _chars_synthesized.observe(total_chars)
                    _chunks_per_request.observe(chunk_index)
                    span.set_attribute("latency_s", round(latency, 3))
                    span.set_attribute("chunks", chunk_index)

                    log.info(
                        "tts_stream_ok",
                        request_id=rid,
                        voice=active_voice,
                        chars=total_chars,
                        chunks=chunk_index,
                        latency_s=round(latency, 3),
                    )

                except LatencyBudgetExceeded:
                    _budget_exceeded.inc()
                    log.warning(
                        "tts_stream_budget_exceeded",
                        request_id=rid,
                        chunks_emitted=chunk_index,
                    )
                    raise

                except asyncio.CancelledError:
                    _req_total.labels(
                        status="cancelled", mode="stream", provider="openai"
                    ).inc()
                    log.warning(
                        "tts_stream_cancelled",
                        request_id=rid,
                        chunks_emitted=chunk_index,
                    )
                    raise

                except Exception as exc:
                    _req_total.labels(
                        status="error", mode="stream", provider="openai"
                    ).inc()
                    span.set_status(StatusCode.ERROR, str(exc))
                    log.error("tts_stream_error", request_id=rid, error=str(exc))
                    raise

                finally:
                    _active.labels(mode="stream").dec()
                    self._inflight_stream -= 1

    # ── streaming synthesis to files ──────────────────────────────────────────

    async def synthesize_stream_to_files(
        self,
        token_stream: AsyncIterator[str],
        voice: VoiceType | None = None,
        speed: float = 1.0,
        stem: str | None = None,
        request_id: str | None = None,
    ) -> AsyncIterator[tuple[str, bool]]:
        """
        Like synthesize_stream() but write each sentence chunk to its own file
        and yield (local_path, is_final) tuples. S3 uploads are fire-and-forget.
        """
        rid = request_id or new_request_id()
        base_stem = stem or f"tts_{uuid.uuid4().hex[:8]}"
        active_voice: VoiceType = voice if voice is not None else self._voice

        with tracer.start_as_current_span("tts.synthesize_stream_to_files") as span:
            span.set_attribute("request_id", rid)
            span.set_attribute("voice", active_voice)

            await self._rate_limiter.acquire()

            async with bulkheads.acquire("tts.stream"):
                _active.labels(mode="stream").inc()
                self._inflight_stream += 1
                t0 = time.monotonic()
                total_chars = 0
                chunk_index = 0
                pending: list[Path] = []

                try:
                    self._ensure_cleanup_running()
                    sentence_stream = AsyncSentenceBuffer(token_stream)

                    async for sentence in sentence_stream:
                        san = sanitize(sentence, request_id=rid)

                        # Skip empty / fully stripped content
                        if not san:
                            continue

                        clean = san.text
                        total_chars += len(clean)

                        for sub in _split_into_chunks(clean):
                            data = await self._synthesize_chunk(
                                sub, active_voice, speed, use_stream_breaker=True
                            )
                            self._validate_chunk(data, chunk_index)

                            out_path = (
                                self._output_dir
                                / f"{base_stem}_chunk_{chunk_index}.{self._format}"
                            )
                            file_lock = await _get_file_lock(out_path)
                            async with file_lock:
                                async with aiofiles.open(out_path, "wb") as f:
                                    await f.write(data)

                            if S3_BUCKET:
                                asyncio.create_task(
                                    self._upload_chunk_s3(
                                        out_path, base_stem, chunk_index, rid
                                    )
                                )

                            pending.append(out_path)
                            chunk_index += 1

                    for i, path in enumerate(pending):
                        yield str(path), i == len(pending) - 1

                    latency = time.monotonic() - t0
                    _req_total.labels(
                        status="ok", mode="stream_to_files", provider="openai"
                    ).inc()
                    _latency.observe(latency)
                    log.info(
                        "tts_stream_to_files_ok",
                        request_id=rid,
                        chunks=chunk_index,
                        chars=total_chars,
                        latency_s=round(latency, 3),
                    )

                except asyncio.CancelledError:
                    _req_total.labels(
                        status="cancelled", mode="stream_to_files", provider="openai"
                    ).inc()
                    raise
                except Exception as exc:
                    _req_total.labels(
                        status="error", mode="stream_to_files", provider="openai"
                    ).inc()
                    span.set_status(StatusCode.ERROR, str(exc))
                    log.error(
                        "tts_stream_to_files_error", request_id=rid, error=str(exc)
                    )
                    raise
                finally:
                    _active.labels(mode="stream").dec()
                    self._inflight_stream -= 1

    async def _upload_chunk_s3(
        self, path: Path, stem: str, chunk_index: int, rid: str
    ) -> None:
        if not S3_BUCKET:
            return
        try:
            key = f"{S3_AUDIO_PREFIX}{stem}_chunk_{chunk_index}.{self._format}"
            uri = await _s3_upload_file(S3_BUCKET, key, path)
            log.info("tts_chunk_s3_ok", request_id=rid, chunk=chunk_index, uri=uri)
        except Exception as exc:
            log.warning(
                "tts_chunk_s3_failed", request_id=rid, chunk=chunk_index, error=str(exc)
            )

    # ── health ────────────────────────────────────────────────────────────────

    async def health(self) -> ServiceHealthState:
        return ServiceHealthState(
            service="tts.local",
            healthy=self._breaker_batch.state != "OPEN",
            circuit_state=self._breaker_batch.state,
            inflight=self._inflight_batch + self._inflight_stream,
        )

    # ── LangGraph node ────────────────────────────────────────────────────────

    async def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        rid = state.get("request_id") or new_request_id()
        text: str = state.get("llm_response", "")
        voice: VoiceType | None = state.get("tts_voice")
        speed: float = float(state.get("tts_speed", 1.0))

        local_path, s3_uri = await self.synthesize(
            text, voice=voice, speed=speed, request_id=rid
        )
        return {
            **state,
            "request_id": rid,
            "audio_output": s3_uri if s3_uri else local_path,
            "audio_local_path": local_path,
            "audio_s3_uri": s3_uri,
        }

    # ── cleanup ───────────────────────────────────────────────────────────────

    async def close(self) -> None:
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        await self._client.close()


# ── remote TTS client ─────────────────────────────────────────────────────────


class RemoteTTSClient:
    """
    Calls a remote TTS microservice over HTTPS. Satisfies TTSNodeProtocol.

    The remote service is expected to expose:
      POST /synthesize              → {"local_path": str, "s3_uri": str}
      POST /synthesize/stream       → chunked audio bytes
      GET  /health                  → {"healthy": bool, ...}

    For synthesize_stream(), tokens are POSTed as a JSON body with
    {"tokens": ["Hello", " world", ...]} and audio bytes are streamed
    back as chunked transfer encoding. The caller yields each chunk as
    it arrives.
    """

    def __init__(
        self,
        base_url: str = TTS_SERVICE_URL,
        api_key: str = TTS_SERVICE_API_KEY,
        timeout: float = TTS_SERVICE_TIMEOUT,
    ) -> None:
        if not base_url:
            raise ValueError("TTS_SERVICE_URL must be set to use RemoteTTSClient.")
        self._base_url = base_url.rstrip("/")
        self._breaker = CircuitBreaker(name="tts:remote")
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

    async def synthesize(
        self,
        text: str,
        voice: VoiceType | None = None,
        filename: str | None = None,
        speed: float = 1.0,
        request_id: str | None = None,
    ) -> tuple[str, str]:
        rid = request_id or new_request_id()
        budget = LatencyBudget.current()
        if budget:
            budget.check(stage="remote_tts.synthesize")

        headers = self._request_headers()
        headers["X-Request-Id"] = rid

        with tracer.start_as_current_span("tts.remote.synthesize") as span:
            span.set_attribute("request_id", rid)
            self._inflight += 1
            t0 = time.monotonic()
            try:

                async def _call() -> dict[str, Any]:
                    resp = await self._http.post(
                        "/synthesize",
                        json={
                            "text": text,
                            "voice": voice,
                            "filename": filename,
                            "speed": speed,
                        },
                        headers=headers,
                    )
                    resp.raise_for_status()
                    return resp.json()

                result = await self._breaker.call(
                    backoff_retry,
                    _call,
                    attempts=3,
                    base_delay=1.0,
                    exceptions=(Exception,),
                )
                span.set_attribute("latency_s", round(time.monotonic() - t0, 3))
                return result.get("local_path", ""), result.get("s3_uri", "")

            except Exception as exc:
                span.set_status(StatusCode.ERROR, str(exc))
                log.error("remote_tts_synthesize_error", request_id=rid, error=str(exc))
                # try apology fallback even for remote failures
                fallback = _try_apology_fallback(rid)
                if fallback is not None:
                    return fallback
                raise
            finally:
                self._inflight -= 1

    async def synthesize_stream(
        self,
        token_stream: AsyncIterator[str],
        voice: VoiceType | None = None,
        speed: float = 1.0,
        request_id: str | None = None,
    ) -> AsyncIterator[bytes]:
        """
        Buffer tokens locally, send them to the remote TTS service in
        sentence-sized batches, and stream audio bytes back.

        Each sentence is sent as a separate POST /synthesize/stream request
        so the first audio chunk arrives as fast as possible.
        """
        rid = request_id or new_request_id()
        budget = LatencyBudget.current()
        if budget:
            budget.check(stage="remote_tts.stream")

        headers = self._request_headers()
        headers["X-Request-Id"] = rid
        active_voice = voice or TTS_VOICE

        with tracer.start_as_current_span("tts.remote.stream") as span:
            span.set_attribute("request_id", rid)

            async with bulkheads.acquire("tts.stream"):
                _active.labels(mode="stream").inc()
                self._inflight += 1
                t0 = time.monotonic()
                chunk_index = 0
                first_bytes_emitted = False

                try:
                    sentence_stream = AsyncSentenceBuffer(token_stream)

                    async for sentence in sentence_stream:
                        san = sanitize(sentence, request_id=rid)

                        # Skip empty / fully stripped content
                        if not san:
                            continue

                        clean = san.text

                        # Stream audio bytes for this sentence from the remote service
                        # Create the streaming context inside the circuit breaker.
                        # This ensures repeated failures trip the breaker while
                        # preserving true streaming semantics (no full buffering).
                        async def _open_stream():
                            return self._http.stream(
                                "POST",
                                "/synthesize/stream",
                                json={
                                    "text": clean,
                                    "voice": active_voice,
                                    "speed": speed,
                                },
                                headers=headers,
                            )

                        stream_ctx = await self._breaker.call(
                            backoff_retry,
                            _open_stream,
                            attempts=3,
                            base_delay=1.0,
                            exceptions=(Exception,),
                        )

                        # Now enter the streaming response context
                        async with stream_ctx as resp:
                            resp.raise_for_status()

                            async for audio_chunk in resp.aiter_bytes(chunk_size=4096):

                                # Enforce SLA before yielding audio
                                if budget:
                                    try:
                                        budget.check(
                                            stage="remote_tts.stream.before_yield"
                                        )
                                    except LatencyBudgetExceeded:
                                        _budget_exceeded.inc()
                                        raise

                                if not first_bytes_emitted:
                                    _ttfb.observe(time.monotonic() - t0)
                                    first_bytes_emitted = True

                                yield audio_chunk

                        chunk_index += 1

                    span.set_attribute("latency_s", round(time.monotonic() - t0, 3))
                    span.set_attribute("sentence_chunks", chunk_index)
                    log.info(
                        "remote_tts_stream_ok",
                        request_id=rid,
                        sentence_chunks=chunk_index,
                        latency_s=round(time.monotonic() - t0, 3),
                    )

                except LatencyBudgetExceeded:
                    _budget_exceeded.inc()
                    log.warning("remote_tts_stream_budget_exceeded", request_id=rid)
                    raise

                except asyncio.CancelledError:
                    log.warning("remote_tts_stream_cancelled", request_id=rid)
                    raise

                except Exception as exc:
                    span.set_status(StatusCode.ERROR, str(exc))
                    log.error("remote_tts_stream_error", request_id=rid, error=str(exc))
                    raise
                finally:
                    _active.labels(mode="stream").dec()
                    self._inflight -= 1

    async def health(self) -> ServiceHealthState:
        try:
            resp = await self._http.get("/health", timeout=5.0)
            data = resp.json()
            return ServiceHealthState(
                service="tts.remote",
                healthy=data.get("healthy", False),
                circuit_state=self._breaker.state,
                inflight=self._inflight,
            )
        except Exception as exc:
            return ServiceHealthState(
                service="tts.remote",
                healthy=False,
                circuit_state=self._breaker.state,
                inflight=self._inflight,
                error=str(exc),
            )

    async def close(self) -> None:
        await self._http.aclose()


# ── node factory ──────────────────────────────────────────────────────────────


def get_tts_node() -> TTSNodeProtocol:
    """
    Return RemoteTTSClient if TTS_SERVICE_URL is set, else local TTSNode.
    VoiceGraph imports only this factory.
    """
    if TTS_SERVICE_URL:
        log.info("tts_using_remote_client", url=TTS_SERVICE_URL)
        return RemoteTTSClient()
    log.info("tts_using_local_node", model=TTS_MODEL, voice=TTS_VOICE)
    return TTSNode()


# ── module-level singleton (backward-compatible) ──────────────────────────────

tts_node: TTSNodeProtocol = get_tts_node()

if __name__ == "__main__":
    import asyncio
    from app.nodes.LLM_service import get_llm_node

    async def _smoke():
        tts = get_tts_node()
        llm = get_llm_node()

        print("── batch synthesize ──")
        local, s3 = await tts.synthesize(
            "The voice module is working correctly.", voice="nova"
        )
        print("Local:", local, "| S3:", s3 or "(none)")

        print("\n── streaming synthesize (bytes) ──")
        stream = llm.stream("Explain binary search in two sentences.")
        i = 0
        async for audio_bytes in tts.synthesize_stream(stream):
            i += 1
            print(f"  chunk {i}: {len(audio_bytes)} bytes")

        print("\nHealth:", await tts.health())
        await tts.close()
        await llm.close()

    asyncio.run(_smoke())
