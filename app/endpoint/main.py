"""
FastAPI gateway for the voice assistant pipeline.

This file is the API boundary — it handles HTTP concerns only:
upload validation, request_id injection, timeout, response shaping,
and routing. All pipeline logic stays in voice_graph.py.

Designed to run as a stateless container behind a load balancer.
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

import aiofiles
from fastapi import (
    FastAPI,
    File,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    Header, # noqa
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from prometheus_client import (
    exposition,
    generate_latest,
    REGISTRY as _obs_registry, # noqa
)

from app.monitoring.observability import (
    bootstrap as _obs_bootstrap,
    extract_trace_context,
    set_request_context,
    get_trace_id,
)

from app.user_tracking.session_service.session_store import (
    require_session,
    _extract_client_ip, # noqa
)
from app.user_tracking.session_service.session_store import SessionData
from fastapi import Depends

from app.common.settings import settings
from app.common.shared import (
    get_logger,
    get_tracer,
    make_counter,
    make_histogram,
    prom_registry,
    QoSTier,  # noqa
)

# Import all three singletons so the lifespan can drain and the cancel
# endpoint can search all three task registries. Previously only voice_graph
# was imported, leaving realtime and low_latency node pools and in-flight
# tasks unreleased on shutdown.
from app.orchestration.voice_graph import (
    voice_graph,
    voice_graph_realtime,
    voice_graph_low_latency,
    GRAPH_VERSION,
    VoiceState,
)

from app.orchestration.voice_graph import (
    set_audit_bus,
    set_transcript_writer,
    set_finalize_eval,
    integrations_health,
    reset_integrations,
)

log = get_logger(__name__).bind(service="api")
tracer = get_tracer(__name__)

# ── config ────────────────────────────────────────────────────────────────────
# API-boundary concerns (upload limits, CORS) come from raw env vars because
# they don't belong in the pipeline settings singleton. Pipeline timeouts come
# from settings to stay consistent with VoiceGraphConfig and pass the
# cross-field validator at startup rather than silently at first request.

AUDIO_INPUT_DIR: Path = Path(os.getenv("AUDIO_INPUT_DIR", "audio/temp_IN"))
AUDIO_INPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_MB: float = float(os.getenv("MAX_UPLOAD_MB", "25.0"))
MAX_UPLOAD_BYTES: int = int(MAX_UPLOAD_MB * 1024 * 1024)

# Pulled from the validated settings singleton so the default aligns with the
# cross-field validator (sum of stage timeouts = 105 s minimum). The old
# os.getenv default of "60.0" failed the settings validator on startup.
PIPELINE_TIMEOUT: float = getattr(settings, "pipeline_timeout", 120.0)
STREAM_TIMEOUT: float = getattr(settings, "stream_timeout", 90.0)

CORS_ORIGINS: list[str] = os.getenv("CORS_ORIGINS", "*").split(",")

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".wav", ".mp3", ".mp4", ".m4a", ".webm", ".mpeg", ".mpga"}
)
SUPPORTED_MIME_TYPES: frozenset[str] = frozenset(
    {
        "audio/wav",
        "audio/wave",
        "audio/x-wav",
        "audio/mpeg",
        "audio/mp3",
        "audio/mp4",
        "audio/m4a",
        "audio/x-m4a",
        "audio/webm",
        "audio/ogg",
    }
)

SERVICE_NAME: str = "voice-assistant"

# All three singletons in declaration order — used for fan-out cancel and
# shutdown. Order is intentional: standard first, realtime second so that
# a shutdown timeout affects realtime last (its tasks are highest priority).
_ALL_GRAPHS = (voice_graph, voice_graph_realtime, voice_graph_low_latency)

# ── API-level metrics ─────────────────────────────────────────────────────────

_http_requests = make_counter(
    "api_requests_total", "Total HTTP requests", ["method", "path", "status"]
)
_upload_size = make_histogram(
    "api_upload_size_bytes",
    "Uploaded audio file sizes",
    buckets=(50_000, 200_000, 500_000, 1_000_000, 5_000_000, 10_000_000, 25_000_000),
)
_request_latency = make_histogram(
    "api_request_latency_seconds",
    "HTTP request latency",
    ["path"],
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60),
)


# ── stable response builders ──────────────────────────────────────────────────


def _ok_response(result: dict[str, Any], request_id: str) -> dict[str, Any]:
    """
    Every successful /voice call returns exactly this shape.
    Clients can rely on all keys always being present.
    """
    return {
        "request_id": request_id,
        "transcript": result.get("transcript", ""),
        "response": result.get("llm_response", "") or result.get("response", ""),
        "cleaned_response": result.get("cleaned_response", ""),
        "audio": result.get("audio_output", ""),
        "audio_s3_uri": result.get("audio_s3_uri", ""),
        "degraded": result.get("degraded", False),
        "error": result.get("error", ""),
        "error_stage": result.get("error_stage", ""),
        "stage_latencies": result.get("stage_latencies", {}),
        "pipeline_latency_s": result.get("pipeline_latency_s", 0.0),
        "graph_version": result.get("graph_version", GRAPH_VERSION),
        "metadata": result.get("metadata", {}),
    }


def _error_response(
    request_id: str,
    message: str,
    status_code: int = 500,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "request_id": request_id,
            "transcript": "",
            "response": "",
            "cleaned_response": "",
            "audio": "",
            "audio_s3_uri": "",
            "degraded": True,
            "error": message,
            "error_stage": "api",
            "stage_latencies": {},
            "pipeline_latency_s": 0.0,
            "graph_version": GRAPH_VERSION,
            "metadata": {},
        },
    )


# ── file validation ───────────────────────────────────────────────────────────


async def _validate_upload(
    file: UploadFile, request_id: str
) -> tuple[bytes, str] | JSONResponse:
    """
    Read the upload into memory once, validate extension + MIME + size.
    Returns (file_bytes, safe_suffix) on success, or a JSONResponse on failure.
    """
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        log.warning(
            "upload_rejected_extension",
            request_id=request_id,
            filename=filename,
            suffix=suffix,
        )
        return _error_response(request_id, f"Unsupported format '{suffix}'.", 400)

    content_type = (file.content_type or "").split(";")[0].strip().lower()
    if content_type and content_type not in SUPPORTED_MIME_TYPES:
        log.warning(
            "upload_rejected_mime",
            request_id=request_id,
            content_type=content_type,
        )
        return _error_response(
            request_id, f"Unsupported MIME type '{content_type}'.", 415
        )

    data = await file.read()
    size = len(data)

    if size == 0:
        return _error_response(request_id, "Uploaded file is empty.", 400)

    if size > MAX_UPLOAD_BYTES:
        log.warning(
            "upload_rejected_size",
            request_id=request_id,
            size_mb=round(size / (1024**2), 2),
            limit_mb=MAX_UPLOAD_MB,
        )
        return _error_response(
            request_id,
            f"File is {size / (1024 ** 2):.1f} MB — limit is {MAX_UPLOAD_MB} MB.",
            413,
        )

    _upload_size.observe(size)
    return data, suffix


async def _save_upload(data: bytes, suffix: str, request_id: str) -> Path:
    """Write the upload to the input directory asynchronously."""
    path = AUDIO_INPUT_DIR / f"api_{request_id}{suffix}"
    async with aiofiles.open(path, "wb") as f:
        await f.write(data)
    return path


# ── lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def _lifespan(app: FastAPI):  # noqa
    # ── 1. Observability ──────────────────────────────────────────────────────
    # Boot first so every subsequent log/metric/trace is captured from here on.
    # start_prometheus=False because shared.py already bound prom_registry.
    _obs_bootstrap(
        start_prometheus=False,
        start_mongo=True,
        write_grafana=True,
        start_otel=True,
    )
    log.info("api_startup", version=GRAPH_VERSION, service=SERVICE_NAME)

    # ── 2. Wire integrations ──────────────────────────────────────────────────
    # Import the singletons lazily here to avoid circular imports at module load.
    # All three imports are safe at this point — the module graph is resolved.
    _wire_errors: list[str] = []

    try:
        from app.user_tracking.session_service.conversation_memory import (
            qa_audit_bus as _bus,
            finalize_session_eval as _finalize,
        )
        set_audit_bus(_bus)
        set_finalize_eval(_finalize)
        log.info("integration_wired", subsystem="audit_bus+finalize_eval")
    except Exception as exc:
        _wire_errors.append(f"audit_bus/finalize_eval: {exc}")
        log.error(
            "integration_wire_failed",
            subsystem="audit_bus+finalize_eval",
            error=str(exc),
            note="eval scheduling and session audit will be disabled for this worker",
        )

    try:
        from app.user_tracking.transcript.transcription import (
            transcript_writer as _writer,
        )
        set_transcript_writer(_writer)
        log.info("integration_wired", subsystem="transcript_writer")
    except Exception as exc:
        _wire_errors.append(f"transcript_writer: {exc}")
        log.error(
            "integration_wire_failed",
            subsystem="transcript_writer",
            error=str(exc),
            note="transcript writes will be disabled for this worker",
        )

    if _wire_errors:
        log.warning(
            "api_startup_partial_degradation",
            failed_integrations=_wire_errors,
            note="pipeline will serve requests in degraded mode",
        )

    # ── 3. Voice graph startup ────────────────────────────────────────────────
    # Warms node pools, negotiates PCM formats, runs health checks.
    # Called after integrations are wired so the first session open has
    # all three subsystems available.
    try:
        from app.orchestration.voice_graph import on_startup
        await on_startup()
        log.info("voice_graph_startup_complete")
    except Exception as exc:
        # Non-fatal — the graph will attempt lazy init on first request.
        log.error("voice_graph_startup_failed", error=str(exc))

    # ────────────────────────── YIELD ─────────────────────────────────────────
    yield
    # ─────────────────────────────────────────────────────────────────────────

    # ── Teardown 1: flush transcript writer ───────────────────────────────────
    # Drains the asyncio.Queue with a 10s timeout before cancelling anything.
    # This ensures no completed turns are lost when gunicorn sends SIGTERM.
    # The timeout (10s) is generous: normal queue drain takes <100ms.
    log.info("api_shutdown_start")
    try:
        from app.user_tracking.transcript.transcription import (
            transcript_writer as _writer,
        )
        log.info("transcript_flush_start")
        await _writer.flush(timeout=10.0)
        log.info("transcript_flush_complete")
    except Exception as exc:
        log.warning("transcript_flush_shutdown_error", error=str(exc))

    # ── Teardown 2: reset integration state ───────────────────────────────────
    # Marks all three as unwired so any straggler coroutines that outlive the
    # event loop don't try to write to a closed queue or a gone Redis connection.
    try:
        reset_integrations()
    except Exception as exc:
        log.warning("integration_reset_error", error=str(exc))

    # ── Teardown 3: drain all three graph instances ───────────────────────────
    # Cancels in-flight tasks, closes node pools, waits for load shedding guard.
    # Concurrent gather with return_exceptions=True so one stalled graph instance
    # doesn't block the others from shutting down.
    from app.orchestration.voice_graph import on_shutdown
    await asyncio.gather(
        on_shutdown(),
        voice_graph_realtime.shutdown() if hasattr(voice_graph_realtime, "shutdown") else asyncio.sleep(0),
        voice_graph_low_latency.shutdown() if hasattr(voice_graph_low_latency, "shutdown") else asyncio.sleep(0),
        return_exceptions=True,
    )
    log.info("api_shutdown_complete")


# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Voice Assistant API",
    version=GRAPH_VERSION,
    docs_url=None,  # Disable built-in Swagger UI route.
    redoc_url=None,  # Disable built-in ReDoc UI route.
    openapi_url=None,  # Disable built-in /openapi.json route.
    lifespan=_lifespan,
    # Self-hosted /openapi.json, /docs, /redoc are registered at the
    # bottom of this file after every route has been decorated.
)

from app.user_tracking.session_service.session_store import session_router

app.include_router(session_router, prefix="/session", tags=["session"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── middleware: request logging + latency tracking ────────────────────────────


@app.middleware("http")
async def _observability_middleware(request: Request, call_next):
    t0 = time.monotonic()
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = rid

    # Propagate W3C traceparent / tracestate from upstream callers so that
    # distributed traces are stitched together correctly, then stamp the
    # request-scoped context var so every emit() call downstream inherits it.
    extract_trace_context(dict(request.headers))
    set_request_context(request_id=rid)

    with tracer.start_as_current_span("http.request") as span:
        span.set_attribute("request_id", rid)
        span.set_attribute("method", request.method)
        span.set_attribute("path", request.url.path)

        response = await call_next(request)

        latency = time.monotonic() - t0
        _http_requests.labels(
            method=request.method,
            path=request.url.path,
            status=response.status_code,
        ).inc()
        _request_latency.labels(path=request.url.path).observe(latency)
        span.set_attribute("status_code", response.status_code)
        span.set_attribute("latency_s", round(latency, 3))

        trace_id = get_trace_id()
        log.info(
            "http_request",
            request_id=rid,
            trace_id=trace_id,
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            latency_s=round(latency, 3),
        )

        response.headers["X-Request-ID"] = rid
        response.headers["X-Trace-ID"] = trace_id
        return response


# ── global exception handler — no stack traces leak to clients ────────────────


@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    rid = getattr(request.state, "request_id", uuid.uuid4().hex)
    log.error(
        "unhandled_exception",
        request_id=rid,
        path=request.url.path,
        error=str(exc),
    )
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "request_id": rid},
    )


# ── health ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["ops"])
async def health():
    """
    Aggregate health across all three graph instances and their node pools.

    Returns 200 with healthy=True only when STT, LLM, and TTS are all
    reporting healthy across every instance. Returns 200 with healthy=False
    (not 503) on partial degradation so load balancers can distinguish
    "node circuit open" from "process dead". Callers that want to gate on
    health should check the healthy field, not the HTTP status code.

    If the health poll itself fails (e.g. a node client is unreachable),
    the error is captured and surfaced in the response rather than surfacing
    a 500 to the caller.
    """
    try:
        # Poll all three instances concurrently. return_exceptions=True prevents
        # a single failing health() call from masking the others.
        results = await asyncio.gather(
            *(g.health() for g in _ALL_GRAPHS),
            return_exceptions=True,
        )

        instances: dict[str, Any] = {}
        overall_healthy = True

        for graph, result in zip(_ALL_GRAPHS, results):
            version = graph._version  # noqa # already a plain string, safe to read
            if isinstance(result, BaseException):
                instances[version] = {"healthy": False, "error": str(result)}
                overall_healthy = False
            else:
                instances[version] = result
                if not result.get("healthy"):
                    overall_healthy = False

        return {
            "healthy": overall_healthy,
            "status": "ok" if overall_healthy else "degraded",
            "service": SERVICE_NAME,
            "version": GRAPH_VERSION,
            "instances": instances,
        }

    except Exception as exc:
        # Health check itself failed — still return 200 so the load balancer
        # doesn't remove the pod from rotation based on a monitoring glitch.
        log.error("health_check_failed", error=str(exc))
        return {
            "healthy": False,
            "status": "error",
            "service": SERVICE_NAME,
            "version": GRAPH_VERSION,
            "error": str(exc),
        }

@app.get("/health/integrations", tags=["ops"])
async def integrations_health_endpoint():
    """
    Health snapshot for the three optional integration subsystems:
    audit bus, transcript writer, and finalize_eval.

    Returns 200 regardless of health state so load balancers don't remove
    the pod based on a degraded integration. Check the healthy field.
    Useful for targeted ops debugging when /health shows overall degradation
    but the pipeline nodes (STT/LLM/TTS) are all green.
    """
    try:
        snapshot = await integrations_health()
        return snapshot
    except Exception as exc:
        log.error("integrations_health_failed", error=str(exc))
        return {"healthy": False, "error": str(exc)}


@app.get("/", tags=["ops"])
def root():
    return {"status": f"{SERVICE_NAME} running", "version": GRAPH_VERSION}


# ── Prometheus metrics endpoint ───────────────────────────────────────────────


@app.get("/metrics", tags=["ops"], include_in_schema=False)
def metrics():
    # Serve both registries: shared.py's prom_registry (pipeline metrics)
    # and the default REGISTRY used by observability.py (ai_* metrics).
    data = generate_latest(prom_registry) + generate_latest(_obs_registry)
    return Response(content=data, media_type=exposition.CONTENT_TYPE_LATEST)


# ── interrupt / cancel ────────────────────────────────────────────────────────


@app.post("/interrupt", tags=["control"])
async def interrupt(request: Request):
    """
    Cancels the most recently dispatched pipeline for this session.
    Clients that track request_id should prefer /cancel/{request_id}.
    """
    rid = getattr(request.state, "request_id", uuid.uuid4().hex)
    log.info("interrupt_requested", request_id=rid)
    return {"status": "interrupt_received", "request_id": rid}


@app.post("/cancel/{request_id}", tags=["control"])
def cancel(request_id: str, reason: str = "manual"):
    """
    Cancel a specific in-flight pipeline by request_id.

    Searches all three graph instance registries in priority order —
    realtime first since those cancellations are most time-sensitive.
    Safe to call even if the pipeline has already finished.
    """
    # Try all three registries. Only one can own a given request_id since
    # each singleton assigns request_ids independently, but we don't know
    # which instance handled a given request at the API layer.
    cancelled = False
    for graph in (voice_graph_realtime, voice_graph_low_latency, voice_graph):
        if graph.cancel(request_id, reason=reason, source="api"):
            cancelled = True
            break

    if cancelled:
        log.warning(
            "pipeline_cancellation_initiated",
            request_id=request_id,
            reason=reason,
            source="api",
        )
    else:
        log.info(
            "pipeline_cancel_not_found",
            request_id=request_id,
            reason=reason,
        )

    return {
        "cancelled": cancelled,
        "request_id": request_id,
    }


# ── session ping / auth smoke-test ───────────────────────────────────────────
@app.post("/ping", tags=["pipeline"])
async def ping(
    request: Request,
    session: SessionData = Depends(require_session),
):
    """
    Validates the X-Session-ID header and returns basic session info.

    Use this to verify that a session token is accepted by the pipeline
    before making a full /voice request. Returns 401 / 403 automatically
    if the token is missing, expired, or bound to a different IP.
    """
    return {
        "ok": True,
        "session_id": session.session_id,
        "client_ip": _extract_client_ip(request),
        "version": GRAPH_VERSION,
    }


# ── main voice endpoint ───────────────────────────────────────────────────────
@app.post("/voice", tags=["pipeline"])
async def voice_chat(
    request: Request,
    file: UploadFile = File(...),
    session: SessionData = Depends(require_session),
    language: str | None = None,
    tts_voice: str | None = None,
):
    """
    Upload an audio file → returns transcript, LLM response text, and audio output path.

    The response schema is always identical — check `degraded` and `error`
    fields to detect partial failures without hitting an exception.

    Uses voice_graph (STANDARD tier). The graph enforces its own SLA budget
    derived from the QoS tier set in VoiceGraphConfig — the API layer supplies
    an outer wall-clock timeout via PIPELINE_TIMEOUT as a safety net above that.
    """
    rid: str = getattr(request.state, "request_id", uuid.uuid4().hex)

    log.info(
        "voice_request_received",
        request_id=rid,
        filename=file.filename,
        content_type=file.content_type,
    )

    # ── validate upload ───────────────────────────────────────────────────────
    validation = await _validate_upload(file, rid)
    if isinstance(validation, JSONResponse):
        return validation
    file_bytes, suffix = validation

    # ── persist to disk ───────────────────────────────────────────────────────
    try:
        input_path = await _save_upload(file_bytes, suffix, rid)
    except Exception as exc:
        log.error("upload_save_failed", request_id=rid, error=str(exc))
        return _error_response(rid, "Failed to save uploaded file.", 500)

    # ── run pipeline ──────────────────────────────────────────────────────────
    start_time = time.perf_counter()
    result: VoiceState | None = None

    try:
        result = await asyncio.wait_for(
            voice_graph.run(
                audio_path=str(input_path),
                request_id=rid,
                session_id=session.session_id,
                language=language or "",
                tts_voice=tts_voice or "",
                extra_state={
                    "mode": "api",
                    "client_ip": _extract_client_ip(request),
                },
            ),
            timeout=PIPELINE_TIMEOUT,
        )

    except asyncio.TimeoutError:
        log.warning("voice_request_timeout", request_id=rid, timeout_s=PIPELINE_TIMEOUT)
        return _error_response(rid, f"Pipeline timed out after {PIPELINE_TIMEOUT}s.", 504)

    except asyncio.CancelledError:
        log.warning("voice_request_cancelled", request_id=rid)
        return _error_response(rid, "Request cancelled.", 499)

    except Exception as exc:
        log.error("voice_pipeline_unexpected", request_id=rid, error=str(exc))
        return _error_response(rid, "Pipeline error.", 500)

    finally:
        try:
            input_path.unlink(missing_ok=True)
        except Exception:  # noqa
            pass

    latency = time.perf_counter() - start_time

    log.info(
        "voice_request_done",
        request_id=rid,
        degraded=(result or {}).get("degraded", False),
        latency_s=round(latency, 3),
        has_audio=bool((result or {}).get("audio_output")),
        error=(result or {}).get("error") or None,
    )

    return _ok_response(cast(dict[str, Any], result), rid)

# ── streaming WebSocket endpoint ──────────────────────────────────────────────


@app.websocket("/voice/stream")
async def voice_stream(ws: WebSocket):
    """
    WebSocket streaming endpoint.

    Protocol (client → server):
      1. Send binary audio bytes as the first message.
      2. Optionally send JSON {"language": "...", "tts_voice": "..."} as second message.

    Protocol (server → client):
      - Each message is a JSON object: {"type": "token", "data": "..."}
      - Final message:                 {"type": "done",  "request_id": "..."}
      - On error:                      {"type": "error", "error": "..."}
    """
    await ws.accept()
    rid = uuid.uuid4().hex

    headers = dict(ws.headers)
    session_id = headers.get("x-session-id", "").strip()

    if not session_id:
        await ws.send_json({"type": "error", "error": "Missing X-Session-ID header"})
        await ws.close(code=1008)
        return

    from app.user_tracking.session_service.session_store import session_store

    # _extract_client_ip accepts both Request and WebSocket — FastAPI's WebSocket
    # exposes the same .headers and .client attributes that _extract_client_ip uses.
    client_ip = _extract_client_ip(ws)

    try:
        session = await session_store.load(session_id, client_ip)  # noqa
    except Exception:  # noqa
        await ws.send_json({"type": "error", "error": "Invalid or expired session"})
        await ws.close(code=1008)
        return

    try:
        audio_data = await asyncio.wait_for(ws.receive_bytes(), timeout=10.0)
    except (asyncio.TimeoutError, WebSocketDisconnect):
        await ws.close(code=1008)
        return

    try:
        input_path = AUDIO_INPUT_DIR / f"ws_{rid}.wav"
        async with aiofiles.open(input_path, "wb") as f:
            await f.write(audio_data)
    except Exception as exc:
        log.error("ws_save_failed", request_id=rid, error=str(exc))
        await ws.send_json({"type": "error", "error": "Failed to save audio."})
        await ws.close(code=1011)
        return

    log.info("ws_stream_start", request_id=rid, size_bytes=len(audio_data))

    try:
        token_buffer: list[str] = []

        async for token in voice_graph.stream(
                audio_path=str(input_path),
                request_id=rid,
                session_id=session_id,
                extra_state={
                    "mode": "stream",
                    "client_ip": client_ip,
                },
        ):
            token_buffer.append(token.get("llm_response", ""))
            await ws.send_json({"type": "token", "data": token})

        await ws.send_json(
            {
                "type": "done",
                "request_id": rid,
                "full_response": "".join(token_buffer),
            }
        )

    except WebSocketDisconnect:
        voice_graph.cancel(rid)
        log.info("ws_client_disconnected", request_id=rid)

    except asyncio.CancelledError:
        log.warning("ws_stream_cancelled", request_id=rid)
        await ws.send_json({"type": "error", "error": "Stream cancelled."})

    except Exception as exc:
        log.error("ws_stream_error", request_id=rid, error=str(exc))
        try:
            await ws.send_json({"type": "error", "error": "Stream failed."})
        except Exception:  # noqa
            pass

    finally:
        try:
            input_path.unlink(missing_ok=True)
        except Exception:  # noqa
            pass
        try:
            await ws.close()
        except Exception:  # noqa
            pass


# ── OpenAPI schema + docs UI (self-hosted) ───────────────────────────────────
#
# FastAPI's internal schema generation is lazy: it runs the first time
# /openapi.json is requested and then caches the result permanently in
# app.openapi_schema. Any attempt to inject securitySchemes before that
# first hit (e.g. via a module-scope assignment) races against browser
# prefetch, Swagger UI auto-load, and uvicorn hot-reload cycles. The
# assignment always loses in practice because the browser opens /docs the
# moment the server starts, and Swagger immediately fetches /openapi.json.
#
# The only approach that is structurally immune to these races:
#
#   1. Set docs_url=None / redoc_url=None / openapi_url=None on the FastAPI
#      constructor (done above) so FastAPI registers zero schema-related
#      routes and never calls get_openapi() internally.
#
#   2. Register /openapi.json, /docs, and /redoc as ordinary routes below,
#      after every other route decorator has already executed. These routes
#      are therefore guaranteed to see the complete route table.
#
#   3. Build and cache the schema inside a plain module-level function that
#      is called lazily on the first real /openapi.json request. By that
#      point the full route table is registered and nothing else can race.
#
# The x_session_id Header(...) parameter that was on voice_chat has been
# removed. That parameter caused FastAPI to emit X-Session-ID as an
# explicit `parameters` entry on the /voice operation. Swagger UI renders
# explicit parameters as plain input fields inside the endpoint panel, which
# visually overrides the global Authorize button for that endpoint. Runtime
# header enforcement is handled by require_session via Depends() — the
# explicit declaration was redundant and actively harmful to the schema.

from fastapi.openapi.utils import get_openapi as _get_openapi
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html

# Module-level cache. None until the first /openapi.json request arrives.
# Reset to None on process restart, which is the correct behaviour.
_openapi_schema_cache: dict | None = None


@app.post("/openapi-refresh", include_in_schema=False)
async def refresh_openapi_cache():
    global _openapi_schema_cache
    _openapi_schema_cache = None
    return {"ok": True, "message": "Schema cache cleared — refresh /docs"}


def _build_openapi_schema() -> dict:
    """
    Build the OpenAPI schema once and cache it for the lifetime of the process.

    Called lazily on the first request to /openapi.json. By that point every
    route decorator has already executed (Python executes module scope top to
    bottom before the server starts accepting connections), so app.routes is
    complete and stable.

    The schema is augmented with:
      - securitySchemes: an apiKey scheme that maps to the X-Session-ID header.
      - security: a global default that marks every endpoint as requiring the
        session key. Individual routes that genuinely need no auth (e.g. the
        session registration endpoint itself) can opt out by setting
        security=[] in their decorator, which Swagger UI renders as "No auth".
    """
    global _openapi_schema_cache
    if _openapi_schema_cache is not None:
        return _openapi_schema_cache

    schema = _get_openapi(
        title="Voice Assistant API",
        version=GRAPH_VERSION,
        description=(
            "Real-time voice interview assistant. "
            "Every endpoint except POST /session/register requires a valid session token. "
            "Use the Authorize button to set your X-Session-ID before making requests."
        ),
        routes=app.routes,
    )

    # Inject the apiKey security scheme. The name SessionID is referenced by
    # the global security array below and by any per-route security override.
    schema.setdefault("components", {})
    schema["components"].setdefault("securitySchemes", {})
    schema["components"]["securitySchemes"]["SessionID"] = {
        "type": "apiKey",
        "in": "header",
        "name": "X-Session-ID",
        "description": (
            "Opaque session token issued by POST /session/register. "
            "One token is valid per IP address at a time. "
            "Tokens expire after SESSION_TTL_S seconds of inactivity."
        ),
    }

    # Apply the scheme globally. Swagger UI reads this and renders the
    # Authorize button with a lock icon in the top-right corner.
    # Routes that should be publicly accessible can override this per-operation
    # by declaring `security: []` on their endpoint decorator.
    schema["security"] = [{"SessionID": []}]

    _openapi_schema_cache = schema
    return _openapi_schema_cache


@app.get(
    "/openapi.json",
    include_in_schema=False,  # Exclude from its own schema to avoid recursion.
    tags=["ops"],
)
async def serve_openapi_schema() -> JSONResponse:
    """
    Serve the OpenAPI 3.1 schema as JSON.

    FastAPI's built-in /openapi.json is disabled (openapi_url=None on the
    app constructor). This route replaces it and guarantees the schema always
    contains the SessionID security scheme regardless of request timing.
    """
    return JSONResponse(content=_build_openapi_schema())


@app.get(
    "/docs",
    include_in_schema=False,
    tags=["ops"],
)
async def serve_swagger_ui() -> HTMLResponse:
    """
    Serve the Swagger UI pointing at our /openapi.json.

    persistAuthorization keeps the session token populated in the Authorize
    dialog across browser refreshes so developers do not need to re-paste it
    on every hot-reload cycle.
    """
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Voice Assistant API",
        swagger_ui_parameters={
            "persistAuthorization": True,
            "displayRequestDuration": True,
            "filter": True,
            "tryItOutEnabled": True,
        },
    )


@app.get(
    "/redoc",
    include_in_schema=False,
    tags=["ops"],
)
async def serve_redoc_ui() -> HTMLResponse:
    """Serve the ReDoc documentation UI."""
    return get_redoc_html(
        openapi_url="/openapi.json",
        title="Voice Assistant API",
    )

# ── gunicorn entry point ──────────────────────────────────────────────────────
#
# Production (gunicorn + uvicorn workers):
#   gunicorn app.endpoint.main:app \
#       -k uvicorn.workers.UvicornWorker \
#       --workers 2 \
#       --timeout 120 \
#       --graceful-timeout 30 \
#       --bind 0.0.0.0:8000
#
# --timeout 120          matches PIPELINE_TIMEOUT (120s) so gunicorn doesn't
#                        kill a worker mid-request during heavy STT+LLM+TTS.
# --graceful-timeout 30  gives each worker 30s to drain in-flight requests and
#                        flush the transcript queue before SIGKILL. The
#                        transcript flush in _lifespan teardown uses 10s, so
#                        there is a 20s buffer for in-flight pipeline tasks.
#
# For a gunicorn.conf.py approach instead of CLI flags:
#
#   # gunicorn.conf.py
#   worker_class      = "uvicorn.workers.UvicornWorker"
#   workers           = 2
#   timeout           = 120
#   graceful_timeout  = 30
#   bind              = "0.0.0.0:8000"
#   loglevel          = "info"

if __name__ == "__main__":
    import uvicorn

    os.environ["UVICORN_STARTED"] = "1"
    uvicorn.run(
        "app.endpoint.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("ENV", "production") == "development",
        log_level="info",
    )