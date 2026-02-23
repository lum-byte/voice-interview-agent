# Voice Assistant Pipeline — Complete Technical Reference

> A production-grade, real-time voice interview assistant built on OpenAI Whisper (STT), GPT (LLM), and OpenAI TTS, orchestrated with LangGraph, backed by Redis and MongoDB, and observed via Prometheus, Grafana, Loki, Tempo, and OpenTelemetry. Supports both a desktop push-to-talk mode and a fully stateless HTTP/WebSocket API mode.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Repository Layout](#3-repository-layout)
4. [Configuration & Settings](#4-configuration--settings)
5. [Shared Infrastructure (`shared.py`)](#5-shared-infrastructure-sharedpy)
6. [Logging (`log_config.py`)](#6-logging-log_configpy)
7. [Pipeline Orchestration (`voice_graph.py`)](#7-pipeline-orchestration-voice_graphpy)
8. [STT Node (`STT_service.py`)](#8-stt-node-stt_servicepy)
9. [LLM Node (`LLM_service.py`)](#9-llm-node-llm_servicepy)
10. [TTS Node (`TTS_service.py`)](#10-tts-node-tts_servicepy)
11. [Sanitize Node (`sanitize.py`)](#11-sanitize-node-sanitizepy)
12. [Session Store (`session_store.py`)](#12-session-store-session_storepy)
13. [Conversation Memory (`conversation_memory.py`)](#13-conversation-memory-conversation_memorypy)
14. [Evaluation Engine (`evaluation_engine.py`)](#14-evaluation-engine-evaluation_enginepy)
15. [Transcript Writer (`transcription.py`)](#15-transcript-writer-transcriptionpy)
16. [FastAPI Gateway (`main.py`)](#16-fastapi-gateway-mainpy)
17. [Desktop Controller (`controller.py`)](#17-desktop-controller-controllerpy)
18. [Audio Recorder (`recorder.py`)](#18-audio-recorder-recorderpy)
19. [Audio Player (`player.py`)](#19-audio-player-playerpy)
20. [Observability Stack (`observability.py`)](#20-observability-stack-observabilitypy)
21. [Pipeline Wrapper (`pipeline.py`)](#21-pipeline-wrapper-pipelinepy)
22. [Startup Display (`startup_display.py`)](#22-startup-display-startup_displaypy)
23. [Docker Compose & Infrastructure](#23-docker-compose--infrastructure)
24. [Execution Modes](#24-execution-modes)
25. [Data Flow: End-to-End Walkthrough](#25-data-flow-end-to-end-walkthrough)
26. [Resilience Patterns](#26-resilience-patterns)
27. [QoS Tiers & Graph Variants](#27-qos-tiers--graph-variants)
28. [Session Lifecycle](#28-session-lifecycle)
29. [Environment Variables Reference](#29-environment-variables-reference)
30. [Running the System](#30-running-the-system)
31. [Dependency Map](#31-dependency-map)

---

## 1. System Overview

This system is a real-time voice interview assistant. A user speaks into a microphone (or uploads an audio file via HTTP); the audio is transcribed to text, passed to an LLM that maintains full conversation history, and the LLM's response is synthesised back to speech and played or returned to the caller. The entire pipeline — transcription, reasoning, synthesis — completes in under 15 seconds end-to-end under normal conditions, and in under 7 seconds on the low-latency graph variant.

The pipeline is designed to handle two very different operating contexts simultaneously:

**Desktop mode** (`APP_MODE=desktop`): A researcher or interviewer runs the server locally. They hold a configurable push-to-talk key on their keyboard, speak, release the key, and hear the AI response through their speakers within seconds. The entire interaction is local and keyboard-driven. No browser required.

**API mode** (`APP_MODE=api`): The server exposes a FastAPI HTTP interface. A web frontend, mobile app, or automation tool sends a session registration request, then POST requests with audio files. The pipeline runs and returns a JSON payload containing the transcript, LLM response text, and audio output path. A WebSocket endpoint (`/voice/stream`) supports token-level streaming for lower perceived latency.

Both modes run the same `VoiceGraph` pipeline under the hood. The only difference is how audio gets in and how results get back out.

---

## 2. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          INPUT LAYER                                  │
│                                                                       │
│  Desktop Mode                         API Mode                        │
│  ┌──────────────┐                    ┌────────────────────────────┐   │
│  │ controller.py│ ← keyboard (PTT)   │ main.py (FastAPI)          │   │
│  │ recorder.py  │ ← microphone       │  POST /voice               │   │
│  └──────┬───────┘                    │  WebSocket /voice/stream   │   │
│         │                            └────────────┬───────────────┘   │
└─────────┼──────────────────────────────────────────┼──────────────────┘
          │ audio file path                          │ audio file path
          ▼                                          ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATION LAYER                              │
│                                                                       │
│   voice_graph.py  (LangGraph StateGraph)                             │
│                                                                       │
│   START → [stt] → [llm] → [sanitize] → [tts] → END                  │
│               ↓       ↓                   ↓                           │
│          [stt_err] [llm_err]         [tts_err]                       │
│               ↓       ↓                   ↓                           │
│          [retry?]  [retry?]         [error_terminal]                  │
│                                                                       │
│   Three graph variants: voice_graph / voice_graph_realtime           │
│                         / voice_graph_low_latency                    │
└──────┬───────────────────┬──────────────────────┬────────────────────┘
       │                   │                      │
       ▼                   ▼                      ▼
┌────────────┐    ┌─────────────────┐    ┌──────────────────┐
│ STT_service│    │  LLM_service    │    │  TTS_service     │
│ (Whisper)  │    │  (GPT via       │    │  (OpenAI TTS)    │
│            │    │   LangChain)    │    │                  │
│ CircuitBkr │    │ Redis cache     │    │ CircuitBkr       │
│ RateLimiter│    │ Stampede lock   │    │ RateLimiter      │
│ Bulkhead   │    │ Fallback model  │    │ S3 upload        │
│ S3 support │    │ Circuit breaker │    │ Local file TTL   │
│ Streaming  │    │ Streaming       │    │                  │
└────────────┘    └────────┬────────┘    └──────────────────┘
                           │
              ┌────────────┼───────────────┐
              │            │               │
              ▼            ▼               ▼
     ┌──────────────┐ ┌─────────┐ ┌───────────────────┐
     │session_store │ │conv_mem │ │evaluation_engine  │
     │  (Redis +    │ │ory.py   │ │ (async, off-path) │
     │   LRU fallbk)│ │         │ │                   │
     └──────────────┘ └─────────┘ └───────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                    OBSERVABILITY LAYER                                │
│                                                                       │
│  observability.py → structlog → Rich console + JSON file             │
│                   → Prometheus metrics (/metrics endpoint)            │
│                   → MongoDB event documents                           │
│                   → OpenTelemetry spans → OTel Collector → Tempo     │
│                                                                       │
│  docker-compose.yaml provisions:                                     │
│  Redis · MongoDB · Prometheus · Grafana · Tempo · Loki · Promtail    │
│  OTel Collector · node-exporter · cAdvisor · redis-exporter         │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. Repository Layout

```
.
├── .env                          # Environment configuration (never commit secrets)
├── docker-compose.yaml           # Full observability stack
│
├── app/
│   ├── common/
│   │   ├── settings.py           # Pydantic Settings — validated at import time
│   │   ├── shared.py             # Circuit breaker, rate limiter, OTel, logging
│   │   └── log_config.py         # Dual-sink structured logging (Rich + JSON)
│   │
│   ├── orchestration/
│   │   ├── voice_graph.py        # LangGraph pipeline engine (the core)
│   │   └── pipeline.py           # Thin OO wrapper around voice_graph
│   │
│   ├── nodes/
│   │   ├── STT_service.py        # Whisper transcription node
│   │   ├── LLM_service.py        # GPT chat node with Redis cache
│   │   ├── TTS_service.py        # OpenAI TTS synthesis node
│   │   └── sanitize.py           # Output sanitizer and size limiter
│   │
│   ├── user_tracking/
│   │   ├── session_service/
│   │   │   ├── session_store.py        # IP-locked Redis session store
│   │   │   └── conversation_memory.py  # History injection + topic tracking
│   │   └── transcript/
│   │       └── transcription.py        # Dual-sink transcript writer
│   │
│   ├── eval/
│   │   └── evaluation_engine.py  # Off-path turn scoring engine
│   │
│   ├── monitoring/
│   │   └── observability.py      # Event taxonomy, emitters, Prometheus, OTel, Mongo
│   │
│   ├── audio_essentials/
│   │   ├── player.py             # Low-latency local audio playback
│   │   └── recorder.py           # Microphone capture (push-to-talk)
│   │
│   ├── startup/
│   │   └── startup_display.py    # Animated Rich boot banner
│   │
│   └── endpoint/
│       └── main.py               # FastAPI gateway
│
├── app/orchestration/controller.py   # Desktop PTT controller
│
├── audio/
│   ├── audio_INPUT/              # Uploaded/recorded audio (cleaned after each run)
│   └── audio_OUTPUT/             # TTS synthesised audio files
│
├── transcripts/                  # Per-session .txt transcript files
├── logs/                         # JSON structured log files
└── infra/                        # Config files for Docker services
    ├── redis/
    ├── prometheus/
    ├── loki/
    ├── otel-collector/
    └── grafana/
```

---

## 4. Configuration & Settings

**File:** `app/common/settings.py`

The `Settings` class is a Pydantic `BaseSettings` singleton that reads every environment variable the entire system uses, validates them at import time, and exits immediately with a clean error message if anything is wrong. No module in the system reads `os.environ` directly (except for a handful of bootstrap-time constants) — everything goes through `settings`.

```python
from app.common.settings import settings

api_key   = settings.openai_api_key.get_secret_value()
redis_url = settings.redis_url
llm_model = settings.llm_model
```

**Why this matters:** A fat-fingered env var (e.g., `LLM_TEMPERATURE=abc`) is caught the instant the process starts, not three minutes later when the first request hits the LLM node. The error message is human-readable, not a raw Python traceback.

### Key Validation Rules

| What is checked | Behaviour on failure |
|---|---|
| `OPENAI_API_KEY` missing | Hard exit |
| `SESSION_SECRET_KEY` missing in non-desktop mode | Hard exit |
| `LLM_TEMPERATURE` outside 0.0–2.0 | Hard exit |
| `STT_MAX_FILE_MB` > 25.0 | Hard exit (Whisper API ceiling) |
| `TTS_VOICE` not in valid set | Hard exit |
| `PIPELINE_TIMEOUT` < sum of stage timeouts | Hard exit (cross-field validator) |
| `REDIS_MAX_CONN` < `LLM_MAX_CONCURRENT` | Hard exit (pool starvation check) |
| `CORS_ORIGINS=*` in production | Warning, not exit |
| `HOST=localhost` in production | Warning, not exit |
| Unknown LLM model name | Warning, not exit |

### Cross-Field Validators

```python
# pipeline_timeout must exceed the sum of all three stage timeouts
# otherwise the outer wall-clock guard always fires before the per-stage guards
if self.pipeline_timeout <= stage_sum:
    raise ValueError(
        f"PIPELINE_TIMEOUT ({self.pipeline_timeout}s) must be greater than "
        f"GRAPH_STT_TIMEOUT + GRAPH_LLM_TIMEOUT + GRAPH_TTS_TIMEOUT ({stage_sum}s)."
    )
```

### Derived Properties

Settings exposes computed properties so callers never need to duplicate the calculation:

```python
settings.is_production            # True when ENV=production
settings.s3_enabled               # True when either STT or TTS S3 bucket is set
settings.graph_stage_timeout_sum  # STT + LLM + TTS timeout sum
settings.node_bottleneck_concurrency  # min(stt, llm, tts) concurrent caps
settings.uvicorn_reload           # True only in development
```

### Startup Checks

```python
await settings.check_redis()    # Ping Redis; raise if unreachable
await settings.check_openai()   # Call models.list(); raise if key invalid
settings.ensure_directories()  # Create audio_INPUT, audio_OUTPUT dirs
```

Call these from the FastAPI lifespan hook or `__main__` before accepting traffic.

### Secret Handling

`OPENAI_API_KEY`, `AWS_ACCESS_KEY_ID`, and `AWS_SECRET_ACCESS_KEY` are declared as `SecretStr`. This means they never appear in logs, `repr()`, or any auto-generated API schema. Accessing them requires an explicit `.get_secret_value()` call.

---

## 5. Shared Infrastructure (`shared.py`)

**File:** `app/common/shared.py`

This module is the plumbing that every other module imports. It provides structured logging, OpenTelemetry, Prometheus, a circuit breaker, a token-bucket rate limiter, a bounded LRU cache, a latency budget, a load shedding guard, a bulkhead pool, and QoS tier definitions — all in one place so no module reaches into multiple utility modules for these fundamentals.

### Request ID Propagation

Every request is assigned a UUID hex string that flows through the entire pipeline via a `contextvars.ContextVar`:

```python
from app.common.shared import new_request_id, current_request_id

rid = new_request_id()          # generates and stores in contextvar
rid = current_request_id()      # reads from contextvar (generates if missing)
```

This means any log line emitted from any node during a request automatically carries the request ID without being explicitly passed everywhere.

### Circuit Breaker

The `CircuitBreaker` class wraps any async callable. It tracks consecutive failures. After `failure_threshold` failures it opens the circuit and raises `CircuitBreakerOpen` immediately for `recovery_timeout` seconds, giving the downstream service time to recover.

```python
cb = CircuitBreaker(
    name="openai_llm",
    failure_threshold=5,
    recovery_timeout=30.0,
)

# In the LLM node:
result = await cb.call(llm_client.generate, prompt)
```

State transitions:
- `CLOSED` → `OPEN`: after `failure_threshold` consecutive failures
- `OPEN` → `HALF_OPEN`: after `recovery_timeout` seconds
- `HALF_OPEN` → `CLOSED`: on first success
- `HALF_OPEN` → `OPEN`: on failure in probe call

### Token-Bucket Rate Limiter

```python
rl = RateLimiter(rate=10.0, burst=20.0)
await rl.acquire()   # blocks until a token is available
```

`rate` is sustained tokens per second. `burst` is the maximum number of tokens that can accumulate when the limiter is idle (short-term surge capacity). This prevents accidental quota exhaustion under load spikes.

### `LatencyBudget`

The single most important voice-UX mechanism in the system. A `LatencyBudget` is created at the start of each request with an SLA deadline. It is stored in a `contextvars.ContextVar` so every node can check it without being explicitly passed a deadline object.

```python
with LatencyBudget.start(budget_ms=10_000, tier=QoSTier.STANDARD):
    # Every node call inside here can check:
    budget = LatencyBudget.current()
    budget.check(stage="stt")   # raises LatencyBudgetExceeded if time is up
```

When `LatencyBudgetExceeded` is raised inside a node, the graph routes immediately to `error_terminal` without retrying — a retry after a blown SLA budget only makes things worse. This is fundamentally different from a transient exception, which the retry router handles.

### `LoadSheddingGuard`

Each `VoiceGraph` instance owns a `LoadSheddingGuard` with its own `max_inflight` counter. When a new `run()` call arrives and inflight requests already equal `max_inflight`, the guard raises `LoadSheddingRejected` immediately. This is a deliberate design decision: it's better to tell a caller "try again later" fast than to queue them into a multi-minute backlog.

```python
guard = LoadSheddingGuard(max_inflight=20)
async with guard:
    result = await pipeline.run(...)
# LoadSheddingRejected raised instantly if at capacity
```

Because each of the three graph singletons (`voice_graph`, `voice_graph_realtime`, `voice_graph_low_latency`) has its own guard, a burst of standard requests cannot crowd out the realtime pool.

### `BulkheadPool`

A named-semaphore registry. Different node types each claim their own semaphore so a flood of STT requests cannot exhaust the LLM concurrency budget:

```python
bulkheads.acquire("stt_batch", limit=20)
bulkheads.acquire("llm_stream", limit=10)
```

### `InMemoryLRU`

A bounded async-safe LRU dictionary. Used as an automatic fallback when Redis is unreachable. Declared explicitly in `session_store.py` and `LLM_service.py`. When the Redis circuit breaker opens, reads and writes fall through to the LRU and a `degraded_mode` context manager reduces accepted concurrency to avoid overloading the model APIs.

```python
lru = InMemoryLRU(maxsize=1024)
await lru.set("key", "value")
value = await lru.get("key")   # None if evicted
```

### `QoSTier`

```python
class QoSTier(Enum):
    REALTIME = "realtime"   # tightest timeouts, zero retries
    STANDARD = "standard"   # balanced defaults
    BATCH    = "batch"       # relaxed timeouts, higher retry budget
```

Used by `LatencyBudget` to set the SLA deadline and by `LoadSheddingGuard` to decide shed priority.

### OpenTelemetry Setup

Three-layer approach to prevent the "Failed to detach context" error that commonly plagues asyncio + OTel integrations:

**Layer 1 — NoOp tracer when `OTEL_ENABLED=false`:** `get_tracer()` returns `_NOOP_TRACER`. Its `start_as_current_span()` is a plain `contextmanager` that yields without ever calling `otel_ctx.attach()` or `detach()`. The error is structurally impossible on this path.

**Layer 2 — Patch `ContextVarsRuntimeContext.detach`:** Swallows `ValueError` before OTel's own error logger fires. This fixes the rare cross-task context mismatch when `OTEL_ENABLED=true` and a span is started inside a task copied from another context (which is what `run_coroutine_threadsafe()` does in the desktop controller).

**Layer 3 — `logging.Filter` on `opentelemetry.context`:** The absolute backstop. Suppresses the specific "Failed to detach context" log message at the stdlib level regardless of whether layers 1 or 2 fired.

---

## 6. Logging (`log_config.py`)

**File:** `app/common/log_config.py`

Two logging modes controlled by `LOG_MODE` env var:

### `LOG_MODE=standard` (human-readable)

Activates the `_DualSinkRenderer` which forks each log event into two independent sinks:

**Sink A — Rich console:**
```
[10:33:49] [INFO]  ◈  stt_ok                    language=english  lat=3.301  size_mb=0.060
[10:33:53] [INFO]  ◈  llm_ok                    model=gpt-5-mini  lat=3.278  cached=False
[10:33:57] [INFO]  ♪  tts_ok                    voice=nova  chars=261  lat=4.706
```

Each event gets an icon and colour based on its prefix. STT events get `◈` in cyan, TTS events get `♪` in blue, pipeline events get `⚙` in white, session events get `⬡` in yellow, and so on. Key-value pairs are rendered compactly with aliases (`latency_s` → `lat`, `session_id` → `sid`, etc.).

**Sink B — JSON file (`LOG_FILE`):** One JSON object per line, identical to the verbose/structlog format. Safe for Datadog, Loki, CloudWatch.

The event dict is deep-copied before either sink mutates it, so both sinks always see the complete original record.

### `LOG_MODE=verbose` (default)

Pure JSON to stdout. Safe for CI pipelines and log aggregators. No Rich dependency at runtime.

### Usage

```python
from app.common.shared import get_logger
log = get_logger(__name__)

log.info("stt_ok", language="english", latency_s=3.3)
log.warning("session_ip_rebound", change_number=2)
log.error("graph_llm_failed", error=str(exc))
```

---

## 7. Pipeline Orchestration (`voice_graph.py`)

**File:** `app/orchestration/voice_graph.py`

This is the heart of the system. It implements the STT → LLM → TTS pipeline as a LangGraph `StateGraph`, wiring nodes, error handlers, retry loops, routing functions, timeout guards, observability hooks, load shedding, and cancellation support into a single cohesive engine.

### Graph Topology

```
START
  └─► stt
        ├── [ok]     → llm
        └── [error]  → stt_error
              ├── [retries remain, no abort] → stt   ← retry cycle
              └── [exhausted / abort]        → error_terminal → END

      llm
        ├── [ok]     → sanitize → tts
        │                         ├── [ok, IS_DEV]  → audio_sink_dev → END
        │                         ├── [ok, !IS_DEV] → END
        │                         └── [error]       → tts_error → error_terminal → END
        └── [error]  → llm_error
              ├── [retries remain, no abort] → llm   ← retry cycle
              └── [exhausted / abort]        → error_terminal → END
```

**Two graph cycles exist:** `stt → stt_error → stt` and `llm → llm_error → llm`. LangGraph handles cycles natively. Each cycle terminates when the retry router returns `error_terminal` (retries exhausted) or succeeds.

**`audio_sink_dev`** is only wired into the graph when `ENV=development`. The node object does not exist in production compiled graphs at all — zero overhead, no conditional check at request time.

### `VoiceState` — Shared State Schema

Every node in the graph reads from and writes to a `VoiceState` `TypedDict`. All fields are optional (except `audio_path`) so any node can safely call `.get()` without `KeyError`:

```python
class VoiceState(TypedDict, total=False):
    audio_path:           str    # Required input
    request_id:           str    # UUID hex, injected by _prepare_state
    session_id:           str    # From caller; links to session_store
    client_ip:            str    # For session IP-lock validation
    mode:                 str    # "api" | "stream" | "realtime"
    user_input:           str    # Whisper transcript
    llm_response:         str    # Raw LLM output
    cleaned_response:     str    # Post-sanitize output
    audio_output:         str    # Local path or S3 URI to TTS audio
    stt_retries:          int    # Retry counter for STT stage
    llm_retries:          int    # Retry counter for LLM stage
    abort_reason:         str    # Non-empty → skip retry, go to terminal
    session_turn_appended: bool  # Guard against double turn persistence
    degraded:             bool   # True if any stage failed gracefully
    stage_latencies:      dict   # {"stt": 3.3, "llm": 3.2, "tts": 4.7}
```

### `VoicePipelineResult` — Stable Output Contract

Every field is always present. Callers never need to guard against `KeyError`:

```python
class VoicePipelineResult(TypedDict):
    request_id:          str
    transcript:          str
    llm_response:        str
    cleaned_response:    str
    audio_output:        str    # local file path
    audio_s3_uri:        str    # empty when S3 not configured
    error:               str
    error_stage:         str
    degraded:            bool
    stage_latencies:     dict
    pipeline_latency_s:  float
    graph_version:       str
    metadata:            dict   # stt_retries, llm_retries, etc.
```

### `VoiceGraphConfig` — Per-Instance Tuning

Each `VoiceGraph` instance carries its own config, resolved from the settings singleton at construction time:

```python
@dataclass
class VoiceGraphConfig:
    stt_timeout:            float  # Per-stage wall-clock guard
    llm_timeout:            float
    tts_timeout:            float
    max_inflight:           int    # LoadSheddingGuard capacity
    default_tier:           QoSTier
    max_stt_retries:        int
    max_llm_retries:        int
    min_prompt_chars:       int    # Gate: don't call LLM on "uh"
    max_transcript_chars:   int    # Truncate before LLM call
    max_llm_response_chars: int    # Cap before sanitize
    max_tts_chars:          int    # Cap before TTS (OpenAI: 4096)
    stt_llm_queue_depth:    int    # For stream_full() concurrent pipeline
    llm_tts_queue_depth:    int
```

### Graph Construction: `_build_graph_for_instance()`

The graph is built **once per `VoiceGraph` instance** inside `__init__`. Every node function is a closure over the instance's `stt`, `llm`, `tts`, and `cfg` — not module-level globals. This is what makes the three singletons genuinely independent: injecting test doubles or custom node implementations works correctly across all execution paths.

```python
def _build_graph_for_instance(
    self,
    stt: STTNodeProtocol,
    llm: LLMNodeProtocol,
    tts: TTSNodeProtocol,
    cfg: VoiceGraphConfig,
    is_dev: bool,
) -> RunnableSerializable:
    # All node functions defined as closures here
    async def node_stt(state: VoiceState) -> VoiceState:
        # uses stt (closed over), cfg (closed over)
        ...
    # ... node_llm, node_tts, node_sanitize, routing functions ...
    builder = StateGraph(VoiceState)
    builder.add_node("stt", node_stt)
    # ... wire all edges ...
    return builder.compile()
```

### Execution Modes

**`run(state, timeout)`** — Standard batch execution. Awaits the full graph. Returns `VoicePipelineResult`. Used for `/voice` HTTP endpoint and desktop controller.

```python
result = await voice_graph.run(
    {"audio_path": "/tmp/input.m4a", "session_id": "abc123", "mode": "api"},
    timeout=120.0,
)
```

**`stream(state)`** — Token streaming. Runs STT → LLM as streaming, yields LLM tokens as they arrive. Does not run TTS. Used for the WebSocket endpoint.

```python
async for token in voice_graph.stream(state):
    await ws.send_json({"type": "token", "data": token})
```

**`stream_full(state)`** — Concurrent pipeline. STT runs first; as soon as the transcript is ready, LLM streams tokens into a `BoundedPipelineQueue`; as each sentence-boundary chunk accumulates, TTS synthesises it and yields audio bytes — all concurrently. This is the lowest-latency path to first audio.

```python
async for audio_bytes in voice_graph_realtime.stream_full(state):
    await ws.send_bytes(audio_bytes)
```

### Per-Stage Retry Logic

Each stage has a dedicated error handler node and routing function:

```
node_stt fails
    → node_stt_error: increments state["stt_retries"]
    → route_after_stt_error:
        if abort_reason set:  → "error_terminal"
        if retries <= max:    → "stt"          (retry loop)
        if retries > max:     → "error_terminal"
```

The comparison is `retries <= max_retries` (not `<`) because `node_stt_error` has already incremented before the router runs:
- 1st failure → `stt_retries` = 1, `max_stt_retries` = 1 → 1 ≤ 1 → retry
- 2nd failure → `stt_retries` = 2, `max_stt_retries` = 1 → 2 > 1 → terminal

### Graceful Degradation

No stage failure surfaces a raw exception to the caller. Every error path eventually reaches `node_error_terminal`, which:

1. Logs the failure with full context
2. Sets `degraded = True`
3. Fills `llm_response` and `cleaned_response` with a stage-appropriate apology string if they are empty
4. Returns a complete `VoicePipelineResult` with the apology

```python
APOLOGY_STT = "I couldn't catch that. Could you try again?"
APOLOGY_LLM = "I'm having trouble thinking right now. Please try again in a moment."
APOLOGY_TTS = "I have a response but couldn't convert it to audio right now."
```

This means callers always get a well-formed response. They check `degraded` and `error_stage` to decide whether to show an error UI.

### Cancellation

Each in-flight `run()` call is tracked in `self._active_tasks: dict[str, asyncio.Task]`. External cancellation by `request_id`:

```python
cancelled = voice_graph.cancel(
    request_id="abc123",
    reason="user_interrupt",
    source="controller",
)
```

The cancellation context (`reason`, `source`, `requested_at`) is attached to the task object so any downstream `CancelledError` handler can inspect why the task was cancelled.

### Prometheus Metrics

```
voice_pipeline_total                    — total runs by version/status/tier
voice_pipeline_stage_errors_total       — errors by stage
voice_pipeline_stage_retries_total      — retries by stage
voice_pipeline_stage_latency_seconds    — per-stage latency histogram
voice_pipeline_latency_seconds          — total pipeline latency histogram
voice_pipeline_cancellations_total      — by stage
voice_pipeline_active                   — currently in-flight gauge
voice_pipeline_degraded_total           — graceful fallback counter
voice_pipeline_load_shed_total          — requests rejected by shedder
voice_pipeline_budget_breached_total    — SLA budget violations by stage
voice_pipeline_stream_full_active       — concurrent stream_full sessions
```

### Three Singletons

```python
# Balanced defaults — one retry per stage
voice_graph = VoiceGraph(version="v2")

# Tight: 10s STT / 15s LLM / 8s TTS, zero retries, REALTIME tier
voice_graph_realtime = VoiceGraph(
    version="realtime",
    config=VoiceGraphConfig(
        stt_timeout=10.0, llm_timeout=15.0, tts_timeout=8.0,
        max_inflight=30, default_tier=QoSTier.REALTIME,
        max_stt_retries=0, max_llm_retries=0,
    ),
)

# Medium: 15s / 20s / 12s, one retry, 50 inflight
voice_graph_low_latency = VoiceGraph(
    version="low_latency",
    config=VoiceGraphConfig(
        stt_timeout=15.0, llm_timeout=20.0, tts_timeout=12.0,
        max_inflight=50, default_tier=QoSTier.STANDARD,
        max_stt_retries=1, max_llm_retries=1,
    ),
)
```

---

## 8. STT Node (`STT_service.py`)

**File:** `app/nodes/STT_service.py`

Wraps the OpenAI Whisper API with every production resilience layer. The graph only ever calls the `STTNodeProtocol` interface — whether that resolves to a local node or a remote HTTP client is determined at startup by `get_stt_node()`.

### `STTNodeProtocol`

```python
@runtime_checkable
class STTNodeProtocol(Protocol):
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
        ...
    ) -> AsyncIterator[STTSegment]: ...

    async def health(self) -> ServiceHealthState: ...
    async def close(self) -> None: ...
```

### `STTNode` — Local Implementation

**Validation:** Before any API call, the node validates that the audio file exists, has a supported extension, is not empty, and does not exceed `STT_MAX_FILE_MB`. Path traversal attempts (`../`) are rejected.

**Concurrency:** `asyncio.Semaphore(STT_MAX_CONCURRENT)` prevents more than N simultaneous Whisper calls. Combined with the token-bucket `RateLimiter`, this gives two-layer burst protection.

**Circuit breaker:** Wraps the Whisper API call. Three consecutive failures open the breaker for 30 seconds.

**Retries with jitter:** `backoff_retry()` wraps the underlying API call with exponential backoff and random jitter to prevent retry storms.

**Confidence logging:** After transcription, `avg_logprob` across all segments is logged. Values below -1.0 indicate very low confidence (background noise, heavily accented speech, wrong language hint).

**S3 integration:** If `STT_S3_BUCKET` is set, the source audio is downloaded from S3 rather than read from local disk. The completed transcript is uploaded back to S3 under `STT_S3_TRANSCRIPT_PREFIX`.

**Streaming (`transcribe_stream`):** Splits the audio file into overlapping N-second `.wav` chunks, submits them to Whisper in parallel (bounded by `STT_STREAM_MAX_PARALLEL`), and yields `STTSegment` objects as each chunk completes. For files shorter than `STT_STREAM_SINGLE_PASS_THRESHOLD_S` (default 12s), it falls back to a single-pass non-streamed call — chunking overhead outweighs the latency benefit for short inputs.

**Fast path (`transcribe_fast`):** Uses `response_format="text"` which skips segment metadata and returns plain text faster.

**Local Whisper fallback:** If `STT_LOCAL_FALLBACK_URL` is set and the primary OpenAI endpoint fails, the node tries the local endpoint before the circuit breaker opens.

### `RemoteSTTClient` — Distributed Mode

When `STT_SERVICE_URL` is set, `get_stt_node()` returns a `RemoteSTTClient` instead of `STTNode`. The client calls the remote service over HTTPS using `httpx`:

- Binary audio posted as `multipart/form-data`
- W3C `traceparent` header injected for distributed trace continuity
- `X-Latency-Budget-Ms` header forwarded so the remote service can self-abort if the SLA is already blown
- Streaming uses SSE (`text/event-stream`), each event a JSON-serialised `STTSegment`
- Circuit breaker wraps all remote calls
- `backoff_retry` with 3 attempts and 1.5s base delay

### `STTResult`

```python
class STTResult(TypedDict):
    text:               str    # Full transcript
    language:           str    # Detected language
    duration_s:         float  # Audio duration
    processing_s:       float  # Wall-clock Whisper latency
    source:             str    # "local" or "s3"
    s3_transcript_key:  str    # S3 key if uploaded, else ""
```

### `STTSegment` (streaming)

```python
class STTSegment(TypedDict):
    text:        str
    language:    str
    start:       float   # Seconds from audio start
    end:         float
    avg_logprob: float   # Confidence indicator
    chunk_index: int
    is_final:    bool    # True on the last segment
```

### Prometheus Metrics

```
stt_requests_total              — by status/mode/provider
stt_latency_seconds             — end-to-end latency histogram
stt_time_to_first_segment_seconds  — stream TTFS
stt_file_size_mb                — upload size distribution
stt_chunks_per_stream           — chunks per streaming call
stt_active_requests             — in-flight gauge
stt_circuit_breaker_open        — 1 when breaker OPEN
stt_latency_budget_exceeded_total
```

---

## 9. LLM Node (`LLM_service.py`)

**File:** `app/nodes/LLM_service.py`

Wraps OpenAI chat completions via LangChain, with a Redis response cache, stampede locks, dual circuit breakers (stream and batch), token-bucket rate limiting, fallback model, and `InMemoryLRU` fallback when Redis is down.

### `LLMNodeProtocol`

```python
@runtime_checkable
class LLMNodeProtocol(Protocol):
    async def generate(
        self,
        prompt: str,
        request_id: str | None = None,
    ) -> dict[str, Any]: ...

    async def stream(
        self,
        prompt: str,
        request_id: str | None = None,
    ) -> AsyncIterator[str]: ...

    async def health(self) -> ServiceHealthState: ...
    async def close(self) -> None: ...
```

### `LLMNode` — Local Implementation

**System prompt:** `SYSTEM_PROMPT` is a module-level constant injected into every call as a `SystemMessage`. The graph's `node_llm` prepends conversation history and the current `user_input` as a `HumanMessage`. The LLM node itself never builds the full message list — that is the graph's responsibility.

**Versioned cache key:** The Redis cache key encodes the model name, temperature, a hash of the system prompt, and a hash of the full prompt:

```python
key = make_versioned_cache_key(
    prefix="llm:v3",
    model=PRIMARY_MODEL,
    temperature=TEMPERATURE,
    system_hash=sha256(SYSTEM_PROMPT),
    prompt_hash=sha256(prompt),
)
```

Config changes (model switch, temperature change, system prompt edit) automatically bust all cached entries — no manual cache flush required.

**Cache stampede lock:** When two concurrent requests arrive with identical prompts and neither is cached yet, without locking both would call the OpenAI API. The node uses a Redis `SETNX` lock keyed on the prompt hash:

```python
# Acquire lock → call API → store result → release lock
# Second caller: waits on lock → reads cached result → returns
```

**Fallback model:** When the primary model's circuit breaker opens (3 consecutive failures), requests are rerouted to `LLM_FALLBACK_MODEL` automatically. The response carries `model_used` so callers can see which model actually answered.

**Dual circuit breakers:** `_batch_breaker` and `_stream_breaker` are independent. A slow or failing streaming endpoint does not trip the batch breaker that serves synchronous `/voice` requests.

**`InMemoryLRU` fallback:** If the Redis circuit breaker opens, cache reads fall through to an in-process LRU (bounded at 512 entries). `degraded_mode` is entered, which halves admitted concurrency to avoid flooding the model APIs during the Redis outage window.

**Token usage logging:** Every response logs `prompt_tokens`, `completion_tokens`, and `total_tokens`. These feed the Prometheus histogram `llm_tokens_total` and the MongoDB event document.

**`generate()` return shape:**

```python
{
    "response":           str,    # LLM output text
    "model_used":         str,    # actual model that answered
    "cached":             bool,   # True if served from Redis/LRU
    "prompt_tokens":      int,
    "completion_tokens":  int,
    "latency_s":          float,
}
```

### `RemoteLLMClient` — Distributed Mode

When `LLM_SERVICE_URL` is set, `get_llm_node()` returns a `RemoteLLMClient`. The client posts to the remote service's `/generate` endpoint. Streaming uses SSE. OTel headers and `X-Latency-Budget-Ms` are forwarded. The response shape is identical to the local node.

### `stream_with_metadata()`

Wraps `stream()` and emits a final `{"type": "metadata", ...}` dict after the last token. Used by WebSocket handlers that need both the token stream and final usage stats.

### Prometheus Metrics

```
llm_requests_total                  — by status/mode/model
llm_latency_seconds                 — histogram
llm_time_to_first_token_seconds     — streaming TTFT
llm_tokens_total                    — prompt + completion tokens
llm_cache_hits_total
llm_cache_misses_total
llm_fallback_total                  — primary → fallback switches
llm_circuit_open                    — gauge
llm_active_requests                 — gauge
llm_latency_budget_exceeded_total
```

---

## 10. TTS Node (`TTS_service.py`)

**File:** `app/nodes/TTS_service.py`

Wraps the OpenAI TTS API. Synthesises text to audio and writes the result to a local file. Optionally uploads to S3. Implements a background cleanup task that removes old local files based on `TTS_LOCAL_FILE_TTL`.

### `TTSNodeProtocol`

```python
@runtime_checkable
class TTSNodeProtocol(Protocol):
    async def synthesise(
        self,
        text: str,
        voice: str | None = None,
        request_id: str | None = None,
    ) -> TTSResult: ...

    async def stream_synthesis(
        self,
        text: str,
        voice: str | None = None,
        request_id: str | None = None,
    ) -> AsyncIterator[bytes]: ...

    async def health(self) -> ServiceHealthState: ...
    async def close(self) -> None: ...
```

### `TTSNode` — Local Implementation

**Chunking:** OpenAI TTS accepts a maximum of 4096 characters per call. `GRAPH_MAX_TTS_CHARS` (default 4000) keeps the total below this ceiling. If the response exceeds the node's per-chunk limit, it is split on sentence boundaries and synthesised in parallel calls, then the chunks are concatenated in order.

**Output file naming:** Each file is named `tts_{uuid_hex[:8]}.{format}` and written to `TTS_OUTPUT_DIR`. The full path is stored in `TTSResult.local_path` and returned to the graph as `audio_local_path`.

**S3 upload:** If `TTS_S3_BUCKET` is set, the synthesised file is uploaded after local write. `TTSResult.s3_uri` carries the `s3://bucket/prefix/filename` URI.

**Background file cleanup:** A `threading.Timer` schedules deletion of each file after `TTS_LOCAL_FILE_TTL` seconds. If S3 is enabled, local files are short-lived working copies that can be cleaned up quickly.

**Circuit breaker + rate limiter:** Same pattern as STT. Three failures open the breaker for 30 seconds. The token-bucket limiter prevents accidental API quota exhaustion.

**`TTSResult`:**

```python
class TTSResult(TypedDict):
    local_path:   str     # Absolute path to the generated audio file
    s3_uri:       str     # S3 URI or "" if not uploaded
    voice:        str     # Voice used (nova, alloy, etc.)
    chars:        int     # Characters synthesised
    chunks:       int     # Number of API calls made
    latency_s:    float
```

### `RemoteTTSClient` — Distributed Mode

When `TTS_SERVICE_URL` is set, `get_tts_node()` returns a `RemoteTTSClient`. The client posts text to the remote service and receives audio bytes in the response body. Streaming synthesis uses SSE with binary chunks.

### Prometheus Metrics

```
tts_requests_total
tts_latency_seconds
tts_chars_total
tts_chunks_per_request
tts_file_size_bytes
tts_active_requests
tts_circuit_open
tts_s3_uploads_total
tts_latency_budget_exceeded_total
```

---

## 11. Sanitize Node (`sanitize.py`)

**File:** `app/nodes/sanitize.py`

A pure function that runs between LLM and TTS. It never hard-fails — worst case it returns the input unchanged. Its jobs:

1. **Size cap:** Truncates text to `max_chars` on a sentence boundary where possible (so TTS doesn't synthesise a mid-sentence cut). Sets `truncated=True` in the result if truncation occurred.

2. **Prompt injection detection:** Scans for known injection patterns (`ignore previous instructions`, `you are now`, `[SYSTEM]`, etc.). Logs a warning and strips or neutralises the offending content.

3. **Control character stripping:** Removes null bytes, form feeds, vertical tabs, and other non-printable characters that confuse TTS.

4. **Length normalisation:** Collapses runs of whitespace and trims.

```python
from app.nodes.sanitize import sanitize

result = sanitize(raw_text, max_chars=4000, request_id=rid)
print(result.text)      # cleaned text
print(result.truncated) # True if it was cut
print(result.warnings)  # list of detected issues
```

The graph uses `node_sanitize` which wraps this function, logs via `SanitizeEmitter`, and records the OTel span.

---

## 12. Session Store (`session_store.py`)

**File:** `app/user_tracking/session_service/session_store.py`

An IP-locked, Redis-backed session store. One active session per IP address at any given time. Raw IPs are never stored — only an HMAC-SHA256 derived fingerprint.

### Security Design

```
client_ip  →  canonical_ip()  →  ip_hash = HMAC-SHA256(SESSION_SECRET_KEY, canonical_ip)
                                  session_id = SHA256(ip_hash + urandom(16))
```

- `ip_hash` is one-way: given the hash, you cannot recover the IP without the secret key
- `session_id` is non-reversible: it does not encode the IP
- `SESSION_SECRET_KEY` rotation invalidates all existing sessions automatically (hashes no longer match)
- Raw IPs never appear in Redis, logs, or API responses — only partially masked display strings (`127.0.*.*`)

### Redis Key Layout

```
session:lock:v1:{ip_hash}      NX sentinel — only one session per IP
session:data:v1:{session_id}   JSON blob — conversation state
session:meta:v1:{ip_hash}      Maps ip_hash → session_id for lookup
```

All keys carry `TTL = SESSION_TTL_S + SESSION_GRACE_S`. The grace period ensures an in-flight request that arrives just as the TTL expires still sees the session data.

### IP Change Handling

Sessions allow up to `SESSION_MAX_IP_CHANGES` (default 3) IP address changes. This accommodates legitimate network changes (WiFi → mobile, NAT reassignment) while flagging actual session hijacking attempts:

```
change 1 → WARN: session_ip_rebound (changes_remaining=2)
change 2 → WARN: session_ip_rebound (changes_remaining=1)
change 3 → ERROR: session_suspended_ip_limit
           → session ended, session_id invalidated
           → next request gets SessionIPChangeLimitExceeded
```

### `SessionData`

```python
@dataclass
class SessionData:
    session_id:      str
    ip_hash:         str
    created_at:      float
    last_active:     float
    turns:           list[ConversationTurn]   # rolling window
    suspended:       bool
    ip_change_count: int
    ip_change_log:   list[IPChangeEntry]
    metadata:        dict
```

`ConversationTurn` holds `(user: str, assistant: str, ts: float)`. The rolling window is capped at `SESSION_MAX_TURNS` (default 20). Oldest turns are pruned first. `to_langchain_messages()` serialises the window to `[HumanMessage, AIMessage, ...]` pairs for LLM injection.

### `InMemoryLRU` Fallback

If Redis is unreachable, `session_store` falls back to an in-process LRU (bounded at `SESSION_LRU_SIZE`). Cross-process session uniqueness cannot be maintained in this mode — it is clearly documented as a degraded operating mode. The LRU fallback is logged as a warning and the `session_active` Prometheus gauge is updated.

### FastAPI Routes

Mounted at `/session` prefix in `main.py`:

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/session/register` | None | Register new session, returns `session_id` |
| `POST` | `/session/end` | `X-Session-ID` | End session, release IP lock |
| `GET` | `/session/status` | `X-Session-ID` | Turn count, elapsed time, IP change log |
| `GET` | `/session/health` | None | Session store health |

### `require_session` Dependency

```python
from app.user_tracking.session_service.session_store import require_session

@app.post("/voice")
async def voice_chat(session: SessionData = Depends(require_session)):
    ...
```

This FastAPI dependency extracts `X-Session-ID` from the request header, resolves it against the session store, validates that the caller's IP matches, and returns the `SessionData`. On any failure it raises the appropriate `HTTPException` (401, 403, 404, or 409).

### Prometheus Metrics

```
session_created_total
session_ended_total       — by reason (explicit_end, ip_change_limit, ttl_expiry)
session_rejected_total    — by reason (locked, ip_unresolvable)
session_turns_total
session_lru_hits_total
session_lock_conflicts_total
session_active            — gauge
session_history_turns     — histogram of turn count at LLM call time
```

---

## 13. Conversation Memory (`conversation_memory.py`)

**File:** `app/user_tracking/session_service/conversation_memory.py`

Sits between `voice_graph.py` and `session_store.py`. Every pipeline call goes through two operations: `resolve()` before the LLM call, and `commit()` after the LLM response.

### `resolve(session_id, user_text)`

1. Loads `SessionData` from `session_store` (or LRU fallback)
2. Builds the `InterviewState` snapshot (topics covered, current topic, question history, hint budget)
3. Constructs the context prefix string that summarises interview progress
4. Returns `MemoryContext` containing the full message list and the interview state

```python
memory_ctx = await conversation_memory.resolve(
    session_id=session_id,
    user_text=transcript,
)
# memory_ctx.messages         — LangChain message list
# memory_ctx.interview_state  — InterviewState snapshot
# memory_ctx.turn_index       — how many turns so far
```

### `commit(session_id, user_text, assistant_text)`

1. Appends the completed turn to `session_store`
2. Updates `InterviewState` (marks topic as covered, increments question/hint counters)
3. Checks if history is approaching the rolling window cap → if so, triggers compression
4. Writes to `transcript_writer`

### History Compression

When the session's turn count approaches `SESSION_MAX_TURNS`, older turns are collapsed into a summary injected as a `SystemMessage`. This means nothing is truly lost to pruning — the LLM always has context about what was discussed early in the interview, just in compressed form rather than verbatim.

### `InterviewState`

```python
@dataclass
class InterviewState:
    current_topic:    str
    topics_covered:   list[str]
    questions_asked:  set[str]        # fingerprints for dedup
    hint_budget:      dict[str, int]  # hints remaining per topic
    follow_up_depth:  int             # consecutive follow-ups on same topic
```

### Question Dedup

Each question the AI asks is fingerprinted (SHA256 of lowercased, punctuation-stripped text). The fingerprint is stored in `InterviewState.questions_asked`. Before the context prefix is built, these fingerprints are checked to prevent the interviewer from asking the same question twice in a session.

### Prometheus Metrics

```
memory_resolves_total
memory_commits_total
memory_fallback_total        — LRU fallback activations
memory_compressions_total    — history compression events
memory_history_depth         — histogram of turn count at resolve() time
```

---

## 14. Evaluation Engine (`evaluation_engine.py`)

**File:** `app/eval/evaluation_engine.py`

Scores each candidate answer against a structured rubric using a dedicated reasoning model. Runs **entirely off the critical path** — all scoring is `asyncio.create_task()` fire-and-forget. TTS latency is never affected.

### Off-Path Design

```python
# In voice_graph.py, after session_store.append_turn():
evaluation_engine.schedule_turn(
    session_id=session_id,
    question=_question,
    candidate_ans=transcript,
    turn_index=len(_reloaded.turns) - 1,
    request_id=rid,
)
# Returns immediately. Scoring happens in background.
```

The `schedule_turn()` method uses a `weakref` to the engine to avoid preventing garbage collection, creates a background task, and returns. If the event loop is shut down, in-flight evaluations are cancelled cleanly.

### Cost Controls

Every mechanism exists to cut token cost without degrading signal quality:

**Adaptive sampling:** Full evaluation for the first `EVAL_ALWAYS_EVALUATE_TURNS` turns (default 3), then `1-in-EVAL_SAMPLE_RATE` sampling. Chatty candidates don't burn 10× the budget.

**Per-session token budget:** A hard cap stored in Redis (`eval:budget:v1:{session_id}`). Once a session exhausts its budget, evaluation stops silently with a log warning.

**Minimum answer gate:** Answers shorter than `EVAL_MIN_ANSWER_CHARS` (default 40 characters) are skipped. "Yes", "I don't know", and one-liners don't need scoring.

**Prompt truncation:** Transcripts and LLM responses are capped at `EVAL_MAX_PROMPT_CHARS` before the scoring API call.

**Tight completion cap:** `max_tokens` on the scoring chain constrains output size. Structured JSON scores don't need paragraphs.

**Token-bucket rate limiter:** Separate from the main LLM limiter so evaluation cannot consume the interview quota.

**Dedicated bulkhead:** Max `EVAL_MAX_CONCURRENT` parallel scoring calls.

**Circuit breaker:** Three consecutive scoring failures open the breaker and pause evaluation until recovery.

**Content-hash dedup:** Identical `(question, answer)` pairs return cached scores from Redis (`eval:dedup:v1:{session_id}:{turn_index}`).

### Scoring Schema

```python
@dataclass
class TurnScore:
    turn_index:       int
    technical:        float    # 0.0–1.0
    communication:    float
    problem_solving:  float
    overall:          float
    strengths:        list[str]
    improvements:     list[str]
    model_used:       str
    tokens_used:      int
    eval_latency_s:   float
    skipped:          bool
    skip_reason:      str
```

Scores are stored in Redis (`eval:score:v1:{session_id}:{turn_index}`) and can be retrieved at any time during or after the session.

### Prometheus Metrics

```
eval_scheduled_total
eval_completed_total
eval_skipped_total          — by reason
eval_failed_total
eval_latency_seconds
eval_tokens_total
eval_budget_exhausted_total
eval_circuit_open
eval_active
```

---

## 15. Transcript Writer (`transcription.py`)

**File:** `app/user_tracking/transcript/transcription.py`

An async dual-sink transcript writer. All public methods are fire-and-forget coroutines — they enqueue the write and return immediately. A single background task drains the queue and does the actual I/O.

### Two Sinks

**Sink A — Human-readable `.txt` file (per session):**
```
────────────────────────────────────────────────────────────
Session  : 7aab4993867a2f04c64a36fd51875b12...
Started  : 2026-02-22 10:33:43 UTC
────────────────────────────────────────────────────────────
[10:33:49] [7aab4993]   Tell me about system design.
[10:33:53] [AI]         System design involves...

```

Named `session_{sid[:16]}.txt` and written to `TRANSCRIPT_DIR`. Simple, grep-able, shareable with candidates for review.

**Sink B — Observability (`emit()`):**

Each turn emits an `ObsEvent` with `EventKind.TRANSCRIPT_TURN`. This fans out to structlog (JSON log), Prometheus (`ai_transcript_turns_total`), MongoDB, and OTel simultaneously.

### Usage

```python
from app.user_tracking.transcript.transcription import transcript_writer

await transcript_writer.open_session(session_id)     # writes header
await transcript_writer.write_turn(
    session_id=session_id,
    user_text="Tell me about system design.",
    assistant_text="System design involves...",
    request_id=rid,
)
await transcript_writer.close_session(session_id)    # writes footer
await transcript_writer.flush(timeout=5.0)           # graceful shutdown drain
```

### Queue and Backpressure

The internal `asyncio.Queue` has a depth of `TRANSCRIPT_QUEUE_DEPTH` (default 512). If the queue fills up (e.g., disk I/O is blocked), `_enqueue()` drops the entry and calls `TranscriptEmitter.queue_drop()` — the pipeline is never back-pressured by transcript I/O.

---

## 16. FastAPI Gateway (`main.py`)

**File:** `app/endpoint/main.py`

The HTTP boundary. Handles upload validation, session auth, request routing, timeout, and response shaping. All pipeline logic stays in `voice_graph.py`.

### Application Setup

```python
app = FastAPI(
    title="Voice Assistant API",
    docs_url=None,       # self-hosted at /docs
    redoc_url=None,
    openapi_url=None,    # self-hosted at /openapi.json
    lifespan=_lifespan,
)
```

The OpenAPI schema is built lazily on the first `/openapi.json` request (not at import time) to avoid a race with uvicorn's reloader process. The schema is patched to include the `X-Session-ID` `apiKey` security scheme.

### Lifespan

```python
@asynccontextmanager
async def _lifespan(app):
    _obs_bootstrap(start_prometheus=False, start_mongo=True, write_grafana=True, start_otel=True)
    log.info("api_startup", version=GRAPH_VERSION)
    yield
    # On shutdown: drain all three graph instances concurrently
    await asyncio.gather(*(g.shutdown() for g in _ALL_GRAPHS), return_exceptions=True)
```

`start_prometheus=False` because `shared.py` already bound its own registry to `prom_registry`. Both registries are served together on `/metrics`.

### Upload Validation (`_validate_upload`)

Before the pipeline runs, every audio upload is validated:
1. File extension must be in `SUPPORTED_EXTENSIONS` (`.wav`, `.mp3`, `.mp4`, `.m4a`, `.webm`, `.mpeg`, `.mpga`)
2. MIME type checked against `SUPPORTED_MIME_TYPES`
3. File size must be nonzero and ≤ `MAX_UPLOAD_MB`

### Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/voice` | `X-Session-ID` | Full pipeline, returns JSON with audio path |
| `WS` | `/voice/stream` | `X-Session-ID` header | WebSocket token streaming |
| `POST` | `/session/register` | None | Register new session |
| `POST` | `/session/end` | `X-Session-ID` | End session |
| `GET` | `/session/status` | `X-Session-ID` | Session status |
| `POST` | `/ping` | `X-Session-ID` | Auth smoke-test |
| `POST` | `/cancel/{request_id}` | None | Cancel in-flight pipeline |
| `POST` | `/interrupt` | None | Cancel most recent pipeline |
| `GET` | `/health` | None | Aggregate node health |
| `GET` | `/metrics` | None | Prometheus metrics |
| `GET` | `/docs` | None | Swagger UI |
| `GET` | `/openapi.json` | None | OpenAPI schema |

### `_ok_response` — Stable Response Contract

```python
{
    "request_id":         "ed175235780f19ee...",
    "transcript":         "Tell me about system design.",
    "response":           "System design involves...",
    "cleaned_response":   "System design involves...",
    "audio":              "audio/audio_OUTPUT/tts_ebaa5613.mp3",
    "audio_s3_uri":       "",
    "degraded":           false,
    "error":              "",
    "error_stage":        "",
    "stage_latencies":    {"stt": 3.3, "llm": 3.3, "tts": 4.7},
    "pipeline_latency_s": 11.3,
    "graph_version":      "v2",
    "metadata":           {"stt_retries": 0, "llm_retries": 0}
}
```

All fields are always present. `degraded=true` with a non-empty `error` field signals a graceful fallback.

### WebSocket Protocol (`/voice/stream`)

```
Client → Server:
  Message 1: binary audio bytes
  Message 2 (optional): JSON {"language": "en", "tts_voice": "nova"}

Server → Client:
  {"type": "token",    "data": "System"}
  {"type": "token",    "data": " design"}
  ...
  {"type": "done",     "request_id": "...", "full_response": "..."}
  {"type": "error",    "error": "..."} on failure
```

### Observability Middleware

Every HTTP request passes through `_observability_middleware`:
1. Assigns or reads `X-Request-ID`
2. Extracts W3C `traceparent`/`tracestate` from upstream callers
3. Starts an OTel span for the HTTP request
4. Records `api_requests_total`, `api_request_latency_seconds`
5. Adds `X-Request-ID` and `X-Trace-ID` to every response header

---

## 17. Desktop Controller (`controller.py`)

**File:** `app/orchestration/controller.py`

The desktop input layer. Handles keyboard events, audio recording, session management, and pipeline dispatch. All pipeline logic stays in `voice_graph.py`.

### Architecture

A single asyncio event loop runs on a **background thread** for the lifetime of the process. The main thread runs the keyboard polling loop synchronously. Pipeline dispatches use `run_coroutine_threadsafe()` — one loop, zero teardown overhead per PTT press, clean cancellation semantics.

```
Main thread (synchronous):
  keyboard.is_pressed(PTT_KEY) loop
    ↓ PTT held
    record_audio_until_released()   ← blocks main thread during recording
    ↓ PTT released
    run_coroutine_threadsafe(_dispatch(path), loop)

Background asyncio loop thread:
  _dispatch(path):
    session = await session_store.register(local_ip)
    result  = await voice_graph.run(state)
    if result["audio_output"]:
        play_audio(result["audio_output"])
```

### PTT Flow

1. `keyboard.is_pressed(PTT_KEY)` detects the press
2. Any in-flight pipeline is cancelled (`voice_graph.cancel()`)
3. Any audio currently playing is stopped (`stop_all()`)
4. `record_audio_until_released()` captures microphone audio until the key is released
5. If the recording is non-empty, it is saved to `audio_INPUT/` and dispatched
6. `_dispatch()` runs the full pipeline asynchronously and plays the result

### Session Registration

The controller registers a new session using the machine's outbound IP:

```python
def _get_local_ip() -> str:
    """UDP connect trick — no packet sent, but OS selects correct source interface."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
```

The session is re-registered per dispatch if the existing session has expired. The session ID is carried in all pipeline state so conversation history persists across PTT presses within a session.

### Startup Display

In `LOG_MODE=standard`, the controller renders an animated boot banner before entering the PTT loop:

```
  Voice Assistant
  ───────────────────────────────────────────
  STT             ████████████████████████  whisper-1               ✓
  LLM             ████████████████████████  gpt-5-mini              ✓
  TTS             ████████████████████████  tts-1 / nova            ✓
  Session Store   ████████████████████████  redis + lru             ✓
  ───────────────────────────────────────────
          PTT   Hold H to talk
          Exit  ESC
  ───────────────────────────────────────────
```

### Graceful Shutdown

`EXIT_KEY` press or `SIGTERM`/`SIGINT`:
1. Sets `_shutdown_event`
2. Cancels active pipeline task
3. Drains active audio playback
4. Calls `evaluation_engine.shutdown()` to flush pending scores
5. Calls `transcript_writer.flush()` to drain pending writes
6. Calls `voice_graph.shutdown()` to cancel all tasks and close node connections
7. Calls `loop.stop()`; joins the loop thread with `LOOP_JOIN_TIMEOUT`

### Config

```
PTT_KEY               = h       (hold to talk)
EXIT_KEY              = esc
POLL_INTERVAL_S       = 0.03    (~33 Hz key polling)
DEBOUNCE_S            = 0.35    (prevents key bounce double-trigger)
INTERRUPT_DRAIN_S     = 0.05    (drain window after cancel signal)
LOOP_DRAIN_S          = 0.10    (drain window before loop.stop())
LOOP_JOIN_TIMEOUT     = 3.0     (thread join timeout)
```

---

## 18. Audio Recorder (`recorder.py`)

**File:** `app/audio_essentials/recorder.py`

Captures microphone audio while the PTT key is held. Returns raw `.wav` bytes.

**Implementation:** Uses `sounddevice.InputStream` with a callback that appends PCM frames to a thread-safe buffer. Recording stops when the PTT key is released. The buffer is assembled into a `.wav` file via the `wave` module.

**Empty recording detection:** If the total recorded duration is below a threshold (e.g., 0.2 seconds), the recording is considered empty (accidental key tap) and the pipeline is not dispatched. `ControllerEmitter.empty_recording()` fires to increment the `ai_controller_empty_recordings_total` counter.

**Silence detection (optional):** If `RECORDER_SILENCE_THRESHOLD_DB` is set, the recorder tracks RMS power and stops early if prolonged silence is detected (user finished speaking but forgot to release the key).

---

## 19. Audio Player (`player.py`)

**File:** `app/audio_essentials/player.py`

Two playback paths sharing the physical device. Mutually exclusive by design.

### File Path: `play_audio(path)`

Decodes the audio file (MP3, WAV, etc.) using `soundfile` and plays it via `sounddevice.play()` on a worker thread. `on_complete` callback is called when playback finishes. Supports `gain_db` adjustment.

### Stream Path: `play_audio_bytes(buf)`

For `stream_full()` partial-TTS streaming. Each call receives a chunk of PCM audio bytes and enqueues them onto a bounded queue. A persistent `sounddevice.OutputStream` is opened once and kept alive between chunks, eliminating the 50–150 ms PortAudio device-open cost on every sentence.

**Latency optimisations:**

1. **Persistent OutputStream** — opened once per streaming session, not per chunk. Eliminates device-open cost.
2. **Dedicated writer thread + bounded queue** — `play_audio_bytes()` decodes bytes → PCM (~1–5 ms) then enqueues and returns in ~μs. The 20–50 ms `write()` call happens on the writer thread, off the async event loop.
3. **`latency='low'`** — asks PortAudio for its minimum safe hardware buffer, saving 10–30 ms of output-buffer depth.
4. **Silent warmup chunk** — written immediately after stream open to prime the driver pipeline so the first real audio chunk doesn't pay the cold-start penalty.
5. **No lock on the write path** — the writer thread owns the `OutputStream` exclusively.

### Control

```python
stop_audio()      # stops file-path playback
stop_stream()     # drains queue, stops stream-path playback
stop_all()        # both paths
is_playing()      # True if file-path playback active
is_streaming()    # True if stream-path active
```

---

## 20. Observability Stack (`observability.py`)

**File:** `app/monitoring/observability.py`

The single observability kernel that every component in the system uses to emit events. A single `emit()` call fans out simultaneously to four independent sinks.

### Four Sinks

1. **structlog** → Rich console + JSON file (via `log_config.py`)
2. **Prometheus** → counters, histograms, gauges on `/metrics`
3. **MongoDB** → one structured document per event for historical analysis
4. **OpenTelemetry** → span attributes/events for trace correlation

### `ObsEvent`

```python
@dataclass
class ObsEvent:
    kind:           str | EventKind   # event type
    service:        str               # "stt" | "llm" | "tts" | ...
    session_id:     str = ""
    request_id:     str = ""
    ts:             float = 0.0       # defaults to time.time()
    transcript:     str = ""
    transcript_chars: int = 0
    latency_ms:     float = 0.0
    model:          str = ""
    error:          str = ""
    extra:          dict = field(default_factory=dict)
```

### `EventKind` Taxonomy

```python
class EventKind(str, Enum):
    # Pipeline lifecycle
    PIPELINE_START  = "pipeline_start"
    PIPELINE_DONE   = "pipeline_done"
    PIPELINE_RETRY  = "pipeline_retry"

    # Stage events
    STT_START = "stt_start";  STT_OK = "stt_ok";  STT_FAILED = "stt_failed"
    LLM_START = "llm_start";  LLM_OK = "llm_ok";  LLM_FAILED = "llm_failed"
    TTS_START = "tts_start";  TTS_OK = "tts_ok";  TTS_FAILED = "tts_failed"

    # Session
    SESSION_REGISTERED = "session_registered"
    SESSION_ENDED      = "session_ended"

    # Evaluation
    EVAL_SCORED = "eval_scored"
    EVAL_SKIPPED = "eval_skipped"

    # Transcript
    TRANSCRIPT_TURN = "transcript_turn"
    # ... etc.
```

### Emitter Classes

Each service has a typed emitter class with static methods:

```python
STTEmitter.start(session_id, request_id, model, ...)
STTEmitter.ok(session_id, request_id, latency_ms, language, ...)
STTEmitter.failed(session_id, request_id, error, error_type)

LLMEmitter.start(...)
LLMEmitter.ok(..., cache_hit, model_used, prompt_tokens, ...)
LLMEmitter.failed(...)

TTSEmitter.start(...)
TTSEmitter.ok(..., voice, chars, chunks, ...)
TTSEmitter.failed(...)

PipelineEmitter.start(...)
PipelineEmitter.done(...)
PipelineEmitter.retry(...)

SessionEmitter.registered(...)
SessionEmitter.ended(...)
SessionEmitter.ip_rebound(...)

ControllerEmitter.ptt_pressed(...)
ControllerEmitter.ptt_released(...)
ControllerEmitter.empty_recording(...)
```

### MongoDB Sink

The MongoDB writer runs on a **background thread** with an `asyncio.Queue`. Every `emit()` call puts a document on the queue without blocking. The writer thread drains the queue and inserts documents into `MONGO_DB.MONGO_COLLECTION`. Documents are full `ObsEvent` serialisations plus timestamp, session, and request IDs.

### OTel Span Context Managers

```python
async with pipeline_span(session_id, request_id) as span:
    span.set_attribute("version", GRAPH_VERSION)

async with stt_span(session_id, request_id, model) as span:
    result = await stt_node.transcribe(audio_path)
    span.set_attribute("language", result["language"])
    span.set_attribute("latency_s", result["processing_s"])
```

### Bootstrap

```python
from app.monitoring.observability import bootstrap

bootstrap(
    start_prometheus=True,   # bind /metrics HTTP server
    start_mongo=True,        # start MongoDB writer thread
    write_grafana=True,      # write provisioning YAML + dashboard JSON
    start_otel=True,         # initialize TracerProvider + MeterProvider
)
```

Protected by `_bootstrap_lock` and `_bootstrapped` flag — idempotent, safe to call multiple times. The `_bootstrapped` flag is module-level and resets on process restart, so each new process bootstraps exactly once.

### Grafana Dashboard Auto-Provisioning

`write_grafana_dashboard()` writes all required files to the paths that Grafana's file provisioner watches:

```
infra/grafana/dashboards/ai_pipeline.json        — main dashboard
infra/grafana/provisioning/datasources/
    prometheus.yaml
    tempo.yaml
    loki.yaml
infra/grafana/provisioning/dashboards/ai_pipeline.yaml
```

The dashboard includes panels for active pipelines, active sessions, pipeline P99 latency, stage latency percentiles, load shedding and degraded rates, STT edge cases, audio duration distribution, LLM latency, TTS latency, circuit breaker states, rate limiter events, evaluation scores, and session IP change events.

---

## 21. Pipeline Wrapper (`pipeline.py`)

**File:** `app/orchestration/pipeline.py`

A thin object-oriented wrapper around `voice_graph` for callers that prefer an OO interface or need a simple integration point for testing:

```python
from app.orchestration.pipeline import VoicePipeline

pipeline = VoicePipeline()
result = await pipeline.run("audio/input.wav")

print(result["transcript"])
print(result["llm_response"])
print(result["audio_output"])
```

This is intentionally minimal. For full control over session_id, mode, language, cancellation, and streaming, use `voice_graph` directly.

---

## 22. Startup Display (`startup_display.py`)

**File:** `app/startup/startup_display.py`

Renders an animated boot banner in `LOG_MODE=standard` using Rich. Called once from `controller.py` before the PTT loop starts:

```python
from app.startup.startup_display import show_boot_sequence

show_boot_sequence([
    {"label": "STT",           "model": settings.stt_model,                    "status": "ok"},
    {"label": "LLM",           "model": settings.llm_model,                    "status": "ok"},
    {"label": "TTS",           "model": f"{settings.tts_model}/{settings.tts_voice}", "status": "ok"},
    {"label": "Session Store", "model": "redis + lru",                          "status": "ok"},
], ptt_key="H", exit_key="ESC")
```

Each module gets an animated progress bar that fills in 24 steps at ~18 ms each, then shows a status icon (`✓`, `⚠`, `—`, `✗`). After the bars complete, the key-binding summary is displayed.

---

## 23. Docker Compose & Infrastructure

**File:** `docker-compose.yaml`

Provisions the complete observability stack. The Python app runs on the **host** (not in Docker). Prometheus reaches it via `host.docker.internal:${PROMETHEUS_PORT}`.

### Three Isolated Networks

```
backend      — Redis, MongoDB, redis-exporter
               Data tier, completely isolated from observability tier
               A compromised observability container cannot reach session data

observability — Prometheus, Grafana, Tempo, OTel, Loki, Promtail
               Telemetry tier — all metrics/traces/logs pipeline

metrics      — node-exporter, cAdvisor, Prometheus
               Host-metrics tier, separate from backend
```

### Services

| Service | Image | Ports | Purpose |
|---|---|---|---|
| `redis` | `redis:7.2-alpine` | 6379 | Session store + LLM cache |
| `redis-exporter` | `oliver006/redis_exporter` | 9121 | Redis → Prometheus |
| `mongo` | `mongo:7` | 27017 | Observability event sink |
| `prometheus` | `prom/prometheus:v2.50` | 9090 | Metrics scrape + storage |
| `grafana` | `grafana/grafana:10.4.0` | 3000 | Dashboards |
| `tempo` | `grafana/tempo:2.4.0` | 3200, 4317 | OTel trace storage |
| `otel` | `otel/opentelemetry-collector-contrib` | 4318, 4319 | OTel collector relay |
| `loki` | `grafana/loki:2.9.4` | 3100 | Log aggregation |
| `promtail` | `grafana/promtail:2.9.4` | — | Docker logs → Loki |
| `node-exporter` | `prom/node-exporter:v1.7` | 9100 | Host OS metrics |
| `cadvisor` | `gcr.io/cadvisor/cadvisor:v0.47` | 8080 | Container metrics |

### Redis Configuration Highlights

```
--requirepass ${REDIS_PASSWORD}      # auth
--appendonly yes                     # AOF persistence (≤1s data loss)
--appendfsync everysec
--maxmemory 512mb                    # hard cap, prevents OOM
--maxmemory-policy allkeys-lru       # evict LRU on memory pressure
--io-threads 4                       # parallelise network I/O
--io-threads-do-reads yes
--activedefrag yes                   # background jemalloc defrag
--notify-keyspace-events Kxg         # session expiry notifications
```

Redis serves three workloads:
- `llm:v3:*` — LLM response cache (TTL = `LLM_CACHE_TTL`)
- `llm:stampede:*` — cache stampede locks (TTL = 30s)
- `session:lock:v1:*`, `session:data:v1:*`, `session:meta:v1:*` — session state

### Tempo Configuration Highlights

```yaml
metrics_generator:
  processor:
    service_graphs:   # span → Prometheus service graph metrics
    span_metrics:     # span → Prometheus histogram metrics
      dimensions:
        - ai.pipeline.stage
        - ai.pipeline.qos_tier
        - gen_ai.response.model
        - ai.stt.language
        - ai.tts.voice
```

Tempo generates Prometheus metrics from span data, enabling Grafana's service map panel and log-to-trace correlation via `tracesToMetrics`.

### Promtail Pipeline

Scrapes all Docker container logs via the Docker API and ships them to Loki:

1. **Docker envelope unwrap:** `{"log": "...", "stream": "...", "time": "..."}` → inner log line
2. **Timestamp:** Uses Docker's timestamp as the Loki log timestamp for accurate correlation
3. **JSON parse:** Extracts `trace_id`, `span_id`, `session_id`, `level`, `kind`
4. **Label promotion:** Only `level` and `kind` become Loki index labels (low cardinality). `trace_id` and `session_id` stay in the log body to avoid index explosion — query with `| json | trace_id="<value>"`
5. **Output:** Sets log line to the inner structlog JSON, discarding the Docker envelope

### Loki Datasource `derivedFields`

Two `derivedFields` are configured so clicking a `trace_id` in a Loki log line opens that trace in Tempo:

```yaml
derivedFields:
  - matcherRegex: '"trace_id":"([a-f0-9]{32})"'   # JSON format
    name: TraceID
    url: '${__value.raw}'
    datasourceUid: tempo
  - matcherRegex: 'trace_id=([a-f0-9]{32})'         # standard format
    name: TraceID
    url: '${__value.raw}'
    datasourceUid: tempo
```

### Required `.env` Keys for Docker Stack

```
GRAFANA_PASSWORD=
REDIS_PASSWORD=
MONGO_USERNAME=admin
MONGO_PASSWORD=
PROMETHEUS_PORT=9091
```

> **Windows note:** `promtail`, `node-exporter`, and `cadvisor` require Linux kernel interfaces and will fail on Windows Docker Desktop. Comment them out — everything else works fine.

---

## 24. Execution Modes

### `mode="api"` (HTTP, batch)

Used by `POST /voice`. Full STT → LLM → TTS pipeline. Returns `VoicePipelineResult` as JSON. Audio is written to `audio_OUTPUT/`. The response includes the local file path.

### `mode="stream"` (WebSocket, token streaming)

Used by `GET /voice/stream`. STT + LLM stream only (no TTS). Each LLM token is sent as a WebSocket JSON message. Final message includes `full_response`. Callers are responsible for their own TTS if needed.

### `mode="realtime"` (stream_full, concurrent pipeline)

STT, LLM, and TTS run concurrently using `BoundedPipelineQueue` inter-stage queues. As soon as the STT transcript is ready, LLM starts streaming. As each sentence-boundary token chunk accumulates, TTS synthesises it and yields audio bytes. This achieves the lowest time-to-first-audio at the cost of higher per-request resource use.

### Desktop vs API mode

| Aspect | Desktop (`APP_MODE=desktop`) | API (`APP_MODE=api`) |
|---|---|---|
| Input | Keyboard PTT + microphone | HTTP file upload or WebSocket |
| Output | Local speaker playback | JSON response + audio file path |
| Session | Auto-registered from local IP | Manual `POST /session/register` |
| SECRET_KEY | Auto-generated ephemeral | Must be set in `.env` |
| Audio sink | `player.play_audio()` | File on disk or S3 |

---

## 25. Data Flow: End-to-End Walkthrough

### API Mode: Single Request

```
1. Client: POST /voice
   Headers: X-Session-ID: 7aab4993...
   Body: multipart/form-data; file=Recording.m4a

2. main.py: _observability_middleware
   → assigns request_id = "ed175235..."
   → extracts traceparent, sets OTel context
   → starts http.request span

3. main.py: voice_chat()
   → require_session dependency: loads SessionData from Redis
   → _validate_upload: checks extension, MIME, size
   → _save_upload: writes to audio/audio_INPUT/api_{request_id}.m4a

4. voice_graph.run(state, timeout=120.0)
   → _prepare_state: assigns request_id, resolves QoSTier
   → LoadSheddingGuard.__aenter__: checks inflight ≤ 20
   → LatencyBudget.start(budget_ms=..., tier=STANDARD)

5. node_stt:
   → STTEmitter.start(...)
   → stt_span context manager
   → stt.transcribe(audio_path, language=None)
     → RateLimiter.acquire()
     → asyncio.Semaphore (max 20 concurrent)
     → CircuitBreaker.call(whisper_api_call)
     → openai.audio.transcriptions.create(...)
     → returns STTResult{text="Tell me about system design."}
   → truncate to max_transcript_chars
   → STTEmitter.ok(..., latency_ms=3301)
   → returns state update: user_input="Tell me about system design."

6. route_after_stt: no error → "llm"

7. node_llm:
   → LLMEmitter.start(...)
   → llm_span context manager
   → conversation_memory.resolve(session_id, transcript)
     → session_store.load(session_id, client_ip)
     → builds history_block from session.turns
     → builds context_prefix from InterviewState
   → builds prompt = context_prefix + history_block + transcript
   → llm.generate(prompt, request_id=rid)
     → check LatencyBudget (remaining time)
     → RateLimiter.acquire()
     → check Redis cache: MISS
     → acquire stampede lock (SETNX)
     → ChatOpenAI.ainvoke([SystemMessage, HumanMessage])
     → store response in Redis (TTL=3600)
     → release stampede lock
     → returns {response="System design involves...", cached=False, ...}
   → session_store.append_turn(session_id, transcript, response)
   → conversation_memory.commit(session_id, transcript, response)
   → transcript_writer.write_turn(...)   ← fire-and-forget
   → evaluation_engine.schedule_turn(...)  ← fire-and-forget
   → sanitize(raw_response, max_chars=4096)
   → LLMEmitter.ok(..., cache_hit=False, latency_ms=3278)
   → returns state update: llm_response="System design involves..."

8. route_after_llm: no error → "sanitize"

9. node_sanitize:
   → sanitize(): strip control chars, injection check, size cap
   → SanitizeEmitter.ok(...)
   → returns state update: cleaned_response="System design involves..."

10. node_tts:
    → TTSEmitter.start(...)
    → tts_span context manager
    → tts.synthesise(cleaned_response, voice="nova")
      → RateLimiter.acquire()
      → CircuitBreaker.call(openai_tts_call)
      → openai.audio.speech.create(model="tts-1", voice="nova", input=text)
      → write bytes to audio/audio_OUTPUT/tts_ebaa5613.mp3
      → returns TTSResult{local_path="audio/...", s3_uri="", ...}
    → TTSEmitter.ok(..., latency_ms=4706)
    → returns state update: audio_output="audio/.../tts_ebaa5613.mp3"

11. [IS_DEV only] node_playback_dev:
    → play_audio("audio/.../tts_ebaa5613.mp3")

12. _extract_result(final_state) → VoicePipelineResult
    pipeline_done logged: lat=11.346s

13. main.py: voice_chat() returns _ok_response(result, rid)
    input_path.unlink()   ← cleans up uploaded file

14. HTTP response: 200 OK, JSON body

Background (fire-and-forget):
  transcript_writer → writes session_{sid[:16]}.txt + emits ObsEvent
  evaluation_engine → scores turn, stores in Redis eval:score:v1:...
```

---

## 26. Resilience Patterns

### Circuit Breaker

Every external API call (Whisper, OpenAI chat, OpenAI TTS, Redis) is wrapped in a `CircuitBreaker`. The breaker transitions:

```
CLOSED (normal) → [N failures] → OPEN (fast-fail) → [timeout] → HALF_OPEN (probe) → CLOSED or OPEN
```

When OPEN, callers get `CircuitBreakerOpen` immediately — no waiting. The graph handles this as a transient failure and routes to the retry handler.

### Retry Loops

STT and LLM stages have configurable retry counts (`max_stt_retries`, `max_llm_retries`). Each retry is a full re-execution of the stage node. The retry router checks `abort_reason` first — if set (e.g., `budget_exceeded`, `path_traversal`), it bypasses retries entirely and routes to `error_terminal`.

### Latency Budget

The most impactful resilience mechanism for voice UX. A per-request SLA deadline is set at pipeline entry. Every node checks remaining time before starting work. When the budget is exhausted, the node raises `LatencyBudgetExceeded`, which is caught by the graph and treated as an abort — no retries, straight to `error_terminal`. This prevents the pipeline from wasting resources on requests that are already too late.

### Load Shedding

Each graph instance has a `LoadSheddingGuard`. When at capacity, new requests are rejected instantly with a 503-equivalent response. This protects the system from cascading overload — it's better to tell 10% of callers to retry than to let the entire system degrade.

### Graceful Degradation

Every stage can fail without crashing the pipeline. The `error_terminal` node fills in apology strings. The caller always gets a well-formed response, just with `degraded=true`.

### InMemoryLRU Fallback

When Redis is unreachable:
- Session store falls back to process-local LRU (cross-process uniqueness lost)
- LLM cache falls back to process-local LRU (cache hit rate drops)
- `degraded_mode` reduces concurrency to protect model APIs

### Backoff Retry with Jitter

`backoff_retry()` wraps upstream API calls with exponential backoff and random jitter:

```python
await backoff_retry(
    api_call,
    attempts=3,
    base_delay=1.5,
    exceptions=(openai.RateLimitError, httpx.TimeoutException),
)
```

Jitter prevents retry storms when multiple callers all hit an API limit simultaneously.

---

## 27. QoS Tiers & Graph Variants

| | `voice_graph` | `voice_graph_low_latency` | `voice_graph_realtime` |
|---|---|---|---|
| STT timeout | 30s | 15s | 10s |
| LLM timeout | 45s | 20s | 15s |
| TTS timeout | 30s | 12s | 8s |
| Max inflight | 20 | 50 | 30 |
| STT retries | 1 | 1 | 0 |
| LLM retries | 1 | 1 | 0 |
| Default tier | STANDARD | STANDARD | REALTIME |

**Why zero retries on realtime?** A retry after a blown SLA budget only pushes the caller further past the deadline. Realtime callers need either a fast answer or a fast apology — not a long wait followed by a slow answer.

**Why 50 max inflight on low_latency but only 30 on realtime?** Realtime requests are by definition high-priority and resource-intensive. Keeping the pool small ensures each gets adequate CPU/IO budget. Low-latency requests are less time-critical and can share more capacity.

---

## 28. Session Lifecycle

```
1. POST /session/register
   → extract client IP
   → ip_hash = HMAC-SHA256(SECRET_KEY, canonical_ip)
   → Redis SETNX session:lock:v1:{ip_hash}  → 409 if already locked
   → session_id = SHA256(ip_hash + urandom(16))
   → Redis SETEX session:data:v1:{session_id}  (TTL = TTL_S + GRACE_S)
   → Redis SETEX session:meta:v1:{ip_hash}
   → transcript_writer.open_session(session_id)
   → Response: {session_id, registered_from_ip, ttl_s}

2. [0..N] POST /voice (X-Session-ID: {session_id})
   → require_session: load session, validate IP
   → voice_graph.run(state)
     → conversation history injected from session.turns
     → after LLM: session_store.append_turn()
     → after LLM: evaluation_engine.schedule_turn()
   → Response: {transcript, response, audio, ...}

3. POST /session/end
   OR SESSION_TTL_S seconds of inactivity
   → session_store.end(session_id)
   → Redis DEL session:data:v1:{session_id}
   → Redis DEL session:lock:v1:{ip_hash}   ← IP is now free
   → transcript_writer.close_session(session_id)
   → SessionEmitter.ended(...)
```

---

## 29. Environment Variables Reference

### Required

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key (sk-...) |
| `SESSION_SECRET_KEY` | 32-byte hex secret for IP HMAC (required in non-desktop mode) |

### LLM

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | `gpt-4o-mini` | Primary model |
| `LLM_FALLBACK_MODEL` | `gpt-3.5-turbo` | Fallback when primary circuit opens |
| `LLM_TEMPERATURE` | `0.7` | Sampling temperature (0.0–2.0) |
| `LLM_MAX_CONCURRENT` | `50` | Max simultaneous LLM calls |
| `LLM_RATE_PER_SEC` | `20.0` | Sustained calls/sec |
| `LLM_RATE_BURST` | `40.0` | Burst headroom |
| `LLM_CACHE_TTL` | `3600` | Redis cache TTL in seconds |

### Redis

| Variable | Default | Description |
|---|---|---|
| `REDIS_ENABLED` | `true` | Set false to disable Redis |
| `REDIS_URL` | `redis://localhost:6379` | Connection URL |
| `REDIS_MAX_CONN` | `200` | Connection pool size |
| `REDIS_PASSWORD` | — | Used in Docker stack |

### STT

| Variable | Default | Description |
|---|---|---|
| `STT_MODEL` | `whisper-1` | Whisper model |
| `STT_MAX_FILE_MB` | `25.0` | Max audio file size (Whisper hard ceiling) |
| `STT_MAX_CONCURRENT` | `20` | Max simultaneous Whisper calls |
| `STT_RATE_PER_SEC` | `10.0` | |
| `STT_RATE_BURST` | `20.0` | |
| `STT_S3_BUCKET` | `""` | S3 bucket for audio/transcripts |
| `STT_SERVICE_URL` | `""` | Remote STT service URL (enables RemoteSTTClient) |

### TTS

| Variable | Default | Description |
|---|---|---|
| `TTS_MODEL` | `tts-1` | `tts-1` or `tts-1-hd` |
| `TTS_VOICE` | `nova` | `alloy` / `echo` / `fable` / `onyx` / `nova` / `shimmer` |
| `TTS_FORMAT` | `mp3` | Output audio format |
| `TTS_OUTPUT_DIR` | `audio/audio_OUTPUT` | Local output directory |
| `TTS_LOCAL_FILE_TTL` | `3600.0` | Seconds before local files are deleted |
| `TTS_S3_BUCKET` | `""` | S3 bucket for audio upload |
| `TTS_SERVICE_URL` | `""` | Remote TTS service URL |

### Graph / Orchestration

| Variable | Default | Description |
|---|---|---|
| `VOICE_GRAPH_VERSION` | `v2` | Version label in metrics/responses |
| `GRAPH_STT_TIMEOUT` | `30.0` | STT stage timeout (seconds) |
| `GRAPH_LLM_TIMEOUT` | `45.0` | LLM stage timeout |
| `GRAPH_TTS_TIMEOUT` | `30.0` | TTS stage timeout |
| `GRAPH_MAX_INFLIGHT` | `20` | Load shedder capacity |
| `GRAPH_MAX_TRANSCRIPT_CHARS` | `2000` | Transcript cap before LLM |
| `GRAPH_MAX_LLM_RESPONSE_CHARS` | `4096` | Response cap before sanitize |
| `GRAPH_MAX_TTS_CHARS` | `4000` | TTS input cap (OpenAI ceiling: 4096) |
| `GRAPH_MIN_PROMPT_CHARS` | `25` | Min chars before LLM fires |
| `GRAPH_STT_LLM_QUEUE_DEPTH` | `8` | stream_full() inter-stage queue |
| `GRAPH_LLM_TTS_QUEUE_DEPTH` | `16` | stream_full() inter-stage queue |

### Session

| Variable | Default | Description |
|---|---|---|
| `SESSION_TTL_S` | `1800` | Session idle TTL (30 min) |
| `SESSION_GRACE_S` | `300` | Grace period on top of TTL |
| `SESSION_MAX_TURNS` | `20` | Rolling history window |
| `SESSION_MAX_IP_CHANGES` | `3` | Max IP changes before suspension |
| `SESSION_LRU_SIZE` | `1024` | LRU fallback capacity |
| `APP_MODE` | `desktop` | `desktop` or `api` |

### Gateway

| Variable | Default | Description |
|---|---|---|
| `AUDIO_INPUT_DIR` | `audio/audio_INPUT` | Upload staging directory |
| `MAX_UPLOAD_MB` | `25.0` | Max upload size |
| `PIPELINE_TIMEOUT` | `120.0` | Outer wall-clock guard (must exceed stage sum) |
| `STREAM_TIMEOUT` | `90.0` | WebSocket streaming timeout |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |
| `ENV` | `production` | `production` / `development` / `test` |

### Controller (Desktop)

| Variable | Default | Description |
|---|---|---|
| `CONTROLLER_PTT_KEY` | `h` | Push-to-talk key |
| `CONTROLLER_EXIT_KEY` | `esc` | Clean exit key |
| `CONTROLLER_POLL_INTERVAL_S` | `0.03` | Key poll rate (~33 Hz) |
| `CONTROLLER_DEBOUNCE_S` | `0.35` | Anti-bounce delay |
| `CONTROLLER_INTERRUPT_DRAIN_S` | `0.05` | Drain after cancel signal |
| `CONTROLLER_LOOP_DRAIN_S` | `0.10` | Drain before loop.stop() |
| `CONTROLLER_LOOP_JOIN_TIMEOUT` | `3.0` | Thread join timeout |

### OpenTelemetry

| Variable | Default | Description |
|---|---|---|
| `OTEL_ENABLED` | `false` | Master OTel switch |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4319` | gRPC collector endpoint |
| `OTEL_SERVICE_NAME` | `voice-pipeline` | Service name in spans |
| `OTEL_TRACE_SAMPLE_RATE` | `1.0` | Fraction of traces sampled |
| `OTEL_METRIC_INTERVAL_MS` | `30000` | OTel metric push interval |

### Observability

| Variable | Default | Description |
|---|---|---|
| `LOG_MODE` | `verbose` | `standard` (Rich) or `verbose` (JSON) |
| `LOG_FILE` | `logs/voice_assistant.log` | JSON log output path |
| `MONGO_URI` | `mongodb://localhost:27017` | MongoDB connection |
| `MONGO_DB` | `ai_observability` | Database name |
| `MONGO_COLLECTION` | `pipeline_events` | Collection name |
| `PROMETHEUS_PORT` | `9091` | App /metrics port (must not be 9090) |
| `GRAFANA_OUT_DIR` | `infra/grafana` | Grafana provisioning path |

---

## 30. Running the System

### Prerequisites

- Python 3.11+
- Docker Desktop (for the observability stack)
- `pip install -r requirements.txt`

### 1. Configure Environment

```bash
cp .env.example .env
# Edit .env — at minimum:
#   OPENAI_API_KEY=sk-...
#   SESSION_SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
#   REDIS_PASSWORD=$(python -c "import secrets; print(secrets.token_hex(24))")
#   MONGO_PASSWORD=$(python -c "import secrets; print(secrets.token_hex(24))")
#   GRAFANA_PASSWORD=your_password_here
```

### 2. Start Observability Stack

```bash
docker compose up -d

# Windows: comment out promtail, node-exporter, cadvisor in docker-compose.yaml first
```

### 3. Verify Configuration

```bash
python -m app.common.settings   # prints redacted config summary
```

### 4a. API Mode

```bash
python -m app.endpoint.main --no-reload
# Server at http://0.0.0.0:8000
# Docs at http://localhost:8000/docs
```

Workflow:
```bash
# Register session
curl -X POST http://localhost:8000/session/register
# → {"session_id": "7aab4993...", ...}

# Send audio
curl -X POST http://localhost:8000/voice \
  -H "X-Session-ID: 7aab4993..." \
  -F "file=@recording.m4a"
# → {transcript, response, audio, stage_latencies, ...}

# End session
curl -X POST http://localhost:8000/session/end \
  -H "X-Session-ID: 7aab4993..."
```

### 4b. Desktop Mode

```bash
APP_MODE=desktop LOG_MODE=standard python -m app.orchestration.controller
# Hold H to talk, ESC to exit
```

### 5. Dashboards

| URL | Credentials |
|---|---|
| http://localhost:3000 | admin / `GRAFANA_PASSWORD` |
| http://localhost:9090 | — |
| http://localhost:3200 | — |
| http://localhost:8000/docs | — |

### Development Mode

```bash
ENV=development LOG_MODE=standard python -m app.endpoint.main --no-reload
```

In development mode:
- `node_playback_dev` is compiled into the graph (local audio playback)
- `IS_DEV=true` throughout the codebase
- Warnings for production-unsafe settings are suppressed

**MODES:**

**Interactive CLI:**
```
 .\run.ps1 → interactive CLI
```
or

**Manual — uvicorn local:**
```
 docker compose -f docker-compose.yaml up -d --scale app=0
 dotenv -f .env -f .env.local run -- uvicorn app.endpoint.main:app --reload
```
or

**Manual — uvicorn Docker:**
```
 docker compose up -d
```
or

**Manual — gunicorn Docker:**
```
docker compose -f docker-compose.yaml up -d
```

> **Note:** `--reload` is safe to use. OTel and Prometheus bootstrap already guard against double-init — the warnings you may see on reloader startup are harmless and suppressed in development mode.

---

## 31. Dependency Map

```
settings.py
  ← shared.py (reads settings via getattr)
  ← STT_service.py
  ← LLM_service.py
  ← TTS_service.py
  ← voice_graph.py
  ← main.py
  ← controller.py

shared.py
  ← STT_service.py
  ← LLM_service.py
  ← TTS_service.py
  ← voice_graph.py
  ← session_store.py
  ← conversation_memory.py
  ← evaluation_engine.py
  ← transcription.py
  ← main.py
  ← controller.py

log_config.py
  ← shared.py (configure_logging called at import)

observability.py
  ← voice_graph.py (emitters, span ctx managers)
  ← STT_service.py
  ← LLM_service.py
  ← TTS_service.py
  ← session_store.py
  ← transcription.py
  ← evaluation_engine.py
  ← controller.py
  ← main.py (bootstrap)

session_store.py
  ← conversation_memory.py (imports session_store)
  ← voice_graph.py (lazy import inside node_llm)
  ← main.py (session_router)
  ← controller.py

conversation_memory.py
  ← voice_graph.py (lazy import inside node_llm)

evaluation_engine.py
  ← voice_graph.py (lazy import inside node_llm)
  ← controller.py (shutdown)

transcription.py
  ← voice_graph.py (lazy import inside node_llm)
  ← controller.py (open/close session, flush)

STT_service.py → voice_graph.py (STTNodeProtocol)
LLM_service.py → voice_graph.py (LLMNodeProtocol)
TTS_service.py → voice_graph.py (TTSNodeProtocol)
sanitize.py    → voice_graph.py (node_sanitize)

voice_graph.py → main.py (voice_graph, voice_graph_realtime, voice_graph_low_latency)
voice_graph.py → pipeline.py
voice_graph.py → controller.py

player.py  → voice_graph.py (play_audio, stop_all)
           → controller.py
recorder.py → controller.py

startup_display.py → controller.py

docker-compose.yaml → external: Redis, MongoDB, Prometheus, Grafana, Tempo, Loki
```

**Lazy imports in `node_llm`:** `session_store`, `evaluation_engine`, `conversation_memory`, and `transcript_writer` are imported inside the node function body, not at module level. This avoids circular import issues (those modules import `voice_graph` indirectly via `settings` or `shared`) and ensures they are only loaded when actually needed.

---

*This document covers every file, every class, every design decision, and every integration point in the Voice Assistant Pipeline codebase. For questions about a specific subsystem, refer to the section index above.*