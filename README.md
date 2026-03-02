# 🎙️ Voice Interview Pipeline

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-orange?style=for-the-badge)](https://github.com/langchain-ai/langgraph)
[![OpenAI](https://img.shields.io/badge/OpenAI-Whisper%20%7C%20GPT%20%7C%20TTS-412991?style=for-the-badge&logo=openai)](https://openai.com)
[![Redis](https://img.shields.io/badge/Redis-Session%20%7C%20Cache-DC382D?style=for-the-badge&logo=redis)](https://redis.io)
[![MongoDB](https://img.shields.io/badge/MongoDB-Observability-47A248?style=for-the-badge&logo=mongodb)](https://mongodb.com)
[![Prometheus](https://img.shields.io/badge/Prometheus-Metrics-E6522C?style=for-the-badge&logo=prometheus)](https://prometheus.io)
[![OpenTelemetry](https://img.shields.io/badge/OpenTelemetry-Tracing-000000?style=for-the-badge&logo=opentelemetry)](https://opentelemetry.io)

**A production-grade, real-time voice interview system with PCM-native audio processing, structured QA state management, adaptive evaluation, and three-layer observability.**

</div>

---

## Table of Contents

1. [Overview](#1-overview)
2. [System Architecture](#2-system-architecture)
3. [Repository Layout](#3-repository-layout)
4. [Installation & Environment Setup](#4-installation--environment-setup)
5. [Configuration Reference](#5-configuration-reference)
6. [Execution Modes](#6-execution-modes)
7. [Pipeline Deep Dive](#7-pipeline-deep-dive)
   - [Graph Topology](#71-graph-topology)
   - [STT Node](#72-stt-node-stt_servicepy)
   - [LLM Node](#73-llm-node-llm_servicepy)
   - [Sanitize Node](#74-sanitize-node-sanitizepy)
   - [TTS Node](#75-tts-node-tts_servicepy)
8. [PCM Audio Engine](#8-pcm-audio-engine)
   - [Format Negotiation](#81-format-negotiation)
   - [PCM-Native Realtime Path](#82-pcm-native-realtime-path)
   - [Audio Components](#83-audio-components)
9. [QA Interview Engine](#9-qa-interview-engine)
   - [Session State Machine](#91-session-state-machine)
   - [QA Data Schema](#92-qa-data-schema)
   - [LLM Modes](#93-llm-modes)
10. [Evaluation Engine](#10-evaluation-engine)
11. [Session Management](#11-session-management)
    - [Session Store](#111-session-store)
    - [Session Lifecycle Manager](#112-session-lifecycle-manager)
12. [Transcript Writer](#12-transcript-writer)
13. [QA Audit Bus](#13-qa-audit-bus)
14. [Observability Stack](#14-observability-stack)
    - [Three-Layer Architecture](#141-three-layer-architecture)
    - [Metrics Reference](#142-metrics-reference)
    - [Grafana Dashboards](#143-grafana-dashboards)
15. [Resilience Patterns](#15-resilience-patterns)
    - [Circuit Breakers](#151-circuit-breakers)
    - [Rate Limiting & Bulkheads](#152-rate-limiting--bulkheads)
    - [Load Shedding](#153-load-shedding)
    - [Latency Budget](#154-latency-budget)
16. [Distributed Service Mode](#16-distributed-service-mode)
17. [Feature Flags](#17-feature-flags)
18. [API Reference](#18-api-reference)
19. [Desktop Controller (PTT)](#19-desktop-controller-ptt)
20. [Kafka Integration](#20-kafka-integration)
21. [Deployment](#21-deployment)
22. [Performance Tuning](#22-performance-tuning)
23. [Troubleshooting](#23-troubleshooting)
24. [Development Guide](#24-development-guide)

---

## 1. Overview

The Voice Interview Pipeline is a **production-grade AI interviewer** that conducts structured technical interviews entirely over voice. A candidate speaks; the system listens, understands, evaluates, and responds — all in near-real-time.

### What It Does

```
Candidate speaks → Mic capture → VAD gate → Speech enhancement
   → Whisper STT → QA state machine → GPT question engine
   → TTS synthesis → Speaker output

Meanwhile (off critical path):
   → Evaluation scoring per answer
   → Dual-sink transcript (txt + observability)
   → MongoDB audit log
```

### Key Capabilities

| Capability | Details |
|---|---|
| **Voice-first** | Push-to-talk (desktop) or HTTP/WebSocket (cloud) |
| **Structured interviews** | Domain rotation, level-adaptive questions, ATS intro extraction |
| **PCM-native audio** | Zero file I/O between stages, 300–600 ms latency reduction |
| **Barge-in detection** | Candidate can interrupt AI mid-sentence |
| **Real-time evaluation** | Off-path scoring with adaptive sampling, budget caps, circuit breaker |
| **Dual-sink transcripts** | Human .txt + structured MongoDB/observability |
| **Three graph instances** | Standard, Realtime (tight SLA), Low-latency (balanced) |
| **IP-locked sessions** | Redis-backed, HMAC-derived session fingerprints |
| **50k LOC codebase** | 19 core modules, 7,555 lines in voice_graph alone |

---

## 2. System Architecture

### High-Level System Diagram

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                         VOICE INTERVIEW PIPELINE                              │
│                                                                               │
│  ┌──────────────┐         ┌───────────────────────────────────────────────┐   │
│  │   CLIENT     │         │              FastAPI Gateway (main.py)        │   │
│  │              │─HTTP───▶│  POST /voice        POST /session/register    │   │
│  │  Web / App   │─WS─────▶│  WS   /voice/stream POST /session/end         │   │
│  │  Desktop PTT │         │  GET  /health       GET  /metrics             │   │
│  └──────────────┘         │  DEL  /cancel/{id}  POST /session/refresh     │   │
│                           └────────────────┬──────────────────────────────┘   │
│                                            │                                  │
│                           ┌────────────────▼──────────────────────────────┐   │
│                           │          voice_graph.py (Orchestration)       │   │
│                           │                                               │   │
│                           │  ┌─────────┐   ┌─────────┐   ┌─────────┐      │   │
│                           │  │voice_   │   │voice_   │   │voice_   │      │   │
│                           │  │graph    │   │graph_   │   │graph_   │      │   │
│                           │  │(balanced│   │realtime │   │low_lat  │      │   │
│                           │  │1 retry) │   │(0 retry │   │ency     │      │   │
│                           │  │         │   │tight SLA│   │         │      │   │
│                           │  └────┬────┘   └────┬────┘   └────┬────┘      │   │
│                           │       └─────────────┴─────────────┘           │   │
│                           │                    │                          │   │
│  ┌─────────────────────────────────────────────┼───────────────────────┐  │   │
│  │                PIPELINE STAGES              │                       │  │   │
│  │                                             ▼                       │  │   │
│  │   ┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌────────┐    │  │   │
│  │   │  node_   │───▶│  node_   │───▶│  node_       │───▶│node_   │    │  │   │
│  │   │  stt     │    │  llm     │    │  sanitize    │    │tts     │    │  │   │
│  │   │          │    │          │    │              │    │        │    │  │   │
│  │   │Whisper   │    │GPT-5/4o  │    │27-step clean │    │tts-1-  │    │  │   │
│  │   │Faster-W  │    │QA engine │    │TTS-safe text │    │hd      │    │  │   │
│  │   └──────────┘    └──────────┘    └──────────────┘    └────────┘    │  │   │
│  │         │               │                                  │        │  │   │
│  │    ┌────▼────┐    ┌─────▼─────┐                      ┌─────▼───┐    │  │   │
│  │    │stt_error│    │llm_error  │                      │tts_error│    │  │   │
│  │    │(retry / │    │(retry /   │                      │(apology │    │  │   │
│  │    │ abort)  │    │  abort)   │                      │  audio) │    │  │   │
│  │    └─────────┘    └───────────┘                      └─────────┘    │  │   │
│  └─────────────────────────────────────────────────────────────────────┘  │   │
│                                                                               │
│  ┌──────────────┐  ┌─────────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  QA Engine   │  │  Eval Engine    │  │  Transcript  │  │  Session     │    │
│  │              │  │                 │  │  Writer      │  │  Store       │    │
│  │ qa_controller│  │ evaluation_     │  │              │  │              │    │
│  │ .py          │  │ engine.py       │  │transcription │  │session_store │    │
│  │              │  │                 │  │.py           │  │.py           │    │
│  │ 4 stages:    │  │ Off-path scoring│  │ .txt + obs   │  │ Redis/LRU    │    │
│  │ greeting     │  │ adaptive sample │  │ dual sink    │  │ IP-locked    │    │
│  │ intro (ATS)  │  │ budget cap      │  │ async queue  │  │ HMAC hash    │    │
│  │ interview    │  │ circuit breaker │  │              │  │              │    │
│  │ complete     │  │                 │  │              │  │              │    │
│  └──────────────┘  └─────────────────┘  └──────────────┘  └──────────────┘    │
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────────┐ │
│  │                        OBSERVABILITY STACK                               │ │
│  │                                                                          │ │
│  │   structlog (console+JSON)   Prometheus (:9090)   MongoDB (90-day TTL)   │ │
│  │   OTel spans (OTLP/gRPC)    Grafana dashboards    Rich TUI (live)        │ │
│  └──────────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────────────┘
```

### Module Dependency Graph

```
                           main.py
                          /       \
                    controller.py  FastAPI routes
                         |              |
                   voice_graph.py ◄─────┘
                  /     |      \
           STT_   LLM_  TTS_   audio_engine.py
           service service service     |
              |      |      |    ┌─────┴──────┐
              └──────┴──────┘    recorder.py  player.py
                     |           opus_ffmpeg_io.py
               sanitize.py
                     |
            ┌────────┼────────┐
      qa_controller  │   evaluation_engine.py
            │   session_store │
     conversation_  │         │
       memory.py  transcription.py
                     │
               observability.py
                     │
               shared.py ◄── settings.py
```

---

## 3. Repository Layout

```
.
├── app/
│   ├── audio_essentials/
│   │   ├── audio_engine.py          # PCM format, VAD, ring buffer, streams (7,141 LOC)
│   │   ├── opus_ffmpeg_io.py        # FFmpeg Opus encode/decode (2,619 LOC)
│   │   ├── recorder.py              # Microphone capture, PTT recording (998 LOC)
│   │   └── player.py                # Audio playback, PCM-native path (900 LOC)
│   │
│   ├── common/
│   │   ├── settings.py              # Central config, startup validation (1,000 LOC)
│   │   └── shared.py                # Circuit breakers, bulkheads, OTel, LRU (1,047 LOC)
│   │
│   ├── eval/
│   │   └── evaluation_engine.py     # Off-path answer scoring (1,206 LOC)
│   │
│   ├── interview/
│   │   └── qa_controller.py         # QA state machine, domain rotation (3,661 LOC)
│   │
│   ├── monitoring/
│   │   └── observability.py         # 3-layer observability stack (8,178 LOC)
│   │
│   ├── nodes/
│   │   ├── STT_service.py           # Whisper node + PCM integration (3,046 LOC)
│   │   ├── LLM_service.py           # GPT node, ATS + Interviewer modes (3,336 LOC)
│   │   ├── TTS_service.py           # TTS node + PCM synthesis (3,056 LOC)
│   │   └── sanitize.py              # 27-step TTS sanitizer (951 LOC)
│   │
│   ├── orchestration/
│   │   └── voice_graph.py           # LangGraph pipeline orchestrator (7,555 LOC)
│   │
│   └── user_tracking/
│       ├── session_service/
│       │   └── session_store.py     # IP-locked Redis sessions (1,209 LOC)
│       └── transcript/
│           ├── transcription.py     # Dual-sink transcript writer (350 LOC)
│           └── conversation_memory.py # QA audit bus (2,011 LOC)
│
├── controller.py                    # Desktop PTT keyboard controller (615 LOC)
├── main.py                          # FastAPI gateway (1,029 LOC)
├── transcripts/                     # Per-session .txt transcripts (auto-created)
├── audio/
│   ├── temp_IN/                     # Uploaded audio uploads
│   └── temp_OUT/                    # TTS output files
├── grafana/                         # Auto-generated dashboard JSON
├── .env.example
└── README.md
```

**Total codebase: ~49,908 lines across 19 core modules.**

---

## 4. Installation & Environment Setup

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | 3.12 recommended |
| Redis | 7.0+ | Required for sessions; LRU fallback available |
| MongoDB | 6.0+ | Optional; disables structured event storage if absent |
| FFmpeg | 6.0+ | Required for Opus encode/decode and PCM transcoding |
| PortAudio | 19+ | Required for mic/speaker (desktop mode only) |
| CUDA | Optional | Faster-Whisper GPU acceleration |

### Installation

```bash
# Clone and enter repo
git clone https://github.com/your-org/voice-interview-pipeline.git
cd voice-interview-pipeline

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install system dependencies (Ubuntu/Debian)
sudo apt-get install -y ffmpeg portaudio19-dev libsndfile1

# macOS
brew install ffmpeg portaudio libsndfile

# Verify FFmpeg is available
ffmpeg -version
```

### Environment Setup

```bash
cp .env.example .env
# Edit .env with your credentials
```

**Minimum required `.env`:**

```bash
# ── Core credentials (REQUIRED) ──────────────────────────────────────────────
OPENAI_API_KEY=sk-...

# ── Session security (REQUIRED) ───────────────────────────────────────────────
SESSION_SECRET_KEY=your-256-bit-hex-key-here   # min 32 chars

# ── Redis (strongly recommended) ─────────────────────────────────────────────
REDIS_URL=redis://localhost:6379/0
```

### Start the Server

```bash
# Development
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production
gunicorn main:app -k uvicorn.workers.UvicornWorker \
  --workers 4 --bind 0.0.0.0:8000 --timeout 120

# Desktop PTT mode (no server needed)
python controller.py
```

---

## 5. Configuration Reference

All configuration is managed in `settings.py` via `pydantic_settings.BaseSettings`. Every variable is validated at import time — misconfigured containers **fail at startup**, not mid-traffic.

### OpenAI

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *required* | Shared key used by LLM, STT, and TTS nodes |
| `OPENAI_BASE_URL` | (OpenAI default) | Override for local Whisper endpoint |

### LLM Node

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | `gpt-5.2` | Chat model name (warns on unknown, hard-errors invalid) |
| `LLM_MAX_CONCURRENT` | `10` | Parallel in-flight LLM calls |
| `LLM_RATE_LIMIT_RPM` | `500` | Token-bucket requests per minute |
| `LLM_CACHE_TTL_S` | `300` | Redis cache TTL for LLM responses |
| `LLM_SERVICE_URL` | *(empty)* | Remote LLM service URL; local node if unset |

### Redis

| Variable | Default | Description |
|---|---|---|
| `REDIS_URL` | `redis://localhost:6379/0` | Connection string |
| `REDIS_CACHE_DB` | `1` | LLM response cache database index |
| `REDIS_POOL_SIZE` | `20` | Max connections in the pool |

### STT Node

| Variable | Default | Description |
|---|---|---|
| `STT_MODEL` | `whisper-1` | Whisper model identifier |
| `STT_MAX_CONCURRENT` | `5` | Parallel transcription jobs |
| `STT_S3_BUCKET` | *(empty)* | S3 bucket for audio storage; local disk if unset |
| `STT_SERVICE_URL` | *(empty)* | Remote STT URL; local node if unset |

### TTS Node

| Variable | Default | Description |
|---|---|---|
| `TTS_MODEL` | `tts-1-hd` | TTS model (`tts-1` or `tts-1-hd`) |
| `TTS_VOICE` | `nova` | Voice: `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer` |
| `TTS_FORMAT` | `mp3` | Output format; `pcm` enables zero-copy pipeline |
| `TTS_S3_BUCKET` | *(empty)* | S3 bucket for TTS audio storage |
| `TTS_SERVICE_URL` | *(empty)* | Remote TTS URL; local node if unset |

### AWS

| Variable | Default | Description |
|---|---|---|
| `AWS_ACCESS_KEY_ID` | *(empty)* | AWS credential |
| `AWS_SECRET_ACCESS_KEY` | *(empty)* | AWS credential |
| `AWS_REGION` | `us-east-1` | S3 region |

### Graph / Orchestration

| Variable | Default | Description |
|---|---|---|
| `GRAPH_STT_TIMEOUT` | `30.0` | Per-request STT timeout (seconds) |
| `GRAPH_LLM_TIMEOUT` | `45.0` | Per-request LLM timeout (seconds) |
| `GRAPH_TTS_TIMEOUT` | `30.0` | Per-request TTS timeout (seconds) |
| `GRAPH_MAX_INFLIGHT` | `20` | Max concurrent pipeline requests |
| `GRAPH_MAX_STT_RETRIES` | `1` | STT retry count before abort |
| `GRAPH_MAX_LLM_RETRIES` | `1` | LLM retry count before abort |
| `PIPELINE_TIMEOUT` | `120.0` | End-to-end pipeline timeout |
| `STREAM_TIMEOUT` | `90.0` | WebSocket stream timeout |

### FastAPI Gateway

| Variable | Default | Description |
|---|---|---|
| `MAX_UPLOAD_MB` | `25.0` | Max audio upload size in megabytes |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `AUDIO_INPUT_DIR` | `audio/temp_IN` | Upload staging directory |

### Session

| Variable | Default | Description |
|---|---|---|
| `SESSION_SECRET_KEY` | *required* | HMAC key for IP fingerprinting |
| `SESSION_TTL_S` | `3600` | Session TTL in seconds |
| `SESSION_GRACE_S` | `300` | Extra TTL after session end |
| `SESSION_MAX_TURNS` | `50` | Max conversation turns per session |

### Controller (Desktop PTT)

| Variable | Default | Description |
|---|---|---|
| `CONTROLLER_PTT_KEY` | `h` | Hold-to-talk key |
| `CONTROLLER_EXIT_KEY` | `esc` | Clean exit key |
| `CONTROLLER_POLL_INTERVAL_S` | `0.03` | Key poll interval |
| `CONTROLLER_DEBOUNCE_S` | `0.35` | Key debounce threshold |

### Observability

| Variable | Default | Description |
|---|---|---|
| `LOG_MODE` | `verbose` | `standard` or `verbose` |
| `LOG_FILE` | *(empty)* | JSON log file path |
| `MONGO_URI` | *(empty)* | MongoDB connection string |
| `MONGO_DB` | `ai_observability` | Database name |
| `MONGO_ENABLED` | `true` | Toggle MongoDB sink |
| `MONGO_TTL_DAYS` | `90` | Event document TTL |
| `PROMETHEUS_PORT` | `9090` | Prometheus metrics HTTP port |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | *(empty)* | OTel collector gRPC endpoint |
| `OTEL_ENABLED` | `false` | Toggle OTel tracing |

### QA Interview

| Variable | Default | Description |
|---|---|---|
| `QA_QUESTIONS_PER_DOMAIN` | `7` | Target questions per domain |
| `QA_MIN_QUESTIONS` | `5` | Minimum before domain rotation |
| `QA_MAX_QUESTIONS` | `10` | Maximum before forced rotation |

### Evaluation Engine

| Variable | Default | Description |
|---|---|---|
| `EVAL_MODEL` | `gpt-5.2` | Scoring model |
| `EVAL_MAX_CONCURRENT` | `3` | Parallel scoring calls |
| `EVAL_SESSION_BUDGET_TOKENS` | `50000` | Hard token cap per session |
| `EVAL_FULL_EVAL_TURNS` | `5` | Always score first N turns |
| `EVAL_SAMPLE_RATE` | `3` | Then score 1-in-N turns |
| `EVAL_MIN_ANSWER_CHARS` | `40` | Skip scoring below this length |
| `EVAL_ENABLED` | `true` | Toggle evaluation globally |

---

## 6. Execution Modes

The pipeline has five distinct execution modes, all implemented in `VoiceGraph`:

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                         EXECUTION MODE SELECTION                                │
│                                                                                  │
│  HTTP POST /voice        ──────────────────────▶  api mode                      │
│    single-shot request                            await voice_graph.run(state)   │
│    returns JSON when done                         → VoicePipelineResult          │
│                                                                                  │
│  WebSocket /voice/stream ──────────────────────▶  stream mode                   │
│    bidirectional WS                               async for token in             │
│    streams LLM tokens in real-time                  voice_graph.stream(state)    │
│                                                                                  │
│  WebSocket /voice/stream ──────────────────────▶  realtime mode                 │
│    (3 concurrent workers)                         async for chunk in             │
│    streams audio chunks as produced                 voice_graph.stream_full()    │
│                                                                                  │
│  PCM native path         ──────────────────────▶  pcm mode                      │
│    FF_PCM_PIPELINE=true                           async for chunk in             │
│    zero-copy, no disk I/O                           voice_graph.stream_full_pcm()│
│                                                                                  │
│  Desktop controller      ──────────────────────▶  ptt mode                      │
│    hold PTT_KEY to talk                           await voice_graph.run_ptt(     │
│    releases → dispatches                            state, is_held_fn)           │
└────────────────────────────────────────────────────────────────────────────────┘
```

### Mode Comparison

| Mode | Use Case | Latency | Concurrency | Disk I/O |
|---|---|---|---|---|
| `api` | REST clients, batch | Highest | 20 inflight | WAV on disk |
| `stream` | Web clients, partial results | Medium | 20 inflight | WAV on disk |
| `realtime` | Low-latency cloud | Low | Configurable | Minimal |
| `pcm` | Desktop, embedded | Lowest | 1 per instance | Zero |
| `ptt` | Desktop PTT interviews | Low | 1 session | Temp WAV |

### Three Graph Singletons

```python
# voice_graph.py — three instances with genuinely different configs

voice_graph = VoiceGraph(
    cfg=VoiceGraphConfig(
        max_stt_retries=1,
        max_llm_retries=1,
        stt_timeout=30.0,
        llm_timeout=45.0,
        tts_timeout=30.0,
        max_inflight=20,
    )
)

voice_graph_realtime = VoiceGraph(
    cfg=VoiceGraphConfig(
        max_stt_retries=0,      # zero retries — retry under blown SLA makes things worse
        max_llm_retries=0,
        stt_timeout=12.0,       # tight timeouts
        llm_timeout=18.0,
        tts_timeout=12.0,
        max_inflight=10,        # reserved realtime capacity
        qos_tier=QoSTier.REALTIME,
    )
)

voice_graph_low_latency = VoiceGraph(
    cfg=VoiceGraphConfig(
        max_stt_retries=1,
        max_llm_retries=0,
        stt_timeout=20.0,
        llm_timeout=25.0,
        tts_timeout=20.0,
        max_inflight=15,
    )
)
```

---

## 7. Pipeline Deep Dive

### 7.1 Graph Topology

The pipeline is a LangGraph `StateGraph` with conditional edges implementing retry loops and error terminals.

```
                        ┌─────────────────────────────────────────────────────┐
                        │              VOICE PIPELINE GRAPH                   │
                        │                                                     │
                        │   START                                             │
                        │     │                                               │
                        │     ▼                                               │
                        │  ┌──────────────────────────────────────────────┐   │
                        │  │                  node_stt                    │   │
                        │  │                                              │   │ 
                        │  │  1. Validate audio path                      │   │
                        │  │  2. Check latency budget                     │   │
                        │  │  3. Acquire STT bulkhead semaphore           │   │
                        │  │  4. Call Whisper API (with timeout)          │   │
                        │  │  5. Log OTel span, Prometheus, structlog     │   │
                        │  └──────────────┬───────────────────────────────┘   │
                        │                 │                                   │
                        │         route_after_stt()                           │
                        │            ┌───┴──────────────┐                     │
                        │         [ok]                [error]                 │
                        │            │                   │                    │
                        │            ▼                   ▼                    │
                        │     ┌──────────┐     ┌────────────────────┐         │
                        │     │ node_llm │     │  node_stt_error    │         │ 
                        │     └──────────┘     │                    │         │
                        │          │           │  retries += 1      │         │
                        │          │           └────────┬───────────┘         │
                        │          │                    │                     │
                        │          │        route_after_stt_error()           │
                        │          │            ┌───────┴──────────────┐      │
                        │          │     [retry remaining]    [exhausted/abort│
                        │          │            │                       │     │
                        │          │            └────────▶ [back to stt]│     │
                        │          │                                    │     │
                        │          │                                    ▼     │
                        │          │                        ┌────────────────┐│
                        │          │                        │node_error_     ││
                        │          │                        │terminal        ││
                        │          │                        │                ││
                        │          │                        │ Build error    ││
                        │          │                        │ result, emit   ││
                        │          │                        │ final metrics  ││
                        │          │                        └────────┬───────┘│
                        │          │                                 │        │
                        │          ▼                                END       │
                        │  ┌───────────────────────────────────────────────┐  │
                        │  │                  node_llm                     │  │
                        │  │                                               │  │
                        │  │  QA stage routing:                            │  │
                        │  │    greeting → return GREETING_TEXT            │  │
                        │  │    intro    → ATSMode.extract(intro)          │  │
                        │  │    interview→ InterviewerMode.stream_question │  │
                        │  │    complete → return CLOSING_TEXT             │  │
                        │  │                                               │  │
                        │  │  Resilience: cache hit → skip API             │  │
                        │  │  circuit breaker → apology response           │  │
                        │  └──────────────┬────────────────────────────────┘  │
                        │                 │                                   │
                        │         route_after_llm()                           │
                        │            ┌───┴──────────────┐                     │
                        │         [ok]                [error]                 │
                        │            │                   │                    │
                        │            ▼                   ▼                    │
                        │  ┌──────────────────┐  ┌──────────────────────┐     │
                        │  │  node_sanitize   │  │  node_llm_error      │     │
                        │  │                  │  │  retries += 1        │     │
                        │  │  27-step text    │  └─────────┬────────────┘     │
                        │  │  cleaning for    │            │                  │
                        │  │  TTS safety      │  route_after_llm_error()      │
                        │  └────────┬─────────┘       ┌───┴─────────┐         │
                        │           │            [retry]          [abort]     │
                        │           ▼                │                │       │
                        │  ┌──────────────────┐      └──▶ [node_llm]  │       │
                        │  │    node_tts      │                       ▼       │
                        │  │                  │              node_error_      │
                        │  │  synthesize      │              terminal → END   │
                        │  │  pcm_stream /    │                               │
                        │  │  full WAV        │                               │
                        │  └────────┬─────────┘                               │
                        │           │                                         │
                        │  route_after_tts()                                  │
                        │       ┌───┴──────────────┐                          │
                        │    [ok,IS_DEV]      [ok,!IS_DEV]     [error]        │
                        │       │                   │              │          │
                        │       ▼                   ▼              ▼          │
                        │ node_playback_     END        node_tts_error →      │
                        │    dev                        node_error_terminal   │
                        │ (play locally)                → END                 │
                        │       │                                             │
                        │      END                                            │
                        └─────────────────────────────────────────────────────┘
```

### 7.2 STT Node (`STT_service.py`)

The Speech-to-Text node wraps OpenAI Whisper with a full production resilience stack.

```
Audio Input (path / PCMChunk / AsyncIterator[PCMChunk])
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STT NODE INTERNALS                                │
│                                                                             │
│  PCMChunk?  ────▶  PCMChunkWAVEncoder                                       │
│                       │  normalize to int16, enforce mono                   │
│                       │  chunk_to_wav_bytes()                               │
│                       ▼                                                     │
│  File path? ────▶  _validate_audio_path()                                   │
│                       │  check exists, extension, size                      │
│                       ▼                                                     │
│             LatencyBudget.check()  ──── exceeded ──▶ raise LatencyBudget    │
│                       │                               Exceeded              │
│                       ▼                                                     │
│             bulkheads["stt"].acquire()  ── no slots ──▶  503                │
│                       │                                                     │
│                       ▼                                                     │
│             circuit_breaker.call()                                          │
│                OPEN ──────────────────────────────▶ try local fallback      │
│                CLOSED / HALF-OPEN:                                          │
│                       │                                                     │
│                       ▼                                                     │
│             _call_whisper(wav_bytes)                                        │
│               ├── response_format: verbose_json                             │
│               ├── language: auto-detect                                     │
│               ├── timeout: STT_TIMEOUT                                      │
│               └── returns: segments + avg_logprob                           │
│                       │                                                     │
│             PCMConfidenceFilter                                             │
│               └── drops segments with avg_logprob < threshold               │
│                       │                                                     │
│             structlog + OTel span + Prometheus                              │
│             S3 upload (if configured)                                       │
│                       │                                                     │
│                       ▼                                                     │
│             STTResult / PCMSTTResult                                        │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
  transcript text  →  node_llm
```

**Streaming path** (`transcribe_chunk_stream`):

```
AsyncIterator[PCMChunk]  (from VAD gate)
    │
    ├─▶ per chunk: PCMConverter → target format (16 kHz mono int16)
    ├─▶ PCMChunkWAVEncoder → WAV bytes
    ├─▶ _call_whisper() → segments
    └─▶ yield STTSegment (real-time)
```

### 7.3 LLM Node (`LLM_service.py`)

Three operation modes with a shared resilience infrastructure:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LLM NODE ARCHITECTURE                             │
│                                                                             │
│   LLMNode                                                                   │
│    ├── InterviewerMode         (stage == "interview")                       │
│    │     stream_question(LLMInterviewInput) → AsyncIterator[str]            │
│    │     Input ONLY: domain | level | last_q | last_a | switch_flag         │
│    │     Enforced 80-word cap via _ResponseValidator                        │
│    │     Truncates to first question sentence ending in "?"                 │
│    │     Fallback question bank if no valid question found                  │
│    │                                                                        │
│    ├── ATSMode                 (stage == "intro")                           │
│    │     extract(intro_transcript) → str (JSON)                             │
│    │     response_format: json_object                                       │
│    │     Validates schema before returning                                  │
│    │     Rule-based fallback via qa_controller if parse fails               │
│    │                                                                        │
│    └── _SharedInfra                                                         │
│          ├── Redis cache (TTL configurable, stampede lock)                  │
│          ├── InMemoryLRU (fallback when Redis unreachable)                  │
│          ├── CircuitBreaker (3 failures → OPEN, 60s reset)                  │
│          ├── RateLimiter (token bucket, LLM_RATE_LIMIT_RPM)                 │
│          ├── bulkheads["llm"] (max concurrent = LLM_MAX_CONCURRENT)         │
│          ├── OTel span (token usage, cache hit/miss, latency)               │
│          └── Prometheus (ai_llm_requests_total, ai_llm_latency_seconds)     │
│                                                                             │
│   LLMNodeV2(LLMNode) — Extended production features                         │
│    ├── PromptInjectionDetector                                              │
│    ├── ContentPolicyFilter                                                  │
│    ├── QuestionDiversityEnforcer   (avoids repeating question types)        │
│    ├── StaticQuestionBank          (fallback for breaker OPEN state)        │
│    ├── StreamingWordGuard           (rejects unsafe tokens mid-stream)      │
│    ├── EvalModeLLM                 (internal; used by evaluation_engine)    │
│    └── BatchATSProcessor           (parallel intro extraction)              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Cache flow:**

```
  request_id + input hash
       │
       ▼
  Redis GET  ──── HIT  ──▶  return cached response (instant)
       │
      MISS
       │
       ▼
  stampede_lock.acquire()  (prevents N concurrent identical calls)
       │
       ▼
  OpenAI API call
       │
       ▼
  Redis SET (TTL = LLM_CACHE_TTL_S)
```

### 7.4 Sanitize Node (`sanitize.py`)

A 27-step deterministic text cleaning pipeline that transforms any LLM output into TTS-safe speech.

```
Raw LLM text
    │
    ▼  Step 1:  Type coercion         (None/bytes/int → str)
    ▼  Step 2:  Empty fast path        (blank → return early)
    ▼  Step 3:  Null-byte + surrogate  (strip before normalization)
    ▼  Step 4:  Unicode NFKC           (ＡＢＣ → ABC, ＋ → +)
    ▼  Step 5:  HTML entity unescape   (&amp; → &, &nbsp; → space)
    ▼  Step 6:  Script/style strip     (<script>…</script>)
    ▼  Step 7:  HTML/XML tag strip     (<b>, </em>, <?xml …?>)
    ▼  Step 8:  ANSI escape strip      (\x1b[31m, \x1b[0m)
    ▼  Step 9:  Zero-width Unicode     (ZWJ, ZWNJ, BOM, soft hyphen)
    ▼  Step 10: Newline normalization  (\n → ". ", \r → "", \t → " ")
    ▼  Step 11: Control char strip     (\x00–\x08, \x0b, \x0c, DEL)
    ▼  Step 12: Path traversal strip   (../ and ..\\ sequences)
    ▼  Step 13: Prompt injection detect ("ignore previous", "system:")
    ▼  Step 14: SSI/template inject    (<!--#, {{, }}, {%, %})
    ▼  Step 15: Tech token sub         (C++ → "C plus plus", .NET → "dot net")
    ▼  Step 16: Symbol substitution    (& → "and", @ → "at", % → "percent")
    ▼  Step 17: Currency expansion     ($5 → "5 dollars", €10 → "10 euros")
    ▼  Step 18: Markdown block strip   (fenced code, block quotes, hr)
    ▼  Step 19: Markdown inline strip  (**bold**, _italic_, [links](url))
    ▼  Step 20: URL simplification     (https://example.com/path → "example.com")
    ▼  Step 21: Emoji strip            (all Unicode emoji and pictograph ranges)
    ▼  Step 22: Repeated punctuation   ("!!!" → "!", "???" → "?")
    ▼  Step 23: Smart quote normalize  (" " → ", ' ' → ')
    ▼  Step 24: Ellipsis normalize     ("......" → "…", "…" kept)
    ▼  Step 25: Whitespace normalize   (multi-space → single, strip ends)
    ▼  Step 26: Empty guard            (if sanitized is blank → return flag)
    ▼  Step 27: Sentence-boundary cap  (max_chars, break at last sentence)
    │
    ▼
SanitizeResult {
    text:          str          # TTS-safe, ready to synthesize
    truncated:     bool
    original_len:  int
    sanitized_len: int
    was_empty:     bool
    warnings:      list[str]    # tags for each issue class detected
}
```

### 7.5 TTS Node (`TTS_service.py`)

```
SanitizeResult.text
    │
    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                          TTS NODE INTERNALS                              │
│                                                                          │
│  PCMTTSOutputConfig  ─── binds output PCMFormat, PCMPlaybackEnhancer,    │
│                           PCMLatencyTracker for this node instance       │
│                                                                          │
│  LatencyBudget.check()  ──── exceeded ──▶ return apology audio           │
│                                                                          │
│  bulkheads["tts"].acquire()                                              │
│                                                                          │
│  circuit_breaker.call()                                                  │
│         │                                                                │
│         ▼                                                                │
│  [synthesize_pcm_stream mode — TTS_FORMAT=pcm]                           │
│   openai.audio.speech.create()  → stream bytes                           │
│        │                                                                 │
│        ▼                                                                 │
│   tts_pcm_to_chunk()           → PCMChunk per packet                     │
│        │                                                                 │
│        ▼                                                                 │
│   PCMTTSQualityGate            → reject/retry bad chunks                 │
│        │                                                                 │
│        ▼                                                                 │
│   PCMSentenceGapManager        → calibrated silence between sentences    │
│        │                                                                 │
│        ▼                                                                 │
│   PCMPlaybackEnhancer          → limiter + AGC before speaker            │
│        │                                                                 │
│        ▼                                                                 │
│   PCMOutputStream.write(chunk) → speaker  OR  yield to caller            │
│                                                                          │
│  [synthesize mode — TTS_FORMAT=mp3/wav]                                  │
│   openai.audio.speech.create() → full audio bytes                        │
│   write to local disk + S3 upload (if configured)                        │
│   return file path                                                       │
│                                                                          │
│  [failure path]                                                          │
│   Cached apology audio → return pre-rendered fallback                    │
│   PCMStreamToWAVCollector → collect stream to in-memory WAV              │
└──────────────────────────────────────────────────────────────────────────┘
    │
    ▼
  audio_output path / S3 URI / PCMChunks
```

---

## 8. PCM Audio Engine

`audio_engine.py` (7,141 lines) is the single source of truth for all raw PCM audio within the pipeline. It eliminates file I/O between stages and provides typed, immutable audio primitives.

### 8.1 Format Negotiation

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         PCMFormat (frozen dataclass)                         │
│                                                                              │
│   PCMFormat(sample_rate=16000, channels=1, dtype="int16", byte_order="<")    │
│                                                                              │
│   Built-in presets:                                                          │
│     PCMFormat.whisper()           → 16 kHz, mono, int16, little-endian       │
│     PCMFormat.openai_tts()        → 24 kHz, mono, float32, little-endian     │
│     PCMFormat.portaudio_default() → 44100 Hz, stereo, float32                │
│                                                                              │
│   negotiate_format(caps_a, caps_b) → PCMFormat                               │
│     Picks the highest common sample rate and channel count                   │
│     Used in SessionLifecycleManager.open_session() to wire                   │
│     mic → STT and TTS → speaker without mismatches                           │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 PCM-Native Realtime Path

The `stream_full_pcm()` mode is the zero-file-I/O path for the lowest achievable latency:

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                    PCM-NATIVE REALTIME PIPELINE (stream_full_pcm)             │
│                                                                               │
│                                                                               │
│  PortAudio callback                                                           │
│       │                                                                       │
│       ▼                                                                       │
│  PCMInputStream ─────────────────────────▶ asyncio.Queue (raw chunks)         │
│       │  (async-for iterator of PCMChunks)                                    │
│       │                                                                       │
│       ▼                                                                       │
│  PCMSpeechEnhancer                                                            │
│    ├── bandpass filter   (removes sub-100Hz rumble, >8kHz noise)              │
│    ├── noise suppression (spectral subtraction)                               │
│    ├── AGC               (normalise input level)                              │
│    ├── gain gate         (attenuate during silence)                           │
│    └── VAD gate          (suppress non-speech frames, 200ms hangover)         │
│       │                                                                       │
│       │  yields: PCMChunk[speech-only, enhanced]                              │
│       │                                                                       │
│  ┌────┴────────────────────────────────────────────────────────┐              │
│  │                    StageBus: mic→stt                        │              │
│  │  BoundedPipelineQueue(maxdepth=8, overflow=DROP_OLDEST)     │              │
│  │  DLQ, throughput metering, backpressure watermarks          │              │
│  └─────────────────────────┬───────────────────────────────────┘              │
│                            │                                                  │
│                            ▼                                                  │
│  mic_reader() ──────────▶ STT worker                                          │
│                            │                                                  │
│  PCMChunkWAVEncoder        │  chunk → WAV bytes (no temp file)                │
│       │                    │                                                  │
│       ▼                    │                                                  │
│  STTNode.transcribe_chunk()│  WAV bytes → Whisper API → transcript text       │
│       │                    │                                                  │
│       ▼                    │                                                  │
│  PCMConfidenceFilter       │  drops low-logprob segments                      │
│       │                    │                                                  │
│  ┌────┴──────────────────────────────────────────────────────┐                │
│  │                    StageBus: stt→llm                      │                │ 
│  └─────────────────────────┬─────────────────────────────────┘                │
│                            │                                                  │
│                            ▼                                                  │
│  LLM worker (qa path)      │                                                  │
│    _node_llm_qa_path()     │  transcript → next question (streamed tokens)    │
│    InterviewerMode         │                                                  │
│       │                    │                                                  │
│  ┌────┴──────────────────────────────────────────────────────┐                │
│  │                    StageBus: llm→tts                      │                │
│  └─────────────────────────┬─────────────────────────────────┘                │
│                            │                                                  │
│                            ▼                                                  │
│  TTS worker                │                                                  │
│    TTSNode.synthesize_     │  tokens → PCMChunks (streamed)                   │
│      pcm_stream()          │                                                  │
│       │                    │                                                  │
│  PCMPlaybackEnhancer       │  limiter + AGC pre-speaker                       │
│       │                    │                                                  │
│  PCMOutputStream           │  → PortAudio speaker output                      │
│       │                    │                                                  │
│  PCMInterruptDetector ─────┘  monitors for energy burst (barge-in)            │
│    └── FF_BARGE_IN=true                                                       │
│        interrupt → cancel TTS worker → restart mic_reader                     │
└───────────────────────────────────────────────────────────────────────────────┘
```

### 8.3 Audio Components

| Component | File | Purpose |
|---|---|---|
| `PCMFormat` | audio_engine.py | Immutable format descriptor (rate, channels, dtype) |
| `PCMChunk` | audio_engine.py | Timestamped, typed PCM payload (slots dataclass) |
| `PCMChunkPool` | audio_engine.py | Numpy array pool, avoids GC on hot path |
| `PCMRingBuffer` | audio_engine.py | Lock-free power-of-2 circular buffer |
| `PCMConverter` | audio_engine.py | Resample + channel coerce + dtype convert |
| `PCMInputStream` | audio_engine.py | Async mic → PCMChunk iterator (PortAudio bridge) |
| `PCMOutputStream` | audio_engine.py | PCMChunk → speaker (persistent, low-latency) |
| `PCMVADGate` | audio_engine.py | Energy VAD with hangover + pre-roll |
| `PCMSpeechEnhancer` | audio_engine.py | Bandpass → NS → AGC → gate → VAD chain |
| `PCMPlaybackEnhancer` | audio_engine.py | Limiter + AGC before speaker output |
| `PCMInterruptDetector` | audio_engine.py | Barge-in energy burst detection |
| `PCMJitterBuffer` | audio_engine.py | Network jitter compensation |
| `PCMDiagnosticsMonitor` | audio_engine.py | Boundary measurement (clipping, RMS, dropout) |
| `PCMWaveformAnalyzer` | audio_engine.py | Full waveform stats per chunk |
| `PCMLatencyTracker` | audio_engine.py | Per-stage latency with percentiles |
| `PCMFormatRegistry` | audio_engine.py | Global format registry for negotiation |
| `PCMSplitter` | audio_engine.py | Fan-out: one input → N async consumers |
| `PCMStreamBridge` | audio_engine.py | Cross-coroutine PCM stream bridge |
| `PCMPipelineBuilder` | audio_engine.py | Fluent builder for PCM stage chains |
| `PCMDriftCorrector` | audio_engine.py | Clock drift compensation |
| `PCMSilencePadder` | audio_engine.py | Inter-sentence silence insertion |
| `PCMStreamMixer` | audio_engine.py | Mix N PCM streams into one |
| `PCMTranscoder` | audio_engine.py | FFmpeg-based format transcoding |
| `tts_pcm_to_chunk()` | audio_engine.py | Parse raw TTS PCM bytes → PCMChunk |
| `chunk_to_wav_bytes()` | audio_engine.py | PCMChunk → complete WAV bytes |
| `negotiate_format()` | audio_engine.py | Capability negotiation between two format sets |
| `FFmpegPCMInputStream` | opus_ffmpeg_io.py | Opus → PCMChunk (network decode) |
| `FFmpegPCMOutputStream` | opus_ffmpeg_io.py | PCMChunk → Opus (network encode, ABR) |

**Latency optimizations in `player.py`:**

```
Optimization 1: Persistent PCMOutputStream
   Open once at startup, keep alive between sentences
   Saves: 50–150 ms device-open cost per sentence

Optimization 2: Async write scheduling
   play_audio_bytes() returns in ~μs, actual write is async
   Saves: sync write latency from TTS call stack

Optimization 3: latency='low' PortAudio setting
   Minimum safe hardware buffer
   Saves: 10–30 ms output buffer depth

Optimization 4: Silent warmup chunk
   Written immediately after stream open
   Saves: 10–20 ms cold-start jitter on first chunk

Optimization 5: Cached device info
   Queried once at open, not on every call
   Saves: syscall overhead per chunk

Optimization 6: PCM-native path (play_pcm_chunk)
   Bypasses sf.read() entirely for voice_graph.stream_full()
   Saves: decode + format conversion on every chunk
```

---

## 9. QA Interview Engine

### 9.1 Session State Machine

The interview progresses through four stages managed by `qa_controller.py`:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                        QA SESSION STATE MACHINE                                  │
│                                                                                  │
│                                                                                  │
│   ┌───────────────┐                                                              │
│   │   GREETING    │  LLM is bypassed — static greeting text returned             │
│   │               │  No API call, no latency                                     │
│   │ "Hello! I'm   │                                                              │
│   │  your AI      │                                                              │
│   │  interviewer" │                                                              │
│   └───────┬───────┘                                                              │
│           │  candidate speaks (intro)                                            │
│           ▼                                                                      │
│   ┌────────────────┐                                                             │
│   │     INTRO      │  ATSMode.extract(raw_intro_transcript)                      │
│   │                │  Returns JSON:                                              │
│   │  ATS extraction│    { name, level, domains[], notes }                        │
│   │  from intro    │  qa_controller.seed_from_intro() writes QA.json             │
│   │  transcript    │  domain_queue populated, active_domain set                  │
│   └───────┬────────┘                                                             │
│           │  seed complete                                                       │
│           ▼                                                                      │
│   ┌───────────────────────────────────────────────────────────┐                  │
│   │                      INTERVIEW                            │                  │
│   │                                                           │                  │
│   │  ┌────────────┐   commit_turn()   ┌────────────────┐      │                  │
│   │  │  Domain A  │ ───────────────▶  │  Domain A      │      │                  │
│   │  │  q_asked=0 │                   │  q_asked=7     │      │                  │
│   │  │  q_target=7│                   │  complete=true │      │                  │
│   │  └────────────┘                   └───────┬────────┘      │                  │
│   │                                           │ rotate        │                  │
│   │  ┌────────────┐                           ▼               │                  │
│   │  │  Domain B  │ ◀── domain_queue.pop() ───┘               │                  │
│   │  │  q_asked=0 │                                           │                  │
│   │  └────────────┘                                           │                  │
│   │       ...                                                 │                  │
│   │                                                           │                  │
│   │  Per-turn LLM input (ONLY these fields):                  │                  │
│   │    domain | level | last_q | last_a | domain_switch_flag  │                  │
│   │                                                           │                  │
│   │  Nothing else — no history, no eval signals, no state     │                  │
│   └────────────────────────────────────────────────────────┬──┘                  │
│                                                            │                     │
│                                              all domains complete                │
│                                                            ▼                     │
│   ┌───────────────┐                                                              │
│   │   COMPLETE    │  LLM bypassed — static closing text returned                 │
│   │               │  Evaluation batch triggers for remaining domains             │
│   │ "Thank you for│  Session marked complete in Redis                            │
│   │  your time"   │                                                              │
│   └───────────────┘                                                              │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 9.2 QA Data Schema

The `QA.json` document is the single source of truth for all interview state, stored per-session in Redis:

```json
{
    "session_id":       "550e8400-e29b-41d4-a716-446655440000",
    "stage":            "interview",
    "candidate": {
        "raw_intro":    "Hi, I'm Sarah, senior backend engineer...",
        "name":         "Sarah",
        "level":        "advanced",
        "notes":        "5 years Python, led ML platform team, ATS score 87"
    },
    "domains":          ["python", "system_design", "distributed_systems"],
    "domain_queue":     ["system_design", "distributed_systems"],
    "active_domain":    "python",
    "domain_progress": {
        "python": {
            "q_asked":  4,
            "q_target": 7,
            "complete": false
        },
        "system_design": {
            "q_asked":  0,
            "q_target": 7,
            "complete": false
        }
    },
    "turns": [
        {
            "turn_index": 0,
            "domain":     "python",
            "q":          "What's the difference between a list and a tuple?",
            "a":          "Lists are mutable, tuples are immutable...",
            "ts":         1711234567.89
        }
    ],
    "current_question": "How would you optimize a slow Python generator pipeline?",
    "turn_index":       4,
    "domain_switched":  false,
    "total_questions":  4,
    "created_at":       1711234500.0,
    "updated_at":       1711234600.0
}
```

### 9.3 LLM Modes

```
Data Flow Per Turn (interview stage):

  voice_graph.node_llm()
        │
        ├─ 1. qa_controller.get_llm_input(session_id, candidate_answer)
        │        └─▶ LLMInterviewInput:
        │               { domain, level, last_q, last_a, domain_switch_flag }
        │
        ├─ 2. LLM API call (InterviewerMode)
        │        └─▶ stream: "Great answer! Now, how would you..."
        │
        ├─ 3. sanitize(llm_response)
        │
        ├─ 4. qa_controller.commit_turn(session_id, answer, question)
        │        └─▶ CommittedTurn:
        │               { turn_index, domain, q, a, ts, domain_rotated }
        │
        ├─ 5. qa_audit_bus.route_turn(committed_turn, qa_doc)
        │        └─▶ transcript_writer.write_turn()  [async, fire-and-forget]
        │        └─▶ eval_engine.schedule_turn_eval() [async, fire-and-forget]
        │
        └─ 6. [if domain_rotated] qa_audit_bus.trigger_domain_eval()
                   └─▶ eval_engine.schedule_domain_eval() [batch scoring]
```

**Intro phase (ATS extraction):**

```
Candidate speaks intro
        │
        ▼
  STT → raw_intro_transcript
        │
        ▼
  qa_controller.get_intro_input(session_id, intro_text)
        │
        ▼
  ATSMode.extract(intro_text)
   └─▶ GPT API: response_format=json_object
   └─▶ Returns: { name, level, domains[], notes }
        │
        ▼
  qa_controller.seed_from_intro(session_id, ats_result)
   └─▶ Writes domains, level to QA.json
   └─▶ Advances stage → "interview"
   └─▶ Returns first LLMInterviewInput
```

---

## 10. Evaluation Engine

The evaluation engine scores every candidate answer **off the critical path** — TTS latency is never affected.

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                         EVALUATION ENGINE                                      │
│                                                                                │
│  Triggers (all async, fire-and-forget from qa_audit_bus):                      │
│    • Per-turn: schedule_turn_eval(session_id, turn)                            │
│    • Per-domain: schedule_domain_eval(session_id, domain, turns)               │
│                                                                                │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │                      COST CONTROLS (in order)                             │ │
│  │                                                                           │ │
│  │  1. MINIMUM ANSWER GATE                                                   │ │
│  │     answer.len < EVAL_MIN_ANSWER_CHARS (40)?                              │ │
│  │     → skip ("Yes", "I don't know" not worth scoring)                      │ │
│  │                                                                           │ │
│  │  2. ADAPTIVE SAMPLING                                                     │ │
│  │     first EVAL_FULL_EVAL_TURNS (5)? → always score                        │ │
│  │     after that? → score 1 in EVAL_SAMPLE_RATE (3)                         │ │
│  │                                                                           │ │
│  │  3. PER-SESSION BUDGET                                                    │ │
│  │     tokens_consumed >= EVAL_SESSION_BUDGET_TOKENS (50000)?                │ │
│  │     → stop silently (Redis budget counter)                                │ │
│  │                                                                           │ │
│  │  4. CONTENT-HASH CACHE                                                    │ │
│  │     sha256(question + answer) hit in Redis?                               │ │
│  │     → return cached score instantly                                       │ │
│  │                                                                           │ │
│  │  5. DEDUP SENTINEL                                                        │ │
│  │     Redis NX key: eval:dedup:v1:{session_id}:{turn_index}                 │ │
│  │     → prevents double-scoring on retry                                    │ │
│  │                                                                           │ │
│  │  6. DEDICATED BULKHEAD                                                    │ │
│  │     max EVAL_MAX_CONCURRENT (3) parallel scoring calls                    │ │
│  │     A burst of PTT presses won't spawn 20 API calls                       │ │
│  │                                                                           │ │
│  │  7. TOKEN-BUCKET RATE LIMITER                                             │ │
│  │     Separate from main LLM limiter                                        │ │
│  │     Eval can't consume interview quota                                    │ │
│  │                                                                           │ │
│  │  8. CIRCUIT BREAKER                                                       │ │
│  │     3 consecutive failures → OPEN                                         │ │
│  │     Stops evaluation until model recovers                                 │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                │
│  Scoring Model: EVAL_MODEL (default: gpt-5.2)                                  │
│  Scoring Input:                                                                │
│    { domain, level, question, answer }  (truncated to avoid overrun)           │
│                                                                                │
│  Scoring Output (structured JSON):                                             │
│    {                                                                           │
│      "score": 1–10,                                                            │
│      "dimension_scores": { "technical": 8, "communication": 7, ... },          │
│      "feedback": "Strong explanation of GIL, missed mention of asyncio",       │
│      "flags": ["partial", "off_topic"]                                         │
│    }                                                                           │
│                                                                                │
│  Redis key layout:                                                             │
│    eval:score:v1:{session_id}:{turn_index}    JSON blob                        │
│    eval:budget:v1:{session_id}               int tokens consumed               │
│    eval:dedup:v1:{session_id}:{turn_index}   NX sentinel                       │
└────────────────────────────────────────────────────────────────────────────────┘
```

**Dead-Letter Queue for eval failures:**

```
eval_engine.schedule_domain_eval() fails
        │
        ▼
  DLQ entry (in-process bounded queue)
        │
        ▼
  Background retry task
   └─▶ exponential backoff (1s, 2s, 4s, 8s, max 60s)
   └─▶ max attempts: 5
   └─▶ alert via observability if exhausted
   └─▶ no turns silently dropped
```

---

## 11. Session Management

### 11.1 Session Store

`session_store.py` provides IP-locked sessions with Redis as the primary store and an in-process LRU as fallback.

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                          SESSION STORE ARCHITECTURE                           │
│                                                                               │
│  Registration: POST /session/register                                         │
│                                                                               │
│  Client IP                                                                    │
│      │                                                                        │
│      ▼                                                                        │
│  IP extraction priority:                                                      │
│    1. CF-Connecting-IP  (Cloudflare)                                          │
│    2. X-Forwarded-For   (standard proxy)                                      │
│    3. REMOTE_ADDR       (direct connection)                                   │
│      │                                                                        │
│      ▼                                                                        │
│  ip_hash = HMAC-SHA256(SESSION_SECRET_KEY, canonical_ip)                      │
│  (raw IPs never stored)                                                       │
│      │                                                                        │
│      ▼                                                                        │
│  Redis SETNX  session:lock:{ip_hash}                                          │
│    ├── SUCCESS → new session allowed                                          │
│    └── FAIL    → 409 Conflict ("active session exists for this IP")           │
│      │                                                                        │
│      ▼                                                                        │
│  session_id = sha256(ip_hash + os.urandom(16))   (opaque, non-reversible)     │
│      │                                                                        │
│      ▼                                                                        │
│  Redis SETEX  session:data:{session_id}  (TTL = SESSION_TTL_S + GRACE)        │
│  Redis SETEX  session:meta:{ip_hash}     (ip_hash → session_id mapping)       │
│      │                                                                        │
│      ▼                                                                        │
│  Response: { session_id, registered_from_ip (masked), expires_at }            │
│                                                                               │
│  Redis Key Layout:                                                            │
│    session:lock:{ip_hash}     NX sentinel, TTL = TTL + GRACE                  │
│    session:data:{session_id}  JSON conversation state, same TTL               │
│    session:meta:{ip_hash}     session_id string lookup, same TTL              │
│                                                                               │
│  InMemoryLRU Fallback (Redis unreachable):                                    │
│    Process-local, no cross-process guarantee                                  │
│    Concurrency throttled in degraded_mode()                                   │
│    Warning logged, SLA marked degraded in response                            │
└───────────────────────────────────────────────────────────────────────────────┘
```

### 11.2 Session Lifecycle Manager

`SessionLifecycleManager` in `voice_graph.py` handles all resource acquisition and release per session:

```
open_session(session_id)
        │
        ├─ 1. Mic health check            (async, optional via FF_SESSION_LIFECYCLE)
        │       └─▶ recorder.run_startup_health_check()
        │
        ├─ 2. Speaker health check         (async, optional)
        │       └─▶ player.check_audio_health()
        │
        ├─ 3. QA controller session create
        │       └─▶ qa_controller.create_session(session_id)
        │
        ├─ 4. Audit bus open
        │       └─▶ qa_audit_bus.open_session(session_id)
        │
        ├─ 5. Transcript writer begin
        │       └─▶ transcript_writer.open_session(session_id)
        │
        ├─ 6. PCM format negotiation       (mic → STT, TTS → speaker)
        │       └─▶ negotiate_format(mic_caps, stt_caps) → mic_fmt
        │       └─▶ negotiate_format(tts_caps, speaker_caps) → tts_fmt
        │
        └─▶ SessionResources (fmt, start_time, diagnostics, watchdog)


close_session(session_id, reason="complete")
        │
        ├─ 1. QA controller close
        ├─ 2. Audit bus close + flush
        ├─ 3. Transcript writer flush     (wait TRANSCRIPT_QUEUE_DEPTH drain)
        ├─ 4. PCM stream drain            (flush speaker output)
        ├─ 5. LLM session cache evict
        ├─ 6. Temp file cleanup           (remove WAV uploads)
        └─▶ Emit session_close ObsEvent
```

**Pipeline Watchdog:**

```
PipelineWatchdog
    │
    ├── Heartbeat check every WATCHDOG_INTERVAL_S (default: 5s)
    ├── Monitors: StageBus depth, pipeline stage timestamps, error rates
    │
    ├── Strike system:
    │     1 strike  → log warning
    │     2 strikes → Prometheus alert
    │     3 strikes → recovery action
    │
    └── Recovery actions (in escalation order):
          1. mic_restart     (PCMInputStream.restart())
          2. speaker_restart (PCMOutputStream.restart())
          3. pipeline_reset  (cancel all tasks, reinitialize)
```

---

## 12. Transcript Writer

`transcription.py` writes every interview turn to two independent, non-blocking sinks simultaneously.

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                         TRANSCRIPT WRITER (Dual Sink)                         │
│                                                                               │
│  After each committed turn:                                                   │
│    transcript_writer.write_turn(session_id, user_text, assistant_text)        │
│                                                                               │
│                         asyncio.Queue (TRANSCRIPT_QUEUE_DEPTH=512)            │
│                                   │                                           │
│                         background drain task                                 │
│                           (run_in_executor → thread pool)                     │
│                                   │                                           │
│                   ┌───────────────┴──────────────────┐                        │
│                   │                                   │                       │
│                   ▼                                   ▼                       │
│            SINK A — .txt file               SINK B — observability            │
│                                                                               │
│  transcripts/session_{sid[:16]}.txt        emit(ObsEvent(                     │
│                                                kind=TRANSCRIPT_TURN,          │
│  Format:                                       session_id=...,                │
│    ────────────────────────────             ))                                │
│    Session  : {full_uuid}                        │                            │
│    Started  : 2024-03-24 10:17:02 UTC            ├─▶ structlog JSON file      │
│    ────────────────────────────             ├─▶ Prometheus counter            │
│                                             │       ai_transcript_turns_total │
│    [10:17:02] [d056d716]   <user>           ├─▶ MongoDB document              │
│    [10:17:10] [AI]         <assistant>      └─▶ OTel span annotation          │
│                                                                               │
│    [10:18:05] [d056d716]   <user>                                             │
│    [10:18:12] [AI]         <assistant>                                        │
│                                                                               │
│    ────────────────────────────                                               │
│    Session ended : 2024-03-24 10:45:00 UTC                                    │
│    ────────────────────────────                                               │
│                                                                               │
│  Queue full → TranscriptEmitter.queue_drop() → counter increment + log        │
│  Flush on close: await transcript_writer.flush(timeout=5.0)                   │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## 13. QA Audit Bus

`conversation_memory.py` is **not** a conversation memory module — it is a dedicated audit routing bus between the pipeline and its downstream systems.

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                          QA AUDIT BUS (conversation_memory.py)                │
│                                                                               │
│  RESPONSIBILITIES:                                                            │
│    1. Turn routing       → fan out CommittedTurns to transcript + eval        │
│    2. Domain batch mgmt  → accumulate turns, dispatch when quota met          │
│    3. Session lifecycle  → open/close events to transcript + observability    │
│    4. Eval scheduling    → ensure each domain batch scored exactly once       │
│    5. Audit snapshots    → Redis-backed per-session turn log                  │
│    6. Dead-letter queue  → if eval unreachable, retry with backoff            │
│                                                                               │
│  DOES NOT:                                                                    │
│    ✗ Build LangChain message lists  (qa_controller owns LLM input)            │
│    ✗ Maintain rolling windows       (LLM never sees history)                  │
│    ✗ Inject system prompts          (qa_controller owns all prompts)          │
│    ✗ Track topics or hints          (qa_controller.QADocument owns state)     │
│                                                                               │
│  Data Flow:                                                                   │
│                                                                               │
│  voice_graph.node_llm()                                                       │
│       │                                                                       │
│       ├─ qa_controller.commit_turn() → CommittedTurn                          │
│       │                                                                       │
│       ▼                                                                       │
│  qa_audit_bus.route_turn(committed_turn, qa_doc)                              │
│       │                                                                       │
│       ├─▶  transcript_writer.write_turn()          [fire-and-forget]          │
│       ├─▶  eval_engine.schedule_turn_eval()        [fire-and-forget]          │
│       └─▶  Redis: audit:v1:{session_id}:turns      [append JSON record]       │
│                                                                               │
│  [domain quota satisfied]                                                     │
│  qa_audit_bus.trigger_domain_eval(session_id, domain, qa_doc)                 │
│       │                                                                       │
│       ├─ Check: audit:v1:{session_id}:domains_evaled  (idempotency)           │
│       ├─ Mark domain as dispatched (Redis SET NX)                             │
│       └─▶ eval_engine.schedule_domain_eval()       [async, fire-and-forget]   │
│                                                                               │
│  Redis key schema:                                                            │
│    audit:v1:{session_id}:turns           → JSON list of AuditTurnRecord       │
│    audit:v1:{session_id}:domains_evaled  → JSON set of dispatched domains     │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## 14. Observability Stack

### 14.1 Three-Layer Architecture

`observability.py` (8,178 lines) implements a unified three-layer observability stack driven by a single `emit()` call:

```
                    emit(ObsEvent(...))
                           │
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
    LAYER 1             LAYER 2         LAYER 3
    Structured          Prometheus      MongoDB
    Logging             Metrics         Events

 structlog console    Counters         Structured
 + JSON file          Histograms       document per
 (LOG_FILE)           Gauges           event
                                       TTL = 90 days
                      :9090/metrics
                                       Grafana
 Rich TUI live        Grafana          dashboards from
 dashboard            dashboards       auto-generated JSON


                           +
                    LAYER 0 (optional)
                    OpenTelemetry spans
                    OTLP/gRPC export
                    W3C TraceContext
```

**Correlation:** Every event, metric label, and MongoDB document is correlated by:
- `session_id` (session-level correlation)
- `request_id` (per-turn correlation)

### Event Coverage

| Domain | Events Tracked |
|---|---|
| **STT** | transcription, language confidence, audio quality, remote fallback |
| **LLM** | token usage, cache hit/miss, model fallback, stream vs batch |
| **TTS** | synthesis, apology fallback, chunk errors, S3 upload |
| **Eval** | scoring, budget, adaptive sampling, dedup |
| **Sess** | registration, IP change, suspension, TTL expiry |
| **Pipe** | stage retries, aborts, load shedding, SLA breaches |
| **Mem** | history compression, turn overflow, LRU fallback |
| **San** | sanitization warnings, prompt injection, truncation |
| **CB** | circuit breaker state transitions (all services) |
| **RL** | rate limiter exhaustion events |
| **BH** | bulkhead saturation events |
| **Ctrl** | PTT press/release, empty recording, pipeline interrupt |
| **Transcript** | turn written, session open/close, queue drops |

### 14.2 Metrics Reference

**Pipeline metrics:**

| Metric | Type | Labels | Description |
|---|---|---|---|
| `ai_pipeline_requests_total` | Counter | `stage`, `status` | Total pipeline invocations |
| `ai_pipeline_latency_seconds` | Histogram | `stage` | Per-stage latency distribution |
| `ai_pipeline_inflight` | Gauge | — | Currently active pipeline requests |
| `ai_pipeline_shed_total` | Counter | `reason` | Load-shedded requests |

**STT metrics:**

| Metric | Type | Labels | Description |
|---|---|---|---|
| `ai_stt_requests_total` | Counter | `status`, `model` | Transcription attempts |
| `ai_stt_latency_seconds` | Histogram | `model` | STT latency distribution |
| `ai_stt_audio_duration_seconds` | Histogram | — | Input audio duration |
| `ai_stt_language_confidence` | Histogram | `language` | Whisper language confidence |

**LLM metrics:**

| Metric | Type | Labels | Description |
|---|---|---|---|
| `ai_llm_requests_total` | Counter | `status`, `model`, `cache` | LLM API calls |
| `ai_llm_latency_seconds` | Histogram | `model` | LLM response latency |
| `ai_llm_tokens_total` | Counter | `model`, `type` | Token consumption |
| `ai_llm_cache_hits_total` | Counter | — | Redis cache hits |

**TTS metrics:**

| Metric | Type | Labels | Description |
|---|---|---|---|
| `ai_tts_requests_total` | Counter | `status`, `voice` | TTS synthesis attempts |
| `ai_tts_latency_seconds` | Histogram | `voice` | Synthesis latency |
| `ai_tts_chars_total` | Counter | `voice` | Characters synthesized |

**Eval metrics:**

| Metric | Type | Labels | Description |
|---|---|---|---|
| `ai_eval_requests_total` | Counter | `status` | Scoring attempts |
| `ai_eval_skipped_total` | Counter | `reason` | Skipped evaluations |
| `ai_eval_budget_exhausted_total` | Counter | — | Budget cap hits |
| `ai_eval_score` | Histogram | `domain` | Score distributions by domain |

**Transcript metrics:**

| Metric | Type | Labels | Description |
|---|---|---|---|
| `ai_transcript_turns_total` | Counter | — | Turns written to transcript |
| `ai_transcript_drops_total` | Counter | — | Queue-full drops |

### 14.3 Grafana Dashboards

Auto-generated dashboard JSON is written to `GRAFANA_OUT_DIR` (default: `grafana/`) on startup. Five dashboards are provisioned:

1. **Pipeline Overview** — request rate, latency percentiles (p50/p95/p99), error rate, inflight gauge
2. **STT Performance** — audio duration distribution, confidence distribution, language breakdown
3. **LLM Performance** — token rate, cache hit rate, latency heatmap, model fallback events
4. **Session Analytics** — session count, duration distribution, IP change events
5. **Evaluation** — score distributions per domain, budget consumption, circuit breaker state

---

## 15. Resilience Patterns

All resilience primitives live in `shared.py` and are imported by every node.

### 15.1 Circuit Breakers

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CIRCUIT BREAKER STATE MACHINE                    │
│                                                                         │
│                                                                         │
│  ┌─────────────┐   3 consecutive       ┌─────────────┐                  │
│  │   CLOSED    │─── failures ────────▶ │    OPEN     │                  │
│  │  (normal)   │                       │  (blocking) │                  │
│  └─────────────┘                       └──────┬──────┘                  │
│         ▲                                      │                        │
│         │                               60s timeout                     │
│         │                                      │                        │
│         │                                      ▼                        │
│         │                             ┌─────────────────┐               │
│         │                             │   HALF-OPEN     │               │
│  success│                             │  (test call)    │               │
│  ──────────────────────────────────── │                 │               │
│  failure────────────────────────────▶ │ 1 trial request │               │
│                                       │  allowed        │               │
│                                       └─────────────────┘               │
│                                                                         │
│  Per-service breakers (independent):                                    │
│    circuit_breakers["stt"]   → STT API calls                            │
│    circuit_breakers["llm"]   → LLM API calls                            │
│    circuit_breakers["tts"]   → TTS API calls                            │
│    circuit_breakers["eval"]  → Evaluation scoring calls                 │
│    circuit_breakers["redis"] → Redis operations                         │
│    circuit_breakers["mongo"] → MongoDB writes                           │
│                                                                         │
│  On OPEN:                                                               │
│    STT breaker → try local Whisper fallback                             │
│    LLM breaker → return static question bank response                   │
│    TTS breaker → return cached apology audio                            │
│    Eval breaker → skip evaluation, log event                            │
└─────────────────────────────────────────────────────────────────────────┘
```

### 15.2 Rate Limiting & Bulkheads

```
Token-bucket rate limiter (per service):
   ┌──────────────────────────────────────────────┐
   │  RateLimiter(rpm=500)                        │
   │    bucket_tokens = min(burst, tokens + rate) │
   │    acquire(n=1) → wait until token available │
   └──────────────────────────────────────────────┘

Bulkhead pool (per operation type):
   ┌─────────────────────────────────────────────┐
   │  bulkheads["stt"]   = asyncio.Semaphore(5)  │
   │  bulkheads["llm"]   = asyncio.Semaphore(10) │
   │  bulkheads["tts"]   = asyncio.Semaphore(5)  │
   │  bulkheads["eval"]  = asyncio.Semaphore(3)  │
   │                                             │
   │  Separate budgets → STT flood can't starve  │
   │  LLM slots; eval can't steal interview quota│
   └─────────────────────────────────────────────┘
```

### 15.3 Load Shedding

```
LoadSheddingGuard (per VoiceGraph instance):
   ┌─────────────────────────────────────────────────────────┐
   │                                                         │
   │  inflight_count  ≥  max_inflight (20)?                  │
   │        │                                                │
   │       YES → raise LoadSheddingRejected → 503            │
   │        │                                                │
   │        NO → accept request, inflight_count += 1         │
   │             on completion: inflight_count -= 1          │
   │                                                         │
   │  Three singletons → separate inflight caps:             │
   │    voice_graph          max_inflight=20                 │
   │    voice_graph_realtime max_inflight=10 (reserved)      │
   │    voice_graph_low_lat  max_inflight=15                 │
   │                                                         │
   │  A standard request surge cannot exhaust realtime slots │
   └─────────────────────────────────────────────────────────┘
```

### 15.4 Latency Budget

```
LatencyBudget — SLA deadline propagated via context variable:

  Request enters pipeline:
    budget = LatencyBudget(deadline_ms=PIPELINE_TIMEOUT * 1000)
    _latency_budget_cv.set(budget)

  Each node calls:
    budget.check(stage="stt")
      │
      ├── remaining_ms() > 0 → proceed
      └── remaining_ms() ≤ 0 → raise LatencyBudgetExceeded
                                 → return error immediately
                                 → never start a stage that's already late

  Impact: prevents runaway requests from consuming resources
  after the user-visible SLA is already blown.
```

### Backoff Retry

```python
# shared.py — backoff_retry decorator
@backoff_retry(
    max_attempts=3,
    base_delay=0.5,
    max_delay=10.0,
    jitter=True,           # randomize ±25% to avoid thundering herd
    exceptions=(httpx.TimeoutException, openai.RateLimitError),
)
async def _call_whisper(wav_bytes: bytes) -> dict:
    ...
```

---

## 16. Distributed Service Mode

Each pipeline stage can run as an independent microservice. The `voice_graph.py` constructor accepts any object implementing the relevant `Protocol`, and factory functions (`get_stt_node()`, `get_llm_node()`, `get_tts_node()`) select local vs remote based on env vars:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                       DISTRIBUTED SERVICE WIRING                             │
│                                                                              │
│  ENV vars control routing (set before module import):                        │
│    STT_SERVICE_URL=https://stt.internal:8001  → RemoteSTTClient              │
│    LLM_SERVICE_URL=https://llm.internal:8002  → RemoteLLMClient              │
│    TTS_SERVICE_URL=https://tts.internal:8003  → RemoteTTSClient              │
│                                                                              │
│  [No URL set]   → local node (in-process, direct API calls)                  │
│  [URL set]      → remote HTTP client (outbound requests, SSE streaming)      │
│                                                                              │
│  Remote clients inject on every request:                                     │
│    1. W3C TraceContext headers  (OTel distributed tracing stays stitched)    │
│    2. LatencyBudget-ms header   (remaining SLA propagated to remote node)    │
│    3. Request-ID header         (correlation across service boundaries)      │
│                                                                              │
│  STT remote:                                                                 │
│    POST {STT_SERVICE_URL}/transcribe                                         │
│    Content-Type: multipart/form-data (binary audio)                          │
│    Streaming: SSE for chunk transcription                                    │
│                                                                              │
│  LLM remote:                                                                 │
│    POST {LLM_SERVICE_URL}/stream_question                                    │
│    Body: LLMInterviewInput JSON                                              │
│    Response: chunked transfer encoding (token stream)                        │
│                                                                              │
│  TTS remote:                                                                 │
│    POST {TTS_SERVICE_URL}/synthesize_stream                                  │
│    Body: token stream (chunked POST)                                         │
│    Response: chunked audio bytes                                             │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Protocol definitions** (`shared.py`):

```python
class STTNodeProtocol(Protocol):
    async def transcribe(self, audio_path: str, *, request_id: str) -> STTResult: ...
    async def transcribe_chunk(self, chunk: PCMChunk, *, request_id: str) -> str: ...

class LLMNodeProtocol(Protocol):
    async def stream_question(self, input: LLMInterviewInput) -> AsyncIterator[str]: ...
    async def extract_ats(self, intro_text: str) -> str: ...

class TTSNodeProtocol(Protocol):
    async def synthesize(self, text: str, *, request_id: str) -> str: ...
    async def synthesize_pcm_stream(self, tokens: AsyncIterator[str]) -> AsyncIterator[PCMChunk]: ...
```

---

## 17. Feature Flags

All feature flags are resolved at module load time from environment variables. Changing them requires a process restart.

| Flag | Default | Description |
|---|---|---|
| `FF_PCM_PIPELINE` | `false` | Enable zero-copy PCM-native pipeline (`stream_full_pcm` mode) |
| `FF_BARGE_IN` | `false` | Enable barge-in detection (requires `FF_PCM_PIPELINE=true`) |
| `FF_AUDIO_DIAGNOSTICS` | `true in dev` | Enable per-boundary PCM waveform measurements |
| `FF_SESSION_LIFECYCLE` | `true` | Enable full session lifecycle management |
| `FF_QUESTION_PREFETCH` | `true` | Prefetch next question while TTS is speaking |
| `FF_CANARY_PCT` | `0.0` | Fraction (0.0–1.0) of requests routed to PCM pipeline for gradual rollout |

**Canary rollout:**

```
FF_PCM_PIPELINE=false
FF_CANARY_PCT=0.1       # 10% of requests use PCM pipeline

# _should_use_pcm_pipeline(request_id):
#   hash(request_id) % 1.0 < FF_CANARY_PCT → True (use PCM)
```

---

## 18. API Reference

### Endpoints

#### `POST /voice`

Single-shot voice processing. Returns complete result when pipeline finishes.

**Request:**
```
Content-Type: multipart/form-data
X-Session-ID: {session_id}         (required)
X-Request-ID: {uuid}               (optional, auto-generated if absent)
X-QoS-Tier: realtime|standard      (optional, defaults to standard)

file: <audio file>                 (.wav, .mp3, .mp4, .m4a, .webm, .mpeg, .mpga)
                                   max size: MAX_UPLOAD_MB (default 25 MB)
```

**Response `200 OK`:**
```json
{
  "request_id":        "550e8400-...",
  "transcript":        "What's the difference between a list and a tuple?",
  "response":          "Great question! Let's explore Python mutability...",
  "cleaned_response":  "Great question! Let us explore Python mutability...",
  "audio":             "/audio/temp_OUT/abc123.mp3",
  "audio_s3_uri":      "s3://my-bucket/tts/abc123.mp3",
  "degraded":          false,
  "error":             "",
  "error_stage":       "",
  "stage_latencies": {
    "stt": 1.24,
    "llm": 2.87,
    "sanitize": 0.003,
    "tts": 1.56
  },
  "pipeline_latency_s": 5.77,
  "graph_version":     "v1.4.0",
  "metadata":          {}
}
```

**Error responses:**

| Code | Condition |
|---|---|
| `400` | Missing/invalid session ID, unsupported file type |
| `413` | Audio file exceeds `MAX_UPLOAD_MB` |
| `503` | Load shedding (max inflight reached) |
| `504` | Pipeline timeout |

---

#### `WebSocket /voice/stream`

Streaming voice processing. Sends partial results as pipeline stages complete.

**Protocol:**

```
Client → Server: binary audio frame (raw PCM or encoded)
Server → Client: JSON messages:

  { "type": "transcript",   "text": "..." }          # STT complete
  { "type": "llm_token",    "token": "..." }          # LLM streaming token
  { "type": "audio_chunk",  "bytes": "<base64>" }     # TTS audio chunk
  { "type": "done",         "result": {...} }          # Pipeline complete
  { "type": "error",        "message": "...",
                            "stage": "stt" }           # Stage error
```

---

#### `POST /session/register`

Register a new interview session. One active session permitted per IP.

**Response `200`:**
```json
{
  "session_id":          "550e8400-e29b-41d4-a716-446655440000",
  "registered_from_ip":  "192.168.*.1",
  "expires_at":          1711234567.89
}
```

**Response `409`:**
```json
{ "detail": "An active session already exists for this IP address." }
```

---

#### `POST /session/end`

Explicitly end a session and release the IP lock.

**Headers:** `X-Session-ID: {session_id}`

---

#### `POST /session/refresh`

Refresh session TTL (call periodically for long interviews).

**Headers:** `X-Session-ID: {session_id}`

---

#### `DELETE /cancel/{request_id}`

Cancel an in-flight pipeline request across all three graph instances.

**Response `200`:**
```json
{ "cancelled": true, "request_id": "..." }
```

---

#### `GET /health`

Service health check. Returns 200 if healthy, 503 if degraded.

```json
{
  "status": "healthy",
  "graph_version": "v1.4.0",
  "integrations": {
    "audit_bus":        "ok",
    "transcript_writer":"ok",
    "finalize_eval":    "ok"
  },
  "graphs": {
    "standard":     { "inflight": 3, "healthy": true },
    "realtime":     { "inflight": 0, "healthy": true },
    "low_latency":  { "inflight": 1, "healthy": true }
  }
}
```

---

#### `GET /metrics`

Prometheus metrics in text exposition format.

---

### Integration Wiring (Lifespan)

The three downstream integrations are wired in the FastAPI `lifespan` context manager:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await _obs_bootstrap()
    set_audit_bus(qa_audit_bus)
    set_transcript_writer(transcript_writer)
    set_finalize_eval(evaluation_engine.finalize_session)

    yield

    # Shutdown — drain all three singletons
    for graph in _ALL_GRAPHS:
        await graph.session_manager.force_close_all("shutdown")
    await transcript_writer.flush(timeout=10.0)
```

---

## 19. Desktop Controller (PTT)

`controller.py` is the desktop input layer — keyboard events, recording, and pipeline dispatch.

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                          DESKTOP PTT CONTROLLER                               │
│                                                                               │
│  Process startup:                                                             │
│    1. show_boot_sequence()        (Rich TUI boot screen)                      │
│    2. _voice_graph_startup()      (initialize nodes, warm connections)        │
│    3. _register_session()         (register with session_store as local IP)   │
│    4. Start asyncio event loop on background thread                           │
│    5. Enter keyboard poll loop (POLL_INTERVAL_S = 30ms)                       │
│                                                                               │
│  PTT_KEY (default: 'h') held:                                                 │
│    ┌──────────────────────────────────────────────────────────────────────┐   │
│    │  1. _interrupt() called:                                             │   │
│    │       stop_all()                   # stop speaker output             │   │
│    │       cancel active pipeline task  # via _active.take()              │   │
│    │       set _interrupt_flag          # discard queued coroutines       │   │
│    │                                                                      │   │
│    │  2. record_audio_until_released(is_held_fn=keyboard.is_pressed)      │   │
│    │       PCMInputStream → VAD → enhancement → WAV                       │   │
│    │       blocks until key released                                      │   │
│    │                                                                      │   │
│    │  3. [key released]                                                   │   │
│    │       If silence only → empty recording counter, skip dispatch       │   │
│    │       If speech detected → _dispatch(audio_path)                     │   │
│    └──────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
│  _dispatch(audio_path):                                                       │
│    request_id = new_request_id()                                              │
│    _active.set(request_id)                                                    │
│    run_coroutine_threadsafe(_run_pipeline(audio_path), _loop)                 │
│                                                                               │
│  _run_pipeline(audio_path):                                                   │
│    result = await voice_graph.run(state={                                     │
│        "audio_path": audio_path,                                              │
│        "session_id": _session_id,                                             │
│        "request_id": request_id,                                              │
│    })                                                                         │
│    play_pcm_bytes(result.audio_bytes, fmt)                                    │
│    delete_temp_recording(audio_path)                                          │
│                                                                               │
│  EXIT_KEY (default: 'esc'):                                                   │
│    cancel active pipeline                                                     │
│    _voice_graph_shutdown()                                                    │
│    stop_all() + drain_output()                                                │
│    session_store.end_session(_session_id)                                     │
│    sys.exit(0)                                                                │
│                                                                               │
│  Single asyncio event loop on background thread:                              │
│    One loop, zero teardown overhead per PTT press                             │
│    run_coroutine_threadsafe() — safe cross-thread dispatch                    │
│    _interrupt_flag prevents stale coroutines from starting                    │
└───────────────────────────────────────────────────────────────────────────────┘
```

**Controller metrics:**

| Metric | Description |
|---|---|
| `controller_sessions_total` | PTT sessions dispatched |
| `controller_interrupts_total` | Pipelines interrupted by user |
| `controller_empty_recordings_total` | PTT releases with no usable audio |
| `controller_pipeline_errors_total` | Unhandled pipeline errors |
| `controller_recording_duration_seconds` | Mic hold duration per PTT press |

---

## 20. Kafka Integration

`StageBus` in `voice_graph.py` can be replaced with a Kafka-backed implementation for distributed multi-node deployments:

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                    KAFKA STAGE BUS (optional)                                 │
│                                                                               │
│  Default: StageBus wraps BoundedPipelineQueue (in-process, asyncio)           │
│  With Kafka: StageBus wraps KafkaStageBus (distributed, cross-node)           │
│                                                                               │
│  Enable:                                                                      │
│    KAFKA_BOOTSTRAP_SERVERS=broker1:9092,broker2:9092                          │
│    KAFKA_STAGE_BUS_ENABLED=true                                               │
│                                                                               │
│  Topic naming:                                                                │
│    voice.stage.{session_id}.{source}_{dest}                                   │
│    e.g.: voice.stage.550e84.stt_llm                                           │
│                                                                               │
│  Stage consumers:                                                             │
│    STT node publishes to:   voice.stage.*.stt_llm                             │
│    LLM node consumes from:  voice.stage.*.stt_llm                             │
│    LLM node publishes to:   voice.stage.*.llm_tts                             │
│    TTS node consumes from:  voice.stage.*.llm_tts                             │
│                                                                               │
│  Overflow policy → Kafka topic retention policy                               │
│  DLQ → dedicated Kafka dead-letter topic                                      │
│  Backpressure → consumer lag monitoring via Prometheus                        │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## 21. Deployment

### Docker

```dockerfile
FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    ffmpeg portaudio19-dev libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000 9090

CMD ["gunicorn", "main:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "--workers", "4", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--graceful-timeout", "30"]
```

```bash
docker build -t voice-interview-pipeline .
docker run -d \
  --name voice-pipeline \
  -p 8000:8000 \
  -p 9090:9090 \
  -e OPENAI_API_KEY=sk-... \
  -e SESSION_SECRET_KEY=your-key \
  -e REDIS_URL=redis://redis:6379/0 \
  -e MONGO_URI=mongodb://mongo:27017 \
  voice-interview-pipeline
```

### Docker Compose

```yaml
version: "3.9"

services:
  app:
    build: .
    ports:
      - "8000:8000"
      - "9090:9090"
    environment:
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      SESSION_SECRET_KEY: ${SESSION_SECRET_KEY}
      REDIS_URL: redis://redis:6379/0
      MONGO_URI: mongodb://mongo:27017
      LLM_MODEL: gpt-5.2
      TTS_VOICE: nova
    depends_on:
      - redis
      - mongo
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru
    ports:
      - "6379:6379"

  mongo:
    image: mongo:6
    ports:
      - "27017:27017"
    volumes:
      - mongo-data:/data/db

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9091:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - ./grafana:/etc/grafana/provisioning/dashboards

volumes:
  mongo-data:
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: voice-pipeline
spec:
  replicas: 3
  selector:
    matchLabels:
      app: voice-pipeline
  template:
    metadata:
      labels:
        app: voice-pipeline
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
    spec:
      containers:
        - name: app
          image: voice-interview-pipeline:latest
          ports:
            - containerPort: 8000
            - containerPort: 9090
          env:
            - name: OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: openai-secret
                  key: api-key
            - name: SESSION_SECRET_KEY
              valueFrom:
                secretKeyRef:
                  name: session-secret
                  key: key
            - name: REDIS_URL
              value: redis://redis-service:6379/0
          resources:
            requests:
              cpu: "500m"
              memory: "512Mi"
            limits:
              cpu: "2000m"
              memory: "2Gi"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 5
```

---

## 22. Performance Tuning

### Latency Budget Allocation (Default)

```
Total PIPELINE_TIMEOUT = 120s

  Stage           Timeout   Retries    Notes
  ─────────────────────────────────────────────────────────
  STT             30s       1          Retry adds up to 30s extra
  LLM             45s       1          Retry adds up to 45s extra
  TTS             30s       0          No retry (apology fallback instead)
  Sanitize        ~3ms      —          CPU-bound, negligible
  ─────────────────────────────────────────────────────────
  Baseline        105s                 Without retries
  Worst case      180s                 All retries exhausted

Realtime profile (voice_graph_realtime):
  STT: 12s, LLM: 18s, TTS: 12s, 0 retries → 42s worst case
```

### Key Levers

| Lever | Impact | Where to set |
|---|---|---|
| `TTS_FORMAT=pcm` | -100–200ms per turn | `.env` |
| `FF_PCM_PIPELINE=true` | -300–600ms total | `.env` |
| `LLM_MODEL=gpt-5-mini` | -500–1500ms LLM latency | `.env` |
| `FF_QUESTION_PREFETCH=true` | -200–800ms LLM latency (amortized) | `.env` (default on) |
| `GRAPH_LLM_TIMEOUT` | Guards runaway calls | `.env` |
| `GRAPH_MAX_INFLIGHT` | Prevent overload | `.env` |
| `EVAL_SAMPLE_RATE=5` | Reduce eval API cost | `.env` |
| `REDIS_POOL_SIZE=50` | Reduce Redis latency under load | `.env` |

### PCM Pipeline Latency Budget

```
Without FF_PCM_PIPELINE (file-based):
  Mic capture     →  disk write (.wav)         ~50ms overhead
  WAV on disk     →  STT read                  ~10ms overhead
  TTS bytes       →  disk write + read          ~30ms overhead
  Play audio file →  file open + device open   ~100ms overhead
                                               ─────────────
                                               ~190ms removable overhead

With FF_PCM_PIPELINE (zero file I/O):
  PCMChunk        →  WAV encode in memory      ~1ms
  PCMChunks       →  PCMOutputStream (warm)    ~0ms device open
                                               ─────────────
                                               ~191ms saved per turn
```

---

## 23. Troubleshooting

### Common Issues

**Redis connection failure:**
```
WARNING  session_store degraded_mode=True
CRITICAL SessionSecretKey not configured — refusing to start
```
→ Check `REDIS_URL`. System falls back to `InMemoryLRU` but **cross-process session guarantees are lost**. Set `REDIS_URL` for production.

---

**STT circuit breaker OPEN:**
```
WARNING  stt_circuit_breaker state=OPEN fallback=local_whisper
```
→ Whisper API is returning errors. Check `OPENAI_API_KEY` validity. Confirm rate limits. If `OPENAI_BASE_URL` is set, verify local endpoint is healthy.

---

**Load shedding 503:**
```
INFO   load_shed request_id=... max_inflight=20 current=20
```
→ Increase `GRAPH_MAX_INFLIGHT` or scale horizontally. Check if `voice_graph_realtime` is saturated separately.

---

**TTS timeout:**
```
ERROR  tts_timeout timeout=30.0 stage=tts
```
→ Increase `GRAPH_TTS_TIMEOUT`. Check if sanitized text is very long (truncated by sanitizer? Check `SanitizeResult.truncated`). Verify TTS API response times in Prometheus.

---

**Empty recordings (PTT mode):**
```
INFO  controller_empty_recording request_id=...
```
→ PTT was pressed but VAD gate detected only silence. Check mic input level. Verify `PCMVADGate` threshold isn't too aggressive for the recording environment. Run `recorder.run_startup_health_check()`.

---

**PCM pipeline unavailable:**
```
RuntimeError: PCM pipeline is not enabled. Set VOICE_FF_PCM_PIPELINE=1
              or configure FF_CANARY_PCT for gradual rollout.
```
→ Set `FF_PCM_PIPELINE=true` in `.env`. For gradual rollout: set `FF_CANARY_PCT=0.1` (10% of traffic).

---

**MongoDB writes dropping:**
```
WARNING  mongo_queue_full events_dropped=47
```
→ Increase `MONGO_QUEUE_DEPTH`. Increase `MONGO_BATCH_SIZE`. Verify MongoDB write latency is acceptable. Consider sharding if sustained.

---

**ATS extraction fails (bad JSON from LLM):**
```
WARNING  ats_parse_failed fallback=rule_based
```
→ ATSMode validates schema and falls back to rule-based extraction. The fallback extracts keywords from the raw intro text. Check `LLM_MODEL` — newer models have better JSON adherence.

---

### Health Check Endpoints

```bash
# Service health
curl http://localhost:8000/health

# Prometheus metrics
curl http://localhost:9090/metrics | grep ai_

# Session debug (requires X-Session-ID header)
curl -H "X-Session-ID: {sid}" http://localhost:8000/debug/session

# Recording health (desktop mode)
python -c "
import asyncio
from app.audio_essentials.recorder import run_startup_health_check
result = asyncio.run(run_startup_health_check())
print(result)
"
```

---

## 24. Development Guide

### Project Structure for Tests

```
tests/
├── unit/
│   ├── test_sanitize.py         # 27-step sanitizer pipeline
│   ├── test_qa_controller.py    # State machine transitions
│   ├── test_session_store.py    # Redis + LRU fallback
│   └── test_audio_engine.py     # PCMFormat, PCMChunk, converters
├── integration/
│   ├── test_voice_graph.py      # Full pipeline with mock nodes
│   ├── test_evaluation.py       # Eval with mock LLM
│   └── test_transcript.py       # Dual-sink writer
└── fixtures/
    ├── sample_audio.wav         # 3s test audio
    └── mock_nodes.py            # Protocol-compliant mock STT/LLM/TTS
```

### Implementing a Custom Node

Any class satisfying the relevant `Protocol` can be injected:

```python
from app.common.shared import STTNodeProtocol
from app.audio_essentials.audio_engine import PCMChunk

class MySTTNode:
    """Custom STT implementation."""

    async def transcribe(self, audio_path: str, *, request_id: str):
        # ... your implementation
        return STTResult(transcript="...", language="en", confidence=0.95)

    async def transcribe_chunk(self, chunk: PCMChunk, *, request_id: str) -> str:
        # PCM-native path
        return "..."

# Inject into voice graph
from app.orchestration.voice_graph import VoiceGraph, VoiceGraphConfig

graph = VoiceGraph(
    stt=MySTTNode(),
    cfg=VoiceGraphConfig(),
)
```

### Running a Single Pipeline Turn

```python
import asyncio
from app.orchestration.voice_graph import voice_graph, VoiceState

async def test_turn():
    state: VoiceState = {
        "audio_path": "tests/fixtures/sample_audio.wav",
        "session_id": "test-session-001",
        "request_id": "req-001",
    }
    result = await voice_graph.run(state)
    print(result["transcript"])
    print(result["response"])
    print(f"Latency: {result['pipeline_latency_s']:.2f}s")

asyncio.run(test_turn())
```

### Extending the QA State Machine

To add a new interview stage, edit `qa_controller.py`:

```python
class QAStage(str, Enum):
    GREETING  = "greeting"
    INTRO     = "intro"
    INTERVIEW = "interview"
    COMPLETE  = "complete"
    MY_STAGE  = "my_stage"    # ← add here
```

Then add routing logic in `voice_graph.py → node_llm()`:

```python
if stage == QAStage.MY_STAGE:
    return await my_stage_handler(state)
```

### Structured Event Emission

```python
from app.monitoring.observability import emit, ObsEvent, EventKind

# In any pipeline node:
emit(ObsEvent(
    kind="my_custom_event",
    service="my_service",
    session_id=session_id,
    request_id=request_id,
    ts=time.time(),
    extra={"my_field": my_value},
))
# → structlog, Prometheus, MongoDB, OTel — all at once
```

### Debugging PCM Pipeline

```python
from app.audio_essentials.audio_engine import PCMLatencyTracker, PCMWaveformAnalyzer

# Get per-stage latency report
from app.audio_essentials.recorder import get_recording_latency_report
report = get_recording_latency_report()
# {
#   "capture_start": {"mean_ms": 12.3, "p95_ms": 18.7},
#   "vad_pass":      {"mean_ms": 0.4,  "p95_ms": 0.8},
#   "wav_encode":    {"mean_ms": 1.1,  "p95_ms": 1.9},
# }

# Get waveform stats for a PCMChunk
from app.audio_essentials.audio_engine import PCMWaveformAnalyzer, PCMFormat
analyzer = PCMWaveformAnalyzer(PCMFormat.whisper())
stats = analyzer.analyze(chunk)
# WaveformStats(rms=0.12, peak=0.87, clipping_ratio=0.002, dc_offset=0.001)
```

---

## Appendix: Full Component Interaction Timeline

```
                    COMPLETE REQUEST TIMELINE (interview turn)
                    ─────────────────────────────────────────

t=0ms     PTT key held / HTTP request received
          │
          ├─ IP extracted, session_id validated (Redis lookup ~1ms)
          │
t=1ms     ─ Pipeline starts: LoadSheddingGuard.acquire()
          │
          ├─ LatencyBudget set (120s deadline)
          │
t=2ms     ─ node_stt enters:
          │   bulkheads["stt"].acquire()
          │   circuit_breaker["stt"].check()
          │
t=3ms     ─ Whisper API call starts (network + inference)
          │
t=1240ms  ─ STT returns: transcript = "What does GIL mean?"
          │   PCMConfidenceFilter passes (logprob > threshold)
          │   STTEmitter.success() → structlog + Prometheus + MongoDB + OTel
          │
t=1241ms  ─ node_llm enters:
          │   qa_controller.get_llm_input(session_id, "What does GIL mean?")
          │   → LLMInterviewInput { domain="python", level="advanced", ... }
          │
          │   [Cache check: Redis GET cache:llm:{hash}] → MISS
          │
          │   LLM API streaming call starts
          │
t=2100ms  ─ First LLM token arrives: "Excellent"
t=2250ms  ─ Token stream complete: "Excellent question! The GIL..."
          │   _ResponseValidator: valid question found ✓
          │   LLMEmitter.success() → all sinks
          │
t=2251ms  ─ node_sanitize:
          │   sanitize_for_tts(llm_response) → SanitizeResult
          │   27 steps, ~2ms CPU
          │
t=2253ms  ─ node_tts enters:
          │   PCMTTSOutputConfig check
          │   TTS API streaming starts
          │
t=2800ms  ─ First TTS audio chunk arrives
          │   PCMTTSQualityGate: RMS=0.14, peak=0.71 → OK
          │   PCMSentenceGapManager: no gap (first sentence)
          │   PCMPlaybackEnhancer: limiter pass-through (no clip)
          │   PCMOutputStream.write(chunk) → speaker starts
          │
t=3200ms  ─ TTS complete
          │   TTSEmitter.success()
          │
          ├─ [fire-and-forget async tasks]:
          │     qa_audit_bus.route_turn() →
          │       transcript_writer.write_turn()     (async queue)
          │       eval_engine.schedule_turn_eval()   (async task)
          │
t=3201ms  ─ Pipeline result built
          │   pipeline_latency_s = 3.2s
          │   PipelineEmitter.complete()
          │
t=3202ms  ─ Response returned to caller
          │   JSON with transcript, response, audio path, stage_latencies
          │
          ├─ [background, off critical path]:
t=3500ms      transcript_writer drains queue → .txt + MongoDB
t=4000ms      eval_engine scores answer → Redis score key
              adaptive_sampling: turn 4 of 5 always-score → SCORED
              budget: 847 tokens consumed (remaining: 49,153)
```

---

<div align="center">

**Built for production. Designed for clarity. Instrumented for everything.**

Total codebase: ~49,908 lines | 19 modules | 5 execution modes | 3-layer observability

</div>
