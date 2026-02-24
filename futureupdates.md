# FUTUREUPDATES.md — Voice Interview Agent Roadmap

> **Phase**: Final maturation  
> **Constraint**: No scope creep. No 180° pivots. Every item deepens the existing rail.  
> **Rail**: `controller → LLM(question engine) → candidate answer → controller → evaluation LLM`

---

## Guiding principle

The system already does the hard thing: structured, domain-rotated, difficulty-scaled technical interviews with real-time voice I/O and fire-and-forget evaluation. The remaining work is not about adding new capabilities — it's about making the existing loop **tighter**, **harder to break**, and **cheaper to run at scale**.

Three pillars, in priority order:

```
1. LLM TRACK     — keep the question engine focused, never off-script
2. CONCURRENCY   — N sessions on one box without corruption or starvation
3. SERVER LOAD   — graceful degradation, not graceful collapse
```

---

## PILLAR 1 — LLM TRACK

The question-engine LLM receives exactly 5 fields: `domain | level | last_answer | last_question | domain_switched`. That's the narrowest contract possible. The risk isn't the LLM seeing too much — it's the LLM **drifting** with what little it gets.

---

### 1.1 · Answer-quality gate before the question LLM

**Where it fits**: Between "candidate answer" and "Question LLM" in the loop.

Right now every candidate answer, no matter how empty or off-topic, reaches the question LLM. The LLM then has to infer from a garbage `last_answer` what to ask next — and sometimes it adjusts difficulty based on that garbage.

Add a lightweight classifier (can be rule-based, doesn't need a model) that tags the answer before it reaches the question LLM:

```
SOLID_ANSWER    → pass through, business as usual
PARTIAL_ANSWER  → tag in user_parts so LLM can probe deeper on same subtopic
OFF_TOPIC       → tag in user_parts so LLM stays on domain, doesn't adjust difficulty
SILENCE / EMPTY → skip the question LLM entirely, use a fallback probe from the bank
```

This is not a new pipeline stage. It's a 4-way branch inside `QAControllerV2.get_llm_input()` that annotates the existing `user_parts` payload. The LLM already respects `domain_switched` as a routing hint — this adds `answer_quality` as a second hint.

**Why it matters for LLM track**: Without this, a candidate saying "I don't know" five times causes the difficulty scaler to ratchet down to trivial questions, and the LLM starts asking "What does HTML stand for?" to a senior candidate. The quality gate keeps the LLM on its intended difficulty curve.

---

### 1.2 · Response structure enforcement (not just word count)

`_ResponseValidator` currently checks word count (80-word cap) and multi-question detection. Add two more checks that don't require a model call:

**a) Preamble detection** — The LLM sometimes opens with "Great answer!" or "That's a good point about..." before asking the next question. This is chat-mode bleed. A regex check for affirmation patterns at the start of the response, followed by stripping them, keeps the interviewer voice clean. The candidate hears only the question.

**b) Question-type drift detection** — The system prompt says "technical question." Occasionally the LLM drifts into behavioral questions ("Tell me about a time when...") or meta-questions ("How do you feel about..."). A lightweight keyword classifier on the generated question can flag this and trigger a re-generation or fallback. Log it — if it happens often for a specific domain, the system prompt for that domain needs tightening.

Both of these live inside `_ResponseValidator` in `qa_controller.py`. No new files, no new pipeline stages.

---

### 1.3 · Per-domain system prompt specialization

Currently `INTERVIEWER_SYSTEM_PROMPT` and `DOMAIN_SWITCH_SYSTEM_PROMPT` are generic across all domains. The LLM gets domain context only through the `domain: Python` line in the user message.

Add a `DOMAIN_PROMPTS: dict[str, str]` registry (keyed by domain slug) that appends domain-specific interviewing guidance to the system prompt. Examples:

- **Python**: "Focus on language-specific idioms, not generic OOP. Prefer questions about generators, context managers, GIL implications, and type annotations over questions that apply equally to Java."
- **System Design**: "Ask about tradeoffs, not just architecture. Every question should have a 'what happens when X fails' follow-up dimension."
- **SQL**: "Prefer questions that require thinking about query plans and indexes, not just syntax recall."

These are 2–3 sentence suffixes appended inside `LLMInputBuilder.build()` based on `domain_label`. They're stored alongside `DOMAIN_REGISTRY` in `qa_controller.py` — no new module.

**Why it matters**: The generic prompt produces generic questions. A Python interviewer that asks "What is polymorphism?" is technically on-domain but practically useless. Domain-specific prompt suffixes are the cheapest way to improve question quality without touching the model or the architecture.

---

### 1.4 · Fingerprint-aware difficulty transitions

`_DifficultyScaler` adjusts difficulty based on `q_index` (position within the domain). But it doesn't consider the fingerprint history — whether the LLM is accidentally circling back to the same subtopic at different difficulty levels.

Feed the last 3 fingerprints from `_QuestionFingerprinter.get_recent_fingerprints()` into the difficulty decision. If the last 3 questions all fingerprinted to the same cluster (e.g., all about "list comprehensions" in Python), force the next question to target a different subtopic regardless of difficulty. This is a 5-line addition inside `LLMInputBuilder.build()` where `fp_hints` is already being fetched.

---

## PILLAR 2 — CONCURRENCY

The system handles concurrent sessions through Redis-backed state, asyncio semaphores, and session locks. The gaps are at the boundaries between these mechanisms.

---

### 2.1 · Session-level pipeline serialization

`voice_graph._node_llm_qa_path()` processes one turn at a time per invocation, but nothing prevents two overlapping WebSocket messages for the same session from spawning two concurrent graph runs. The `_session_lock` in `qa_controller` protects the state write, but the LLM call itself is outside the lock — two concurrent LLM calls for the same session waste tokens and can produce conflicting questions.

Add a per-session `asyncio.Lock` at the `voice_graph` layer (not inside `qa_controller`) that serializes entire graph runs for the same `session_id`. This is distinct from the `_session_lock` which only protects Redis writes. The graph-level lock ensures the full STT→LLM→TTS pipeline runs atomically per session.

Implementation: a `defaultdict(asyncio.Lock)` in `VoiceGraph`, keyed by `session_id`, acquired at the top of `_node_llm_qa_path()` and released at the end. Weak-ref the locks so they're garbage-collected when the session ends.

---

### 2.2 · Eval backpressure signal

`QAAuditBus` dispatches evaluation tasks fire-and-forget via `asyncio.create_task()`. The DLQ handles failures, but there's no backpressure — if 50 sessions all rotate domains simultaneously, 50 eval tasks spawn at once and hit the OpenAI rate limiter wall.

Add a bounded `asyncio.Semaphore` (configurable, default 5) to `QAAuditBus._dispatch_eval()` that caps concurrent eval API calls. Tasks that can't acquire the semaphore queue in FIFO order (asyncio does this natively). This is the same pattern already used in STT/TTS/LLM nodes — extend it to eval.

The semaphore count should be separate from `EVAL_MAX_CONCURRENT` in `evaluation_engine.py` because the audit bus dispatches domain-batch evals (multiple Q/A pairs per call), not per-turn evals. The bus semaphore caps concurrent domain-batch dispatches; the engine semaphore caps concurrent scoring calls within each batch.

---

### 2.3 · Session-aware load shedding

The load shedder in `voice_graph.py` counts total in-flight requests across all sessions. Under pressure, it rejects the newest request — which could be a turn from a 20-minute-deep interview, while admitting a brand-new session's first turn.

Make the shedder session-aware: once a session has completed its intro and entered the interview stage, its turns get priority over new session registrations. Implementation: check `QAStage` from the session doc (already loaded) before the shed decision. New sessions are shed first; in-progress interviews are shed last.

This is a policy change inside the existing `_LoadShedder` class in `voice_graph.py`, not a new component.

---

### 2.4 · Graceful session handoff on worker restart

Gunicorn worker recycling (`max_requests`) kills the Python process mid-session. The session state in Redis survives, but any in-flight `asyncio.shield()`-protected `commit_turn()` is interrupted, and the next worker that picks up the session may see a stale `current_question`.

Add a pre-shutdown hook in `gunicorn_conf.py` that:
1. Sets a `draining` flag on the `VoiceGraph` instances (already `_ALL_GRAPHS` in `main.py`)
2. Waits up to N seconds for in-flight `commit_turn()` calls to complete (they're already `shield()`ed)
3. Only then allows the worker to exit

Gunicorn's `worker_exit` hook is the right place. The drain timeout should be `graph_llm_timeout + 2s` — enough for the slowest LLM call to finish and commit.

---

## PILLAR 3 — SERVER LOAD

The infra stack (Redis, Prometheus, Tempo, Loki, Grafana, OTel, Mongo) is already production-grade. The load concerns are in the application layer — specifically how the app behaves when external APIs slow down.

---

### 3.1 · Adaptive LLM timeout based on queue depth

`graph_llm_timeout` is a static value from settings. When 40 sessions are active and the OpenAI API is slow, every session waits the full timeout before failing — then all 40 retry simultaneously, creating a thundering herd.

Make the effective LLM timeout adaptive: `effective_timeout = base_timeout × (1 - (inflight / max_inflight) × 0.5)`. When the pipeline is 80% full, timeouts drop to 60% of base — sessions fail faster, freeing capacity for others. This is a 10-line change in `VoiceGraph._run_with_timeout()`.

The `LatencyBudget` already propagates deadline awareness per-request. The adaptive timeout adjusts the budget at the source so the entire downstream chain (LLM → TTS) tightens proportionally.

---

### 3.2 · Redis connection pool partitioning

Currently `qa_controller`, `session_store`, `evaluation_engine`, and `LLM_service` all share the same Redis connection pool (`REDIS_MAX_CONN=200`). A stampede of LLM cache lookups can starve session store operations, causing session validation timeouts on perfectly healthy requests.

Partition the pool: LLM cache gets 60% (120 connections), session store gets 25% (50 connections), eval gets 10% (20 connections), reserve 5% (10 connections) for admin/monitoring. Each pool is a separate `aioredis.ConnectionPool` instance. The `_QARedisClient` in `qa_controller.py` already wraps Redis access — it just needs to accept a pool argument instead of creating its own.

This is a settings change (`REDIS_POOL_LLM_MAX`, `REDIS_POOL_SESSION_MAX`, `REDIS_POOL_EVAL_MAX`) and a wiring change in each module's Redis client constructor. No architectural change.

---

### 3.3 · TTS sentence coalescing under load

When the pipeline is under load, `AsyncSentenceBuffer` in `TTS_service.py` flushes every sentence boundary. Each flush is a separate TTS API call. Under load, this means 5–8 API calls per turn (one per sentence).

Add a load-aware coalescing mode: when `tts_active_requests` exceeds a threshold (say, 70% of `tts_max_concurrent`), the sentence buffer accumulates 2–3 sentences before flushing. First-byte latency increases by ~1s, but API call count drops by 60%, and the TTS rate limiter stops being the bottleneck.

This is a conditional inside `AsyncSentenceBuffer.__anext__()` that checks the gauge value before deciding whether to flush. The gauge already exists (`_active`).

---

### 3.4 · Evaluation sampling rate tied to server load

`evaluation_engine.py` has adaptive sampling (full eval for first N turns, then 1-in-K). But the sampling rate is per-session, not per-server. When 50 sessions are active, 50 × (first N turns) all hit the eval API simultaneously.

Add a server-level eval throttle: when `stt_active_requests + llm_active_requests` exceeds 80% of capacity, increase the sampling K globally (e.g., from 1-in-3 to 1-in-6). This doesn't affect eval quality meaningfully — the most informative turns (first few, domain switches) are already force-evaluated regardless of sampling.

Implementation: `EvalEngine` checks the Prometheus gauge values (already exported) before each scoring decision. 5 lines in `_should_evaluate()`.

---

## TOOLING — What to add and where

These are not new features. They're developer and operator tools that make the existing system easier to debug, tune, and observe.

---

### T1 · Session replay CLI

**What**: A command-line tool that takes a `session_id` and reconstructs the full interview timeline from Redis + MongoDB + transcript files.

**Output**:
```
SESSION d056d716  |  Python (sr) → Java (mid) → System Design (sr)
─────────────────────────────────────────────────────────────────
[00:00] GREETING  → "Welcome to the interview..."
[00:12] INTRO     → ATS extracted: Python, Java, System Design | level: senior
[00:15] Q1 Python → "Explain the difference between..." (difficulty: medium)
[00:45] A1        → "In Python, generators use..." (quality: SOLID, 47 words)
[00:46] EVAL      → {correctness: 4, depth: 3, relevance: 5} (sampled: yes)
[01:02] Q2 Python → "How does the GIL affect..." (difficulty: hard)
...
[12:30] COMPLETE  → 14 questions, 3 domains, avg eval 3.8/5
```

**Where it lives**: A standalone script in `tools/session_replay.py`. Reads from `QAAuditBus.get_audit_turns()`, `evaluation_engine` Redis keys, and the transcript `.txt` file. No new dependencies.

**Why**: Right now debugging a weird session requires grep across 4 log sources. This tool stitches them into one timeline.

---

### T2 · Prompt drift monitor

**What**: A background task (runs once per hour, not per-request) that samples the last 100 LLM-generated questions from MongoDB/Redis audit records and checks for:
- Preamble ratio (% of questions that start with affirmation)
- Behavioral question ratio (% that are "tell me about a time...")
- Duplicate fingerprint ratio (% that repeat within the same session)
- Average word count vs. the 80-word cap
- Domain adherence (does the question actually match the domain label?)

**Output**: Prometheus gauges (`llm_prompt_drift_preamble_ratio`, etc.) that feed into a Grafana panel.

**Where it lives**: A method on `QAAuditBus` that runs on a `asyncio` timer, reading from the existing audit store. Results are emitted via `observability.emit()` — same pipeline as everything else.

**Why**: The LLM can drift slowly over thousands of sessions in ways that no single-session test catches. This is the smoke detector.

---

### T3 · Load test harness

**What**: A script that simulates N concurrent interview sessions end-to-end — including the WebSocket handshake, audio upload, and multi-turn conversation loop. Uses pre-recorded audio snippets (or silence + text injection for STT bypass).

**Where it lives**: `tools/loadtest.py`. Uses `httpx` + `websockets` against the live API. Configurable: `--sessions 50 --turns-per-session 10 --ramp-up-seconds 30`.

**Why**: The system has semaphores, rate limiters, circuit breakers, load shedders, and bulkheads — but nobody has tested what happens when all of them activate simultaneously on the same box. This tool answers that question before production does.

---

### T4 · Domain question bank warm-up

**What**: A startup task that pre-generates 5 questions per domain per difficulty level using the question LLM, stores them in Redis with a `fallback:` prefix, and refreshes them daily.

**Where it lives**: Called from the FastAPI lifespan hook in `main.py`. Uses the existing `LLMNode.stream_question()` with a synthetic `LLMInterviewInput`.

**Why**: `_ResponseValidator` already has a fallback question bank for when the LLM fails. But the current bank is static and hand-written. A warm-up pass produces domain-specific, difficulty-appropriate fallbacks that are indistinguishable from live questions. If the LLM goes down mid-interview, the candidate never knows.

---

### T5 · Eval calibration snapshot

**What**: A weekly job that takes 20 randomly sampled (question, answer, eval_score) triples from the audit store, re-evaluates them with the scoring model, and computes the score drift between original and re-evaluation.

**Where it lives**: `tools/eval_calibration.py`. Reads from eval Redis keys, calls `evaluation_engine.score_turn()` directly.

**Why**: The scoring model's behavior can shift between OpenAI model versions. A 0.5-point drift in average score across a model update means every candidate in that window was mis-scored. This catches it within a week instead of when a hiring manager complains.

---

## What NOT to add

These are things that look tempting but would be scope creep or a directional pivot:

| Temptation | Why not |
|---|---|
| **Multi-model LLM routing** (Claude for questions, GPT for eval) | The architecture supports it (protocol-based nodes), but the complexity of maintaining two prompt sets, two rate limiters, and two failure modes isn't justified until single-model quality hits a ceiling. You're not there yet. |
| **Real-time candidate emotion detection** (voice tone analysis) | This is a different product. The current system evaluates answers, not affect. Adding emotion detection changes the evaluation contract and introduces bias concerns that require months of calibration. |
| **Collaborative interviewing** (multiple AI interviewers) | The session model is 1:1. Making it N:1 requires rethinking session state, turn ordering, and the evaluation contract. This is a v2 rewrite, not a v1.x maturation. |
| **Custom question upload by hiring managers** | This breaks the "LLM generates questions" contract and turns the system into a question delivery platform. If you go here, you're building a different product. |
| **Video integration** (webcam feed) | The architecture is voice-in, voice-out. Adding video means a new STT-equivalent for facial recognition, new privacy concerns, new infra (WebRTC), and new evaluation dimensions. This is a separate product line. |

---

## Ship order

```
IMMEDIATE (before ship):
  patchfix.md bugs #1 and #2

NEXT SPRINT:
  patchfix.md bugs #3–6
  1.1  Answer-quality gate
  2.1  Session-level pipeline serialization
  3.2  Redis connection pool partitioning

SPRINT +2:
  1.2  Response structure enforcement
  1.3  Per-domain system prompt specialization
  2.2  Eval backpressure signal
  T1   Session replay CLI

SPRINT +3:
  1.4  Fingerprint-aware difficulty transitions
  2.3  Session-aware load shedding
  3.1  Adaptive LLM timeout
  3.3  TTS sentence coalescing
  T3   Load test harness

SPRINT +4:
  2.4  Graceful session handoff
  3.4  Eval sampling tied to server load
  T2   Prompt drift monitor
  T4   Domain question bank warm-up
  T5   Eval calibration snapshot
```
