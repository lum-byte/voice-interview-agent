"""
Gunicorn configuration for the Voice Assistant API.

Run with:
    gunicorn -c gunicorn_conf.py app.endpoint.main:app

Or for development (single worker, auto-reload):
    gunicorn -c gunicorn_conf.py app.endpoint.main:app --workers 1 --reload

Requirements:
    pip install gunicorn uvicorn[standard]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REALISTIC CAPACITY MODEL — READ BEFORE TUNING WORKERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  This pipeline is almost entirely I/O-bound (waiting on OpenAI APIs).
  Observed per-request latencies from production logs:
      STT  ~1–3 s
      LLM  ~4–5 s
      TTS  ~4–7 s
      ──────────
      Total ~5–14 s end-to-end

  One Gunicorn node can THEORETICALLY handle:
    workers × GRAPH_MAX_INFLIGHT concurrent pipelines
    e.g.  8 workers × 20 inflight = 160 simultaneous pipelines

  The REAL bottleneck is your OpenAI API key rate limits, not Gunicorn.
  At 1k concurrent pipelines each firing 3 API calls simultaneously,
  you need ~3k concurrent OpenAI requests. Standard tiers don't allow that.

  Horizontal scaling checklist for high user counts:
    ✓ nginx / Caddy in front (TLS termination, connection pooling)
    ✓ Multiple Gunicorn instances across machines (K8s, ECS, etc.)
    ✓ API key pooling across instances (rotate keys per request)
    ✓ Redis Cluster (not single-node) for session + LLM cache
    ✓ GRAPH_MAX_INFLIGHT tuned per-machine based on your OpenAI quota
    ✓ HPA (Horizontal Pod Autoscaler) on CPU + request queue depth

  Rough capacity by tier (single machine, 8-core):
    OpenAI Standard   →  ~50–200 concurrent pipelines (rate-limit bound)
    OpenAI Enterprise →  ~500–1k concurrent pipelines (API-key-pool bound)
    50k concurrent    →  ~50–100 Gunicorn nodes minimum + orchestration

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
POST_FORK DESIGN — WHY EACH RESET EXISTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  With preload_app=True, Gunicorn imports the entire application in the
  master process before forking workers.  Python's fork() copies the full
  master memory space — including global singletons, open file descriptors,
  socket connections, background threads, and OTel provider globals — into
  every child.

  Each of the five reset steps below addresses a distinct post-fork hazard:

  [1] OTel API globals (opentelemetry.trace / opentelemetry.metrics)
      The OTel API uses module-level Once() sentinels to enforce "set once"
      semantics for TracerProvider and MeterProvider. These sentinels are
      copied into every worker. When the worker's lifespan calls
      set_tracer_provider() / set_meter_provider(), the Once sees it's
      already been called and emits "Overriding … is not allowed", then
      silently returns — so NO spans or OTLP metrics ever reach Tempo.
      Fix: null both the provider slot AND its Once sentinel so each worker
      installs its own real OTLP exporters during lifespan startup.

      NOTE: The MeterProvider warning ("Overriding of current MeterProvider
      is not allowed") comes from opentelemetry.metrics (the API module),
      NOT opentelemetry.sdk.metrics. The previous code reset the wrong
      module, leaving this warning unfixed.

  [2] OTel SDK provider instances (_OtelLayer internal state)
      Even after the API globals are cleared, observability.py's _OTEL
      singleton still holds stale references to the forked TracerProvider
      and MeterProvider. These providers hold BatchSpanProcessors with
      background threads that didn't survive the fork — any span export
      attempt would silently no-op or deadlock on the orphaned thread lock.
      Fix: clear _tracer_prov, _meter_prov, the tracer/meter caches, and
      the _initialized guard so bootstrap() rebuilds them from scratch.

  [3] OTel propagator global (opentelemetry.propagate)
      W3C TraceContext propagation also lives in a module-level global. A
      stale reference here won't cause crashes (propagators are stateless)
      but it does prevent re-registration on workers where the bootstrap
      order differs from the master. Cleared defensively.

  [4] _MongoWriter thread + client (the silent data-loss bug)
      threading.Thread objects and pymongo MongoClient connections are NOT
      fork-safe. After fork:
        - The writer thread simply doesn't exist in the child process.
          Its thread ID is gone. _started=True prevents _MONGO.start() from
          ever running in the worker, so MongoDB silently drops every event.
        - The inherited MongoClient holds file descriptors to the TCP socket
          that was open in the master. Using it from a forked process causes
          undefined behaviour (BSON decode errors, connection pool corruption,
          eventual silent loss of all MongoDB events).
      Fix: close the inherited client, reset all _MongoWriter state, and
      clear _bootstrap_lock/_bootstrapped so the worker's lifespan will call
      _MONGO.start() and create fresh connections from scratch.

  [5] Redis connection pool (aioredis / redis-py)
      Similar to MongoDB: any Redis connection pool open in the master is
      forked into every worker. The master may have opened connections during
      import-time initialisation. Using a forked pool causes RESP protocol
      desynchronisation across workers. We close and nullify any inherited
      pool so the worker's first await creates a clean pool.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import multiprocessing
import os

# ── binding ───────────────────────────────────────────────────────────────────

host = os.getenv("HOST", "0.0.0.0")
port = os.getenv("PORT", "8000")
bind = f"{host}:{port}"

# ── workers ───────────────────────────────────────────────────────────────────

worker_class = "uvicorn.workers.UvicornWorker"

# UvicornWorker is async — each worker runs a full asyncio event loop and
# handles thousands of concurrent I/O-bound connections without blocking.
# For this pipeline (almost entirely waiting on OpenAI APIs): 1 per CPU core.
workers = int(os.getenv("GUNICORN_WORKERS", multiprocessing.cpu_count()))

# threads=1 is correct for a pure-async app. asyncio handles all concurrency.
threads = int(os.getenv("GUNICORN_THREADS", "1"))

# 2048 gives real headroom for connection bursts / retry storms.
backlog = int(os.getenv("GUNICORN_BACKLOG", "2048"))

# SO_REUSEPORT: kernel-level load balancing across workers — avoids the
# thundering herd through the master's single accept() loop.
reuse_port = True

# Load the app in the master before forking workers.
# Benefits: CoW RAM sharing, faster worker startup.
# The post_fork() hook below handles all fork-safety hazards this creates.
preload_app = True

# ── timeouts ──────────────────────────────────────────────────────────────────

# Must be > PIPELINE_TIMEOUT (120s). 150s gives 30s headroom for network
# variance. Gunicorn SIGKILLs any worker that exceeds this.
timeout = int(os.getenv("GUNICORN_TIMEOUT", "150"))

# Matches timeout so any in-flight pipeline can always finish before SIGKILL.
graceful_timeout = int(os.getenv("GUNICORN_GRACEFUL_TIMEOUT", "150"))

# Behind nginx: 5s is fine (nginx manages browser keepalives).
# Directly internet-exposed: drop to 2s to free file descriptors faster.
keepalive = int(os.getenv("GUNICORN_KEEPALIVE", "5"))

# ── lifecycle ─────────────────────────────────────────────────────────────────

# 5000 with large jitter staggers restarts across workers so they never
# all restart simultaneously (avoids a thundering-herd on your OpenAI quota).
max_requests = int(os.getenv("GUNICORN_MAX_REQUESTS", "5000"))
max_requests_jitter = int(os.getenv("GUNICORN_MAX_REQUESTS_JITTER", "500"))

# ── security hardening ────────────────────────────────────────────────────────

limit_request_line = int(os.getenv("GUNICORN_LIMIT_REQUEST_LINE", "4094"))
limit_request_fields = int(os.getenv("GUNICORN_LIMIT_REQUEST_FIELDS", "100"))
limit_request_field_size = int(os.getenv("GUNICORN_LIMIT_REQUEST_FIELD_SIZE", "8192"))

# ── proxy / load balancer ─────────────────────────────────────────────────────

# CRITICAL for session IP tracking: without this every request appears to come
# from the proxy IP and SESSION_MAX_IP_CHANGES fires on every request.
# "127.0.0.1"  — nginx on same machine (safe default)
# "10.0.0.0/8" — VPC load balancer → EC2/pod
# "*"           — trust all (only inside a fully controlled network)
forwarded_allow_ips = os.getenv("GUNICORN_FORWARDED_ALLOW_IPS", "127.0.0.1")
secure_scheme_headers = {"X-Forwarded-Proto": "https"}

# ── logging ───────────────────────────────────────────────────────────────────

loglevel = os.getenv("GUNICORN_LOG_LEVEL", "info")
accesslog = os.getenv("GUNICORN_ACCESS_LOG", "-")   # "-" = stdout
errorlog  = os.getenv("GUNICORN_ERROR_LOG", "-")    # "-" = stderr
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s %(f)s %(a)s %(D)sμs'

# ── process naming ────────────────────────────────────────────────────────────

proc_name = "voice-assistant"


# ── helpers ───────────────────────────────────────────────────────────────────

def _reset_otel_api_globals(server) -> list[str]:
    """
    Reset the OpenTelemetry API-layer provider globals.

    Both opentelemetry.trace and opentelemetry.metrics use a Once() sentinel
    to enforce "set only once" semantics for the global provider. After fork,
    these sentinels are already tripped in every worker, so lifespan's
    set_tracer_provider() / set_meter_provider() silently no-ops and no spans
    or OTLP metrics ever leave the process.

    We null both the provider slot and the sentinel so each worker installs
    fresh OTLP exporters during lifespan startup.

    Returns a list of warning strings for any resets that didn't fully apply
    (e.g. attribute names changed in a newer OTel version).
    """
    warnings = []

    try:
        from opentelemetry.util._once import Once
    except ImportError:
        # Older OTel versions (< 1.15) don't have opentelemetry.util._once.
        # Fall back to a duck-typed reset: try to create a fresh Once-like
        # object by resetting the boolean flag the old implementation used.
        Once = None

    # ── Tracer provider (opentelemetry.trace) ─────────────────────────────────
    try:
        import opentelemetry.trace as _ot_trace

        # Null the cached provider reference.
        if hasattr(_ot_trace, "_TRACER_PROVIDER"):
            _ot_trace._TRACER_PROVIDER = None
        else:
            warnings.append("opentelemetry.trace._TRACER_PROVIDER not found — "
                            "OTel version may have renamed this attribute.")

        # Reset the Once sentinel that gates set_tracer_provider().
        if hasattr(_ot_trace, "_TRACER_PROVIDER_SET_ONCE"):
            _ot_trace._TRACER_PROVIDER_SET_ONCE = Once() if Once else _ot_trace._TRACER_PROVIDER_SET_ONCE
            if Once is None:
                # Attempt the legacy boolean approach used in otel-api < 1.15.
                sentinel = _ot_trace._TRACER_PROVIDER_SET_ONCE
                for attr in ("_done", "_called", "_has_been_called"):
                    if hasattr(sentinel, attr):
                        setattr(sentinel, attr, False)
        else:
            warnings.append("opentelemetry.trace._TRACER_PROVIDER_SET_ONCE not found.")

        # Some versions also keep a _PROXY_TRACER_PROVIDER that must be cleared
        # so the proxy's get_tracer() calls route to the new real provider.
        if hasattr(_ot_trace, "_PROXY_TRACER_PROVIDER"):
            _ot_trace._PROXY_TRACER_PROVIDER = None

    except Exception as exc:
        warnings.append(f"opentelemetry.trace reset raised: {exc}")

    # ── Meter provider (opentelemetry.metrics — the API module) ───────────────
    # IMPORTANT: The "Overriding of current MeterProvider is not allowed"
    # warning comes from opentelemetry.metrics (API), NOT opentelemetry.sdk.metrics.
    # The previous code reset the SDK module — the wrong one.
    #
    # In opentelemetry-api >= 1.12 the globals moved to the private submodule
    # opentelemetry.metrics._internal — the top-level opentelemetry.metrics
    # module re-exports set_meter_provider() from there but does NOT hold the
    # provider slots itself. We probe all known locations across versions so
    # this works regardless of which otel-api release is installed:
    #
    #   < 1.12 : globals on opentelemetry.metrics directly
    #   >= 1.12: globals on opentelemetry.metrics._internal
    #   some intermediate builds also used opentelemetry._metrics
    _meter_reset_done = False
    for _metrics_mod_path in (
        "opentelemetry.metrics._internal",   # otel-api >= 1.12  ← most likely
        "opentelemetry._metrics",            # some intermediate builds
        "opentelemetry.metrics",             # otel-api < 1.12
    ):
        try:
            import importlib as _importlib
            _ot_metrics_mod = _importlib.import_module(_metrics_mod_path)

            _found_provider   = hasattr(_ot_metrics_mod, "_METER_PROVIDER")
            _found_set_once   = hasattr(_ot_metrics_mod, "_METER_PROVIDER_SET_ONCE")
            _found_proxy      = hasattr(_ot_metrics_mod, "_PROXY_METER_PROVIDER")

            if not (_found_provider or _found_set_once):
                # This submodule doesn't hold the provider globals — try next.
                continue

            if _found_provider:
                _ot_metrics_mod._METER_PROVIDER = None

            if _found_set_once:
                _ot_metrics_mod._METER_PROVIDER_SET_ONCE = Once() if Once else _ot_metrics_mod._METER_PROVIDER_SET_ONCE
                if Once is None:
                    # Legacy boolean sentinel (otel-api < 1.12).
                    _sentinel = _ot_metrics_mod._METER_PROVIDER_SET_ONCE
                    for _attr in ("_done", "_called", "_has_been_called"):
                        if hasattr(_sentinel, _attr):
                            setattr(_sentinel, _attr, False)

            if _found_proxy:
                _ot_metrics_mod._PROXY_METER_PROVIDER = None

            _meter_reset_done = True
            break  # found and reset — no need to probe further

        except (ImportError, ModuleNotFoundError):
            continue  # submodule doesn't exist in this otel-api version
        except Exception as exc:
            warnings.append(f"{_metrics_mod_path} reset raised: {exc}")
            break

    if not _meter_reset_done:
        # None of the known submodules held the provider globals.
        # The warning will still fire — surface it so it's actionable.
        warnings.append(
            "MeterProvider global not found in any known submodule "
            "(opentelemetry.metrics._internal / opentelemetry._metrics / "
            "opentelemetry.metrics). The 'Overriding of current MeterProvider' "
            "warning may still appear. Check your opentelemetry-api version and "
            "update the submodule probe list above."
        )

    # Also clear the proxy on the top-level opentelemetry.metrics namespace
    # regardless of where the canonical globals live — belt-and-suspenders.
    try:
        import opentelemetry.metrics as _ot_metrics_top
        if hasattr(_ot_metrics_top, "_PROXY_METER_PROVIDER"):
            _ot_metrics_top._PROXY_METER_PROVIDER = None
    except Exception:
        pass

    # ── Propagator (opentelemetry.propagate) ──────────────────────────────────
    # Stateless in practice, but clearing it lets each worker re-register
    # the W3C TraceContext propagator cleanly and prevents stale references
    # to the forked composite propagator object.
    try:
        import opentelemetry.propagate as _ot_propagate

        if hasattr(_ot_propagate, "_DEFAULT_PROPAGATOR"):
            _ot_propagate._DEFAULT_PROPAGATOR = None
        # Also clear the global composite propagator used by inject/extract.
        if hasattr(_ot_propagate, "_propagator"):
            _ot_propagate._propagator = None

    except Exception as exc:
        warnings.append(f"opentelemetry.propagate reset raised: {exc}")

    return warnings


def _reset_otel_sdk_layer(server) -> list[str]:
    """
    Reset observability.py's _OtelLayer singleton.

    Even after the API globals are cleared, the _OTEL singleton in
    observability.py still holds:
      - stale TracerProvider / MeterProvider object references
      - a tracer cache (_tracers dict) and meter cache (_meters dict)
        full of objects tied to the dead providers
      - _initialized=True, which prevents bootstrap() from reinitialising

    Any call to _OTEL.get_tracer() after fork but before lifespan would
    return a tracer backed by a BatchSpanProcessor whose background thread
    doesn't exist — spans would queue up forever and never be exported.

    Returns a list of warning strings for partial resets.
    """
    warnings = []
    try:
        import app.monitoring.observability as _obs
        import threading as _threading

        otel = _obs._OTEL  # noqa: SLF001

        # Clear provider references so they can't be used post-fork.
        otel._tracer_prov = None
        otel._meter_prov  = None

        # Clear tracer/meter caches — they hold references into the dead providers.
        otel._tracers.clear()
        otel._meters.clear()

        # Clear propagator reference.
        otel._propagator = None

        # Allow bootstrap() to reinitialise this worker's providers.
        otel._initialized = False
        otel._lock = _threading.Lock()

    except AttributeError as exc:
        warnings.append(f"_OtelLayer field missing (renamed?): {exc}")
    except Exception as exc:
        warnings.append(f"_OtelLayer reset raised: {exc}")

    return warnings


def _reset_bootstrap_flags(server) -> list[str]:
    """
    Reset observability.py's module-level bootstrap guard.

    bootstrap() is idempotent via (_bootstrapped, _bootstrap_lock). After
    fork both are copied from the master. If the master ever called bootstrap()
    (which it may during import in preload_app mode), _bootstrapped=True in
    every worker and the worker's lifespan bootstrap() call is a no-op —
    leaving OTel uninitialised, MongoDB unconnected, and Grafana not written.
    """
    warnings = []
    try:
        import app.monitoring.observability as _obs
        import threading as _threading

        _obs._bootstrapped    = False          # type: ignore[attr-defined]
        _obs._bootstrap_lock  = _threading.Lock()  # type: ignore[attr-defined]

    except AttributeError as exc:
        warnings.append(f"bootstrap flag field missing (renamed?): {exc}")
    except Exception as exc:
        warnings.append(f"bootstrap flag reset raised: {exc}")

    return warnings


def _reset_mongo_writer(server) -> list[str]:
    """
    Reset _MongoWriter so the worker starts a fresh connection + thread.

    After fork:
      - The writer thread simply does not exist in the child. Its OS thread
        ID is gone. _started=True blocks _MONGO.start() forever → every
        MongoDB event silently dropped in workers 2-N.
      - The inherited MongoClient holds TCP socket FDs opened by the master.
        Using them from a forked process causes BSON decode errors and
        connection pool corruption. Must be closed before the FD is reused.

    We close the inherited client (best-effort), then reset all _MongoWriter
    fields to their __init__ defaults so start() runs cleanly in the worker.
    """
    warnings = []
    try:
        import app.monitoring.observability as _obs
        import queue as _queue
        import threading as _threading

        writer = _obs._MONGO  # noqa: SLF001

        # Close the inherited MongoClient. This is best-effort — the client
        # may be in an inconsistent state if the master was mid-operation
        # at the point of fork(), so we swallow any error here.
        if writer._client is not None:
            try:
                writer._client.close()
            except Exception:
                pass

        # Reset all fields to __init__ defaults so start() runs cleanly.
        writer._client  = None
        writer._col     = None
        writer._thread  = None
        writer._started = False
        writer._drops   = 0
        writer._lock    = _threading.Lock()

        # Drain the inherited queue — events queued before fork belong to the
        # master process's context (different PIDs, no active thread to drain).
        # Carrying them into the worker risks duplicate inserts if the master
        # also drains its queue, and wastes the worker's first flush cycle.
        new_q: _queue.Queue = _queue.Queue(maxsize=writer.MONGO_QUEUE_DEPTH)
        writer._queue = new_q

    except AttributeError as exc:
        warnings.append(f"_MongoWriter field missing (renamed?): {exc}")
    except Exception as exc:
        warnings.append(f"_MongoWriter reset raised: {exc}")

    return warnings


def _reset_redis_pools(server) -> list[str]:
    """
    Close any Redis connection pools inherited from the master.

    redis-py (sync) and aioredis / redis.asyncio open TCP sockets during
    pool creation. fork() duplicates these sockets into every worker. The
    master and all workers now share the same underlying TCP connections —
    concurrent RESP commands from different processes interleave on the wire,
    causing protocol desynchronisation and silent data corruption.

    We close inherited pools here. The worker's first await on a Redis
    command creates a fresh pool from scratch (lazy init).

    Note: asyncio event loop has NOT been created yet at post_fork time
    (UvicornWorker creates it later). Async close methods are not callable
    here, so we use the sync close() path where available.
    """
    warnings = []

    # ── redis-py (sync client, used by session_store and LLM cache) ───────────
    try:
        from app.common import shared as _shared

        for attr_name in ("_redis_pool", "_sync_redis", "_redis_client", "_pool"):
            pool = getattr(_shared, attr_name, None)
            if pool is not None:
                try:
                    pool.close()
                    if hasattr(pool, "disconnect"):
                        pool.disconnect()
                except Exception:
                    pass
                setattr(_shared, attr_name, None)

    except ImportError:
        _shared = None  # shared not imported yet — nothing to close
    except Exception as exc:
        warnings.append(f"redis-py pool reset raised: {exc}")

    # ── aioredis / redis.asyncio (async client) ───────────────────────────────
    # These are typically lazily created on first await, so they shouldn't
    # exist yet at post_fork time. Cleared defensively in case an import-time
    # side effect created one.
    try:
        import redis.asyncio as _redis_async

        # redis.asyncio doesn't expose a module-level pool singleton, but some
        # applications cache one. Check the most common locations.
        for module_path in (
            "app.common.shared",
            "app.user_tracking.session_service.session_store",
            "app.orchestration.voice_graph",
        ):
            try:
                import importlib
                mod = importlib.import_module(module_path)
                for attr in dir(mod):
                    obj = getattr(mod, attr, None)
                    if isinstance(obj, (_redis_async.Redis, _redis_async.ConnectionPool)):
                        try:
                            # Sync close path — safe before event loop exists.
                            if hasattr(obj, "connection_pool"):
                                obj.connection_pool.disconnect()
                            elif hasattr(obj, "disconnect"):
                                obj.disconnect()
                        except Exception:
                            pass
                        try:
                            setattr(mod, attr, None)
                        except (AttributeError, TypeError):
                            pass
            except (ImportError, ModuleNotFoundError):
                pass

    except ImportError:
        _redis_async = None  # redis.asyncio not installed
    except Exception as exc:
        warnings.append(f"redis.asyncio pool reset raised: {exc}")

    return warnings


# ── hooks ─────────────────────────────────────────────────────────────────────


def on_starting(server):
    server.log.info(
        f"Starting Voice Assistant API — {workers} workers × {threads} threads "
        f"on {bind} (timeout={timeout}s, preload_app={preload_app})"
    )


def post_fork(server, worker):
    """
    Called in each worker immediately after Gunicorn forks from the master.

    Resets all state that is unsafe after fork so the worker's FastAPI
    lifespan can call bootstrap() cleanly and build fresh connections,
    threads, and OTel providers.

    The five reset steps — and WHY each is needed — are documented in the
    module-level docstring above and in each helper's own docstring.
    """
    all_warnings: list[str] = []

    # Reset order matters:
    #   1. API globals first (so nothing can call set_*_provider with stale Once)
    #   2. SDK / singleton layer next (clears stale provider refs + caches)
    #   3. Bootstrap flags (allows lifespan to re-run bootstrap())
    #   4. MongoDB writer (thread + client both unsafe post-fork)
    #   5. Redis pools (TCP sockets unsafe post-fork)

    all_warnings += _reset_otel_api_globals(server)
    all_warnings += _reset_otel_sdk_layer(server)
    all_warnings += _reset_bootstrap_flags(server)
    all_warnings += _reset_mongo_writer(server)
    all_warnings += _reset_redis_pools(server)

    for w in all_warnings:
        server.log.warning(f"[post_fork:{worker.pid}] {w}")

    server.log.info(
        f"Worker {worker.pid} forked — "
        f"OTel providers reset, MongoDB writer reset, Redis pools closed. "
        f"Ready for lifespan bootstrap."
        + (f" ({len(all_warnings)} partial-reset warning(s))" if all_warnings else "")
    )


def child_exit(server, worker):
    """
    Called in the MASTER when a worker exits (cleanly or via signal).

    Best place to decrement any master-side accounting of live worker count.
    Nothing here right now but the hook is wired up so you can add alerting
    (e.g. PagerDuty webhook when workers crash faster than max_requests would
    explain) without touching gunicorn's internal worker management.
    """
    server.log.info(f"Worker {worker.pid} exited.")


def worker_abort(worker):
    """
    Called when a worker is killed by the timeout watchdog (SIGABRT).

    Distinct from worker_exit — this means a request was actively in flight
    and took longer than `timeout` seconds. Worth alerting on in production:
    it means a user's pipeline was killed mid-response.

    Note: Gunicorn passes only `worker` here (arity=1), unlike most other
    hooks which receive (server, worker). No server.log available — use
    the worker's own log attribute instead.
    """
    worker.log.warning(
        f"Worker {worker.pid} aborted by timeout watchdog "
        f"(timeout={timeout}s) — a pipeline was killed mid-flight. "
        f"Consider raising GUNICORN_TIMEOUT or optimising the slow stage."
    )


def on_exit(server):
    server.log.info("Voice Assistant API shutting down.")