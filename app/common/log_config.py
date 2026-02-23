"""
log_config.py  —  Two-mode structured logging.

  LOG_MODE=standard   →  Rich-coloured console  +  JSON written to LOG_FILE.
  LOG_MODE=verbose    →  JSON to stdout only (default). Safe for CI / prod
                          log aggregators (Datadog, Loki, CloudWatch).
  LOG_MODE=both       →  Alias for standard (dual-sink is always active in
                          standard mode).

Environment variables
─────────────────────
  LOG_MODE   standard | verbose | both    (default: verbose)
  LOG_FILE   path to JSON log file        (default: logs/voice_assistant.log)

Usage
─────
  Replace the structlog.configure(...) block in shared.py with:

      from app.common.log_config import configure_logging
      configure_logging()

  Then set the env var before launching:

      LOG_MODE=standard  python -m app.orchestration.controller   # pretty console + JSON file
      LOG_MODE=verbose   python -m app.orchestration.controller   # JSON stdout only
      python -m app.orchestration.controller                      # JSON stdout only (default)

Dual-sink detail
────────────────
  In standard mode the _DualSinkRenderer processor:
    1. Deep-copies the event_dict so neither sink sees the other's mutations.
    2. Renders the Rich line to stdout via _CONSOLE.print().
    3. Serialises the original dict to JSON and appends it to LOG_FILE.
    4. Raises structlog.DropEvent() so the PrintLoggerFactory doesn't
       double-print anything to stdout.
"""

from __future__ import annotations

import logging
import os

import structlog
from rich.console import Console
from rich.theme import Theme

# ── mode selection ────────────────────────────────────────────────────────────

LOG_MODE: str = os.getenv(
    "LOG_MODE", "verbose"
).lower()  # "standard" | "verbose" | "both"
LOG_FILE: str = os.getenv("LOG_FILE", "logs/voice_assistant.log")

# "both" is just an alias — dual-sink is always active in standard mode
if LOG_MODE == "both":
    LOG_MODE = "standard"

# ── colour theme ──────────────────────────────────────────────────────────────

_THEME = Theme(
    {
        "ts": "dim white",
        "lvl.info": "bold bright_green",
        "lvl.warn": "bold yellow",
        "lvl.error": "bold bright_red",
        "lvl.debug": "dim cyan",
        "evt": "white",
        "kv.key": "dim white",
        "kv.val": "bright_white",
        "icon.stt": "cyan",
        "icon.llm": "magenta",
        "icon.tts": "blue",
        "icon.rec": "bright_green",
        "icon.ses": "yellow",
        "icon.pipe": "white",
        "icon.ok": "bright_green",
        "icon.warn": "yellow",
        "icon.misc": "dim white",
    }
)

_CONSOLE = Console(theme=_THEME, highlight=False)

# ── icon + colour routing ─────────────────────────────────────────────────────

_EVENT_MAP: list[tuple[str, str, str]] = [
    # (event_prefix,        icon,  rich_style)
    ("stt", "◈", "icon.stt"),
    ("llm", "◈", "icon.llm"),
    ("tts", "◈", "icon.tts"),
    ("stream_full_ok", "✓", "icon.ok"),
    ("stream_ok", "⇢", "icon.stt"),
    ("recorder", "●", "icon.rec"),
    ("controller_recording", "●", "icon.rec"),
    ("player", "♪", "icon.tts"),
    ("session", "⬡", "icon.ses"),
    ("controller_session", "⬡", "icon.ses"),
    ("pipeline", "⚙", "icon.pipe"),
    ("sanitize", "⌖", "icon.misc"),
    ("eval", "⊛", "icon.ses"),
    ("retry", "↺", "icon.warn"),
    ("controller_shutdown", "⏻", "icon.warn"),
    ("controller_signal", "⚡", "icon.warn"),
    ("controller_ready", "✓", "icon.ok"),
]

_LEVEL_STYLE: dict[str, str] = {
    "info": "lvl.info",
    "warning": "lvl.warn",
    "error": "lvl.error",
    "debug": "lvl.debug",
    "warn": "lvl.warn",
}

# KV keys that are always too noisy for standard mode
_KV_SUPPRESS = {"request_id", "level", "timestamp", "event"}

# KV keys that are shown with shortened names for compactness
_KV_ALIAS: dict[str, str] = {
    "audio_path": "path",
    "detected_language": "lang",
    "avg_logprob": "conf",
    "latency_s": "latency",
    "duration_s": "dur",
    "session_id": "sid",
    "circuit_state": "cb",
    "ip_masked": "ip",
}


def _pick_icon(event: str) -> tuple[str, str]:
    for prefix, icon, style in _EVENT_MAP:
        if event.startswith(prefix):
            return icon, style
    return "·", "icon.misc"


def _kv_string(event_dict: dict) -> str:
    parts: list[str] = []
    for raw_key, val in event_dict.items():
        if raw_key in _KV_SUPPRESS or val is None or val == "" or val is False:
            continue
        key = _KV_ALIAS.get(raw_key, raw_key)
        parts.append(f"[kv.key]{key}=[/kv.key][kv.val]{val}[/kv.val]")
    return "  ".join(parts)


# ── JSON file sink ────────────────────────────────────────────────────────────

import json
import threading
from pathlib import Path

_file_lock = threading.Lock()


def _write_json_line(event_dict: dict) -> None:
    """Append one JSON line to LOG_FILE, creating the directory if needed."""
    try:
        path = Path(LOG_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event_dict, default=str)
        with _file_lock:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception:  # noqa — never let logging crash the app
        pass


# ── dual-sink renderer ────────────────────────────────────────────────────────


class _DualSinkRenderer:
    """
    Forks each log record into two independent sinks:

    Sink A — Rich console (stdout)
        Pretty coloured single-line format:
        [HH:MM:SS] [LEVEL]  <icon>  <event·····················>  key=val

    Sink B — JSON file (LOG_FILE)
        One JSON object per line, identical to verbose/structlog output.
        Safe for Datadog, Loki, CloudWatch, or any log aggregator.

    The event_dict is deep-copied before either sink mutates it, so both
    sinks always see the full original record. DropEvent is raised at the
    end so the PrintLoggerFactory writes nothing to stdout itself.
    """

    def __call__(self, _logger: object, _method: str, event_dict: dict) -> str:
        import copy

        json_dict = copy.deepcopy(event_dict)  # Sink B gets a pristine copy
        rich_dict = event_dict  # Sink A may mutate freely

        # ── Sink A: Rich console ──────────────────────────────────────────────
        ts = rich_dict.pop("timestamp", "")
        level = rich_dict.pop("level", "info").lower()
        event = rich_dict.pop("event", "")

        time_str = ts[11:19] if len(ts) >= 19 else ts
        icon, icon_style = _pick_icon(event)
        lvl_style = _LEVEL_STYLE.get(level, "lvl.info")
        level_label = ("WARN" if level == "warning" else level.upper())[:5]
        kv = _kv_string(rich_dict)

        line = (
            f"[ts]\\[{time_str}][/ts] "
            f"[{lvl_style}][{level_label:<4}][/{lvl_style}]  "
            f"[{icon_style}]{icon}[/{icon_style}]  "
            f"[evt]{event:<36}[/evt]  "
            f"{kv}"
        )
        _CONSOLE.print(line)

        # ── Sink B: JSON file ─────────────────────────────────────────────────
        _write_json_line(json_dict)

        raise structlog.DropEvent()


# ── public entry point ────────────────────────────────────────────────────────


def configure_logging() -> None:
    """
    Call once at process startup (before any logger is obtained).
    Replaces the structlog.configure() block in shared.py.
    """

    def _inject_request_id(_logger: object, _method: str, event_dict: dict) -> dict:
        # Import here to avoid circular import — shared.py owns the context var.
        from app.common.shared import current_request_id  # noqa

        event_dict["request_id"] = current_request_id()
        return event_dict

    base_processors = [
        structlog.contextvars.merge_contextvars,
        _inject_request_id,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.ExceptionRenderer(),
    ]

    if LOG_MODE == "standard":
        final_processor = _DualSinkRenderer()
    else:
        final_processor = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[*base_processors, final_processor],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )
