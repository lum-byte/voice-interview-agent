"""
startup_display.py  —  Animated boot banner for standard mode.

Shown once during controller init to replace the wall of
stt_using_local_node / llm_using_local_node JSON lines.

Usage (inside controller.py  listen_loop(), before the session block):
─────────────────────────────────────────────────────────────────────
    from app.common.log_config import LOG_MODE
    from app.common.startup_display import show_boot_sequence

    if LOG_MODE == "standard":
        show_boot_sequence([
            {"label": "STT",           "model": settings.stt_model,        "status": "ok"},
            {"label": "LLM",           "model": settings.llm_model,        "status": "ok"},
            {"label": "TTS",           "model": f"{settings.tts_model} / {settings.tts_voice}", "status": "ok"},
            {"label": "Session Store", "model": "redis + lru",             "status": "ok"},
        ])
"""

from __future__ import annotations

import time

from rich.console import Console
from rich.panel import Panel  # noqa
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,  # noqa
    TextColumn,
    TimeElapsedColumn,  # noqa
)
from rich.table import Table
from rich.text import Text  # noqa
from rich.theme import Theme

_THEME = Theme(
    {
        "banner": "bold bright_cyan",
        "ok": "bright_green",
        "warn": "yellow",
        "skip": "dim white",
        "label": "bold white",
        "model": "dim white",
        "bar": "cyan",
        "bar.done": "bright_green",
        "sep": "dim white",
        "key": "bold bright_white",
        "val": "dim white",
    }
)

_CONSOLE = Console(theme=_THEME, highlight=False)

_STATUS_ICON = {
    "ok": ("[ok]✓[/ok]", "bar.done"),
    "warn": ("[warn]⚠[/warn]", "yellow"),
    "skip": ("[skip]—[/skip]", "dim white"),
    "err": ("[red]✗[/red]", "red"),
}


def show_boot_sequence(
    modules: list[dict],
    *,
    app_name: str = "Voice Assistant",
    ptt_key: str = "H",
    exit_key: str = "ESC",
) -> None:
    """
    Render an animated module-loading banner.

    modules — list of dicts:
        label   str   display name, e.g. "STT"
        model   str   model/version string, e.g. "whisper-1"
        status  str   "ok" | "warn" | "skip" | "err"
        note    str   (optional) extra annotation, e.g. "LRU-only"
    """
    _CONSOLE.print()
    _CONSOLE.rule(f"[banner]{app_name}[/banner]")
    _CONSOLE.print()

    with Progress(
        TextColumn("  "),
        TextColumn("[label]{task.description:<18}[/label]"),
        BarColumn(
            bar_width=22,
            style="bar",
            complete_style="bar.done",
            finished_style="bar.done",
        ),
        TextColumn(" "),
        TextColumn("[model]{task.fields[model]:<22}[/model]"),
        TextColumn("{task.fields[status_icon]}"),
        TextColumn(" {task.fields[note]}"),
        console=_CONSOLE,
        transient=False,
    ) as progress:
        for mod in modules:
            label = mod.get("label", "Module")
            model = mod.get("model", "")
            status = mod.get("status", "ok")
            note = mod.get("note", "")

            icon_markup, _bar_style = _STATUS_ICON.get(status, _STATUS_ICON["ok"])

            task = progress.add_task(
                label,
                total=24,
                model=model,
                status_icon="",  # filled after bar completes
                note="",
            )

            # animate bar fill
            for _ in range(24):
                time.sleep(0.018)
                progress.advance(task)

            progress.update(
                task, status_icon=icon_markup, note=f"[dim]{note}[/dim]" if note else ""
            )

    # ── session / key-binding summary ─────────────────────────────────────────
    _CONSOLE.print()
    _CONSOLE.rule("[sep]─[/sep]")

    info = Table.grid(padding=(0, 2))
    info.add_column(style="key", justify="right")
    info.add_column(style="val")
    info.add_row("PTT", f"Hold [key]{ptt_key.upper()}[/key] to talk")
    info.add_row("Exit", f"[key]{exit_key.upper()}[/key]")
    _CONSOLE.print(info, justify="center")

    _CONSOLE.rule("[sep]─[/sep]")
    _CONSOLE.print()
