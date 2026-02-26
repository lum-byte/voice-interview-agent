# PATCHFIX 3 — `player.py`

**Date:** 2026-02-26
**File:** `player.py`
**Bugs:** 4

---

### BUG-001 · `get_output_stream()` — PCMOutputStream never started

| | |
|---|---|
| **File** | `player.py` |
| **Line** | 259–262 |
| **Problem** | `PCMOutputStream` is constructed but `start()` is never called. `write()` immediately raises `RuntimeError("PCMOutputStream.write() called before start()")` on the first chunk. Nothing plays. |
| **Fix** | |

```python
  _pcm_output_stream = PCMOutputStream(
      preferred_fmt=output_fmt,
      converter=get_converter(),
  )
+ asyncio.get_event_loop().run_until_complete(_pcm_output_stream.start())
```

---

### BUG-002 · `stop_stream()` — calls `stop()` instead of `stop_playback()`

| | |
|---|---|
| **File** | `player.py` |
| **Line** | 562 |
| **Problem** | `stop()` sends `_SHUTDOWN` and blocks up to 3 seconds joining the writer thread — permanently destroys the stream. The next `write()` raises `RuntimeError`. The correct call is `stop_playback()` which sends `_STOP`, silences pending audio, and leaves the thread alive for the next chunk. |
| **Fix** | |

```python
- asyncio.create_task(output_stream.stop())
+ asyncio.create_task(output_stream.stop_playback())
```

---

### BUG-003 · `play_pcm_chunk` + `play_audio_bytes` — `create_task(write(...))` should be `await`

| | |
|---|---|
| **File** | `player.py` |
| **Lines** | 419, 536 |
| **Problem** | `asyncio.create_task()` raises `RuntimeError` if there is no running event loop. `write()` is a coroutine that just calls `put_nowait()` and returns in microseconds — fire-and-forget buys nothing and adds a crash vector. |
| **Fix** | |

```python
# Line 419
- asyncio.create_task(get_output_stream().write(chunk))
+ await get_output_stream().write(chunk)

# Line 536
- asyncio.create_task(get_output_stream().write(pcm_chunk))
+ await get_output_stream().write(pcm_chunk)
```

Both functions must be `async def` if not already.

---

### BUG-004 · `play_pcm_chunk` — redundant float32 conversion before `write()`

| | |
|---|---|
| **File** | `player.py` |
| **Lines** | 390–396 |
| **Problem** | `play_pcm_chunk` manually converts the chunk to float32 before calling `write()`. `PCMOutputStream.write()` converts to float32 internally anyway. The conversion runs twice on every chunk for no reason. |
| **Fix** | Remove the manual conversion entirely and pass the enhanced chunk straight to `write()`: |

```python
- converter = get_converter()
- target_fmt = PCMFormat(
-     sample_rate=chunk.fmt.sample_rate,
-     channels=chunk.fmt.channels,
-     dtype="float32",
- )
- chunk = converter.convert(chunk, target_fmt)
```

---

## Summary

| # | Line | Problem | Fix |
|---|------|---------|-----|
| BUG-001 | 259–262 | `PCMOutputStream` constructed but never started | Call `start()` after construction |
| BUG-002 | 562 | `stop()` permanently destroys stream instead of pausing | Change to `stop_playback()` |
| BUG-003 | 419, 536 | `create_task(write(...))` crashes outside event loop | Change to `await write(...)` |
| BUG-004 | 390–396 | Double float32 conversion — `write()` already does it | Remove manual `converter.convert()` call |