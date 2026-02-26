# WIRINGFIX — STT / PCM / audio_engine Cross-Module Wiring Audit

**Files:** `player.py` · `recorder.py` · `STT_service.py` · `audio_engine.py`
**Date:** 2026-02-26
**Scope:** Integration wiring only — how STT, recorder, and player consume audio_engine primitives

---

## P0 — CRITICAL (Broken wiring, device contention, silent failures)

### BUG-W001 · `PCMOutputStream` is dead code; any caller opens a competing PortAudio device

| | |
|---|---|
| **File** | `player.py` |
| **Lines** | 114 (import), 273–274 (singleton), 286–308 (`get_output_stream()`), 631 (`put_nowait`) |
| **Symptom** | `get_output_stream()` creates a `PCMOutputStream` singleton that is never written to. `play_pcm_chunk()` routes through `_write_queue` → `_writer_loop` → a raw `sd.OutputStream`. Any external caller invoking `get_output_stream()` opens a second PortAudio output device concurrently with the writer thread's stream — both fight for the same hardware. |

**Root Cause:** The module docstring, public API table (line 61), and integration comment (line 44) all claim the PCM-native path feeds `PCMOutputStream`. In reality `play_pcm_chunk()` enqueues an `_AudioChunk` onto `_write_queue` (line 631) consumed by `_writer_loop` managing its own raw `sd.OutputStream`. `_pcm_output_stream` is constructed on demand but receives zero `.write()` calls anywhere in the file.

**Fix:**
```python
# Option A — route play_pcm_chunk through PCMOutputStream (make the path async):
# Replace line 631:
-     _write_queue.put_nowait(audio_chunk)
+     asyncio.create_task(get_output_stream().write(chunk))  # chunk is already float32 PCMChunk

# Then delete _writer_loop, _write_queue, _writer_open_stream, _writer_close_stream
# and all related state (_stream, _stream_sr, _stream_ch, _stream_open_event).

# Option B — remove PCMOutputStream entirely from this module (honest short-term fix):
# Line 114:
-     PCMOutputStream,

# Lines 273-274:
-     _pcm_output_stream: PCMOutputStream | None = None
-     _pcm_output_stream_lock = threading.Lock()

# Lines 286-308: delete get_output_stream() entirely.
# Update docstring lines 12-16 and 61 to remove all PCMOutputStream references.
```

**Why:** The current state is a trap — calling `get_output_stream()` from outside (e.g. `voice_graph.stream_full()`) silently opens a second PortAudio stream on the same device. Behaviour under dual-open is platform-dependent: silent failure, distorted output, or a hard PortAudio crash. Option A is the correct long-term fix; Option B is the honest short-term fix.

---

## P1 — HIGH (Wrong results, broken detection)

### BUG-W002 · `push_reference()` scale mismatch silently disables echo suppression in the barge-in detector

| | |
|---|---|
| **Files** | `player.py`, `audio_engine.py` |
| **Lines** | `player.py:353` (wrong fmt), `player.py:606–616` (float32 chunk pushed to int16-calibrated detector), `audio_engine.py:3760` (scale assignment) |
| **Symptom** | Echo suppression in `PCMInterruptDetector` never fires. During TTS playback, speaker bleed into the mic can trigger false barge-in interrupts because the reference RMS never clears the suppression threshold. |

**Root Cause:** `get_interrupt_detector()` (player.py:353) initialises the detector with `fmt = PCMFormat.whisper()` (int16). Inside `PCMInterruptDetector.__init__` (audio_engine.py:3760):

```python
self._scale = 32768.0 if fmt.dtype == "float32" else 1.0
# → self._scale = 1.0   (int16 path chosen)
# → reference_threshold = 500.0 is in int16 amplitude units
```

Then in `play_pcm_chunk()`, the chunk is converted to float32 **before** being pushed as the reference (lines 599–616):

```python
chunk = converter.convert(chunk, target_fmt)   # line 606 — chunk is now float32, RMS ≈ 0.01–0.3
...
_interrupt_detector.push_reference(chunk)      # line 616 — float32 RMS * scale(1.0) ≈ 0.01–0.3
```

`0.01–0.3 > 500.0` is permanently false, so `_playback_rms > _ref_thresh` never triggers and echo suppression is dead.

**Fix:**
```python
# player.py line 353 — use the playback format so the detector's scale is float32-calibrated:
-             input_fmt = get_format_registry().get("whisper") or PCMFormat.whisper()
+             input_fmt = get_format_registry().get("openai_tts") or PCMFormat.openai_tts()
              _interrupt_detector = PCMInterruptDetector(
                  fmt=input_fmt,
                  on_interrupt=on_interrupt,
              )
```

This changes `self._scale = 32768.0` inside the detector (audio_engine.py:3760), amplifying float32 RMS back to int16 scale so it's correctly compared against `reference_threshold = 500.0`.

**Why:** The barge-in detector is the only defence against the agent interrupting its own speech. With echo suppression broken, loud TTS output leaking into the mic fires `on_interrupt`, causing the agent to cancel mid-sentence in any non-anechoic room.

---

## P2 — MEDIUM (Misleading API, always-empty return values)

### BUG-W003 · `_playback_latency_tracker` is never observed; `get_playback_latency_report()` always returns `{}`

| | |
|---|---|
| **File** | `player.py` |
| **Lines** | 283 (declaration), 376–385 (getter), 450 (missing write observe), 637 (missing enqueue observe) |
| **Symptom** | `get_playback_latency_report()` creates a `PCMLatencyTracker` and immediately returns `.get_latency_report()` on a tracker with zero observations. Always `{}`. |

**Root Cause:** `_playback_latency_tracker` is lazily initialised inside the getter on line 383–384. The two sites that need to call `.observe()` — `play_pcm_chunk()` (enqueue at line 637) and `_writer_loop` (write at line 450) — never obtain the instance because the tracker only exists inside the getter's local initialisation.

**Fix:**
```python
# Line 283 — promote to module-level eager init so all sites share one instance:
-     _playback_latency_tracker: PCMLatencyTracker | None = None
+     _playback_latency_tracker: PCMLatencyTracker = PCMLatencyTracker()

# Line 637 — in play_pcm_chunk(), after put_nowait:
      _chunk_enqueue_lat.observe(time.monotonic() - t0)
+     _playback_latency_tracker.observe(chunk, "enqueue")

# Line 450 — in _writer_loop(), after _stream.write():
      _write_lat.observe(write_s)
+     _playback_latency_tracker.observe(diag_chunk, "write")

# Lines 382-384 — simplify getter now that tracker is always initialised:
  def get_playback_latency_report() -> dict[str, dict[str, float]]:
-     global _playback_latency_tracker
-     if _playback_latency_tracker is None:
-         _playback_latency_tracker = PCMLatencyTracker()
      return _playback_latency_tracker.get_latency_report()
```

**Why:** This is a production diagnostics endpoint. Returning `{}` permanently hides playback queue stalls, slow `write()` calls, and high enqueue-to-play jitter — exactly the data needed to debug audio dropout complaints.

---

## P2 — MEDIUM (Silent functional degradation)

### BUG-W004 · `PCMNoiseSuppressor` receives int16 chunks in the recorder pipeline; silence threshold is never satisfied — suppressor is a silent passthrough

| | |
|---|---|
| **Files** | `recorder.py`, `audio_engine.py` |
| **Lines** | `recorder.py:328–330` (int16 fmt passed to enhancer), `audio_engine.py:5340–5342` (float_fmt created but no converter stage inserted), `audio_engine.py:3068` (unenforced float32 contract), `audio_engine.py:3127` (threshold miscalibrated for int16) |
| **Symptom** | Noise suppression has zero effect. No crash, no error — just raw noisy mic audio sent straight to Whisper as if the suppressor doesn't exist. |

**Root Cause:** `recorder.py` constructs `PCMSpeechEnhancer` with `fmt=_RECORDING_FMT` (int16, line 330). Inside `PCMSpeechEnhancer.__init__` (audio_engine.py:5341) a `float_fmt` is created for the `PCMNoiseSuppressor` instance, but **no converter stage is inserted into the pipeline builder** between bandpass and suppressor:

```python
# audio_engine.py 5340-5342 (current):
if enable_ns:
    float_fmt = PCMFormat(sample_rate=fmt.sample_rate, channels=fmt.channels, dtype="float32")
    builder.with_noise_suppressor(PCMNoiseSuppressor(float_fmt))
    # ↑ float_fmt describes the suppressor's expected input, but the pipeline
    #   still delivers int16 chunks from the bandpass stage above it.
```

`PCMBandpassFilter.process()` preserves dtype (`out.astype(chunk.fmt.dtype)` → still int16). Inside `PCMNoiseSuppressor.process()`, `data.astype(float64)` yields values in 0–32768 range. `rms < silence_threshold(0.01)` is never true, the noise PSD estimate never updates from its initial `1e-6`, and the suppression gain is permanently ~1.0.

**Fix:**
```python
# audio_engine.py lines 5340-5342 — insert a converter stage before the suppressor:
          if enable_ns:
              float_fmt = PCMFormat(sample_rate=fmt.sample_rate, channels=fmt.channels, dtype="float32")
+             if fmt.dtype != "float32":
+                 builder.add(PCMConverter(quality="auto"), "int16_to_float32")
              builder.with_noise_suppressor(PCMNoiseSuppressor(float_fmt))

# audio_engine.py line 3068 — enforce the float32 contract at construction time:
  def __init__(self, fmt: PCMFormat, ...):
+     if fmt.dtype != "float32":
+         raise ValueError(
+             f"PCMNoiseSuppressor requires float32 input; got {fmt.dtype!r}. "
+             "Add a PCMConverter stage before this processor."
+         )
```

**Why:** Noise suppression is the primary reason the enhancement pipeline exists in office/HVAC environments. Without it, the recorder sends raw noisy audio to Whisper, producing degraded transcripts and burning extra API tokens on hallucinated filler segments. The `float_fmt` object at line 5341 shows the intent was correct — only the converter insertion step was missed.

---

## Summary

| Priority | Bug | File(s) | Lines | One-line description |
|----------|-----|---------|-------|----------------------|
| **P0** | W001 | `player.py` | 114, 273–274, 286–308, 631 | `PCMOutputStream` created but never written to; competing PortAudio device on first external call |
| **P1** | W002 | `player.py`, `audio_engine.py` | 353, 606–616, 3760 | int16/float32 scale mismatch kills echo suppression; false barge-in interrupts in any loud room |
| **P2** | W003 | `player.py` | 283, 376–385, 450, 637 | Latency tracker never observed; `get_playback_latency_report()` always returns `{}` |
| **P2** | W004 | `recorder.py`, `audio_engine.py` | 328–330, 5340–5342, 3068, 3127 | Missing int16→float32 converter before `PCMNoiseSuppressor`; suppressor is a silent passthrough |

### Recommended Fix Order

1. **W002** — single line change; immediately fixes live barge-in detection
2. **W004** — two additions; restores noise suppression for all mic recordings
3. **W001** — choose Option A or B; eliminates the competing-device trap
4. **W003** — lowest risk; promotes tracker to module-level and adds two observe calls
