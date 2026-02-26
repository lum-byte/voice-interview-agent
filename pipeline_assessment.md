# Voice Pipeline — Wiring & Bug Assessment Report

**Scope:** All 6 modules — `audio_engine.py`, `STT_service.py`, `TTS_service.py`, `player.py`, `recorder.py`, `LLM_service.py`

---

## 1. Pipeline Wiring Diagram vs. Actual Wiring

```
Target:
Mic ──► PCMInputStream ──► [AGC, NS, VAD...] ──► PCMSpeechEnhancer ──► STT
                                                                          │
                                                                        LLM
                                                                          │
Speaker ◄── PCMOutputStream ◄── [Limiter, Pad...] ◄── PCMPlaybackEnhancer ◄── TTS
```

### Segment-by-segment wiring status

| Segment | Wired? | Notes |
|---|---|---|
| **Mic → PCMInputStream** | ✅ | `recorder.py`: `async with PCMInputStream(fmt=fmt) as stream:` |
| **PCMInputStream → PCMSpeechEnhancer** | ✅ | `recorder.py` feeds collected chunks into `PCMSpeechEnhancer.stream()` |
| **PCMSpeechEnhancer [AGC, NS, VAD chain]** | ✅ (with bugs) | Composed via `PCMPipelineBuilder`: bandpass → NS → AGC → gate → VAD |
| **PCMSpeechEnhancer → STT** | ✅ | `STTNode.transcribe_chunk()` / `transcribe_chunk_stream()` accept `PCMChunk` directly |
| **STT → LLM** | ✅ | Text-only handoff; LLM has no PCM knowledge by design |
| **LLM → TTS** | ✅ | Text-only handoff |
| **TTS → PCMPlaybackEnhancer** | ✅ | `TTS_service.py` uses `PCMPCMPipeline` with `PCMPlaybackEnhancer`; `player.py` calls enhancer components directly via `play_pcm_chunk()` |
| **PCMPlaybackEnhancer → PCMOutputStream** | ✅ | Both `TTS_service.py` and `player.py` write to `PCMOutputStream` |
| **PCMOutputStream → Speaker** | ✅ | Writer thread owns `sd.OutputStream` exclusively |
| **AEC reference feed (TTS → AEC → Mic path)** | ❌ **MISSING** | `aec.push_reference()` never called automatically (see BUG-3) |
| **PCMInterruptDetector** | ❌ **NOT WIRED** | Defined but not included in any live pipeline (see BUG-5) |

---

## 2. Bug Assessment — Severity Ranked

---

### 🔴 BUG-1 (CRITICAL): `PCMDynamicsProcessor` Silently Destroys int16 Audio

**Location:** `audio_engine.py` — `PCMDynamicsProcessor.process()` lines 3579, 3684  
**Affects:** `PCMPlaybackEnhancer` limiter → **all TTS playback output is silenced**

**Root cause:**

The level detector computes dBFS using raw sample values without normalization:
```python
level_db = 20.0 * np.log10(np.abs(mono) + 1e-12)  # line 3579
```

For `int16` input (`PCMFormat.openai_tts()` = 24kHz, 1ch, int16), `mono` values
range from 0 to 32767. A full-scale int16 value gives:

```
level_db = 20 * log10(32767) ≈ +90 dB
```

The limiter's threshold is `threshold_db=-1.0`. Every sample with amplitude > 1
(i.e., every non-zero int16 sample) gets a target gain reduction of:

```
target_gr = T - level_db = -1 - 90 = -91 dB
```

This applies ~-91 dB of gain reduction to **all audio**. Then:

```python
result = np.clip(result, -1.0, 1.0)  # line 3684
```

Clips the already-crushed values to `[-1, 1]`, then:

```python
result.astype(chunk.fmt.dtype)  # cast back to int16
```

Values of `-1.0`, `0.0`, `1.0` become `-1`, `0`, `1` in int16 — **effectively silence**.

**Call chain:** `player.play_pcm_chunk()` → `enhancer._limiter.process(chunk)` → destruction.  
Same bug in `PCMPlaybackEnhancer.stream()` and `PCMSpeechEnhancer` AGC stage when int16 chunks reach `PCMDynamicsProcessor`.

**Fix:**
```python
# Normalize to [-1, 1] before level detection
scale = 32768.0 if chunk.fmt.dtype == "int16" else 1.0
mono_norm = mono / scale
level_db = 20.0 * np.log10(np.abs(mono_norm) + 1e-12)
# ... processing ...
# Restore scale on output
if chunk.fmt.dtype == "int16":
    result = np.clip(result * scale, -32768, 32767)
else:
    result = np.clip(result, -1.0, 1.0)
```

---

### 🔴 BUG-2 (CRITICAL): `PCMNoiseSuppressor` Silence Detection Always False for int16

**Location:** `audio_engine.py` — `PCMNoiseSuppressor._suppress_frame()` line 3127;  
`PCMSpeechEnhancer.__init__()` lines 5341-5342  
**Affects:** Noise suppression on all mic input — **adaptive noise floor never updates**

**Root cause:**

`PCMSpeechEnhancer` initializes `PCMNoiseSuppressor` with a float32 format but feeds
it int16 chunks from the bandpass stage:

```python
# PCMSpeechEnhancer.__init__() line 5341
float_fmt = PCMFormat(sample_rate=fmt.sample_rate, channels=fmt.channels, dtype="float32")
builder.with_noise_suppressor(PCMNoiseSuppressor(float_fmt))
```

The suppressor is added to the pipeline **after the bandpass filter** which still
outputs `fmt` (int16, Whisper default). The suppressor receives int16 chunks.

Inside `_suppress_frame()`, silence detection uses:
```python
rms = float(np.sqrt(np.mean(frame ** 2)))   # line 3126
is_silence = rms < self._silence_thresh      # thresh = 0.01
```

After `chunk.data.astype(np.float64)`, int16 values remain in `[-32768, 32767]` scale.
A typical noise floor at int16 amplitude 100 gives `rms ≈ 100` — which is never `< 0.01`.

**`is_silence` is permanently `False`**, so:
- Noise PSD estimate updates during warmup (10 frames) then **freezes forever**
- The suppressor operates on a stale noise floor from the very first 10 frames
- Changing room noise / background audio is never re-learned

**Fix:**
```python
# In PCMSpeechEnhancer: ensure float32 before NS, or scale the threshold
if enable_ns:
    float_fmt = PCMFormat(sample_rate=fmt.sample_rate, channels=fmt.channels, dtype="float32")
    # Insert a dtype converter stage before NS
    builder.add(PCMConverter().as_processor(float_fmt), "dtype_to_float32")
    builder.with_noise_suppressor(PCMNoiseSuppressor(float_fmt))
```

---

### 🟠 BUG-3 (HIGH): AEC Reference Feed Not Wired to TTS Playback

**Location:** `audio_engine.py` — `build_voice_agent_pipeline()` line 5495;  
`TTS_service.py`; `player.py`  
**Affects:** Acoustic Echo Cancellation is completely dormant in all normal pipelines

**Root cause:**

`build_voice_agent_pipeline()` returns the `PCMEchoCanceller` as a separate object:
```python
return speech_enhancer, playback_enhancer, aec, drift
```

For AEC to function, `aec.push_reference(tts_chunk)` must be called for every chunk
sent to the speaker. Neither `TTS_service.py`, `player.py`, nor `PCMOutputStream` call this.

The factory's own docstring shows the correct wiring but as an **opt-in pattern that no
existing code implements**:
```python
# From build_voice_agent_pipeline docstring (reference only, not executed anywhere):
async def _tts_with_ref(s):
    async for c in s:
        aec.push_reference(c)  # ← nobody calls this
        yield c
```

**Fix:** Wrap `PCMOutputStream.write()` to automatically call `aec.push_reference(chunk)`
when an AEC is configured, or wire it into the `PCMPlaybackEnhancer.stream()` method.

---

### 🟠 BUG-4 (MEDIUM): `PCMNoiseGate` Runs lfilter Twice — Dead Code

**Location:** `audio_engine.py` — `PCMNoiseGate.process()` lines 2953–2956

**Root cause:** Two lfilter calls compute `zi_ea` and `zi_er` whose results are
immediately discarded. The identical filters are re-applied two lines later:

```python
# Lines 2953-2956 — DEAD CODE (results thrown away)
zi_ea, _ = _scipy_signal.lfilter(b_a, A_a, abs_x, zi=...)
zi_er, _ = _scipy_signal.lfilter(b_r, A_r, abs_x, zi=...)

# Lines 2959-2962 — CORRECT calls (used downstream)
env_attack, _ = _scipy_signal.lfilter(b_a, A_a, abs_x, zi=...)
env_release, _ = _scipy_signal.lfilter(b_r, A_r, abs_x, zi=...)
```

**Effect:** Every `PCMNoiseGate.process()` call does 4 lfilter passes instead of 2.
At 16 kHz / 60ms blocks, this is ~17 wasted passes per second on the hot path.

**Fix:** Delete lines 2953–2956 entirely.

---

### 🟠 BUG-5 (MEDIUM): `PCMInterruptDetector` Not Wired into Any Live Pipeline

**Location:** `player.py` (references module var `_interrupt_detector`); `audio_engine.py`

**Root cause:**

`player.play_pcm_chunk()` has this reference feed:
```python
if _interrupt_detector is not None:         # line 615
    _interrupt_detector.push_reference(chunk)
```

But `_interrupt_detector` is initialized as `None` and there is no initialization
code path exposed in `player.py` or `recorder.py`. The `PCMSpeechEnhancer` pipeline
builder also has no `with_interrupt_detector()` step.

**Effect:** Barge-in detection never activates; users cannot interrupt agent speech.

**Fix:**
1. Add `_interrupt_detector` initialization in `player.py`'s module setup or a
   dedicated `configure_interrupt_detector(fmt, on_interrupt)` function.
2. Add the detector to `PCMSpeechEnhancer`'s mic-side pipeline as an async passthrough
   (it already has a `stream()` method ready for this).

---

### 🟡 BUG-6 (LOW): `PCMAGCProcessor` Silently Promotes Chunk dtype to float32

**Location:** `audio_engine.py` — `PCMAGCProcessor.process()` lines 2812–2815

**Root cause:**

The AGC always emits a float32 chunk regardless of input dtype:
```python
return PCMChunk(
    data=limited.astype(np.float32),
    fmt=PCMFormat(sample_rate=chunk.fmt.sample_rate,
                  channels=chunk.fmt.channels, dtype="float32"),  # always float32
    ...
)
```

In `PCMSpeechEnhancer`, AGC is initialized with `fmt` (int16) but emits float32.
Downstream `PCMNoiseGate(fmt)` — initialized expecting int16 — receives float32 chunks.

The gate adapts silently via `chunk.fmt.dtype` on arrival, so no crash occurs, but
the stages downstream of AGC operate in float32 while all their internal state (thresholds,
envelope followers) was calibrated for int16 scale. The noise gate's `_threshold_lin`
computed at init time and RMS values at runtime are now in mismatched scales.

**Fix:** Return `chunk.fmt.dtype` from AGC (match input format), or document explicitly
that AGC is a float32 barrier and initialize all downstream stages with float32.

---

### 🟡 BUG-7 (LOW): Duplicate `queue` Import in `audio_engine.py`

**Location:** `audio_engine.py` lines 58 and 65

```python
import queue as _stdlib_queue   # line 58
import queue                    # line 65 — same module, different alias
```

Both refer to the stdlib `queue` module. The two aliases are then used inconsistently:
`_stdlib_queue.Queue`, `_stdlib_queue.Full`, `queue.Full`, `queue.Queue` — all
interchangeable but visually confusing when tracing exception handlers.

**Fix:** Keep only `import queue` and use `queue.Queue`, `queue.Full` everywhere.

---

## 3. Summary Table

| # | Severity | Module | Description | Audio Impact |
|---|---|---|---|---|
| BUG-1 | 🔴 CRITICAL | `audio_engine.py` | `PCMDynamicsProcessor` destroys int16 audio via mis-scaled dBFS + hard clip | **TTS playback silenced** |
| BUG-2 | 🔴 CRITICAL | `audio_engine.py` | `PCMNoiseSuppressor` silence detection broken for int16 → noise floor frozen | NS never adapts; mic quality degraded |
| BUG-3 | 🟠 HIGH | `audio_engine.py` / `player.py` / `TTS_service.py` | AEC reference push never called → echo cancellation dormant | Echo in mic uncancelled |
| BUG-4 | 🟠 MEDIUM | `audio_engine.py` | `PCMNoiseGate` runs lfilter twice (dead code at lines 2953-2956) | 2× CPU waste per chunk |
| BUG-5 | 🟠 MEDIUM | `player.py` / `audio_engine.py` | `PCMInterruptDetector` never initialized; barge-in detection non-functional | No barge-in |
| BUG-6 | 🟡 LOW | `audio_engine.py` | AGC always promotes chunk to float32; downstream stages calibrated for int16 | Potential threshold mismatch |
| BUG-7 | 🟡 LOW | `audio_engine.py` | Duplicate `queue` import (`_stdlib_queue` + `queue`) | None; code hygiene |

---

## 4. Quick-Fix Priority

1. **BUG-1** — Normalize to `[-1, 1]` before dBFS math in `PCMDynamicsProcessor`.
2. **BUG-2** — Insert float32 converter stage before `PCMNoiseSuppressor`, or scale `silence_thresh` by 32768 for int16.
3. **BUG-3** — Wrap `PCMOutputStream.write()` or `PCMPlaybackEnhancer.stream()` to call `aec.push_reference()` automatically.
4. **BUG-4** — Delete lines 2953–2956 in `PCMNoiseGate.process()`.
5. **BUG-5** — Expose an init function for `_interrupt_detector` and add to `PCMSpeechEnhancer`.
6. **BUG-6** — Make AGC preserve input dtype or document the float32 promotion contract.
7. **BUG-7** — Remove duplicate import.
