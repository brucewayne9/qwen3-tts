# Qwen3-TTS Research Document

This document summarizes the Qwen3-TTS model capabilities, architecture, and Python API for building a drop-in replacement server for F5-TTS.

---

## Overview

**Qwen3-TTS** is an open-source TTS model series developed by the Qwen team at Alibaba Cloud. Released January 2026.

### Key Features

- **Multi-language support**: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian
- **Voice cloning**: Clone any voice from a reference audio clip
- **Voice design**: Create voices from natural language descriptions
- **Custom voices**: Pre-built speaker voices with emotion/style control
- **Streaming generation**: Ultra-low latency (97ms end-to-end)
- **Instruction control**: Natural language control over tone, emotion, prosody

---

## Released Models

| Model | Size | Purpose | HuggingFace ID |
|-------|------|---------|----------------|
| **Tokenizer** | - | Audio encode/decode | `Qwen/Qwen3-TTS-Tokenizer-12Hz` |
| **Base** | 1.7B | Voice cloning | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` |
| **Base** | 0.6B | Voice cloning (smaller) | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` |
| **CustomVoice** | 1.7B | Pre-built speakers + instruct | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` |
| **CustomVoice** | 0.6B | Pre-built speakers (smaller) | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` |
| **VoiceDesign** | 1.7B | Voice from description | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` |

### Model Selection for F5-TTS Replacement

For a drop-in replacement of F5-TTS (which does voice cloning from reference audio):
- **Primary choice**: `Qwen3-TTS-12Hz-1.7B-Base` (voice cloning)
- **Alternative**: `Qwen3-TTS-12Hz-0.6B-Base` (smaller, faster)

---

## Installation

### Python Package (Recommended)

```bash
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts
pip install -U qwen-tts
```

### Optional: FlashAttention 2 (reduces GPU memory)

```bash
pip install -U flash-attn --no-build-isolation
# For machines with <96GB RAM and many CPU cores:
MAX_JOBS=4 pip install -U flash-attn --no-build-isolation
```

### From Source

```bash
git clone https://github.com/QwenLM/Qwen3-TTS.git
cd Qwen3-TTS
pip install -e .
```

---

## Python API Reference

### Core Class: `Qwen3TTSModel`

Import and load:

```python
import torch
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",  # or other model
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # optional
)
```

---

## Voice Clone API (Most Relevant for F5-TTS Replacement)

This is the primary API needed for the replacement server.

### `model.generate_voice_clone()`

**Purpose**: Clone a voice from reference audio and synthesize new text.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | str or list[str] | Yes | Text to synthesize |
| `language` | str or list[str] | Yes | Language code (e.g., "English", "Chinese", "Auto") |
| `ref_audio` | various | Yes* | Reference audio (see formats below) |
| `ref_text` | str | Yes* | Transcript of reference audio |
| `voice_clone_prompt` | object | No | Pre-computed prompt (alternative to ref_audio/ref_text) |
| `x_vector_only_mode` | bool | No | If True, only use speaker embedding (ref_text not required) |

*Either `ref_audio`+`ref_text` OR `voice_clone_prompt` is required.

#### `ref_audio` Accepted Formats

- **Local file path**: `"/path/to/audio.wav"`
- **URL**: `"https://example.com/audio.wav"`
- **Base64 string**: Base64-encoded audio data
- **Tuple**: `(numpy_array, sample_rate)`

#### Returns

- `wavs`: List of numpy arrays (audio waveforms)
- `sr`: Sample rate (int)

#### Example: Single Inference

```python
import soundfile as sf

ref_audio = "reference.wav"
ref_text = "This is the transcript of the reference audio."

wavs, sr = model.generate_voice_clone(
    text="Hello, this is the text I want to synthesize.",
    language="English",
    ref_audio=ref_audio,
    ref_text=ref_text,
)

sf.write("output.wav", wavs[0], sr)
```

#### Example: Batch Inference

```python
wavs, sr = model.generate_voice_clone(
    text=["Sentence one.", "Sentence two."],
    language=["English", "English"],
    ref_audio=ref_audio,
    ref_text=ref_text,
)

for i, wav in enumerate(wavs):
    sf.write(f"output_{i}.wav", wav, sr)
```

---

## Reusable Voice Clone Prompts

For efficiency when generating multiple outputs with the same reference voice.

### `model.create_voice_clone_prompt()`

**Purpose**: Pre-compute reference features to reuse across multiple generations.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `ref_audio` | various | Yes | Reference audio (same formats as above) |
| `ref_text` | str | Yes* | Transcript of reference audio |
| `x_vector_only_mode` | bool | No | If True, only extract speaker embedding |

#### Returns

- `prompt_items`: Object to pass to `generate_voice_clone(voice_clone_prompt=...)`

#### Example: Reusable Prompt

```python
# Create prompt once
prompt_items = model.create_voice_clone_prompt(
    ref_audio="reference.wav",
    ref_text="Transcript of reference audio.",
    x_vector_only_mode=False,
)

# Reuse for multiple generations
wavs1, sr = model.generate_voice_clone(
    text="First sentence.",
    language="English",
    voice_clone_prompt=prompt_items,
)

wavs2, sr = model.generate_voice_clone(
    text="Second sentence.",
    language="English",
    voice_clone_prompt=prompt_items,
)
```

---

## Custom Voice API (Pre-built Speakers)

For the `CustomVoice` models with built-in speaker voices.

### `model.generate_custom_voice()`

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | str or list[str] | Yes | Text to synthesize |
| `language` | str or list[str] | No | Language ("Auto" for auto-detect) |
| `speaker` | str or list[str] | Yes | Speaker name (e.g., "Vivian", "Ryan") |
| `instruct` | str or list[str] | No | Emotion/style instruction |

#### Helper Methods

- `model.get_supported_speakers()` - List available speakers
- `model.get_supported_languages()` - List supported languages

#### Example

```python
wavs, sr = model.generate_custom_voice(
    text="Hello, how are you today?",
    language="English",
    speaker="Ryan",
    instruct="Very happy and excited.",
)
```

---

## Voice Design API (Create Voice from Description)

For the `VoiceDesign` model.

### `model.generate_voice_design()`

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | str or list[str] | Yes | Text to synthesize |
| `language` | str or list[str] | Yes | Language code |
| `instruct` | str or list[str] | Yes | Natural language voice description |

#### Example

```python
wavs, sr = model.generate_voice_design(
    text="Hello, this is a test.",
    language="English",
    instruct="Male, 30 years old, deep voice, calm and professional tone.",
)
```

---

## Tokenizer API (Audio Encode/Decode)

For direct audio encoding/decoding without TTS.

### `Qwen3TTSTokenizer`

```python
from qwen_tts import Qwen3TTSTokenizer

tokenizer = Qwen3TTSTokenizer.from_pretrained(
    "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    device_map="cuda:0",
)

# Encode audio to tokens
enc = tokenizer.encode("audio.wav")  # or URL

# Decode tokens back to audio
wavs, sr = tokenizer.decode(enc)
```

---

## Generation Parameters

All `generate_*` functions accept additional kwargs from HuggingFace Transformers `model.generate()`:

| Parameter | Description |
|-----------|-------------|
| `max_new_tokens` | Maximum tokens to generate |
| `top_p` | Nucleus sampling parameter |
| `temperature` | Sampling temperature |
| ... | Other HF generate kwargs |

### Evaluation Defaults

From the official evaluation:
- `dtype`: `torch.bfloat16`
- `max_new_tokens`: 2048

---

## Supported Languages

| Language | Code |
|----------|------|
| Chinese | `"Chinese"` |
| English | `"English"` |
| Japanese | `"Japanese"` |
| Korean | `"Korean"` |
| German | `"German"` |
| French | `"French"` |
| Russian | `"Russian"` |
| Portuguese | `"Portuguese"` |
| Spanish | `"Spanish"` |
| Italian | `"Italian"` |
| Auto-detect | `"Auto"` |

---

## Architecture Notes

- **Tokenizer**: Qwen3-TTS-Tokenizer-12Hz (12 Hz acoustic compression)
- **Architecture**: Discrete multi-codebook LM (not DiT-based like F5-TTS)
- **Streaming**: Dual-Track hybrid streaming architecture
- **Latency**: 97ms end-to-end for streaming mode

---

## Mapping F5-TTS Features to Qwen3-TTS

| F5-TTS Feature | Qwen3-TTS Equivalent |
|----------------|---------------------|
| Voice cloning with ref audio | `generate_voice_clone()` with `ref_audio` + `ref_text` |
| Transcription of ref audio | **Not built-in** - need external ASR (e.g., Whisper) |
| Speed control | Not directly supported (may need post-processing) |
| Default voice | Use `CustomVoice` model with built-in speakers |

### Key Differences

1. **No built-in transcription**: F5-TTS has `model.transcribe()`. Qwen3-TTS requires you to provide `ref_text`. You'll need an external ASR model (Whisper recommended).

2. **No speed parameter**: F5-TTS has `speed` parameter. Qwen3-TTS doesn't expose this directly.

3. **Language parameter**: Qwen3-TTS requires explicit `language` parameter (or "Auto").

4. **Output format**: Both output numpy arrays + sample rate. F5-TTS uses 24kHz; Qwen3-TTS sample rate returned by the model.

---

## Dependencies

### Python Package: `qwen-tts`

Installs all required dependencies automatically.

### Recommended

- Python 3.12
- CUDA-compatible GPU
- FlashAttention 2 (optional, reduces memory)

---

## Web UI Demo

Qwen3-TTS includes a built-in demo server:

```bash
# For Base model (voice cloning)
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-Base --ip 0.0.0.0 --port 8000

# For CustomVoice model
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --ip 0.0.0.0 --port 8000
```

---

## Summary: Implementation Plan for F5-TTS Replacement

To build a drop-in replacement server:

1. **Model**: Use `Qwen3-TTS-12Hz-1.7B-Base` for voice cloning
2. **Transcription**: Add Whisper or similar ASR for auto-transcribing reference audio
3. **API mapping**:
   - `/synthesize_speech/` → `generate_voice_clone()`
   - `/upload_audio/` → Store ref audio + transcribe with Whisper
   - `/change_voice/` → `generate_voice_clone()` with transcribed input
   - `/base_tts/` → Could use `CustomVoice` model with default speaker
4. **Speed control**: May need audio post-processing (e.g., librosa time-stretch)
5. **Output**: Convert numpy array to WAV with soundfile
