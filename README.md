# Qwen3-TTS Server

A FastAPI-based text-to-speech server powered by Qwen3-TTS, featuring voice cloning, voice conversion, and multi-language support.

## Features

- **Voice cloning** - Clone any voice from a short reference audio clip
- **Voice conversion** - Transform the voice of existing audio files
- **Multi-language support** - 10 languages: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian
- **Automatic transcription** - Uses Whisper for reference audio transcription
- **Speed control** - Adjust speech speed via time-stretching
- **RunPod ready** - Optimized for deployment on RunPod with network volume support

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/base_tts/` | GET | TTS with default English voice |
| `/synthesize_speech/` | GET | TTS with specified voice |
| `/upload_audio/` | POST | Upload reference voice |
| `/change_voice/` | POST | Voice conversion |

### Endpoint Details

#### `GET /synthesize_speech/`

Generate speech from text using a specified voice.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | Yes | Text to synthesize |
| `voice` | string | Yes | Voice label (filename prefix in `resources/`) |
| `speed` | float | No | Speech speed multiplier (default: 1.0) |

#### `GET /base_tts/`

Generate speech using the default English voice.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | Yes | Text to synthesize |
| `speed` | float | No | Speech speed multiplier (default: 1.0) |

#### `POST /upload_audio/`

Upload an audio file to use as a reference voice.

| Field | Type | Description |
|-------|------|-------------|
| `audio_file_label` | string | Label/name for the voice |
| `file` | file | Audio file (wav, mp3, flac, ogg; max 5MB) |

#### `POST /change_voice/`

Convert the voice of an existing audio file.

| Field | Type | Description |
|-------|------|-------------|
| `reference_speaker` | string | Voice label to convert to |
| `file` | file | Audio file to convert |

## Quick Start

### Using Docker (Recommended)

```bash
# Build the image
docker build -t qwen3-tts_server .

# Run the container
docker run --gpus all -p 7860:7860 qwen3-tts_server
```

### Local Development

```bash
# Create conda environment
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts

# Install PyTorch with CUDA
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install dependencies
pip install -r requirements.txt

# Optional: Install FlashAttention 2 for better performance
pip install flash-attn --no-build-isolation

# Run the server
./start.sh
# or
python -m uvicorn server:app --host 0.0.0.0 --port 7860
```

## Directory Structure

```
.
├── server.py           # Main FastAPI server
├── Dockerfile          # Multi-stage Docker build
├── requirements.txt    # Python dependencies
├── start.sh            # Startup script
├── resources/          # Voice reference files
│   └── demo_speaker0.mp3
└── outputs/            # Generated audio (temporary)
```

## Usage Examples

### Synthesize Speech

```bash
curl "http://localhost:7860/synthesize_speech/?text=Hello%20world&voice=demo_speaker0" \
  --output output.wav
```

### Upload a Voice

```bash
curl -X POST "http://localhost:7860/upload_audio/" \
  -F "audio_file_label=my_voice" \
  -F "file=@/path/to/voice_sample.mp3"
```

### Use Uploaded Voice

```bash
curl "http://localhost:7860/synthesize_speech/?text=Hello%20world&voice=my_voice" \
  --output output.wav
```

### Change Voice of Audio

```bash
curl -X POST "http://localhost:7860/change_voice/" \
  -F "reference_speaker=demo_speaker0" \
  -F "file=@/path/to/input.wav" \
  --output converted.wav
```

## Models

| Model | Purpose | Size |
|-------|---------|------|
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Voice cloning TTS | 1.7B parameters |
| `openai/whisper-base` | Audio transcription | 74M parameters |

Models are pre-downloaded during Docker build and stored in `/root/.cache/`.

## Requirements

- **GPU**: CUDA-compatible GPU with 8GB+ VRAM recommended
- **CUDA**: 12.8+ (for RTX 5090/Blackwell support)
- **Python**: 3.10+

## RunPod Deployment

The Docker image is optimized for RunPod:
- Server files are in `/app/server/` (not `/workspace/`)
- `/workspace/` is left free for network volumes
- Uses `sleep infinity` to keep container alive for web terminal access

## License

See the respective licenses for:
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- [OpenAI Whisper](https://github.com/openai/whisper)
