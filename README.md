# TaskAnalyzer

**Offline batch processor for analyzing first-person POV videos to understand tasks performed.**

Given a video with narrated audio (e.g., a Twitch VOD), this tool:
1. Extracts frames using FFmpeg
2. Extracts and transcribes audio using Whisper
3. Analyzes the frames + transcript using Claude Vision
4. Outputs structured JSON describing the task, steps, and completion status

## Architecture

```
MP4 Video
   ↓
FFmpeg
   ├── frames/ (JPEGs, sparse sampling)
   └── audio.wav
           ↓
        Whisper (local)
           ↓
        transcript.txt
           ↓
Frames + Transcript
           ↓
Claude Vision API
           ↓
task_analysis.json
```

## Requirements

### System Dependencies

- **Python 3.10+**
- **FFmpeg** (for video/audio extraction)

Install FFmpeg on macOS:
```bash
brew install ffmpeg
```

### Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `openai-whisper` - Local speech-to-text
- `anthropic` - Claude API client
- `Pillow` - Image handling
- `python-dotenv` - Environment variables

### API Key

You need an Anthropic API key for Claude Vision:

```bash
export ANTHROPIC_API_KEY=your_key_here
```

Or create a `.env` file:
```
ANTHROPIC_API_KEY=your_key_here
```

## Usage

### Basic Usage

```bash
python main.py /path/to/video.mp4
```

### Options

```bash
# Skip transcription (use existing transcript.txt)
python main.py video.mp4 --skip-transcription

# Validate configuration only
python main.py video.mp4 --validate-only
```

### Output

All output is saved to the `output/` directory:

```
output/
├── frames/           # Extracted video frames (JPEG)
│   ├── frame_001.jpg
│   ├── frame_002.jpg
│   └── ...
├── audio.wav         # Extracted audio
├── transcript.txt    # Whisper transcription
└── task_analysis.json # Final analysis
```

### Output Format

The `task_analysis.json` follows this structure:

```json
{
  "task_name": "Send an email",
  "confidence": 0.85,
  "steps": [
    "Opened email client",
    "Composed new message",
    "Added recipient",
    "Typed message content",
    "Sent the email"
  ],
  "completion_detected": true
}
```

## Configuration

Edit `config.py` to customize:

| Setting | Default | Description |
|---------|---------|-------------|
| `DEFAULT_FRAME_INTERVAL_SECONDS` | 3 | Extract 1 frame every N seconds |
| `MAX_FRAMES` | 50 | Maximum frames to extract |
| `WHISPER_MODEL` | "base" | Whisper model size (tiny/base/small/medium/large) |
| `CLAUDE_MODEL` | "claude-sonnet-4-20250514" | Claude model for vision analysis |

## Module Overview

| Module | Responsibility |
|--------|----------------|
| `config.py` | Configuration constants and validation |
| `frame_extractor.py` | FFmpeg frame extraction |
| `audio_extractor.py` | FFmpeg audio extraction |
| `transcriber.py` | Whisper transcription |
| `task_analyzer.py` | Claude Vision API integration |
| `main.py` | CLI entry point and pipeline orchestration |

### Running Individual Modules

Each module can be tested independently:

```bash
# Test frame extraction
python frame_extractor.py video.mp4

# Test audio extraction
python audio_extractor.py video.mp4

# Test transcription
python transcriber.py output/audio.wav

# Test analysis (requires frames and transcript)
python task_analyzer.py output/frames output/transcript.txt
```

## Design Principles

This is a **learning MVP** designed for:

- **Clarity over cleverness** - Simple, readable code
- **Correctness over completeness** - Core functionality first
- **Debuggability over automation** - Intermediate artifacts saved

### What This Is NOT

- Real-time or live-stream processing
- Twitch API integration
- Multi-session memory
- Production system

## Future Extensions

After graduation criteria are met, the architecture can be extended to:
- Accept live frame sources
- Integrate with streaming handlers
- Move toward near-real-time processing

## License

MIT
