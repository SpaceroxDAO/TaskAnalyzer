"""
TaskAnalyzer Configuration
==========================
Central configuration for the video analysis pipeline.
"""

import os
from pathlib import Path

# Load environment variables from .env file if present
from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# PATHS
# =============================================================================

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Output directory for all generated artifacts
OUTPUT_DIR = PROJECT_ROOT / "output"

# Subdirectories
FRAMES_DIR = OUTPUT_DIR / "frames"
AUDIO_FILE = OUTPUT_DIR / "audio.wav"
TRANSCRIPT_FILE = OUTPUT_DIR / "transcript.txt"
ANALYSIS_FILE = OUTPUT_DIR / "task_analysis.json"

# =============================================================================
# FFMPEG CONFIGURATION
# =============================================================================

# Path to FFmpeg binaries (Homebrew default on macOS)
FFMPEG_PATH = "/opt/homebrew/bin/ffmpeg"
FFPROBE_PATH = "/opt/homebrew/bin/ffprobe"

# Frame extraction settings
DEFAULT_FRAME_INTERVAL_SECONDS = 3  # Extract 1 frame every N seconds
MAX_FRAMES = 50                      # Maximum frames to extract
MIN_FRAMES = 5                       # Minimum frames required
FRAME_FORMAT = "jpg"                 # Output format for frames
FRAME_QUALITY = 2                    # JPEG quality (2 = high quality, lower = better)

# =============================================================================
# WHISPER CONFIGURATION
# =============================================================================

# Whisper model size: tiny, base, small, medium, large
# Smaller = faster, larger = more accurate
WHISPER_MODEL = "base"

# =============================================================================
# VISION LLM CONFIGURATION (Claude)
# =============================================================================

# API key from environment variable
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Model to use for vision analysis
CLAUDE_MODEL = "claude-sonnet-4-20250514"

# Maximum tokens for response
MAX_TOKENS = 1024

# Vision model system prompt (from spec - USE EXACTLY)
VISION_PROMPT = """You are observing a human performing a task from a first-person perspective.

You are given:
1. A transcript of what the person said
2. A sequence of images captured during the task

Your job is to:
- Identify the primary task being performed
- Describe the task in 3–6 high-level steps
- Indicate when the task was completed

Do not guess specific details like recipients, exact content, or credentials.
Focus only on the type of work and the workflow.

Output JSON only in this format:

{
  "task_name": "",
  "confidence": 0.0,
  "steps": [
    "step 1",
    "step 2"
  ],
  "completion_detected": true
}"""

# =============================================================================
# OUTPUT CONTRACT
# =============================================================================

# Expected JSON output structure (for validation)
OUTPUT_SCHEMA = {
    "task_name": str,
    "confidence": float,
    "steps": list,
    "completion_detected": bool
}


def ensure_output_dirs():
    """Create output directories if they don't exist."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    FRAMES_DIR.mkdir(exist_ok=True)


def validate_config():
    """Validate configuration before running pipeline."""
    errors = []

    # Check FFmpeg
    if not Path(FFMPEG_PATH).exists():
        errors.append(f"FFmpeg not found at {FFMPEG_PATH}")

    if not Path(FFPROBE_PATH).exists():
        errors.append(f"FFprobe not found at {FFPROBE_PATH}")

    # Check API key
    if not ANTHROPIC_API_KEY:
        errors.append("ANTHROPIC_API_KEY environment variable not set")

    if errors:
        raise RuntimeError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))

    return True
