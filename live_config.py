"""
TaskAnalyzer Live Stream Configuration
======================================
Configuration for real-time Twitch stream analysis.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os

from config import PROJECT_ROOT, OUTPUT_DIR, ANTHROPIC_API_KEY


# =============================================================================
# LIVE STREAM OUTPUT DIRECTORIES
# =============================================================================

LIVE_OUTPUT_DIR = OUTPUT_DIR / "live"
LIVE_FRAMES_DIR = LIVE_OUTPUT_DIR / "frames"
LIVE_AUDIO_DIR = LIVE_OUTPUT_DIR / "audio"
LIVE_SESSIONS_DIR = LIVE_OUTPUT_DIR / "sessions"


# =============================================================================
# DEFAULT STREAM SETTINGS
# =============================================================================

# Frame capture settings
DEFAULT_FRAME_RATE = 1  # Frames per second to capture
DEFAULT_FRAME_QUALITY = "480p"  # Twitch quality preset

# Audio capture settings
DEFAULT_AUDIO_RATE = 16000  # Sample rate in Hz (matches Whisper)
DEFAULT_AUDIO_CHANNELS = 1  # Mono for speech recognition
DEFAULT_AUDIO_SEGMENT_LENGTH = 5.0  # Seconds per audio segment

# Buffer settings
DEFAULT_FRAME_BUFFER_SIZE = 300  # ~5 minutes at 1fps
DEFAULT_AUDIO_BUFFER_SIZE = 60  # Audio segments to buffer

# Analysis settings
DEFAULT_ANALYSIS_INTERVAL = 30  # Seconds between analyses
DEFAULT_MIN_FRAMES_FOR_ANALYSIS = 10  # Minimum frames before first analysis
DEFAULT_MIN_AUDIO_FOR_ANALYSIS = 10.0  # Minimum seconds of audio


@dataclass
class LiveStreamConfig:
    """Configuration for a live stream capture session."""

    # Stream source
    twitch_url: str

    # Session identification
    session_id: Optional[str] = None

    # Frame capture
    frame_rate: int = DEFAULT_FRAME_RATE
    frame_quality: str = DEFAULT_FRAME_QUALITY
    max_frame_buffer: int = DEFAULT_FRAME_BUFFER_SIZE

    # Audio capture
    audio_rate: int = DEFAULT_AUDIO_RATE
    audio_channels: int = DEFAULT_AUDIO_CHANNELS
    audio_segment_length: float = DEFAULT_AUDIO_SEGMENT_LENGTH
    max_audio_buffer: int = DEFAULT_AUDIO_BUFFER_SIZE

    # Analysis triggers
    analysis_interval: int = DEFAULT_ANALYSIS_INTERVAL
    min_frames_for_analysis: int = DEFAULT_MIN_FRAMES_FOR_ANALYSIS
    min_audio_for_analysis: float = DEFAULT_MIN_AUDIO_FOR_ANALYSIS

    # Output paths (auto-generated based on session_id)
    output_dir: Path = field(init=False)
    frames_dir: Path = field(init=False)
    audio_dir: Path = field(init=False)

    def __post_init__(self):
        """Initialize output directories based on session."""
        import uuid
        from datetime import datetime

        if self.session_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"

        self.output_dir = LIVE_SESSIONS_DIR / self.session_id
        self.frames_dir = self.output_dir / "frames"
        self.audio_dir = self.output_dir / "audio"

    def ensure_dirs(self):
        """Create output directories for this session."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(exist_ok=True)
        self.audio_dir.mkdir(exist_ok=True)

    @property
    def channel_name(self) -> str:
        """Extract channel name from Twitch URL."""
        # Handle formats: twitch.tv/channel, https://twitch.tv/channel, etc.
        url = self.twitch_url.rstrip('/')
        return url.split('/')[-1]

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "twitch_url": self.twitch_url,
            "session_id": self.session_id,
            "channel_name": self.channel_name,
            "frame_rate": self.frame_rate,
            "frame_quality": self.frame_quality,
            "audio_rate": self.audio_rate,
            "audio_channels": self.audio_channels,
            "audio_segment_length": self.audio_segment_length,
            "analysis_interval": self.analysis_interval,
            "output_dir": str(self.output_dir),
        }


def ensure_live_dirs():
    """Create base live stream directories."""
    LIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LIVE_FRAMES_DIR.mkdir(exist_ok=True)
    LIVE_AUDIO_DIR.mkdir(exist_ok=True)
    LIVE_SESSIONS_DIR.mkdir(exist_ok=True)


def validate_live_config():
    """Validate live stream configuration."""
    errors = []

    # Check API key
    if not ANTHROPIC_API_KEY:
        errors.append("ANTHROPIC_API_KEY environment variable not set")

    # Check streamlink availability
    try:
        import streamlink
    except ImportError:
        errors.append("streamlink package not installed (pip install streamlink)")

    # Check numpy availability
    try:
        import numpy as np
    except ImportError:
        errors.append("numpy package not installed")

    if errors:
        raise RuntimeError("Live config errors:\n" + "\n".join(f"  - {e}" for e in errors))

    return True


# Quality presets mapping (from twitch-realtime-handler)
QUALITY_PRESETS = {
    "160p": {"resolution": (320, 160), "description": "Low quality, fast"},
    "160p30": {"resolution": (320, 160), "description": "Low quality, 30fps"},
    "360p": {"resolution": (640, 360), "description": "Medium-low quality"},
    "360p30": {"resolution": (640, 360), "description": "Medium-low quality, 30fps"},
    "480p": {"resolution": (854, 480), "description": "Medium quality (recommended)"},
    "480p30": {"resolution": (854, 480), "description": "Medium quality, 30fps"},
    "720p": {"resolution": (1280, 720), "description": "HD quality"},
    "720p30": {"resolution": (1280, 720), "description": "HD quality, 30fps"},
    "720p60": {"resolution": (1280, 720), "description": "HD quality, 60fps"},
    "1080p": {"resolution": (1920, 1080), "description": "Full HD quality"},
    "1080p60": {"resolution": (1920, 1080), "description": "Full HD, 60fps"},
    "audio_only": {"resolution": None, "description": "Audio only stream"},
}
