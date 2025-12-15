"""
Audio Extractor Module
======================
Extracts audio from video files using FFmpeg.

Responsibilities:
- Extract full audio track from video
- Convert to WAV format for Whisper compatibility
- Save to configured output path
"""

import subprocess
from pathlib import Path

import config


def extract_audio(video_path: str, output_path: Path = None) -> Path:
    """
    Extract audio from a video file as WAV.

    Args:
        video_path: Path to the input video file
        output_path: Path for output audio file (default: config.AUDIO_FILE)

    Returns:
        Path to the extracted audio file

    Raises:
        FileNotFoundError: If video file doesn't exist
        RuntimeError: If FFmpeg fails
    """
    # Validate input
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Setup output path
    output_path = output_path or config.AUDIO_FILE
    output_path = Path(output_path)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[AudioExtractor] Extracting audio from: {video_path.name}")

    # Build FFmpeg command
    # -vn : no video
    # -acodec pcm_s16le : PCM 16-bit audio (WAV format)
    # -ar 16000 : 16kHz sample rate (optimal for Whisper)
    # -ac 1 : mono channel
    cmd = [
        config.FFMPEG_PATH,
        "-i", str(video_path),
        "-vn",                    # No video
        "-acodec", "pcm_s16le",   # PCM 16-bit WAV
        "-ar", "16000",           # 16kHz sample rate
        "-ac", "1",               # Mono
        "-y",                     # Overwrite without asking
        str(output_path)
    ]

    print(f"[AudioExtractor] Running FFmpeg...")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed: {result.stderr}")

    # Validate output
    if not output_path.exists():
        raise RuntimeError(f"Audio file was not created: {output_path}")

    file_size = output_path.stat().st_size
    print(f"[AudioExtractor] Extracted audio to: {output_path}")
    print(f"[AudioExtractor] Audio file size: {file_size / 1024:.1f} KB")

    return output_path


if __name__ == "__main__":
    # Test module independently
    import sys

    if len(sys.argv) < 2:
        print("Usage: python audio_extractor.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    config.ensure_output_dirs()

    try:
        audio_path = extract_audio(video_path)
        print(f"\nExtracted audio: {audio_path}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
