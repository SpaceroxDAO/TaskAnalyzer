"""
Frame Extractor Module
======================
Extracts frames from video files using FFmpeg.

Responsibilities:
- Get video duration using ffprobe
- Calculate optimal frame sampling rate
- Extract frames as JPEG images
- Cap total frames to configured maximum
"""

import subprocess
import shutil
from pathlib import Path
from typing import List

import config


def get_video_duration(video_path: str) -> float:
    """
    Get the duration of a video file in seconds.

    Args:
        video_path: Path to the video file

    Returns:
        Duration in seconds

    Raises:
        RuntimeError: If ffprobe fails
    """
    cmd = [
        config.FFPROBE_PATH,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]

    print(f"[FrameExtractor] Getting video duration...")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    duration = float(result.stdout.strip())
    print(f"[FrameExtractor] Video duration: {duration:.2f} seconds")

    return duration


def calculate_frame_interval(duration: float) -> float:
    """
    Calculate the optimal interval between frames.

    We want to extract frames such that:
    - We get at least MIN_FRAMES frames
    - We don't exceed MAX_FRAMES frames
    - Default interval is DEFAULT_FRAME_INTERVAL_SECONDS

    Args:
        duration: Video duration in seconds

    Returns:
        Interval in seconds between frames
    """
    # Start with default interval
    interval = config.DEFAULT_FRAME_INTERVAL_SECONDS

    # Calculate how many frames we'd get
    estimated_frames = duration / interval

    # If too many frames, increase interval
    if estimated_frames > config.MAX_FRAMES:
        interval = duration / config.MAX_FRAMES
        print(f"[FrameExtractor] Adjusted interval to {interval:.2f}s to cap at {config.MAX_FRAMES} frames")

    # If too few frames, decrease interval
    elif estimated_frames < config.MIN_FRAMES:
        interval = duration / config.MIN_FRAMES
        print(f"[FrameExtractor] Adjusted interval to {interval:.2f}s to get at least {config.MIN_FRAMES} frames")

    return interval


def extract_frames(video_path: str, output_dir: Path = None) -> List[Path]:
    """
    Extract frames from a video file.

    Args:
        video_path: Path to the input video file
        output_dir: Directory to save frames (default: config.FRAMES_DIR)

    Returns:
        List of paths to extracted frame images, sorted by frame number

    Raises:
        FileNotFoundError: If video file doesn't exist
        RuntimeError: If FFmpeg fails
    """
    # Validate input
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Setup output directory
    output_dir = output_dir or config.FRAMES_DIR
    output_dir = Path(output_dir)

    # Clear existing frames
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[FrameExtractor] Extracting frames from: {video_path.name}")

    # Get video duration and calculate interval
    duration = get_video_duration(str(video_path))
    interval = calculate_frame_interval(duration)

    # Build FFmpeg command
    # -vf fps=1/N : extract 1 frame every N seconds
    # -q:v 2 : JPEG quality (2 = high quality)
    output_pattern = output_dir / f"frame_%03d.{config.FRAME_FORMAT}"

    cmd = [
        config.FFMPEG_PATH,
        "-i", str(video_path),
        "-vf", f"fps=1/{interval}",
        "-q:v", str(config.FRAME_QUALITY),
        "-y",  # Overwrite without asking
        str(output_pattern)
    ]

    print(f"[FrameExtractor] Running FFmpeg with interval: {interval:.2f}s")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr}")

    # Collect extracted frames
    frames = sorted(output_dir.glob(f"*.{config.FRAME_FORMAT}"))

    print(f"[FrameExtractor] Extracted {len(frames)} frames to {output_dir}")

    # Validate we got frames
    if len(frames) == 0:
        raise RuntimeError("No frames were extracted from the video")

    return frames


if __name__ == "__main__":
    # Test module independently
    import sys

    if len(sys.argv) < 2:
        print("Usage: python frame_extractor.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    config.ensure_output_dirs()

    try:
        frames = extract_frames(video_path)
        print(f"\nExtracted frames:")
        for f in frames:
            print(f"  {f}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
