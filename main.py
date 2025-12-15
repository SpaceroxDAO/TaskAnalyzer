#!/usr/bin/env python3
"""
TaskAnalyzer - Main Entry Point
===============================

Offline batch processor for analyzing first-person POV videos.
Extracts frames and audio, transcribes speech, and uses Claude Vision
to understand what task was performed.

Usage:
    python main.py <video_path>
    python main.py /path/to/video.mp4

Output:
    - output/frames/        : Extracted video frames (JPEG)
    - output/audio.wav      : Extracted audio
    - output/transcript.txt : Whisper transcription
    - output/task_analysis.json : Final structured analysis
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import config
from frame_extractor import extract_frames
from audio_extractor import extract_audio
from transcriber import transcribe_audio
from task_analyzer import analyze_task


def print_banner():
    """Print application banner."""
    print("=" * 60)
    print("  TaskAnalyzer - Video Task Understanding MVP")
    print("=" * 60)
    print()


def print_stage(stage_num: int, stage_name: str):
    """Print stage header."""
    print()
    print(f"{'─' * 60}")
    print(f"  Stage {stage_num}: {stage_name}")
    print(f"{'─' * 60}")
    print()


def run_pipeline(video_path: str, skip_transcription: bool = False) -> dict:
    """
    Run the full analysis pipeline.

    Pipeline stages:
    1. Extract frames from video (FFmpeg)
    2. Extract audio from video (FFmpeg)
    3. Transcribe audio (Whisper)
    4. Analyze task (Claude Vision)

    Args:
        video_path: Path to the input video file
        skip_transcription: If True, use existing transcript file

    Returns:
        Analysis result dictionary
    """
    start_time = datetime.now()

    # Validate input
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    print(f"Input video: {video_path}")
    print(f"Output directory: {config.OUTPUT_DIR}")
    print()

    # Ensure output directories exist
    config.ensure_output_dirs()

    # =========================================================================
    # Stage 1: Extract Frames
    # =========================================================================
    print_stage(1, "Frame Extraction")

    frames = extract_frames(str(video_path))
    print(f"✓ Extracted {len(frames)} frames")

    # =========================================================================
    # Stage 2: Extract Audio
    # =========================================================================
    print_stage(2, "Audio Extraction")

    audio_path = extract_audio(str(video_path))
    print(f"✓ Extracted audio to {audio_path}")

    # =========================================================================
    # Stage 3: Transcribe Audio
    # =========================================================================
    print_stage(3, "Audio Transcription (Whisper)")

    if skip_transcription and config.TRANSCRIPT_FILE.exists():
        print("[Transcriber] Using existing transcript file")
        transcript = config.TRANSCRIPT_FILE.read_text(encoding="utf-8")
    else:
        transcript = transcribe_audio(str(audio_path))

    print(f"✓ Transcribed {len(transcript)} characters")

    # =========================================================================
    # Stage 4: Analyze Task
    # =========================================================================
    print_stage(4, "Task Analysis (Claude Vision)")

    analysis = analyze_task(frames, transcript)

    # =========================================================================
    # Summary
    # =========================================================================
    elapsed = datetime.now() - start_time

    print()
    print("=" * 60)
    print("  ANALYSIS COMPLETE")
    print("=" * 60)
    print()
    print(f"Time elapsed: {elapsed.total_seconds():.1f} seconds")
    print()
    print("Output files:")
    print(f"  Frames:     {config.FRAMES_DIR}")
    print(f"  Audio:      {config.AUDIO_FILE}")
    print(f"  Transcript: {config.TRANSCRIPT_FILE}")
    print(f"  Analysis:   {config.ANALYSIS_FILE}")
    print()
    print("Analysis Result:")
    print("-" * 40)
    print(json.dumps(analysis, indent=2))
    print("-" * 40)

    return analysis


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Analyze first-person POV videos to understand tasks performed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py video.mp4
  python main.py /path/to/twitch_vod.mp4
  python main.py video.mp4 --skip-transcription

Output:
  The analysis will be saved to output/task_analysis.json
        """
    )

    parser.add_argument(
        "video_path",
        help="Path to the input video file (MP4)"
    )

    parser.add_argument(
        "--skip-transcription",
        action="store_true",
        help="Skip transcription and use existing transcript.txt"
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configuration without running pipeline"
    )

    args = parser.parse_args()

    print_banner()

    # Validate configuration
    try:
        config.validate_config()
        print("✓ Configuration validated")
        print()
    except RuntimeError as e:
        print(f"✗ Configuration error:\n{e}")
        sys.exit(1)

    if args.validate_only:
        print("Validation complete. Exiting.")
        sys.exit(0)

    # Run pipeline
    try:
        analysis = run_pipeline(args.video_path, args.skip_transcription)

        # Exit with success
        sys.exit(0)

    except FileNotFoundError as e:
        print(f"\n✗ File not found: {e}")
        sys.exit(1)

    except ImportError as e:
        print(f"\n✗ Missing dependency: {e}")
        sys.exit(1)

    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
