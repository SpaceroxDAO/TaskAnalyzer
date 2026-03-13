#!/usr/bin/env python3
"""
TaskAnalyzer Live Capture Test
==============================
CLI tool for testing live Twitch stream capture and analysis.

Usage:
    python live_capture_test.py https://twitch.tv/channelname
    python live_capture_test.py https://twitch.tv/channelname --duration 60
    python live_capture_test.py https://twitch.tv/channelname --quality 720p
"""

import argparse
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent))

from live_config import LiveStreamConfig, validate_live_config, ensure_live_dirs
from live_stream import TwitchStreamCapture, check_stream_available, CapturedFrame, CapturedAudio
from stream_buffer import StreamBuffer, AnalysisWindow


class LiveCaptureTest:
    """Test harness for live stream capture."""

    def __init__(self, twitch_url: str, duration: int = 60, quality: str = "480p"):
        self.twitch_url = twitch_url
        self.duration = duration
        self.quality = quality

        self.config = LiveStreamConfig(
            twitch_url=twitch_url,
            frame_quality=quality,
            frame_rate=1,  # 1 fps for testing
            audio_segment_length=5.0,
        )

        self.buffer = StreamBuffer(
            max_frames=300,
            max_audio_segments=60,
            max_buffer_duration=300.0
        )

        self.capture: TwitchStreamCapture = None
        self._running = False
        self._start_time = None

    def on_frame(self, frame: CapturedFrame):
        """Callback for each captured frame."""
        self.buffer.add_frame(frame)
        stats = self.buffer.get_stats()
        elapsed = time.time() - self._start_time
        print(
            f"\r[{elapsed:5.1f}s] Frames: {stats.frames_in_buffer:3d} | "
            f"Audio: {stats.total_audio_duration:5.1f}s | "
            f"Buffer: {stats.buffer_duration_seconds:5.1f}s",
            end="", flush=True
        )

    def on_audio(self, segment: CapturedAudio):
        """Callback for each captured audio segment."""
        self.buffer.add_audio(segment)

    def on_error(self, error: Exception):
        """Callback for capture errors."""
        print(f"\n[ERROR] {error}")

    def run(self):
        """Run the capture test."""
        print("=" * 60)
        print("TaskAnalyzer Live Capture Test")
        print("=" * 60)

        # Validate configuration
        print("\n[1/5] Validating configuration...")
        try:
            validate_live_config()
            print("      ✓ Configuration valid")
        except RuntimeError as e:
            print(f"      ✗ Configuration error: {e}")
            return False

        # Check stream availability
        print(f"\n[2/5] Checking stream: {self.twitch_url}")
        available, qualities = check_stream_available(self.twitch_url)

        if not available:
            print("      ✗ Stream is offline or URL is invalid")
            return False

        print(f"      ✓ Stream is live!")
        print(f"      Available qualities: {', '.join(qualities)}")

        if self.quality not in qualities:
            print(f"      ! Requested quality '{self.quality}' not available")
            # Find best alternative
            for q in ["480p", "360p", "720p", "480p30", "360p30"]:
                if q in qualities:
                    self.quality = q
                    self.config.frame_quality = q
                    print(f"      → Using '{q}' instead")
                    break

        # Create output directories
        print(f"\n[3/5] Setting up capture session...")
        ensure_live_dirs()
        self.config.ensure_dirs()
        print(f"      Session ID: {self.config.session_id}")
        print(f"      Output dir: {self.config.output_dir}")

        # Initialize capture
        print(f"\n[4/5] Starting capture...")
        print(f"      Quality: {self.quality}")
        print(f"      Duration: {self.duration} seconds")
        print(f"      Frame rate: {self.config.frame_rate} fps")
        print(f"      Audio: {self.config.audio_rate}Hz, {self.config.audio_channels}ch")

        self.capture = TwitchStreamCapture(self.config)
        self.capture.on_frame(self.on_frame)
        self.capture.on_audio(self.on_audio)
        self.capture.on_error(self.on_error)

        # Set up signal handler for graceful shutdown
        def signal_handler(sig, frame):
            print("\n\n[!] Interrupted - stopping capture...")
            self._running = False

        signal.signal(signal.SIGINT, signal_handler)

        # Start capture
        try:
            self.capture.start()
            self._running = True
            self._start_time = time.time()

            print("\n      Capturing... (Press Ctrl+C to stop early)")
            print("-" * 60)

            # Wait for duration
            while self._running and (time.time() - self._start_time) < self.duration:
                time.sleep(0.1)

        except ValueError as e:
            print(f"\n      ✗ Failed to start capture: {e}")
            return False
        except Exception as e:
            print(f"\n      ✗ Unexpected error: {e}")
            return False
        finally:
            self.capture.stop()

        # Results
        print("\n")
        print("-" * 60)
        print("\n[5/5] Capture complete!")

        stats = self.buffer.get_stats()
        print(f"\n      Summary:")
        print(f"      - Total frames captured: {stats.total_frames_received}")
        print(f"      - Frames in buffer: {stats.frames_in_buffer}")
        print(f"      - Total audio segments: {stats.total_audio_segments_received}")
        print(f"      - Audio duration: {stats.total_audio_duration:.1f}s")
        print(f"      - Buffer duration: {stats.buffer_duration_seconds:.1f}s")

        # Save sample data
        if stats.frames_in_buffer > 0:
            print(f"\n      Saving sample frames...")
            frame_paths = self.buffer.save_frames_for_analysis(
                self.config.frames_dir,
                count=min(20, stats.frames_in_buffer)
            )
            print(f"      → Saved {len(frame_paths)} frames to {self.config.frames_dir}")

        if stats.total_audio_duration > 0:
            print(f"\n      Saving audio...")
            audio_path = self.config.output_dir / "audio.wav"
            saved_path = self.buffer.save_audio_for_whisper(audio_path)
            if saved_path:
                print(f"      → Saved audio to {saved_path}")

        print(f"\n      All output saved to: {self.config.output_dir}")
        print("=" * 60)

        return True


def main():
    parser = argparse.ArgumentParser(
        description="Test live Twitch stream capture for TaskAnalyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s https://twitch.tv/shroud
    %(prog)s https://twitch.tv/shroud --duration 120
    %(prog)s https://twitch.tv/shroud --quality 720p --duration 30
        """
    )

    parser.add_argument(
        "twitch_url",
        help="Twitch stream URL (e.g., https://twitch.tv/channelname)"
    )

    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=60,
        help="Capture duration in seconds (default: 60)"
    )

    parser.add_argument(
        "--quality", "-q",
        default="480p",
        choices=["160p", "360p", "480p", "720p", "720p60", "1080p", "1080p60"],
        help="Stream quality (default: 480p)"
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check if stream is available, don't capture"
    )

    args = parser.parse_args()

    # Normalize URL
    twitch_url = args.twitch_url
    if not twitch_url.startswith("http"):
        twitch_url = f"https://twitch.tv/{twitch_url}"

    if args.check_only:
        print(f"Checking stream: {twitch_url}")
        available, qualities = check_stream_available(twitch_url)
        if available:
            print(f"✓ Stream is live!")
            print(f"  Available qualities: {', '.join(qualities)}")
            return 0
        else:
            print("✗ Stream is offline or URL is invalid")
            return 1

    test = LiveCaptureTest(
        twitch_url=twitch_url,
        duration=args.duration,
        quality=args.quality
    )

    success = test.run()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
