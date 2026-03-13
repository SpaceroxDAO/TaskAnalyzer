"""
TaskAnalyzer Live Stream Handler
================================
Wrapper around twitch-realtime-handler for capturing live Twitch streams.
Provides unified interface for frame and audio capture.
"""

import queue
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Thread, Event
from typing import Optional, Callable, Union, List, Tuple

import numpy as np

try:
    import streamlink
except ImportError:
    raise ImportError("streamlink package required: pip install streamlink")

from live_config import LiveStreamConfig, QUALITY_PRESETS


@dataclass
class CapturedFrame:
    """A captured video frame with metadata."""
    data: np.ndarray  # RGB numpy array (height, width, 3)
    timestamp: float  # Unix timestamp when captured
    frame_number: int  # Sequential frame number
    width: int
    height: int

    def save(self, path: Path) -> Path:
        """Save frame as JPEG image."""
        from PIL import Image
        img = Image.fromarray(self.data)
        img.save(path, 'JPEG', quality=85)
        return path


@dataclass
class CapturedAudio:
    """A captured audio segment with metadata."""
    data: np.ndarray  # Audio samples as numpy array
    timestamp: float  # Unix timestamp when captured
    segment_number: int  # Sequential segment number
    duration: float  # Duration in seconds
    sample_rate: int
    channels: int

    def save(self, path: Path) -> Path:
        """Save audio segment as WAV file."""
        import wave
        import struct

        # Convert to int16 for WAV
        if self.data.dtype == np.float64 or self.data.dtype == np.float32:
            audio_int = (self.data * 32767).astype(np.int16)
        else:
            audio_int = self.data.astype(np.int16)

        # Flatten if stereo
        if len(audio_int.shape) > 1:
            audio_int = audio_int.flatten()

        with wave.open(str(path), 'w') as wav:
            wav.setnchannels(self.channels)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(self.sample_rate)
            wav.writeframes(audio_int.tobytes())

        return path


class TwitchStreamCapture:
    """
    Unified handler for capturing both video frames and audio from Twitch streams.

    Uses streamlink to get stream URLs and FFmpeg for actual capture,
    inspired by twitch-realtime-handler architecture but simplified for
    our analysis pipeline.
    """

    def __init__(self, config: LiveStreamConfig):
        self.config = config
        self._stream_url: Optional[str] = None

        # State
        self._running = False
        self._stop_event = Event()

        # Frame capture
        self._frame_queue: queue.Queue = queue.Queue(maxsize=config.max_frame_buffer)
        self._frame_thread: Optional[Thread] = None
        self._frame_process: Optional[subprocess.Popen] = None
        self._frame_count = 0

        # Audio capture
        self._audio_queue: queue.Queue = queue.Queue(maxsize=config.max_audio_buffer)
        self._audio_thread: Optional[Thread] = None
        self._audio_process: Optional[subprocess.Popen] = None
        self._audio_segment_count = 0

        # Callbacks
        self._on_frame: Optional[Callable[[CapturedFrame], None]] = None
        self._on_audio: Optional[Callable[[CapturedAudio], None]] = None
        self._on_error: Optional[Callable[[Exception], None]] = None

        # Resolution from quality preset
        if config.frame_quality in QUALITY_PRESETS:
            res = QUALITY_PRESETS[config.frame_quality]["resolution"]
            if res:
                self._width, self._height = res
            else:
                self._width, self._height = 854, 480  # Default
        else:
            self._width, self._height = 854, 480

    def _get_stream_url(self, quality: str) -> str:
        """Get the stream URL from Twitch using streamlink."""
        try:
            streams = streamlink.streams(self.config.twitch_url)
        except streamlink.exceptions.NoPluginError:
            raise ValueError(f"No stream available for {self.config.twitch_url}")

        if not streams:
            raise ValueError(f"Stream is offline: {self.config.twitch_url}")

        if quality not in streams:
            available = list(streams.keys())
            raise ValueError(
                f"Quality '{quality}' not available. Options: {available}"
            )

        return streams[quality].url

    def start(self):
        """Start capturing frames and audio from the stream."""
        if self._running:
            return

        self.config.ensure_dirs()
        self._stop_event.clear()
        self._running = True

        # Get stream URLs
        try:
            video_url = self._get_stream_url(self.config.frame_quality)
            audio_url = self._get_stream_url("audio_only")
        except ValueError as e:
            self._running = False
            raise e

        # Start capture threads
        self._frame_thread = Thread(
            target=self._capture_frames,
            args=(video_url,),
            daemon=True
        )
        self._audio_thread = Thread(
            target=self._capture_audio,
            args=(audio_url,),
            daemon=True
        )

        self._frame_thread.start()
        self._audio_thread.start()

    def stop(self):
        """Stop all capture threads and processes."""
        self._stop_event.set()
        self._running = False

        # Terminate FFmpeg processes
        if self._frame_process:
            self._frame_process.terminate()
            try:
                self._frame_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._frame_process.kill()

        if self._audio_process:
            self._audio_process.terminate()
            try:
                self._audio_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._audio_process.kill()

        # Wait for threads
        if self._frame_thread and self._frame_thread.is_alive():
            self._frame_thread.join(timeout=5)

        if self._audio_thread and self._audio_thread.is_alive():
            self._audio_thread.join(timeout=5)

    def _capture_frames(self, stream_url: str):
        """Capture video frames using FFmpeg."""
        cmd = [
            "ffmpeg",
            "-i", stream_url,
            "-f", "image2pipe",
            "-r", str(self.config.frame_rate),
            "-pix_fmt", "rgb24",
            "-s", f"{self._width}x{self._height}",
            "-vcodec", "rawvideo",
            "-loglevel", "quiet",
            "-"
        ]

        bytes_per_frame = self._width * self._height * 3

        try:
            self._frame_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=10**8
            )

            while not self._stop_event.is_set():
                raw_frame = self._frame_process.stdout.read(bytes_per_frame)
                if not raw_frame or len(raw_frame) < bytes_per_frame:
                    break

                try:
                    frame_data = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
                        (self._height, self._width, 3)
                    )

                    frame = CapturedFrame(
                        data=frame_data,
                        timestamp=time.time(),
                        frame_number=self._frame_count,
                        width=self._width,
                        height=self._height
                    )

                    self._frame_count += 1

                    # Add to queue (non-blocking, drop oldest if full)
                    try:
                        self._frame_queue.put_nowait(frame)
                    except queue.Full:
                        # Drop oldest frame
                        try:
                            self._frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                        self._frame_queue.put_nowait(frame)

                    # Call callback if registered
                    if self._on_frame:
                        self._on_frame(frame)

                except ValueError as e:
                    if self._on_error:
                        self._on_error(e)

        except Exception as e:
            if self._on_error:
                self._on_error(e)

    def _capture_audio(self, stream_url: str):
        """Capture audio segments using FFmpeg."""
        cmd = [
            "ffmpeg",
            "-i", stream_url,
            "-f", "f32le",  # 32-bit float little-endian
            "-acodec", "pcm_f32le",
            "-ar", str(self.config.audio_rate),
            "-ac", str(self.config.audio_channels),
            "-loglevel", "quiet",
            "-"
        ]

        bytes_per_sample = 4  # float32
        samples_per_segment = int(
            self.config.audio_rate *
            self.config.audio_segment_length *
            self.config.audio_channels
        )
        bytes_per_segment = samples_per_segment * bytes_per_sample

        try:
            self._audio_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=10**8
            )

            while not self._stop_event.is_set():
                raw_audio = self._audio_process.stdout.read(bytes_per_segment)
                if not raw_audio or len(raw_audio) < bytes_per_segment:
                    break

                try:
                    audio_data = np.frombuffer(raw_audio, dtype=np.float32)

                    if self.config.audio_channels == 2:
                        audio_data = audio_data.reshape(-1, 2)

                    segment = CapturedAudio(
                        data=audio_data,
                        timestamp=time.time(),
                        segment_number=self._audio_segment_count,
                        duration=self.config.audio_segment_length,
                        sample_rate=self.config.audio_rate,
                        channels=self.config.audio_channels
                    )

                    self._audio_segment_count += 1

                    # Add to queue
                    try:
                        self._audio_queue.put_nowait(segment)
                    except queue.Full:
                        try:
                            self._audio_queue.get_nowait()
                        except queue.Empty:
                            pass
                        self._audio_queue.put_nowait(segment)

                    # Call callback if registered
                    if self._on_audio:
                        self._on_audio(segment)

                except ValueError as e:
                    if self._on_error:
                        self._on_error(e)

        except Exception as e:
            if self._on_error:
                self._on_error(e)

    def get_frame(self, blocking: bool = False, timeout: float = None) -> Optional[CapturedFrame]:
        """Get the next captured frame from the queue."""
        try:
            return self._frame_queue.get(block=blocking, timeout=timeout)
        except queue.Empty:
            return None

    def get_audio(self, blocking: bool = False, timeout: float = None) -> Optional[CapturedAudio]:
        """Get the next captured audio segment from the queue."""
        try:
            return self._audio_queue.get(block=blocking, timeout=timeout)
        except queue.Empty:
            return None

    def get_all_frames(self) -> List[CapturedFrame]:
        """Get all frames currently in the buffer."""
        frames = []
        while True:
            try:
                frames.append(self._frame_queue.get_nowait())
            except queue.Empty:
                break
        return frames

    def get_all_audio(self) -> List[CapturedAudio]:
        """Get all audio segments currently in the buffer."""
        segments = []
        while True:
            try:
                segments.append(self._audio_queue.get_nowait())
            except queue.Empty:
                break
        return segments

    def on_frame(self, callback: Callable[[CapturedFrame], None]):
        """Register a callback for each captured frame."""
        self._on_frame = callback

    def on_audio(self, callback: Callable[[CapturedAudio], None]):
        """Register a callback for each captured audio segment."""
        self._on_audio = callback

    def on_error(self, callback: Callable[[Exception], None]):
        """Register a callback for capture errors."""
        self._on_error = callback

    @property
    def is_running(self) -> bool:
        """Check if capture is currently running."""
        return self._running

    @property
    def frame_count(self) -> int:
        """Get total number of frames captured."""
        return self._frame_count

    @property
    def audio_segment_count(self) -> int:
        """Get total number of audio segments captured."""
        return self._audio_segment_count

    @property
    def frame_buffer_size(self) -> int:
        """Get current number of frames in buffer."""
        return self._frame_queue.qsize()

    @property
    def audio_buffer_size(self) -> int:
        """Get current number of audio segments in buffer."""
        return self._audio_queue.qsize()

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def check_stream_available(twitch_url: str) -> Tuple[bool, List[str]]:
    """
    Check if a Twitch stream is available and return available qualities.

    Returns:
        Tuple of (is_available, list_of_qualities)
    """
    try:
        streams = streamlink.streams(twitch_url)
        if streams:
            return True, list(streams.keys())
        return False, []
    except Exception:
        return False, []
