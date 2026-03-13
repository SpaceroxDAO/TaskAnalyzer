"""
TaskAnalyzer Stream Buffer
==========================
Rolling buffer management for live stream frames and audio.
Provides intelligent sampling for periodic analysis.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import List, Optional, Tuple
import wave
import tempfile

import numpy as np

from live_stream import CapturedFrame, CapturedAudio


@dataclass
class BufferStats:
    """Statistics about buffer state."""
    total_frames_received: int = 0
    total_audio_segments_received: int = 0
    frames_in_buffer: int = 0
    audio_segments_in_buffer: int = 0
    buffer_duration_seconds: float = 0.0
    oldest_frame_age: float = 0.0
    newest_frame_age: float = 0.0
    total_audio_duration: float = 0.0


class StreamBuffer:
    """
    Rolling buffer for live stream data with intelligent sampling.

    Maintains a time-windowed buffer of frames and audio segments,
    providing methods to sample representative data for analysis.
    """

    def __init__(
        self,
        max_frames: int = 300,
        max_audio_segments: int = 60,
        max_buffer_duration: float = 300.0  # 5 minutes
    ):
        self.max_frames = max_frames
        self.max_audio_segments = max_audio_segments
        self.max_buffer_duration = max_buffer_duration

        # Thread-safe buffers using deque for O(1) append/pop
        self._frames: deque = deque(maxlen=max_frames)
        self._audio: deque = deque(maxlen=max_audio_segments)

        # Locks for thread safety
        self._frame_lock = Lock()
        self._audio_lock = Lock()

        # Statistics
        self._total_frames = 0
        self._total_audio = 0

    def add_frame(self, frame: CapturedFrame):
        """Add a frame to the buffer."""
        with self._frame_lock:
            self._frames.append(frame)
            self._total_frames += 1
            self._trim_old_frames()

    def add_audio(self, segment: CapturedAudio):
        """Add an audio segment to the buffer."""
        with self._audio_lock:
            self._audio.append(segment)
            self._total_audio += 1
            self._trim_old_audio()

    def _trim_old_frames(self):
        """Remove frames older than max_buffer_duration."""
        if not self._frames:
            return

        cutoff = time.time() - self.max_buffer_duration
        while self._frames and self._frames[0].timestamp < cutoff:
            self._frames.popleft()

    def _trim_old_audio(self):
        """Remove audio segments older than max_buffer_duration."""
        if not self._audio:
            return

        cutoff = time.time() - self.max_buffer_duration
        while self._audio and self._audio[0].timestamp < cutoff:
            self._audio.popleft()

    def get_frames_for_analysis(
        self,
        count: int = 20,
        time_window: Optional[float] = None
    ) -> List[CapturedFrame]:
        """
        Get a representative sample of frames for analysis.

        Args:
            count: Target number of frames to return
            time_window: Only consider frames from last N seconds (None = all)

        Returns:
            List of frames evenly sampled across the time window
        """
        with self._frame_lock:
            if not self._frames:
                return []

            frames = list(self._frames)

            # Filter by time window if specified
            if time_window is not None:
                cutoff = time.time() - time_window
                frames = [f for f in frames if f.timestamp >= cutoff]

            if not frames:
                return []

            # If we have fewer frames than requested, return all
            if len(frames) <= count:
                return frames

            # Sample evenly across the frames
            step = len(frames) / count
            indices = [int(i * step) for i in range(count)]
            return [frames[i] for i in indices]

    def get_audio_for_analysis(
        self,
        time_window: Optional[float] = None
    ) -> Tuple[Optional[np.ndarray], int, int]:
        """
        Get concatenated audio for transcription.

        Args:
            time_window: Only consider audio from last N seconds (None = all)

        Returns:
            Tuple of (audio_array, sample_rate, channels) or (None, 0, 0) if empty
        """
        with self._audio_lock:
            if not self._audio:
                return None, 0, 0

            segments = list(self._audio)

            # Filter by time window if specified
            if time_window is not None:
                cutoff = time.time() - time_window
                segments = [s for s in segments if s.timestamp >= cutoff]

            if not segments:
                return None, 0, 0

            # Concatenate all audio data
            sample_rate = segments[0].sample_rate
            channels = segments[0].channels

            audio_arrays = [s.data for s in segments]
            concatenated = np.concatenate(audio_arrays)

            return concatenated, sample_rate, channels

    def save_audio_for_whisper(
        self,
        output_path: Path,
        time_window: Optional[float] = None
    ) -> Optional[Path]:
        """
        Save buffered audio to a WAV file suitable for Whisper transcription.

        Args:
            output_path: Path to save the WAV file
            time_window: Only include audio from last N seconds

        Returns:
            Path to saved file, or None if no audio available
        """
        audio_data, sample_rate, channels = self.get_audio_for_analysis(time_window)

        if audio_data is None:
            return None

        # Convert to int16 for WAV (Whisper expects 16kHz mono int16)
        if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
            audio_int = (audio_data * 32767).astype(np.int16)
        else:
            audio_int = audio_data.astype(np.int16)

        # Convert to mono if stereo
        if channels == 2 and len(audio_int.shape) > 1:
            audio_int = audio_int.mean(axis=1).astype(np.int16)
            channels = 1

        # Flatten
        audio_int = audio_int.flatten()

        # Write WAV file
        with wave.open(str(output_path), 'w') as wav:
            wav.setnchannels(1)  # Mono
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(sample_rate)
            wav.writeframes(audio_int.tobytes())

        return output_path

    def save_frames_for_analysis(
        self,
        output_dir: Path,
        count: int = 20,
        time_window: Optional[float] = None
    ) -> List[Path]:
        """
        Save sampled frames as JPEG files for analysis.

        Args:
            output_dir: Directory to save frames
            count: Number of frames to save
            time_window: Only consider frames from last N seconds

        Returns:
            List of paths to saved frame files
        """
        from PIL import Image

        output_dir.mkdir(parents=True, exist_ok=True)
        frames = self.get_frames_for_analysis(count, time_window)

        saved_paths = []
        for i, frame in enumerate(frames):
            filename = f"frame_{i:04d}_{int(frame.timestamp)}.jpg"
            path = output_dir / filename

            img = Image.fromarray(frame.data)
            img.save(path, 'JPEG', quality=85)
            saved_paths.append(path)

        return saved_paths

    def get_buffer_duration(self) -> float:
        """Get the time span covered by buffered data."""
        with self._frame_lock:
            if not self._frames:
                return 0.0

            oldest = self._frames[0].timestamp
            newest = self._frames[-1].timestamp
            return newest - oldest

    def get_audio_duration(self) -> float:
        """Get total duration of buffered audio."""
        with self._audio_lock:
            if not self._audio:
                return 0.0

            return sum(s.duration for s in self._audio)

    def get_stats(self) -> BufferStats:
        """Get current buffer statistics."""
        with self._frame_lock:
            frame_count = len(self._frames)
            if self._frames:
                oldest_frame_age = time.time() - self._frames[0].timestamp
                newest_frame_age = time.time() - self._frames[-1].timestamp
                buffer_duration = self._frames[-1].timestamp - self._frames[0].timestamp
            else:
                oldest_frame_age = 0.0
                newest_frame_age = 0.0
                buffer_duration = 0.0

        with self._audio_lock:
            audio_count = len(self._audio)
            audio_duration = sum(s.duration for s in self._audio)

        return BufferStats(
            total_frames_received=self._total_frames,
            total_audio_segments_received=self._total_audio,
            frames_in_buffer=frame_count,
            audio_segments_in_buffer=audio_count,
            buffer_duration_seconds=buffer_duration,
            oldest_frame_age=oldest_frame_age,
            newest_frame_age=newest_frame_age,
            total_audio_duration=audio_duration
        )

    def clear(self):
        """Clear all buffered data."""
        with self._frame_lock:
            self._frames.clear()

        with self._audio_lock:
            self._audio.clear()

    @property
    def has_enough_for_analysis(self) -> bool:
        """Check if buffer has minimum data for analysis."""
        with self._frame_lock:
            has_frames = len(self._frames) >= 5

        with self._audio_lock:
            audio_duration = sum(s.duration for s in self._audio)
            has_audio = audio_duration >= 5.0

        return has_frames and has_audio


class AnalysisWindow:
    """
    Manages analysis windows for periodic live analysis.

    Tracks what data has been analyzed and provides methods
    to get new data for incremental analysis.
    """

    def __init__(self, window_size: float = 30.0):
        """
        Args:
            window_size: Size of analysis window in seconds
        """
        self.window_size = window_size
        self._last_analysis_time: float = 0.0
        self._analysis_count: int = 0

    def should_analyze(self, buffer: StreamBuffer) -> bool:
        """Check if enough time has passed for a new analysis."""
        if not buffer.has_enough_for_analysis:
            return False

        if self._last_analysis_time == 0:
            return True

        elapsed = time.time() - self._last_analysis_time
        return elapsed >= self.window_size

    def mark_analyzed(self):
        """Mark that an analysis was just performed."""
        self._last_analysis_time = time.time()
        self._analysis_count += 1

    def get_analysis_window(self) -> Tuple[float, float]:
        """Get the time window for the next analysis."""
        now = time.time()
        if self._last_analysis_time == 0:
            # First analysis - use full window
            return now - self.window_size, now
        else:
            # Subsequent - from last analysis to now
            return self._last_analysis_time, now

    @property
    def analysis_count(self) -> int:
        """Get number of analyses performed."""
        return self._analysis_count

    @property
    def time_since_last_analysis(self) -> float:
        """Get seconds since last analysis."""
        if self._last_analysis_time == 0:
            return float('inf')
        return time.time() - self._last_analysis_time
