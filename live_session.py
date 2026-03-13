"""
TaskAnalyzer Live Session Manager
=================================
Manages active live streaming sessions and their lifecycle.
"""

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Callable, Any

from live_config import LiveStreamConfig, LIVE_SESSIONS_DIR, ensure_live_dirs
from live_analyzer import LiveAnalyzer, AnalysisResult


class SessionState(Enum):
    """Possible states for a live session."""
    CREATED = "created"
    CONNECTING = "connecting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class LiveSession:
    """Represents a single live streaming session."""
    session_id: str
    twitch_url: str
    quality: str
    state: SessionState = SessionState.CREATED
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    stopped_at: Optional[float] = None
    error_message: Optional[str] = None

    # Runtime components (not serialized)
    config: Optional[LiveStreamConfig] = field(default=None, repr=False)
    analyzer: Optional[LiveAnalyzer] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        data = {
            "session_id": self.session_id,
            "twitch_url": self.twitch_url,
            "quality": self.quality,
            "state": self.state.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "error_message": self.error_message,
        }

        # Add channel name
        if self.config:
            data["channel_name"] = self.config.channel_name

        # Add runtime stats if analyzer exists
        if self.analyzer:
            summary = self.analyzer.get_session_summary()
            data.update({
                "frames_captured": summary.get("total_frames", 0),
                "audio_duration": summary.get("total_audio", 0),
                "analyses_completed": summary.get("analyses_completed", 0),
                "current_task": summary.get("latest_task", ""),
                "current_confidence": summary.get("latest_confidence", 0.0),
            })

        return data

    @property
    def duration(self) -> float:
        """Get session duration in seconds."""
        if not self.started_at:
            return 0.0
        end_time = self.stopped_at or time.time()
        return end_time - self.started_at


class LiveSessionManager:
    """
    Manages all active live streaming sessions.

    Handles session lifecycle, callbacks, and cleanup.
    """

    def __init__(self):
        self._sessions: Dict[str, LiveSession] = {}
        self._lock = threading.Lock()

        # Callbacks for events
        self._on_status: Optional[Callable[[str, dict], None]] = None
        self._on_frame: Optional[Callable[[str, str, float], None]] = None
        self._on_transcript: Optional[Callable[[str, str, float], None]] = None
        self._on_analysis: Optional[Callable[[str, AnalysisResult], None]] = None
        self._on_error: Optional[Callable[[str, str], None]] = None
        self._on_state_change: Optional[Callable[[str, SessionState], None]] = None

        ensure_live_dirs()

    def set_callbacks(
        self,
        on_status: Optional[Callable[[str, dict], None]] = None,
        on_frame: Optional[Callable[[str, str, float], None]] = None,
        on_transcript: Optional[Callable[[str, str, float], None]] = None,
        on_analysis: Optional[Callable[[str, AnalysisResult], None]] = None,
        on_error: Optional[Callable[[str, str], None]] = None,
        on_state_change: Optional[Callable[[str, SessionState], None]] = None,
    ):
        """Set callback functions for session events."""
        self._on_status = on_status
        self._on_frame = on_frame
        self._on_transcript = on_transcript
        self._on_analysis = on_analysis
        self._on_error = on_error
        self._on_state_change = on_state_change

    def create_session(
        self,
        twitch_url: str,
        quality: str = "480p",
        analysis_interval: float = 30.0,
    ) -> LiveSession:
        """
        Create a new live streaming session.

        Args:
            twitch_url: Twitch stream URL
            quality: Stream quality preset
            analysis_interval: Seconds between analyses

        Returns:
            Created LiveSession object
        """
        # Create config
        config = LiveStreamConfig(
            twitch_url=twitch_url,
            frame_quality=quality,
            frame_rate=1,  # 1 fps for analysis
            audio_segment_length=5.0,
        )

        session = LiveSession(
            session_id=config.session_id,
            twitch_url=twitch_url,
            quality=quality,
            config=config,
        )

        # Create analyzer with callbacks that route through session manager
        def on_status(status: dict):
            if self._on_status:
                self._on_status(session.session_id, status)

        def on_frame(frame_b64: str, timestamp: float):
            if self._on_frame:
                self._on_frame(session.session_id, frame_b64, timestamp)

        def on_transcript(text: str, timestamp: float):
            if self._on_transcript:
                self._on_transcript(session.session_id, text, timestamp)

        def on_analysis(result: AnalysisResult):
            if self._on_analysis:
                self._on_analysis(session.session_id, result)

        def on_error(error: str):
            session.error_message = error
            if self._on_error:
                self._on_error(session.session_id, error)

        session.analyzer = LiveAnalyzer(
            config=config,
            analysis_interval=analysis_interval,
            on_status=on_status,
            on_frame=on_frame,
            on_transcript=on_transcript,
            on_analysis=on_analysis,
            on_error=on_error,
        )

        with self._lock:
            self._sessions[session.session_id] = session

        return session

    def start_session(self, session_id: str) -> bool:
        """
        Start a session's live capture and analysis.

        Args:
            session_id: ID of session to start

        Returns:
            True if started successfully
        """
        session = self.get_session(session_id)
        if not session:
            return False

        if session.state not in [SessionState.CREATED, SessionState.STOPPED]:
            return False

        try:
            session.state = SessionState.CONNECTING
            self._emit_state_change(session)

            session.analyzer.start()

            session.state = SessionState.RUNNING
            session.started_at = time.time()
            self._emit_state_change(session)

            return True
        except Exception as e:
            session.state = SessionState.ERROR
            session.error_message = str(e)
            self._emit_state_change(session)

            if self._on_error:
                self._on_error(session_id, str(e))

            return False

    def stop_session(self, session_id: str) -> bool:
        """
        Stop a session's capture and analysis.

        Args:
            session_id: ID of session to stop

        Returns:
            True if stopped successfully
        """
        session = self.get_session(session_id)
        if not session:
            return False

        if session.state not in [SessionState.RUNNING, SessionState.CONNECTING]:
            return False

        try:
            session.analyzer.stop()
            session.state = SessionState.STOPPED
            session.stopped_at = time.time()
            self._emit_state_change(session)

            # Save session summary
            self._save_session_summary(session)

            return True
        except Exception as e:
            session.error_message = str(e)
            if self._on_error:
                self._on_error(session_id, str(e))
            return False

    def remove_session(self, session_id: str) -> bool:
        """
        Remove a session from the manager.

        Args:
            session_id: ID of session to remove

        Returns:
            True if removed successfully
        """
        session = self.get_session(session_id)
        if not session:
            return False

        # Stop if running
        if session.state == SessionState.RUNNING:
            self.stop_session(session_id)

        with self._lock:
            del self._sessions[session_id]

        return True

    def get_session(self, session_id: str) -> Optional[LiveSession]:
        """Get a session by ID."""
        with self._lock:
            return self._sessions.get(session_id)

    def get_all_sessions(self) -> Dict[str, LiveSession]:
        """Get all sessions."""
        with self._lock:
            return dict(self._sessions)

    def get_active_sessions(self) -> Dict[str, LiveSession]:
        """Get only running sessions."""
        with self._lock:
            return {
                sid: s for sid, s in self._sessions.items()
                if s.state == SessionState.RUNNING
            }

    def get_session_status(self, session_id: str) -> Optional[dict]:
        """Get detailed status for a session."""
        session = self.get_session(session_id)
        if not session:
            return None

        status = session.to_dict()

        # Add buffer stats if running
        if session.analyzer and session.state == SessionState.RUNNING:
            buffer_stats = session.analyzer.buffer.get_stats()
            status["buffer"] = {
                "frames_in_buffer": buffer_stats.frames_in_buffer,
                "audio_segments_in_buffer": buffer_stats.audio_segments_in_buffer,
                "buffer_duration": buffer_stats.buffer_duration_seconds,
            }

            # Add latest analysis
            if session.analyzer.latest_result:
                status["latest_analysis"] = session.analyzer.latest_result.to_dict()

            # Add accumulated context
            status["accumulated_context"] = session.analyzer.accumulated_context.to_dict()

            # Add frame preview
            if session.analyzer.frame_preview:
                status["frame_preview"] = session.analyzer.frame_preview

        return status

    def _emit_state_change(self, session: LiveSession):
        """Emit state change callback."""
        if self._on_state_change:
            self._on_state_change(session.session_id, session.state)

    def _save_session_summary(self, session: LiveSession):
        """Save session summary to disk."""
        if not session.config:
            return

        summary_path = session.config.output_dir / "session_summary.json"
        session.config.ensure_dirs()

        summary = {
            "session_id": session.session_id,
            "twitch_url": session.twitch_url,
            "channel_name": session.config.channel_name,
            "quality": session.quality,
            "created_at": datetime.fromtimestamp(session.created_at).isoformat(),
            "started_at": datetime.fromtimestamp(session.started_at).isoformat() if session.started_at else None,
            "stopped_at": datetime.fromtimestamp(session.stopped_at).isoformat() if session.stopped_at else None,
            "duration_seconds": session.duration,
        }

        if session.analyzer:
            summary["session_summary"] = session.analyzer.get_session_summary()
            summary["all_analyses"] = [r.to_dict() for r in session.analyzer.analysis_results]

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)


# Global session manager instance
_session_manager: Optional[LiveSessionManager] = None


def get_session_manager() -> LiveSessionManager:
    """Get the global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = LiveSessionManager()
    return _session_manager
