"""
TaskAnalyzer Live Analyzer
==========================
Coordinates periodic analysis of live stream data.
Runs Whisper transcription and Claude Vision analysis on buffered data.
"""

import base64
import json
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any
import tempfile

import numpy as np

from live_config import LiveStreamConfig
from live_stream import TwitchStreamCapture, CapturedFrame, CapturedAudio
from stream_buffer import StreamBuffer, AnalysisWindow


@dataclass
class AnalysisResult:
    """Result of a single analysis cycle."""
    timestamp: float
    analysis_number: int

    # Transcription
    transcript: str = ""
    transcript_duration: float = 0.0

    # Task analysis from Claude
    task_name: str = ""
    confidence: float = 0.0
    steps: List[str] = field(default_factory=list)
    completion_detected: bool = False

    # Context extraction
    detected_apps: List[str] = field(default_factory=list)
    detected_urls: List[str] = field(default_factory=list)
    detected_actions: List[str] = field(default_factory=list)
    pain_points: List[str] = field(default_factory=list)

    # Metadata
    frames_analyzed: int = 0
    audio_seconds_analyzed: float = 0.0
    processing_time: float = 0.0

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "analysis_number": self.analysis_number,
            "transcript": self.transcript,
            "transcript_duration": self.transcript_duration,
            "task_name": self.task_name,
            "confidence": self.confidence,
            "steps": self.steps,
            "completion_detected": self.completion_detected,
            "detected_apps": self.detected_apps,
            "detected_urls": self.detected_urls,
            "detected_actions": self.detected_actions,
            "pain_points": self.pain_points,
            "frames_analyzed": self.frames_analyzed,
            "audio_seconds_analyzed": self.audio_seconds_analyzed,
            "processing_time": self.processing_time,
        }


@dataclass
class AccumulatedContext:
    """Context accumulated over the entire session."""
    all_apps: set = field(default_factory=set)
    all_urls: set = field(default_factory=set)
    all_actions: set = field(default_factory=set)
    all_pain_points: List[str] = field(default_factory=list)
    task_history: List[Dict[str, Any]] = field(default_factory=list)
    full_transcript: str = ""

    def add_result(self, result: AnalysisResult):
        """Merge a new analysis result into accumulated context."""
        self.all_apps.update(result.detected_apps)
        self.all_urls.update(result.detected_urls)
        self.all_actions.update(result.detected_actions)
        self.all_pain_points.extend(result.pain_points)

        # Track task changes
        if result.task_name:
            if not self.task_history or self.task_history[-1]["task_name"] != result.task_name:
                self.task_history.append({
                    "task_name": result.task_name,
                    "confidence": result.confidence,
                    "timestamp": result.timestamp,
                    "steps": result.steps,
                })

        # Append transcript
        if result.transcript:
            if self.full_transcript:
                self.full_transcript += "\n" + result.transcript
            else:
                self.full_transcript = result.transcript

    def to_dict(self) -> dict:
        return {
            "all_apps": list(self.all_apps),
            "all_urls": list(self.all_urls),
            "all_actions": list(self.all_actions),
            "all_pain_points": self.all_pain_points,
            "task_history": self.task_history,
            "full_transcript": self.full_transcript,
        }


class LiveAnalyzer:
    """
    Coordinates live stream capture and periodic analysis.

    Manages:
    - Stream capture (frames + audio)
    - Buffer management
    - Periodic analysis triggers
    - Whisper transcription
    - Claude Vision analysis
    - Context extraction and accumulation
    """

    def __init__(
        self,
        config: LiveStreamConfig,
        analysis_interval: float = 30.0,
        on_status: Optional[Callable[[dict], None]] = None,
        on_frame: Optional[Callable[[str, float], None]] = None,
        on_transcript: Optional[Callable[[str, float], None]] = None,
        on_analysis: Optional[Callable[[AnalysisResult], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ):
        self.config = config
        self.analysis_interval = analysis_interval

        # Callbacks
        self._on_status = on_status
        self._on_frame = on_frame
        self._on_transcript = on_transcript
        self._on_analysis = on_analysis
        self._on_error = on_error

        # Components
        self.capture: Optional[TwitchStreamCapture] = None
        self.buffer = StreamBuffer(
            max_frames=config.max_frame_buffer,
            max_audio_segments=config.max_audio_buffer,
            max_buffer_duration=300.0  # 5 minutes
        )
        self.analysis_window = AnalysisWindow(window_size=analysis_interval)

        # State
        self._running = False
        self._analysis_thread: Optional[threading.Thread] = None
        self._status_thread: Optional[threading.Thread] = None
        self._accumulated_context = AccumulatedContext()
        self._analysis_results: List[AnalysisResult] = []
        self._last_frame_preview: Optional[str] = None
        self._last_frame_time: float = 0

        # Load models lazily
        self._whisper_model = None
        self._anthropic_client = None

    def _load_whisper(self):
        """Lazy load Whisper model."""
        if self._whisper_model is None:
            import whisper
            from config import WHISPER_MODEL
            self._whisper_model = whisper.load_model(WHISPER_MODEL)
        return self._whisper_model

    def _load_anthropic(self):
        """Lazy load Anthropic client."""
        if self._anthropic_client is None:
            import anthropic
            from config import ANTHROPIC_API_KEY
            self._anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        return self._anthropic_client

    def _on_captured_frame(self, frame: CapturedFrame):
        """Handle each captured frame."""
        self.buffer.add_frame(frame)

        # Update frame preview every 3 seconds
        now = time.time()
        if now - self._last_frame_time >= 3.0:
            self._last_frame_time = now
            # Encode frame as base64 JPEG for preview
            from PIL import Image
            import io

            img = Image.fromarray(frame.data)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=70)
            b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            self._last_frame_preview = b64

            if self._on_frame:
                self._on_frame(b64, frame.timestamp)

    def _on_captured_audio(self, segment: CapturedAudio):
        """Handle each captured audio segment."""
        self.buffer.add_audio(segment)

    def _on_capture_error(self, error: Exception):
        """Handle capture errors."""
        if self._on_error:
            self._on_error(str(error))

    def _emit_status(self):
        """Emit current status to callback."""
        if not self._on_status:
            return

        stats = self.buffer.get_stats()
        status = {
            "connected": self._running and self.capture and self.capture.is_running,
            "frames_captured": stats.total_frames_received,
            "frames_in_buffer": stats.frames_in_buffer,
            "audio_duration": stats.total_audio_duration,
            "buffer_duration": stats.buffer_duration_seconds,
            "analyses_completed": len(self._analysis_results),
            "current_task": self._analysis_results[-1].task_name if self._analysis_results else "",
        }
        self._on_status(status)

    def _status_loop(self):
        """Background thread to emit status updates."""
        while self._running:
            self._emit_status()
            time.sleep(1.0)

    def _analysis_loop(self):
        """Background thread for periodic analysis."""
        while self._running:
            # Check if we should run analysis
            if self.analysis_window.should_analyze(self.buffer):
                try:
                    result = self._run_analysis()
                    if result:
                        self._analysis_results.append(result)
                        self._accumulated_context.add_result(result)

                        if self._on_analysis:
                            self._on_analysis(result)

                        self.analysis_window.mark_analyzed()
                except Exception as e:
                    if self._on_error:
                        self._on_error(f"Analysis error: {e}")

            time.sleep(1.0)

    def _run_analysis(self) -> Optional[AnalysisResult]:
        """Run a single analysis cycle on buffered data."""
        start_time = time.time()
        analysis_num = len(self._analysis_results) + 1

        result = AnalysisResult(
            timestamp=time.time(),
            analysis_number=analysis_num,
        )

        # Get frames for analysis
        frames = self.buffer.get_frames_for_analysis(
            count=15,
            time_window=self.analysis_interval
        )

        if not frames:
            return None

        result.frames_analyzed = len(frames)

        # Transcribe audio
        transcript = self._transcribe_audio()
        if transcript:
            result.transcript = transcript
            result.audio_seconds_analyzed = self.buffer.get_audio_duration()

            if self._on_transcript:
                self._on_transcript(transcript, time.time())

        # Run Claude Vision analysis
        try:
            vision_result = self._analyze_with_claude(frames, transcript)
            if vision_result:
                result.task_name = vision_result.get("task_name", "")
                result.confidence = vision_result.get("confidence", 0.0)
                result.steps = vision_result.get("steps", [])
                result.completion_detected = vision_result.get("completion_detected", False)
        except Exception as e:
            if self._on_error:
                self._on_error(f"Claude analysis error: {e}")

        # Extract context from transcript
        if transcript:
            context = self._extract_context(transcript)
            result.detected_apps = context.get("apps", [])
            result.detected_urls = context.get("urls", [])
            result.detected_actions = context.get("actions", [])
            result.pain_points = context.get("pain_points", [])

        result.processing_time = time.time() - start_time
        return result

    def _transcribe_audio(self) -> str:
        """Transcribe buffered audio using Whisper."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = Path(f.name)

        try:
            saved_path = self.buffer.save_audio_for_whisper(
                temp_path,
                time_window=self.analysis_interval
            )

            if not saved_path:
                return ""

            model = self._load_whisper()
            result = model.transcribe(str(saved_path))
            return result.get("text", "").strip()
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def _analyze_with_claude(self, frames: List[CapturedFrame], transcript: str) -> dict:
        """Analyze frames with Claude Vision."""
        from config import CLAUDE_MODEL, VISION_PROMPT, MAX_TOKENS

        client = self._load_anthropic()

        # Encode frames as base64
        from PIL import Image
        import io

        image_content = []
        for frame in frames:
            img = Image.fromarray(frame.data)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            image_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": b64,
                }
            })

        # Add transcript as text
        user_content = image_content + [{
            "type": "text",
            "text": f"Transcript of what the person said:\n\n{transcript}" if transcript else "No speech detected."
        }]

        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=MAX_TOKENS,
            system=VISION_PROMPT,
            messages=[{"role": "user", "content": user_content}]
        )

        # Parse JSON response
        response_text = response.content[0].text

        # Try to extract JSON
        try:
            # Handle markdown code blocks
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0]
            else:
                json_str = response_text

            return json.loads(json_str.strip())
        except json.JSONDecodeError:
            return {"task_name": "Unknown", "confidence": 0.0, "steps": [], "completion_detected": False}

    def _extract_context(self, transcript: str) -> dict:
        """Extract context from transcript text."""
        import re

        context = {
            "apps": [],
            "urls": [],
            "actions": [],
            "pain_points": [],
        }

        # Common app names to detect
        app_patterns = [
            r'\b(chrome|firefox|safari|edge)\b',
            r'\b(vs ?code|visual studio|sublime|atom|vim|emacs)\b',
            r'\b(slack|discord|teams|zoom)\b',
            r'\b(excel|word|powerpoint|sheets|docs)\b',
            r'\b(figma|sketch|photoshop|illustrator)\b',
            r'\b(terminal|command line|bash|powershell)\b',
            r'\b(github|gitlab|bitbucket)\b',
            r'\b(jira|trello|asana|notion)\b',
        ]

        lower_transcript = transcript.lower()
        for pattern in app_patterns:
            matches = re.findall(pattern, lower_transcript, re.IGNORECASE)
            context["apps"].extend(matches)

        context["apps"] = list(set(context["apps"]))

        # URL patterns
        url_pattern = r'(?:https?://)?(?:www\.)?([a-zA-Z0-9-]+(?:\.[a-zA-Z]{2,})+)'
        urls = re.findall(url_pattern, transcript)
        context["urls"] = list(set(urls))

        # Action verbs
        action_patterns = [
            r'\b(click(?:ing|ed)?|tap(?:ping|ped)?)\b',
            r'\b(type|typing|typed|enter(?:ing|ed)?)\b',
            r'\b(scroll(?:ing|ed)?|swipe[ds]?)\b',
            r'\b(open(?:ing|ed)?|clos(?:e|ing|ed))\b',
            r'\b(copy|paste|cut|delet(?:e|ing|ed))\b',
            r'\b(search(?:ing|ed)?|find(?:ing)?|found)\b',
            r'\b(select(?:ing|ed)?|choose|chose|chosen)\b',
            r'\b(upload(?:ing|ed)?|download(?:ing|ed)?)\b',
            r'\b(save[ds]?|saving|submit(?:ted|ting)?)\b',
        ]

        for pattern in action_patterns:
            matches = re.findall(pattern, lower_transcript, re.IGNORECASE)
            context["actions"].extend(matches)

        context["actions"] = list(set(context["actions"]))

        # Pain point indicators
        pain_indicators = [
            r"(?:this is |it's |that's )?(?:so |really |very )?(frustrating|annoying|confusing)",
            r"(?:i |we )?(?:can't|cannot|couldn't) (?:find|figure out|understand)",
            r"(?:why (?:is|does|doesn't|won't))",
            r"(?:this (?:doesn't|won't|isn't) work)",
            r"(?:i (?:don't|didn't) (?:know|understand))",
            r"(?:where (?:is|did|do))",
            r"(?:(?:ugh|argh|damn|darn))",
        ]

        for pattern in pain_indicators:
            matches = re.findall(pattern, lower_transcript, re.IGNORECASE)
            if matches:
                # Get surrounding context
                for match in re.finditer(pattern, lower_transcript, re.IGNORECASE):
                    start = max(0, match.start() - 30)
                    end = min(len(transcript), match.end() + 30)
                    context["pain_points"].append(transcript[start:end].strip())

        return context

    def start(self):
        """Start live capture and analysis."""
        if self._running:
            return

        self._running = True
        self.config.ensure_dirs()

        # Initialize capture
        self.capture = TwitchStreamCapture(self.config)
        self.capture.on_frame(self._on_captured_frame)
        self.capture.on_audio(self._on_captured_audio)
        self.capture.on_error(self._on_capture_error)

        # Start capture
        self.capture.start()

        # Start analysis thread
        self._analysis_thread = threading.Thread(
            target=self._analysis_loop,
            daemon=True
        )
        self._analysis_thread.start()

        # Start status thread
        self._status_thread = threading.Thread(
            target=self._status_loop,
            daemon=True
        )
        self._status_thread.start()

    def stop(self):
        """Stop capture and analysis."""
        self._running = False

        if self.capture:
            self.capture.stop()

        if self._analysis_thread:
            self._analysis_thread.join(timeout=5)

        if self._status_thread:
            self._status_thread.join(timeout=2)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def accumulated_context(self) -> AccumulatedContext:
        return self._accumulated_context

    @property
    def analysis_results(self) -> List[AnalysisResult]:
        return self._analysis_results

    @property
    def latest_result(self) -> Optional[AnalysisResult]:
        return self._analysis_results[-1] if self._analysis_results else None

    @property
    def frame_preview(self) -> Optional[str]:
        return self._last_frame_preview

    def get_session_summary(self) -> dict:
        """Get a summary of the entire session."""
        stats = self.buffer.get_stats()
        return {
            "session_id": self.config.session_id,
            "channel": self.config.channel_name,
            "duration": stats.buffer_duration_seconds,
            "total_frames": stats.total_frames_received,
            "total_audio": stats.total_audio_duration,
            "analyses_completed": len(self._analysis_results),
            "accumulated_context": self._accumulated_context.to_dict(),
            "latest_task": self.latest_result.task_name if self.latest_result else "",
            "latest_confidence": self.latest_result.confidence if self.latest_result else 0.0,
        }
