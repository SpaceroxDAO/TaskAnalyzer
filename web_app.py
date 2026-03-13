"""
TaskAnalyzer Web UI
===================
Flask web interface for analyzing videos via drag-and-drop.
Enhanced with rich context extraction and live Twitch stream analysis.
"""

import os
import json
import uuid
import threading
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit

import config
from frame_extractor import extract_frames, get_video_duration
from audio_extractor import extract_audio
from transcriber import transcribe_audio
from task_analyzer import analyze_task
from context_extractor import (
    extract_full_context,
    context_to_dict,
    extract_transcript_context,
    extract_all_frames_context,
    analyze_automation_potential
)

# Live streaming imports
from live_config import LiveStreamConfig, QUALITY_PRESETS
from live_session import (
    LiveSessionManager,
    LiveSession,
    SessionState,
    get_session_manager
)
from live_stream import check_stream_available

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'taskanalyzer-secret-key')

# Initialize SocketIO with eventlet for async support
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Store analysis jobs and their status
jobs = {}

# Upload directory
UPLOAD_DIR = config.PROJECT_ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize session manager with callbacks
session_manager = get_session_manager()


# =============================================================================
# SOCKETIO CALLBACKS FOR LIVE STREAMING
# =============================================================================

def on_live_status(session_id: str, status: dict):
    """Callback when live session status updates."""
    socketio.emit('live:status', {
        'session_id': session_id,
        **status
    })


def on_live_frame(session_id: str, frame_b64: str, timestamp: float):
    """Callback when a new frame is available for preview."""
    socketio.emit('live:frame', {
        'session_id': session_id,
        'frame': frame_b64,
        'timestamp': timestamp
    })


def on_live_transcript(session_id: str, text: str, timestamp: float):
    """Callback when new transcript is available."""
    socketio.emit('live:transcript', {
        'session_id': session_id,
        'text': text,
        'timestamp': timestamp
    })


def on_live_analysis(session_id: str, result):
    """Callback when analysis completes."""
    socketio.emit('live:analysis', {
        'session_id': session_id,
        'result': result.to_dict()
    })


def on_live_error(session_id: str, error: str):
    """Callback when an error occurs."""
    socketio.emit('live:error', {
        'session_id': session_id,
        'error': error
    })


def on_live_state_change(session_id: str, state: SessionState):
    """Callback when session state changes."""
    socketio.emit('live:state', {
        'session_id': session_id,
        'state': state.value
    })


# Set up session manager callbacks
session_manager.set_callbacks(
    on_status=on_live_status,
    on_frame=on_live_frame,
    on_transcript=on_live_transcript,
    on_analysis=on_live_analysis,
    on_error=on_live_error,
    on_state_change=on_live_state_change,
)


# =============================================================================
# BATCH VIDEO ANALYSIS (Original functionality)
# =============================================================================

class AnalysisJob:
    """Tracks the state of an analysis job."""

    def __init__(self, job_id: str, video_path: Path):
        self.job_id = job_id
        self.video_path = video_path
        self.status = "pending"
        self.current_stage = ""
        self.progress = 0
        self.logs = []
        self.error = None
        self.started_at = datetime.now()
        self.completed_at = None

        # Results
        self.video_duration = None
        self.frames = []
        self.audio_path = None
        self.transcript = None
        self.analysis = None

        # Rich context extraction
        self.context = None  # Full context from context_extractor

        # Debug info
        self.api_request = None
        self.api_response = None

    def log(self, message: str):
        """Add a log entry."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")

    def to_dict(self):
        """Convert to JSON-serializable dict."""
        return {
            "job_id": self.job_id,
            "status": self.status,
            "current_stage": self.current_stage,
            "progress": self.progress,
            "logs": self.logs,
            "error": self.error,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "video_duration": self.video_duration,
            "frame_count": len(self.frames),
            "frames": [f.name for f in self.frames],
            "transcript": self.transcript,
            "analysis": self.analysis,
            "context": self.context,
            "api_request": self.api_request,
            "api_response": self.api_response,
        }


def run_analysis(job: AnalysisJob):
    """Run the full analysis pipeline in a background thread."""
    try:
        job.status = "running"

        # Create job-specific output directory
        job_output_dir = config.OUTPUT_DIR / job.job_id
        job_output_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = job_output_dir / "frames"
        audio_file = job_output_dir / "audio.wav"
        transcript_file = job_output_dir / "transcript.txt"
        analysis_file = job_output_dir / "task_analysis.json"
        context_file = job_output_dir / "context.json"

        # Stage 1: Get video duration
        job.current_stage = "Getting video info"
        job.progress = 5
        job.log("Getting video duration...")
        job.video_duration = get_video_duration(str(job.video_path))
        job.log(f"Video duration: {job.video_duration:.2f} seconds")

        # Stage 2: Extract frames
        job.current_stage = "Extracting frames"
        job.progress = 10
        job.log("Extracting frames with FFmpeg...")
        job.frames = extract_frames(str(job.video_path), frames_dir)
        job.log(f"Extracted {len(job.frames)} frames")
        job.progress = 25

        # Stage 3: Extract audio
        job.current_stage = "Extracting audio"
        job.progress = 30
        job.log("Extracting audio with FFmpeg...")
        job.audio_path = extract_audio(str(job.video_path), audio_file)
        job.log(f"Audio extracted: {audio_file.stat().st_size / 1024:.1f} KB")
        job.progress = 40

        # Stage 4: Transcribe
        job.current_stage = "Transcribing audio"
        job.progress = 45
        job.log(f"Loading Whisper model: {config.WHISPER_MODEL}")
        job.transcript = transcribe_audio(str(audio_file), transcript_file)
        job.log(f"Transcription complete: {len(job.transcript)} characters")
        job.progress = 60

        # Stage 5: Extract rich context
        job.current_stage = "Extracting context"
        job.progress = 65
        job.log("Extracting rich context from frames and transcript...")

        frame_interval = job.video_duration / len(job.frames) if job.frames else 3.0
        full_context = extract_full_context(
            frames=job.frames,
            transcript=job.transcript,
            video_duration=job.video_duration,
            frame_interval=frame_interval
        )
        job.context = context_to_dict(full_context)

        # Save context to file
        with open(context_file, "w", encoding="utf-8") as f:
            json.dump(job.context, f, indent=2)

        job.log(f"Context extracted:")
        job.log(f"  - Applications detected: {job.context['transcript']['applications']}")
        job.log(f"  - URLs found: {len(job.context['transcript']['urls'])}")
        job.log(f"  - Action verbs: {len(job.context['transcript']['action_verbs'])}")
        job.log(f"  - Frustration indicators: {len(job.context['transcript']['frustration_indicators'])}")
        job.log(f"  - Automation score: {job.context['automation']['automation_candidate_score']:.2f}")
        job.progress = 75

        # Stage 6: Analyze with Claude
        job.current_stage = "Analyzing with Claude Vision"
        job.progress = 80
        job.log(f"Sending {len(job.frames)} frames to Claude ({config.CLAUDE_MODEL})...")

        # Capture API request info for debugging
        job.api_request = {
            "model": config.CLAUDE_MODEL,
            "max_tokens": config.MAX_TOKENS,
            "frame_count": len(job.frames),
            "transcript_length": len(job.transcript),
            "system_prompt": config.VISION_PROMPT[:200] + "..."
        }

        job.analysis = analyze_task(job.frames, job.transcript, analysis_file)

        job.api_response = {
            "task_name": job.analysis.get("task_name"),
            "confidence": job.analysis.get("confidence"),
            "step_count": len(job.analysis.get("steps", [])),
            "completion_detected": job.analysis.get("completion_detected")
        }

        job.log(f"Analysis complete: {job.analysis.get('task_name')}")
        job.progress = 100

        # Done
        job.status = "completed"
        job.current_stage = "Complete"
        job.completed_at = datetime.now()
        elapsed = (job.completed_at - job.started_at).total_seconds()
        job.log(f"Pipeline finished in {elapsed:.1f} seconds")

    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        job.log(f"ERROR: {e}")
        import traceback
        job.log(traceback.format_exc())


# =============================================================================
# HTTP ROUTES - BATCH ANALYSIS
# =============================================================================

@app.route("/")
def index():
    """Main page with drag-drop upload and live stream interface."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_video():
    """Handle video upload and start analysis."""
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Generate job ID
    job_id = str(uuid.uuid4())[:8]

    # Save uploaded file
    filename = f"{job_id}_{file.filename}"
    video_path = UPLOAD_DIR / filename
    file.save(video_path)

    # Create and start job
    job = AnalysisJob(job_id, video_path)
    job.log(f"Video uploaded: {file.filename}")
    jobs[job_id] = job

    # Run analysis in background thread
    thread = threading.Thread(target=run_analysis, args=(job,))
    thread.daemon = True
    thread.start()

    return jsonify({"job_id": job_id, "status": "started"})


@app.route("/status/<job_id>")
def get_status(job_id):
    """Get the current status of an analysis job."""
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404

    return jsonify(jobs[job_id].to_dict())


@app.route("/frames/<job_id>/<filename>")
def get_frame(job_id, filename):
    """Serve extracted frame images."""
    frames_dir = config.OUTPUT_DIR / job_id / "frames"
    return send_from_directory(frames_dir, filename)


@app.route("/jobs")
def list_jobs():
    """List all jobs."""
    return jsonify({
        job_id: {
            "status": job.status,
            "video": job.video_path.name,
            "started_at": job.started_at.isoformat()
        }
        for job_id, job in jobs.items()
    })


# =============================================================================
# HTTP ROUTES - LIVE STREAMING
# =============================================================================

@app.route("/api/live/check", methods=["POST"])
def check_stream():
    """Check if a Twitch stream is available."""
    data = request.get_json()
    twitch_url = data.get("url", "")

    if not twitch_url:
        return jsonify({"error": "No URL provided"}), 400

    # Normalize URL
    if not twitch_url.startswith("http"):
        twitch_url = f"https://twitch.tv/{twitch_url}"

    available, qualities = check_stream_available(twitch_url)

    return jsonify({
        "available": available,
        "qualities": qualities,
        "url": twitch_url
    })


@app.route("/api/live/sessions", methods=["GET"])
def list_live_sessions():
    """List all live streaming sessions."""
    sessions = session_manager.get_all_sessions()
    return jsonify({
        sid: session.to_dict()
        for sid, session in sessions.items()
    })


@app.route("/api/live/sessions", methods=["POST"])
def create_live_session():
    """Create a new live streaming session."""
    data = request.get_json()
    twitch_url = data.get("url", "")
    quality = data.get("quality", "480p")
    analysis_interval = data.get("analysis_interval", 30)

    if not twitch_url:
        return jsonify({"error": "No URL provided"}), 400

    # Normalize URL
    if not twitch_url.startswith("http"):
        twitch_url = f"https://twitch.tv/{twitch_url}"

    try:
        session = session_manager.create_session(
            twitch_url=twitch_url,
            quality=quality,
            analysis_interval=analysis_interval
        )
        return jsonify({
            "session_id": session.session_id,
            "status": "created",
            "session": session.to_dict()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/live/sessions/<session_id>", methods=["GET"])
def get_live_session(session_id):
    """Get details of a specific live session."""
    status = session_manager.get_session_status(session_id)
    if not status:
        return jsonify({"error": "Session not found"}), 404
    return jsonify(status)


@app.route("/api/live/sessions/<session_id>/start", methods=["POST"])
def start_live_session(session_id):
    """Start a live streaming session."""
    success = session_manager.start_session(session_id)
    if success:
        return jsonify({"status": "started", "session_id": session_id})
    else:
        return jsonify({"error": "Failed to start session"}), 500


@app.route("/api/live/sessions/<session_id>/stop", methods=["POST"])
def stop_live_session(session_id):
    """Stop a live streaming session."""
    success = session_manager.stop_session(session_id)
    if success:
        return jsonify({"status": "stopped", "session_id": session_id})
    else:
        return jsonify({"error": "Failed to stop session"}), 500


@app.route("/api/live/sessions/<session_id>", methods=["DELETE"])
def delete_live_session(session_id):
    """Delete a live streaming session."""
    success = session_manager.remove_session(session_id)
    if success:
        return jsonify({"status": "deleted", "session_id": session_id})
    else:
        return jsonify({"error": "Failed to delete session"}), 404


@app.route("/api/live/qualities")
def get_quality_presets():
    """Get available quality presets."""
    return jsonify(QUALITY_PRESETS)


# =============================================================================
# SOCKETIO EVENT HANDLERS
# =============================================================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print(f"Client connected: {request.sid}")


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print(f"Client disconnected: {request.sid}")


@socketio.on('live:subscribe')
def handle_subscribe(data):
    """Subscribe to updates for a specific session."""
    session_id = data.get('session_id')
    print(f"Client {request.sid} subscribed to session {session_id}")
    # Could implement room-based routing here if needed


@socketio.on('live:unsubscribe')
def handle_unsubscribe(data):
    """Unsubscribe from session updates."""
    session_id = data.get('session_id')
    print(f"Client {request.sid} unsubscribed from session {session_id}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Ensure output directories exist
    config.ensure_output_dirs()

    print("=" * 60)
    print("  TaskAnalyzer Web UI")
    print("=" * 60)
    print()
    print("  Open in browser: http://localhost:5001")
    print()
    print("  Features:")
    print("    BATCH ANALYSIS:")
    print("    - Video frame extraction")
    print("    - Audio transcription (Whisper)")
    print("    - Claude Vision analysis")
    print("    - Rich context extraction")
    print()
    print("    LIVE STREAMING:")
    print("    - Twitch stream capture")
    print("    - Real-time frame preview")
    print("    - Periodic analysis (every 30s)")
    print("    - Live transcript updates")
    print("    - WebSocket real-time updates")
    print()
    print("=" * 60)

    # Use socketio.run instead of app.run for WebSocket support
    socketio.run(app, host="0.0.0.0", port=5001, debug=True, allow_unsafe_werkzeug=True)
