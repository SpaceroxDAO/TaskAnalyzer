"""
TaskAnalyzer Web UI
===================
Simple Flask web interface for analyzing videos via drag-and-drop.
Enhanced with rich context extraction for automation building.
"""

import os
import json
import uuid
import threading
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory

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

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload

# Store analysis jobs and their status
jobs = {}

# Upload directory
UPLOAD_DIR = config.PROJECT_ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


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


@app.route("/")
def index():
    """Main page with drag-drop upload."""
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


if __name__ == "__main__":
    # Ensure output directories exist
    config.ensure_output_dirs()

    print("=" * 60)
    print("  TaskAnalyzer Web UI (Enhanced)")
    print("=" * 60)
    print()
    print("  Open in browser: http://localhost:5001")
    print()
    print("  Features:")
    print("    - Video frame extraction")
    print("    - Audio transcription (Whisper)")
    print("    - Claude Vision analysis")
    print("    - Rich context extraction:")
    print("      * OCR from frames")
    print("      * Named entity extraction")
    print("      * Action verb detection")
    print("      * Frustration/sentiment analysis")
    print("      * Automation potential scoring")
    print()
    print("=" * 60)

    app.run(host="0.0.0.0", port=5001, debug=True, threaded=True)
