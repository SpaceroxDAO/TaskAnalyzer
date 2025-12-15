"""
Transcriber Module
==================
Transcribes audio files using OpenAI's Whisper model.

Responsibilities:
- Load Whisper model (local, offline)
- Transcribe audio to text
- Save transcript to disk
"""

from pathlib import Path

import config


def transcribe_audio(audio_path: str, output_path: Path = None) -> str:
    """
    Transcribe audio file to text using Whisper.

    Args:
        audio_path: Path to the audio file (WAV format)
        output_path: Path to save transcript (default: config.TRANSCRIPT_FILE)

    Returns:
        Transcript text

    Raises:
        FileNotFoundError: If audio file doesn't exist
        ImportError: If whisper is not installed
        RuntimeError: If transcription fails
    """
    # Import whisper here to allow other modules to work without it installed
    try:
        import whisper
    except ImportError:
        raise ImportError(
            "Whisper is not installed. Install with: pip install openai-whisper"
        )

    # Validate input
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Setup output path
    output_path = output_path or config.TRANSCRIPT_FILE
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Transcriber] Loading Whisper model: {config.WHISPER_MODEL}")
    model = whisper.load_model(config.WHISPER_MODEL)

    print(f"[Transcriber] Transcribing audio: {audio_path.name}")
    result = model.transcribe(str(audio_path))

    transcript = result["text"].strip()

    # Save transcript to file
    output_path.write_text(transcript, encoding="utf-8")

    print(f"[Transcriber] Transcript saved to: {output_path}")
    print(f"[Transcriber] Transcript length: {len(transcript)} characters")

    # Print preview
    preview = transcript[:200] + "..." if len(transcript) > 200 else transcript
    print(f"[Transcriber] Preview: {preview}")

    return transcript


if __name__ == "__main__":
    # Test module independently
    import sys

    if len(sys.argv) < 2:
        print("Usage: python transcriber.py <audio_path>")
        sys.exit(1)

    audio_path = sys.argv[1]
    config.ensure_output_dirs()

    try:
        transcript = transcribe_audio(audio_path)
        print(f"\nFull transcript:\n{transcript}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
