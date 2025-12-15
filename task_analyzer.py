"""
Task Analyzer Module
====================
Analyzes video frames and transcript using Claude Vision API.

Responsibilities:
- Encode frames as base64
- Build prompt with frames and transcript
- Call Claude Vision API
- Parse and validate JSON response
"""

import base64
import json
from pathlib import Path
from typing import List, Dict, Any

import config


def encode_image_to_base64(image_path: Path) -> str:
    """
    Encode an image file to base64 string.

    Args:
        image_path: Path to the image file

    Returns:
        Base64 encoded string
    """
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def get_image_media_type(image_path: Path) -> str:
    """
    Get the media type for an image based on extension.

    Args:
        image_path: Path to the image file

    Returns:
        Media type string (e.g., "image/jpeg")
    """
    extension = image_path.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp"
    }
    return media_types.get(extension, "image/jpeg")


def analyze_task(frames: List[Path], transcript: str, output_path: Path = None) -> Dict[str, Any]:
    """
    Analyze frames and transcript to understand the task performed.

    Args:
        frames: List of paths to frame images (in order)
        transcript: Text transcript of narration
        output_path: Path to save analysis JSON (default: config.ANALYSIS_FILE)

    Returns:
        Analysis result as dictionary

    Raises:
        ImportError: If anthropic is not installed
        ValueError: If response is not valid JSON
        RuntimeError: If API call fails
    """
    # Import anthropic here to allow other modules to work without it
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "Anthropic SDK not installed. Install with: pip install anthropic"
        )

    # Validate API key
    if not config.ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY environment variable not set. "
            "Set it with: export ANTHROPIC_API_KEY=your_key"
        )

    # Setup output path
    output_path = output_path or config.ANALYSIS_FILE
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[TaskAnalyzer] Analyzing {len(frames)} frames with transcript")
    print(f"[TaskAnalyzer] Transcript length: {len(transcript)} characters")

    # Build content array with images and transcript
    content = []

    # Add instruction text
    content.append({
        "type": "text",
        "text": f"TRANSCRIPT OF WHAT THE PERSON SAID:\n\n{transcript}\n\nIMAGES FROM THE VIDEO (in chronological order):"
    })

    # Add each frame as an image
    for i, frame_path in enumerate(frames):
        print(f"[TaskAnalyzer] Encoding frame {i + 1}/{len(frames)}: {frame_path.name}")

        image_data = encode_image_to_base64(frame_path)
        media_type = get_image_media_type(frame_path)

        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": image_data
            }
        })

    # Add final instruction
    content.append({
        "type": "text",
        "text": "Based on the transcript and images above, analyze the task. Output JSON only."
    })

    # Create Anthropic client
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    print(f"[TaskAnalyzer] Calling Claude API ({config.CLAUDE_MODEL})...")

    # Make API call
    try:
        response = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=config.MAX_TOKENS,
            system=config.VISION_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ]
        )
    except Exception as e:
        raise RuntimeError(f"Claude API call failed: {e}")

    # Extract response text
    response_text = response.content[0].text.strip()
    print(f"[TaskAnalyzer] Received response ({len(response_text)} characters)")

    # Parse JSON response
    try:
        # Handle case where response might be wrapped in markdown code blocks
        if response_text.startswith("```"):
            # Remove markdown code block
            lines = response_text.split("\n")
            # Remove first line (```json) and last line (```)
            json_lines = [l for l in lines if not l.startswith("```")]
            response_text = "\n".join(json_lines)

        analysis = json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"[TaskAnalyzer] Warning: Response was not valid JSON")
        print(f"[TaskAnalyzer] Raw response: {response_text}")
        raise ValueError(f"Failed to parse JSON response: {e}")

    # Validate response structure
    required_keys = ["task_name", "confidence", "steps", "completion_detected"]
    missing_keys = [k for k in required_keys if k not in analysis]
    if missing_keys:
        print(f"[TaskAnalyzer] Warning: Response missing keys: {missing_keys}")

    # Save analysis to file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)

    print(f"[TaskAnalyzer] Analysis saved to: {output_path}")
    print(f"[TaskAnalyzer] Task identified: {analysis.get('task_name', 'Unknown')}")
    print(f"[TaskAnalyzer] Confidence: {analysis.get('confidence', 0)}")
    print(f"[TaskAnalyzer] Steps: {len(analysis.get('steps', []))}")
    print(f"[TaskAnalyzer] Completion detected: {analysis.get('completion_detected', False)}")

    return analysis


if __name__ == "__main__":
    # Test module independently
    import sys

    if len(sys.argv) < 3:
        print("Usage: python task_analyzer.py <frames_dir> <transcript_file>")
        sys.exit(1)

    frames_dir = Path(sys.argv[1])
    transcript_file = Path(sys.argv[2])

    if not frames_dir.exists():
        print(f"Error: Frames directory not found: {frames_dir}")
        sys.exit(1)

    if not transcript_file.exists():
        print(f"Error: Transcript file not found: {transcript_file}")
        sys.exit(1)

    # Get frames
    frames = sorted(frames_dir.glob(f"*.{config.FRAME_FORMAT}"))
    if not frames:
        print(f"Error: No frames found in {frames_dir}")
        sys.exit(1)

    # Get transcript
    transcript = transcript_file.read_text(encoding="utf-8")

    config.ensure_output_dirs()

    try:
        analysis = analyze_task(frames, transcript)
        print(f"\nAnalysis result:")
        print(json.dumps(analysis, indent=2))
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
