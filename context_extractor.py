"""
Context Extractor Module
========================
Extracts rich context from frames and transcript for automation building.

Capabilities:
- OCR extraction from frames (window titles, URLs, text)
- Named entity extraction from transcript
- Action verb detection
- Sentiment/frustration analysis
- UI element detection
- Screen state analysis
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from collections import Counter

# Try imports - gracefully handle missing dependencies
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# ============================================================================
# Data Classes for Structured Output
# ============================================================================

@dataclass
class FrameContext:
    """Context extracted from a single frame."""
    frame_number: int
    frame_path: str
    timestamp_estimate: float  # seconds into video

    # OCR Results
    ocr_text: str = ""
    window_title: Optional[str] = None
    detected_url: Optional[str] = None
    detected_app: Optional[str] = None

    # UI Elements
    ui_elements: List[Dict] = field(default_factory=list)

    # Screen State
    screen_state: str = "unknown"  # loading, error, success, form, list, etc.
    has_modal: bool = False
    has_error_message: bool = False
    has_success_message: bool = False
    has_loading_indicator: bool = False

    # Visual Features
    dominant_colors: List[str] = field(default_factory=list)
    brightness: float = 0.0


@dataclass
class TranscriptContext:
    """Context extracted from transcript."""
    full_text: str

    # Named Entities
    applications: List[str] = field(default_factory=list)
    urls: List[str] = field(default_factory=list)
    emails: List[str] = field(default_factory=list)
    file_paths: List[str] = field(default_factory=list)
    people_names: List[str] = field(default_factory=list)

    # Action Analysis
    action_verbs: List[Dict] = field(default_factory=list)  # {verb, context, position}
    sequence_markers: List[Dict] = field(default_factory=list)  # first, then, next, finally
    conditional_statements: List[str] = field(default_factory=list)

    # Sentiment/Pain Points
    frustration_indicators: List[Dict] = field(default_factory=list)
    hesitation_markers: List[str] = field(default_factory=list)
    pain_points: List[str] = field(default_factory=list)

    # Repetition Indicators
    repetition_phrases: List[str] = field(default_factory=list)  # "every time", "always"

    # Temporal References
    temporal_references: List[Dict] = field(default_factory=list)


@dataclass
class AutomationContext:
    """High-level automation insights."""
    # Workflow
    estimated_trigger: str = ""
    estimated_frequency: str = ""
    complexity_score: float = 0.0
    automation_candidate_score: float = 0.0

    # Systems
    applications_used: List[str] = field(default_factory=list)
    urls_accessed: List[str] = field(default_factory=list)

    # Decision Points
    decision_points: List[Dict] = field(default_factory=list)
    human_judgment_required: List[str] = field(default_factory=list)

    # Pain Points (prioritized)
    pain_points_ranked: List[Dict] = field(default_factory=list)

    # Suggested Automations
    automation_suggestions: List[Dict] = field(default_factory=list)


@dataclass
class FullContext:
    """Complete context extraction result."""
    frames: List[FrameContext] = field(default_factory=list)
    transcript: Optional[TranscriptContext] = None
    automation: Optional[AutomationContext] = None

    # Summary
    total_frames: int = 0
    video_duration: float = 0.0
    extraction_timestamp: str = ""


# ============================================================================
# Transcript Analysis
# ============================================================================

# Common action verbs in UI interactions
ACTION_VERBS = [
    "click", "tap", "press", "select", "choose", "pick",
    "open", "close", "launch", "start", "exit", "quit",
    "type", "enter", "input", "write", "fill",
    "scroll", "swipe", "drag", "drop", "move",
    "copy", "paste", "cut", "delete", "remove",
    "save", "submit", "send", "upload", "download",
    "search", "find", "look", "check", "verify",
    "wait", "load", "refresh", "reload",
    "log in", "sign in", "log out", "sign out",
    "navigate", "go to", "switch", "tab"
]

# Sequence markers
SEQUENCE_MARKERS = [
    ("first", "start"),
    ("then", "sequence"),
    ("next", "sequence"),
    ("after", "sequence"),
    ("before", "sequence"),
    ("finally", "end"),
    ("lastly", "end"),
    ("now", "current"),
    ("and then", "sequence"),
]

# Frustration indicators
FRUSTRATION_PATTERNS = [
    (r"\bugh\b", "exclamation"),
    (r"\bargh\b", "exclamation"),
    (r"\bdamn\b", "exclamation"),
    (r"\bshoot\b", "exclamation"),
    (r"this is (so )?(annoying|frustrating|tedious|slow)", "complaint"),
    (r"i (hate|can't stand) (this|when)", "complaint"),
    (r"why (does|do|is|won't)", "complaint"),
    (r"(always|every time) (have to|need to|must)", "repetitive_complaint"),
    (r"takes (forever|so long|too long)", "time_complaint"),
    (r"wish (this|it) (would|could|was)", "wish"),
    (r"there('s| is) got to be a better way", "wish"),
]

# Repetition indicators
REPETITION_PATTERNS = [
    r"every (time|day|week|morning|monday)",
    r"i always (have to|need to)",
    r"(again|another one|one more)",
    r"same (thing|process|steps)",
    r"(repeatedly|over and over)",
]

# Conditional patterns
CONDITIONAL_PATTERNS = [
    r"if .+ then",
    r"when .+ (i|we|you)",
    r"unless .+",
    r"in case .+",
    r"depending on",
]

# Common application names
KNOWN_APPS = [
    "chrome", "safari", "firefox", "edge", "brave",
    "gmail", "outlook", "mail", "thunderbird",
    "slack", "teams", "discord", "zoom",
    "excel", "word", "powerpoint", "google sheets", "google docs",
    "notion", "asana", "trello", "jira", "monday",
    "salesforce", "hubspot", "zendesk",
    "terminal", "iterm", "vscode", "visual studio",
    "finder", "explorer", "files",
    "spotify", "youtube", "twitter", "linkedin", "facebook",
]


def extract_transcript_context(transcript: str) -> TranscriptContext:
    """
    Extract rich context from transcript text.

    Args:
        transcript: The transcribed text

    Returns:
        TranscriptContext with extracted information
    """
    ctx = TranscriptContext(full_text=transcript)
    text_lower = transcript.lower()

    # Extract URLs
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    ctx.urls = re.findall(url_pattern, transcript)

    # Also look for spoken URLs like "gmail.com"
    domain_pattern = r'\b[\w-]+\.(com|org|net|io|co|ai|app|dev)\b'
    spoken_domains = re.findall(domain_pattern, text_lower)
    ctx.urls.extend([f"https://{d[0]}.{d[1]}" if isinstance(d, tuple) else d for d in spoken_domains])
    ctx.urls = list(set(ctx.urls))

    # Extract emails
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    ctx.emails = re.findall(email_pattern, transcript)

    # Detect applications mentioned
    for app in KNOWN_APPS:
        if app in text_lower:
            ctx.applications.append(app.title())
    ctx.applications = list(set(ctx.applications))

    # Extract action verbs with context
    for verb in ACTION_VERBS:
        pattern = rf'\b{verb}(s|ed|ing)?\b'
        for match in re.finditer(pattern, text_lower):
            start = max(0, match.start() - 30)
            end = min(len(transcript), match.end() + 30)
            context = transcript[start:end].strip()
            ctx.action_verbs.append({
                "verb": verb,
                "context": f"...{context}...",
                "position": match.start()
            })

    # Sort by position
    ctx.action_verbs.sort(key=lambda x: x["position"])

    # Extract sequence markers
    for marker, marker_type in SEQUENCE_MARKERS:
        pattern = rf'\b{marker}\b'
        for match in re.finditer(pattern, text_lower):
            start = max(0, match.start() - 20)
            end = min(len(transcript), match.end() + 40)
            context = transcript[start:end].strip()
            ctx.sequence_markers.append({
                "marker": marker,
                "type": marker_type,
                "context": f"...{context}...",
                "position": match.start()
            })

    ctx.sequence_markers.sort(key=lambda x: x["position"])

    # Detect frustration
    for pattern, frust_type in FRUSTRATION_PATTERNS:
        for match in re.finditer(pattern, text_lower):
            start = max(0, match.start() - 20)
            end = min(len(transcript), match.end() + 20)
            context = transcript[start:end].strip()
            ctx.frustration_indicators.append({
                "type": frust_type,
                "match": match.group(),
                "context": f"...{context}..."
            })

    # Detect repetition indicators
    for pattern in REPETITION_PATTERNS:
        for match in re.finditer(pattern, text_lower):
            ctx.repetition_phrases.append(match.group())
    ctx.repetition_phrases = list(set(ctx.repetition_phrases))

    # Detect conditional statements
    for pattern in CONDITIONAL_PATTERNS:
        for match in re.finditer(pattern, text_lower):
            start = max(0, match.start() - 10)
            end = min(len(transcript), match.end() + 50)
            ctx.conditional_statements.append(transcript[start:end].strip())

    # Detect hesitation markers
    hesitation_patterns = [r'\bum+\b', r'\buh+\b', r'\bhmm+\b', r'\blet me (see|think)\b', r'\bwait\b']
    for pattern in hesitation_patterns:
        matches = re.findall(pattern, text_lower)
        ctx.hesitation_markers.extend(matches)

    # Extract pain points (complaints + wishes)
    pain_patterns = [
        r"(annoying|frustrating|tedious|slow|painful|difficult|hard|confusing)",
        r"wish .{10,50}",
        r"(hate|can't stand) .{10,30}",
        r"takes (forever|too long|so long)",
    ]
    for pattern in pain_patterns:
        for match in re.finditer(pattern, text_lower):
            start = max(0, match.start() - 20)
            end = min(len(transcript), match.end() + 30)
            ctx.pain_points.append(transcript[start:end].strip())

    return ctx


# ============================================================================
# Frame Analysis (OCR + Visual)
# ============================================================================

def extract_frame_context(
    frame_path: Path,
    frame_number: int,
    frame_interval: float
) -> FrameContext:
    """
    Extract context from a single video frame.

    Args:
        frame_path: Path to the frame image
        frame_number: Frame sequence number
        frame_interval: Seconds between frames

    Returns:
        FrameContext with extracted information
    """
    ctx = FrameContext(
        frame_number=frame_number,
        frame_path=str(frame_path),
        timestamp_estimate=frame_number * frame_interval
    )

    if not PIL_AVAILABLE:
        return ctx

    try:
        img = Image.open(frame_path)

        # OCR if available
        if TESSERACT_AVAILABLE:
            try:
                ctx.ocr_text = pytesseract.image_to_string(img)

                # Try to extract window title (usually top of screen)
                # Crop top 5% of image for title bar
                width, height = img.size
                title_region = img.crop((0, 0, width, int(height * 0.05)))
                title_text = pytesseract.image_to_string(title_region).strip()
                if title_text:
                    ctx.window_title = title_text.split('\n')[0][:100]

                # Look for URL in the text (browser address bar)
                url_patterns = [
                    r'https?://[^\s<>"{}|\\^`\[\]]+',
                    r'www\.[^\s<>"{}|\\^`\[\]]+',
                    r'[a-zA-Z0-9-]+\.(com|org|net|io|co|ai|app)[^\s]*'
                ]
                for pattern in url_patterns:
                    urls = re.findall(pattern, ctx.ocr_text)
                    if urls:
                        ctx.detected_url = urls[0] if isinstance(urls[0], str) else urls[0][0]
                        break

                # Detect application from OCR text
                ocr_lower = ctx.ocr_text.lower()
                for app in KNOWN_APPS:
                    if app in ocr_lower:
                        ctx.detected_app = app.title()
                        break

                # Detect screen states
                ctx.has_error_message = any(word in ocr_lower for word in [
                    'error', 'failed', 'invalid', 'incorrect', 'wrong',
                    'not found', 'denied', 'unauthorized', 'forbidden'
                ])
                ctx.has_success_message = any(word in ocr_lower for word in [
                    'success', 'completed', 'saved', 'sent', 'done',
                    'confirmed', 'approved', 'submitted'
                ])
                ctx.has_loading_indicator = any(word in ocr_lower for word in [
                    'loading', 'please wait', 'processing', 'uploading',
                    'downloading', 'connecting'
                ])

                # Determine screen state
                if ctx.has_error_message:
                    ctx.screen_state = "error"
                elif ctx.has_success_message:
                    ctx.screen_state = "success"
                elif ctx.has_loading_indicator:
                    ctx.screen_state = "loading"
                elif any(word in ocr_lower for word in ['submit', 'save', 'send', 'name:', 'email:', 'password:']):
                    ctx.screen_state = "form"
                else:
                    ctx.screen_state = "content"

            except Exception as e:
                ctx.ocr_text = f"OCR Error: {e}"

        # Basic image analysis with OpenCV if available
        if CV2_AVAILABLE:
            try:
                cv_img = cv2.imread(str(frame_path))
                if cv_img is not None:
                    # Calculate brightness
                    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                    ctx.brightness = float(gray.mean()) / 255.0

                    # Simple modal detection (look for darker overlay)
                    height, width = cv_img.shape[:2]
                    center_region = gray[height//4:3*height//4, width//4:3*width//4]
                    edge_region_top = gray[0:height//8, :]

                    center_brightness = center_region.mean()
                    edge_brightness = edge_region_top.mean()

                    # If center is much brighter than edges, might be a modal
                    if center_brightness > edge_brightness * 1.3:
                        ctx.has_modal = True

            except Exception:
                pass

    except Exception as e:
        ctx.ocr_text = f"Error processing frame: {e}"

    return ctx


def extract_all_frames_context(
    frames: List[Path],
    frame_interval: float = 3.0
) -> List[FrameContext]:
    """
    Extract context from all frames.

    Args:
        frames: List of frame paths
        frame_interval: Seconds between frames

    Returns:
        List of FrameContext objects
    """
    contexts = []
    for i, frame_path in enumerate(frames):
        ctx = extract_frame_context(frame_path, i, frame_interval)
        contexts.append(ctx)
    return contexts


# ============================================================================
# Automation Analysis
# ============================================================================

def analyze_automation_potential(
    frame_contexts: List[FrameContext],
    transcript_context: TranscriptContext
) -> AutomationContext:
    """
    Analyze the workflow for automation potential.

    Args:
        frame_contexts: Context from all frames
        transcript_context: Context from transcript

    Returns:
        AutomationContext with insights
    """
    auto_ctx = AutomationContext()

    # Collect all applications
    apps = set(transcript_context.applications)
    for fc in frame_contexts:
        if fc.detected_app:
            apps.add(fc.detected_app)
    auto_ctx.applications_used = list(apps)

    # Collect all URLs
    urls = set(transcript_context.urls)
    for fc in frame_contexts:
        if fc.detected_url:
            urls.add(fc.detected_url)
    auto_ctx.urls_accessed = list(urls)

    # Identify decision points from conditionals and hesitations
    for cond in transcript_context.conditional_statements:
        auto_ctx.decision_points.append({
            "type": "conditional",
            "description": cond
        })

    if len(transcript_context.hesitation_markers) > 3:
        auto_ctx.decision_points.append({
            "type": "hesitation_pattern",
            "description": f"Multiple hesitations detected ({len(transcript_context.hesitation_markers)} instances)"
        })

    # Calculate complexity score (0-1)
    complexity_factors = [
        len(apps) * 0.1,  # More apps = more complex
        len(transcript_context.conditional_statements) * 0.15,
        len(transcript_context.action_verbs) * 0.02,
        1 if any(fc.has_error_message for fc in frame_contexts) else 0,
        len(transcript_context.hesitation_markers) * 0.05,
    ]
    auto_ctx.complexity_score = min(1.0, sum(complexity_factors))

    # Calculate automation candidate score (0-1, higher = better candidate)
    automation_factors = [
        0.3 if transcript_context.repetition_phrases else 0,  # Repetitive = automate
        0.2 if transcript_context.frustration_indicators else 0,  # Pain = motivation
        0.1 * min(len(transcript_context.action_verbs), 10) / 10,  # Has clear actions
        0.2 if len(apps) <= 3 else 0.1,  # Fewer apps = easier
        0.2 if not transcript_context.conditional_statements else 0.1,  # Simple logic
    ]
    auto_ctx.automation_candidate_score = min(1.0, sum(automation_factors))

    # Rank pain points
    pain_counter = Counter()
    for pain in transcript_context.pain_points:
        pain_counter[pain] += 1
    for frust in transcript_context.frustration_indicators:
        pain_counter[frust["context"]] += 2  # Weight frustration higher

    auto_ctx.pain_points_ranked = [
        {"description": pain, "severity": count}
        for pain, count in pain_counter.most_common(5)
    ]

    # Estimate trigger and frequency from transcript
    text_lower = transcript_context.full_text.lower()
    if "every day" in text_lower or "daily" in text_lower:
        auto_ctx.estimated_frequency = "daily"
    elif "every week" in text_lower or "weekly" in text_lower:
        auto_ctx.estimated_frequency = "weekly"
    elif "every time" in text_lower:
        auto_ctx.estimated_frequency = "event-triggered"
    elif "when" in text_lower and ("email" in text_lower or "message" in text_lower):
        auto_ctx.estimated_frequency = "on-demand"
    else:
        auto_ctx.estimated_frequency = "unknown"

    # Generate automation suggestions
    if transcript_context.repetition_phrases:
        auto_ctx.automation_suggestions.append({
            "type": "scheduled_automation",
            "reason": "Task is performed repeatedly",
            "evidence": transcript_context.repetition_phrases[:3]
        })

    if len(apps) == 1 and apps:
        auto_ctx.automation_suggestions.append({
            "type": "single_app_macro",
            "reason": f"Task uses only {list(apps)[0]}",
            "tool_suggestion": "App-specific automation (e.g., browser extension, app macros)"
        })

    if transcript_context.urls:
        auto_ctx.automation_suggestions.append({
            "type": "web_automation",
            "reason": "Task involves web interactions",
            "urls": transcript_context.urls[:3],
            "tool_suggestion": "Playwright, Puppeteer, or browser RPA"
        })

    if any("copy" in v["verb"] or "paste" in v["verb"] for v in transcript_context.action_verbs):
        auto_ctx.automation_suggestions.append({
            "type": "data_transfer",
            "reason": "Task involves copy/paste operations",
            "tool_suggestion": "API integration or data pipeline"
        })

    return auto_ctx


# ============================================================================
# Main Extraction Function
# ============================================================================

def extract_full_context(
    frames: List[Path],
    transcript: str,
    video_duration: float = 0.0,
    frame_interval: float = 3.0
) -> FullContext:
    """
    Extract complete context from frames and transcript.

    Args:
        frames: List of frame paths
        transcript: Transcript text
        video_duration: Video duration in seconds
        frame_interval: Seconds between frames

    Returns:
        FullContext with all extracted information
    """
    from datetime import datetime

    # Extract transcript context
    transcript_ctx = extract_transcript_context(transcript)

    # Extract frame contexts
    frame_contexts = extract_all_frames_context(frames, frame_interval)

    # Analyze automation potential
    automation_ctx = analyze_automation_potential(frame_contexts, transcript_ctx)

    # Build full context
    full_ctx = FullContext(
        frames=frame_contexts,
        transcript=transcript_ctx,
        automation=automation_ctx,
        total_frames=len(frames),
        video_duration=video_duration,
        extraction_timestamp=datetime.now().isoformat()
    )

    return full_ctx


def context_to_dict(ctx: FullContext) -> Dict[str, Any]:
    """Convert FullContext to JSON-serializable dict."""
    return {
        "frames": [asdict(f) for f in ctx.frames],
        "transcript": asdict(ctx.transcript) if ctx.transcript else None,
        "automation": asdict(ctx.automation) if ctx.automation else None,
        "total_frames": ctx.total_frames,
        "video_duration": ctx.video_duration,
        "extraction_timestamp": ctx.extraction_timestamp
    }


# ============================================================================
# CLI Testing
# ============================================================================

if __name__ == "__main__":
    import sys

    print("Context Extractor Module")
    print("=" * 40)
    print(f"OpenCV available: {CV2_AVAILABLE}")
    print(f"Tesseract available: {TESSERACT_AVAILABLE}")
    print(f"PIL available: {PIL_AVAILABLE}")
    print()

    # Test transcript extraction
    test_transcript = """
    Okay, watch. I'm going to open up Chrome, open up a new tab.
    I'm going to go to gmail.com. I'm going to click compose.
    And I'm going to create an email to myself at test@gmail.com.
    So that will be test. The body will be test.
    This is something I do every day just to test out my email.
    Ugh, this loading is so slow. I wish it would go faster.
    First I click here, then I wait, and finally I submit.
    """

    print("Testing transcript extraction...")
    ctx = extract_transcript_context(test_transcript)
    print(f"  Applications: {ctx.applications}")
    print(f"  URLs: {ctx.urls}")
    print(f"  Emails: {ctx.emails}")
    print(f"  Action verbs: {len(ctx.action_verbs)}")
    print(f"  Sequence markers: {len(ctx.sequence_markers)}")
    print(f"  Frustration indicators: {ctx.frustration_indicators}")
    print(f"  Repetition phrases: {ctx.repetition_phrases}")
    print(f"  Pain points: {ctx.pain_points}")
