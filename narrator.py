"""Narrator module: generates AI commentary text and converts it to speech."""

from __future__ import annotations

import base64
import logging
import os
import re
from typing import Optional

import requests
from openai import AzureOpenAI, OpenAI

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an EXTREMELY passionate, high-energy handball broadcast commentator \
delivering live play-by-play and color commentary. You are ALWAYS fired up — \
your baseline energy is already enthusiastic and loud, and it only goes UP \
from there.

You are watching a LIVE match — you only see what has happened so far, never \
what comes next. React to what you see RIGHT NOW.

Guidelines:
- Keep commentary brief (1-3 sentences, max 40 words).
- **EXPLOSIVE moments** (goals, fast breaks, saves, fouls, penalties): \
GO ABSOLUTELY WILD. Use ALL-CAPS for key words. Scream through the text. \
Examples: "GOOOAAL!! WHAT A ROCKET!! He BURIED that into the top corner!" / \
"OH MY GOD!! THE SAVE OF THE CENTURY!!"
- **Build-up & mid-play**: stay highly energetic, convey urgency and \
anticipation, push the excitement forward. \
Examples: "They're pushing HARD on the counter! Three on two — HERE THEY COME!" / \
"The ball is FLYING around the arc, the defense is SCRAMBLING!"
- **Stoppages & timeouts**: still energetic and animated, never dull. Deliver \
analysis with passion and conviction. \
Examples: "Timeout called and you can FEEL the tension in this arena! \
They HAVE to regroup NOW!" / "What a half! This has been ELECTRIC!"
- Use present tense, as if calling the action live.
- Vary your sentence structure: mix exclamations, questions, and statements.
- Show intense genuine emotion — shock, awe, tension, relief, disbelief.
- Use exclamation marks liberally. This is NOT a library, this is a MATCH!
- Do not repeat or closely echo previous commentary.
- Never describe the image technically (no "I see a frame" or "the image shows").

IMPORTANT — Output format:
Line 1: event importance — one of: [critical], [notable], or [minor]
  • [critical] = goal scored, score changed, penalty, red card — the BIG stuff
  • [notable] = great save, fast break, foul, tactical shift, momentum swing
  • [minor] = general play, ball movement, stoppages, nothing special happening
Line 2: style tag for voice delivery — one of:
  [excited] [shouting] [cheerful] [hopeful] [angry] [terrified] [friendly]
Line 3+: the commentary text.

If NOTHING interesting or new is happening compared to your recent commentary, \
output ONLY the single word:
[SKIP]

Example output for a goal:
[critical]
[shouting]
GOOOAAL!! WHAT AN ABSOLUTE ROCKET INTO THE TOP CORNER!!

Example output for build-up:
[notable]
[excited]
They're pushing HARD on the counter! Three on two — HERE THEY COME!

Example for nothing happening:
[SKIP]
"""

# Regex to parse the importance + style tags from LLM output
_IMPORTANCE_RE = re.compile(
    r"^\[(critical|notable|minor)\]\s*\n?",
    re.IGNORECASE,
)

_STYLE_RE = re.compile(
    r"^\[("
    r"excited|shouting|cheerful|hopeful|angry|terrified|friendly"
    r")\]\s*\n?",
    re.IGNORECASE,
)

# Valid Azure Speech styles for en-US-JasonNeural
_VALID_STYLES = {
    "excited", "shouting", "cheerful", "hopeful", "angry",
    "terrified", "friendly", "default", "sad", "unfriendly", "whispering",
}

# Commentary levels — which event importances to include
COMMENTARY_LEVELS = {
    "important": {"critical"},
    "normal":    {"critical", "notable"},
    "all":       {"critical", "notable", "minor"},
}

_TTS_INSTRUCTIONS = """\
You are an EXTREMELY energetic live sports broadcast commentator. Your voice \
should ALWAYS be loud, animated, and full of adrenaline — like you're \
commentating a championship final:
- Your BASELINE voice is already raised and excited — never drop to a calm or \
quiet tone.
- For big moments (goals, saves, fast breaks) go ALL OUT: YELL, SCREAM, let \
your voice CRACK with excitement. Sound like you just witnessed the impossible.
- During build-up play, keep the energy HIGH — speak fast, sound urgent, like \
something incredible could happen any second.
- Even during stoppages, stay animated and intense — you're pumped, the crowd \
is loud, convey that electricity.
- Vary your pitch dramatically — go from high shouts to intense low growls.
- Quicken your pace when action intensifies, but always stay energetic.
- Add brief explosive pauses before erupting with excitement.
- You should sound like you've had five espressos and are living the best \
moment of your career. NEVER sound bored, flat, or monotone.
"""


def _build_chat_client() -> tuple[AzureOpenAI | OpenAI, str]:
    """Return a chat client and model name based on ``CHAT_BACKEND``.

    Supported backends:

    - ``"azure_openai"`` (default) — standard Azure OpenAI Service.
    - ``"kimi"`` — Kimi model via Azure AI Foundry / Model Inference.
    """
    backend = os.getenv("CHAT_BACKEND", "azure_openai").lower().strip()

    if backend == "kimi":
        return _build_kimi_client()
    return _build_azure_openai_client()


def _build_azure_openai_client() -> tuple[AzureOpenAI, str]:
    """Return an Azure OpenAI chat client and deployment name.

    Requires ``AZURE_OPENAI_ENDPOINT`` and ``AZURE_OPENAI_KEY`` to be set.
    """
    client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_KEY"],
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    )
    model = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4-mini")
    return client, model


def _build_kimi_client() -> tuple[OpenAI, str]:
    """Return a Kimi chat client via Azure AI Foundry Model Inference.

    Requires ``KIMI_ENDPOINT`` and ``KIMI_KEY`` to be set.
    The endpoint should be the base URL, e.g.
    ``https://<resource>.services.ai.azure.com/``.
    """
    endpoint = os.environ["KIMI_ENDPOINT"].rstrip("/")
    api_key = os.environ["KIMI_KEY"]
    model = os.getenv("KIMI_MODEL", "Kimi-K2.5")

    client = OpenAI(
        base_url=f"{endpoint}/openai/v1/",
        api_key=api_key,
    )
    return client, model


def _build_tts_client() -> tuple[AzureOpenAI, str]:
    """Return an Azure OpenAI TTS client and deployment name.

    Uses ``AZURE_TTS_DEPLOYMENT`` to select the speech synthesis deployment.
    Falls back to the main Azure OpenAI endpoint/key if TTS-specific ones are
    not provided.
    """
    endpoint = os.getenv("AZURE_TTS_ENDPOINT", os.environ["AZURE_OPENAI_ENDPOINT"])
    api_key = os.getenv("AZURE_TTS_KEY", os.environ["AZURE_OPENAI_KEY"])
    api_version = os.getenv(
        "AZURE_TTS_API_VERSION",
        os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    )
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )
    model = os.getenv("AZURE_TTS_DEPLOYMENT", "tts")
    return client, model


def _parse_importance_style_text(raw: str) -> tuple[str | None, str, str]:
    """Extract importance, style tag, and clean text from LLM output.

    Returns
    -------
    tuple[str | None, str, str]
        ``(importance, style, text)``.
        *importance* is ``"critical"``, ``"notable"``, ``"minor"``, or
        ``None`` if the LLM output ``[SKIP]``.
        *style* defaults to ``"excited"`` if not found.
        *text* is the commentary without tag prefixes.
    """
    stripped = raw.strip()

    # Check for SKIP
    if stripped.upper().startswith("[SKIP]") or stripped.upper() == "[SKIP]":
        return None, "excited", ""

    # Parse importance
    importance = "notable"  # default
    m = _IMPORTANCE_RE.match(stripped)
    if m:
        importance = m.group(1).lower()
        stripped = stripped[m.end():].strip()

    # Parse style
    style = "excited"
    m = _STYLE_RE.match(stripped)
    if m:
        style = m.group(1).lower()
        stripped = stripped[m.end():].strip()

    if style not in _VALID_STYLES:
        style = "excited"

    return importance, style, stripped


def get_commentary(
    frame_paths: str | list[str],
    timestamp: float,
    previous_comments: Optional[list[str]] = None,
) -> tuple[str | None, str, str]:
    """Generate commentary text for a sequence of video frames.

    Parameters
    ----------
    frame_paths:
        Path to a single JPEG frame or a list of paths representing a
        chronological sequence of frames leading up to the current moment.
    timestamp:
        Timestamp of the latest frame in seconds.
    previous_comments:
        Optional list of recent commentary strings used to avoid repetition.

    Returns
    -------
    tuple[str | None, str, str]
        ``(importance, style, text)``.
        *importance* is ``"critical"``, ``"notable"``, ``"minor"``, or
        ``None`` when the LLM decided to skip (nothing worth saying).
    """
    # Normalise to a list
    if isinstance(frame_paths, str):
        frame_paths = [frame_paths]

    client, model = _build_chat_client()

    minutes = int(timestamp // 60)
    seconds = int(timestamp % 60)
    time_str = f"{minutes:02d}:{seconds:02d}"

    n_frames = len(frame_paths)
    user_text = (
        f"Video timestamp: {time_str}. "
        f"You are seeing {n_frames} chronological frame(s) spanning the last few "
        f"seconds of play. Use ALL of them to understand the flow of action "
        f"before crafting your commentary."
    )
    if previous_comments:
        recent = " | ".join(previous_comments[-3:])
        user_text += f"\nRecent commentary (avoid repeating): {recent}"
    user_text += "\nCall the action like a live broadcast commentator:"

    # Build content parts: text first, then each frame image
    content_parts: list[dict] = [{"type": "text", "text": user_text}]
    for fp in frame_paths:
        with open(fp, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        content_parts.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_data}",
                    "detail": "low",
                },
            }
        )

    # gpt-5.4-mini requires max_completion_tokens; Kimi needs max_tokens
    # with a larger budget for its internal reasoning tokens.
    backend = os.getenv("CHAT_BACKEND", "azure_openai").lower().strip()
    if backend == "kimi":
        token_kwarg = {"max_tokens": 4096}
    else:
        token_kwarg = {"max_completion_tokens": 300}

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": content_parts},
        ],
        temperature=0.85,
        **token_kwarg,
    )

    content = response.choices[0].message.content
    if not content:
        logger.warning("LLM returned empty content (finish_reason=%s)",
                       response.choices[0].finish_reason)
        return None, "excited", ""
    raw = content.strip()
    return _parse_importance_style_text(raw)


def _build_ssml(text: str, style: str, voice: str) -> str:
    """Build an SSML document with ``mstts:express-as`` style wrapping.

    Parameters
    ----------
    text:
        The commentary text to synthesize.
    style:
        Azure Speech style name (e.g. ``"excited"``, ``"shouting"``).
    voice:
        Azure Speech voice name (e.g. ``"en-US-JasonNeural"``).
    """
    # Escape XML special characters in the commentary text
    safe_text = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
    return (
        '<speak xmlns="http://www.w3.org/2001/10/synthesis" '
        'xmlns:mstts="http://www.w3.org/2001/mstts" '
        'xmlns:emo="http://www.w3.org/2009/10/emotionml" '
        'version="1.0" xml:lang="en-US">'
        f'<voice name="{voice}">'
        f'<mstts:express-as style="{style}">'
        f"{safe_text}"
        "</mstts:express-as>"
        "</voice>"
        "</speak>"
    )


def _tts_azure_speech(text: str, output_path: str, style: str = "excited") -> None:
    """Synthesize speech using Azure Cognitive Services Speech REST API.

    Requires ``AZURE_SPEECH_KEY`` and ``AZURE_SPEECH_REGION`` (or
    ``AZURE_SPEECH_ENDPOINT``) environment variables.
    """
    speech_key = os.environ["AZURE_SPEECH_KEY"]
    region = os.getenv("AZURE_SPEECH_REGION", "swedencentral")
    endpoint = os.getenv(
        "AZURE_SPEECH_ENDPOINT",
        f"https://{region}.tts.speech.microsoft.com",
    )
    voice = os.getenv("AZURE_SPEECH_VOICE", "en-US-JasonNeural")

    # Build the TTS REST URL
    # Custom endpoints (*.cognitiveservices.azure.com) need /tts/ prefix;
    # regional endpoints (*.tts.speech.microsoft.com) do not.
    base = endpoint.rstrip("/")
    if "cognitiveservices.azure.com" in base:
        url = f"{base}/tts/cognitiveservices/v1"
    else:
        url = f"{base}/cognitiveservices/v1"

    ssml = _build_ssml(text, style, voice)
    logger.debug("SSML payload:\n%s", ssml)

    headers = {
        "Ocp-Apim-Subscription-Key": speech_key,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "audio-16khz-128kbitrate-mono-mp3",
        "User-Agent": "Commentator/1.0",
    }

    resp = requests.post(url, headers=headers, data=ssml.encode("utf-8"), timeout=30)
    resp.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(resp.content)

    logger.debug("Azure Speech audio saved to %s (%d bytes)", output_path, len(resp.content))


def _tts_openai(text: str, output_path: str, voice: str = "onyx") -> None:
    """Synthesize speech using Azure OpenAI TTS (gpt-4o-mini-tts)."""
    client, model = _build_tts_client()

    with client.audio.speech.with_streaming_response.create(
        model=model,
        voice=voice,
        input=text,
        instructions=_TTS_INSTRUCTIONS,
        response_format="mp3",
    ) as response:
        response.stream_to_file(output_path)

    logger.debug("OpenAI TTS audio saved to %s", output_path)


def text_to_speech(
    text: str,
    output_path: str,
    voice: str = "onyx",
    style: str = "excited",
) -> None:
    """Convert text to speech and save as an MP3 file.

    Supports two backends controlled by ``TTS_BACKEND`` env var:

    - ``"azure_speech"`` — Azure Cognitive Services Speech with SSML style
      tags (``excited``, ``shouting``, ``cheerful``, etc.).
    - ``"openai"`` (default) — Azure OpenAI gpt-4o-mini-tts with
      ``_TTS_INSTRUCTIONS``.

    Parameters
    ----------
    text:
        The commentary text to synthesize.
    output_path:
        Destination file path (should end in ``.mp3``).
    voice:
        TTS voice identifier.  Used by the OpenAI backend.  Ignored by the
        Azure Speech backend (which reads ``AZURE_SPEECH_VOICE`` from env).
    style:
        Azure Speech style for SSML delivery (e.g. ``"excited"``,
        ``"shouting"``).  Defaults to ``"excited"``.
    """
    backend = os.getenv("TTS_BACKEND", "openai").lower().strip()
    logger.info("TTS backend=%s  style=%s  text=%s", backend, style, text[:80])

    if backend == "azure_speech":
        _tts_azure_speech(text, output_path, style=style)
    else:
        _tts_openai(text, output_path, voice=voice)

    logger.debug("TTS audio saved to %s", output_path)
