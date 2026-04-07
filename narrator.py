"""Narrator module: generates AI commentary text and converts it to speech."""

from __future__ import annotations

import base64
import logging
import os
from typing import Optional

from openai import AzureOpenAI

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an EXTREMELY passionate, high-energy handball broadcast commentator \
delivering live play-by-play and color commentary. You are ALWAYS fired up — \
your baseline energy is already enthusiastic and loud, and it only goes UP \
from there.

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
"""

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


def _build_chat_client() -> tuple[AzureOpenAI, str]:
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


def get_commentary(
    frame_paths: str | list[str],
    timestamp: float,
    previous_comments: Optional[list[str]] = None,
) -> str:
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
    str
        The LLM-generated commentary.
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

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": content_parts},
        ],
        max_completion_tokens=120,
        temperature=0.85,
    )

    return response.choices[0].message.content.strip()


def text_to_speech(
    text: str,
    output_path: str,
    voice: str = "onyx",
) -> None:
    """Convert text to speech and save as an MP3 file.

    Parameters
    ----------
    text:
        The text to synthesize.
    output_path:
        Destination file path (should end in ``.mp3``).
    voice:
        TTS voice identifier.  Defaults to ``"onyx"`` for a rich narrator tone.
    """
    client, model = _build_tts_client()

    with client.audio.speech.with_streaming_response.create(
        model=model,
        voice=voice,
        input=text,
        instructions=_TTS_INSTRUCTIONS,
        response_format="mp3",
    ) as response:
        response.stream_to_file(output_path)

    logger.debug("TTS audio saved to %s", output_path)
