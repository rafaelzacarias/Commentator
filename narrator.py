"""Narrator module: generates AI commentary text and converts it to speech."""

from __future__ import annotations

import base64
import logging
import os
from typing import Optional

from openai import AzureOpenAI

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an engaging AI narrator watching a video. Your job is to provide \
concise, vivid commentary about what you see in each video frame.

Guidelines:
- Keep commentary brief (1-2 sentences, max 25 words).
- Focus on the most interesting or notable aspects of the scene.
- Use present tense, as if narrating live action.
- Be engaging and descriptive, like a professional documentary narrator.
- Do not repeat or closely echo previous commentary.
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
    """
    client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_KEY"],
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    )
    model = os.getenv("AZURE_TTS_DEPLOYMENT", "tts")
    return client, model


def get_commentary(
    frame_path: str,
    timestamp: float,
    previous_comments: Optional[list[str]] = None,
) -> str:
    """Generate commentary text for a video frame at the given timestamp.

    Parameters
    ----------
    frame_path:
        Path to the JPEG frame image.
    timestamp:
        Timestamp of the frame in seconds.
    previous_comments:
        Optional list of recent commentary strings used to avoid repetition.

    Returns
    -------
    str
        The LLM-generated commentary.
    """
    client, model = _build_chat_client()

    with open(frame_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    minutes = int(timestamp // 60)
    seconds = int(timestamp % 60)
    time_str = f"{minutes:02d}:{seconds:02d}"

    user_text = f"Video timestamp: {time_str}."
    if previous_comments:
        recent = " | ".join(previous_comments[-3:])
        user_text += f"\nRecent commentary (avoid repeating): {recent}"
    user_text += "\nDescribe what you see in this frame in 1-2 short sentences:"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}",
                            "detail": "low",
                        },
                    },
                ],
            },
        ],
        max_completion_tokens=80,
        temperature=0.7,
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
        response_format="mp3",
    ) as response:
        response.stream_to_file(output_path)

    logger.debug("TTS audio saved to %s", output_path)
