"""Video processor: extracts frames and mixes audio using ffmpeg."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from typing import Optional

from narrator import get_commentary, text_to_speech

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ffprobe / ffmpeg helpers
# ---------------------------------------------------------------------------


def get_video_info(video_path: str) -> dict:
    """Return video metadata from ffprobe as a dictionary.

    Parameters
    ----------
    video_path:
        Path to the input video file.

    Returns
    -------
    dict
        Parsed ffprobe JSON output with ``format`` and ``streams`` keys.

    Raises
    ------
    subprocess.CalledProcessError
        If ffprobe exits with a non-zero status.
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def has_audio_stream(info: dict) -> bool:
    """Return ``True`` if the video contains at least one audio stream."""
    return any(s.get("codec_type") == "audio" for s in info.get("streams", []))


def extract_frame(video_path: str, timestamp: float, output_path: str) -> bool:
    """Extract a single JPEG frame from the video at *timestamp* seconds.

    Parameters
    ----------
    video_path:
        Path to the input video file.
    timestamp:
        Time offset in seconds.
    output_path:
        Destination path for the JPEG file.

    Returns
    -------
    bool
        ``True`` on success, ``False`` if the frame could not be extracted.
    """
    cmd = [
        "ffmpeg",
        "-ss", str(timestamp),
        "-i", video_path,
        "-vframes", "1",
        "-q:v", "4",
        "-y",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0 and os.path.exists(output_path)


def mix_video_audio(
    video_path: str,
    commentary_clips: list[tuple[float, str]],
    output_path: str,
    has_original_audio: bool = True,
    original_audio_volume: float = 0.8,
    commentary_volume: float = 1.0,
) -> None:
    """Re-encode the video with original audio mixed with commentary clips.

    Parameters
    ----------
    video_path:
        Path to the source video file.
    commentary_clips:
        List of ``(timestamp_seconds, audio_file_path)`` tuples.  Each audio
        clip is overlaid at *timestamp_seconds* in the output.
    output_path:
        Destination path for the output video file.
    has_original_audio:
        Whether the source video contains an audio stream to preserve.
    original_audio_volume:
        Volume multiplier for the original audio (0.0–1.0).  Lowering this
        slightly lets the commentary stand out. Defaults to 0.8.
    commentary_volume:
        Volume multiplier for each commentary clip.  Defaults to 1.0.
    """
    if not commentary_clips:
        # Nothing to mix – just copy the video.
        subprocess.run(
            ["ffmpeg", "-i", video_path, "-c", "copy", "-y", output_path],
            check=True,
        )
        return

    # Build input list: [video] + [each commentary clip]
    inputs: list[str] = ["-i", video_path]
    for _, audio_path in commentary_clips:
        inputs.extend(["-i", audio_path])

    # Build filter_complex
    filter_parts: list[str] = []
    mix_streams: list[str] = []

    if has_original_audio:
        filter_parts.append(f"[0:a]volume={original_audio_volume}[orig]")
        mix_streams.append("[orig]")

    for i, (ts, _) in enumerate(commentary_clips):
        delay_ms = int(ts * 1000)
        input_idx = i + 1  # input 0 is the video
        label = f"c{i}"
        filter_parts.append(
            f"[{input_idx}:a]"
            f"adelay={delay_ms}|{delay_ms},"
            f"volume={commentary_volume}"
            f"[{label}]"
        )
        mix_streams.append(f"[{label}]")

    n_streams = len(mix_streams)
    streams_str = "".join(mix_streams)
    filter_parts.append(
        f"{streams_str}amix=inputs={n_streams}:duration=first:normalize=0[aout]"
    )

    filter_complex = ";".join(filter_parts)

    cmd = [
        "ffmpeg",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-y",
        output_path,
    ]

    logger.debug("ffmpeg command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def process_video(
    input_path: str,
    output_path: str,
    interval: int = 10,
    voice: str = "onyx",
) -> None:
    """Process a video: generate AI commentary and mix it into the audio track.

    For every *interval* seconds of video a frame is extracted, sent to the
    vision LLM for commentary, converted to speech, and mixed with the
    original audio before writing the final output video.

    Parameters
    ----------
    input_path:
        Path to the source video file.
    output_path:
        Destination path for the output video file.
    interval:
        How often (in seconds) to generate a commentary clip. Defaults to 10.
    voice:
        TTS voice to use.  Defaults to ``"onyx"``.
    """
    print(f"📹 Analyzing video: {input_path}")

    info = get_video_info(input_path)
    duration = float(info["format"]["duration"])
    audio_present = has_audio_stream(info)

    print(f"   Duration     : {duration:.1f}s")
    print(f"   Audio stream : {'yes' if audio_present else 'no'}")
    print(f"   Interval     : {interval}s")

    # Build list of timestamps (skip the very end to avoid empty frames)
    timestamps = [ts for ts in range(0, int(duration), interval) if ts < duration - 1]

    print(f"\n🎙️  Generating {len(timestamps)} commentary clip(s)…\n")

    commentary_clips: list[tuple[float, str]] = []
    previous_comments: list[str] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, ts in enumerate(timestamps):
            print(f"  [{idx + 1}/{len(timestamps)}] {ts}s", end="", flush=True)

            # Extract frame
            frame_path = os.path.join(tmpdir, f"frame_{idx:04d}.jpg")
            if not extract_frame(input_path, ts, frame_path):
                print(" ⚠️  frame extraction failed – skipping")
                continue

            # Generate commentary text
            try:
                text = get_commentary(frame_path, ts, previous_comments)
            except Exception as exc:
                print(f" ⚠️  commentary failed: {exc}")
                continue

            print(f"\n     💬 {text}")
            previous_comments.append(text)

            # Convert to speech
            audio_path = os.path.join(tmpdir, f"speech_{idx:04d}.mp3")
            try:
                text_to_speech(text, audio_path, voice=voice)
            except Exception as exc:
                print(f"     ⚠️  TTS failed: {exc}")
                continue

            commentary_clips.append((float(ts), audio_path))

        if not commentary_clips:
            print("\n⚠️  No commentary clips were generated. Copying video as-is.")
            subprocess.run(
                ["ffmpeg", "-i", input_path, "-c", "copy", "-y", output_path],
                check=True,
            )
            return

        print(f"\n🎬 Mixing audio and encoding output video…")
        mix_video_audio(
            input_path,
            commentary_clips,
            output_path,
            has_original_audio=audio_present,
        )

    print(f"\n✅ Done!  Output saved to: {output_path}")
