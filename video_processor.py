"""Video processor: extracts frames and mixes audio using ffmpeg."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from typing import Optional

from narrator import COMMENTARY_LEVELS, get_commentary, text_to_speech

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


def extract_frames(
    video_path: str,
    start_time: float,
    end_time: float,
    output_dir: str,
    prefix: str = "seq",
    fps: float | None = None,
) -> list[str]:
    """Extract JPEG frames between *start_time* and *end_time* at a given FPS.

    Parameters
    ----------
    video_path:
        Path to the input video file.
    start_time:
        Start of the time range (seconds, inclusive).
    end_time:
        End of the time range (seconds, inclusive).
    output_dir:
        Directory where the JPEG files will be written.
    prefix:
        Filename prefix for the extracted frame files.
    fps:
        Frames per second to sample.  Defaults to ``FRAME_SAMPLE_FPS`` env
        var (fallback: 2).

    Returns
    -------
    list[str]
        Paths to the successfully extracted JPEG files, in chronological order.
    """
    if fps is None:
        fps = float(os.getenv("FRAME_SAMPLE_FPS", "2"))

    duration = end_time - start_time
    if duration <= 0:
        # Fall back to a single frame at end_time
        path = os.path.join(output_dir, f"{prefix}_0000.jpg")
        if extract_frame(video_path, end_time, path):
            return [path]
        return []

    # Number of frames based on requested FPS (at least 1)
    n_frames = max(1, int(round(duration * fps)))
    step = duration / n_frames

    paths: list[str] = []
    for i in range(n_frames):
        ts = start_time + i * step
        path = os.path.join(output_dir, f"{prefix}_{i:04d}.jpg")
        if extract_frame(video_path, ts, path):
            paths.append(path)

    return paths


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
    interval: int = 3,
    voice: str = "onyx",
    commentary_level: str = "normal",
) -> None:
    """Process a video: scan frequently, generate AI commentary, mix audio.

    Frames are sampled every *interval* seconds (default 3) to catch events
    like goals as soon as they happen.  The LLM classifies each moment as
    ``critical``, ``notable``, or ``minor`` and the *commentary_level*
    setting decides which classes actually produce spoken commentary:

    - ``"important"`` — only ``critical`` events (goals, penalties, red cards)
    - ``"normal"``    — ``critical`` + ``notable`` (saves, fouls, fast breaks)
    - ``"all"``       — everything the LLM deems worth mentioning

    The LLM may also output ``[SKIP]`` when nothing new is happening,
    regardless of the level.  Critical events always bypass any gap cooldown.

    Parameters
    ----------
    input_path:
        Path to the source video file.
    output_path:
        Destination path for the output video file.
    interval:
        How often (in seconds) to sample frames for analysis. Defaults to 3.
        This is the *scan* cadence, NOT the commentary cadence.
    voice:
        TTS voice to use (OpenAI backend).
    commentary_level:
        One of ``"important"``, ``"normal"``, ``"all"``.
    """
    allowed = COMMENTARY_LEVELS.get(commentary_level, COMMENTARY_LEVELS["normal"])

    # Minimum seconds between non-critical comments to avoid constant talking.
    min_gap = int(os.getenv("MIN_COMMENT_GAP", "8"))

    # Optionally persist extracted frames and audio clips for debugging.
    keep_files = os.getenv("KEEP_LOCAL_FILES", "false").lower().strip() in ("1", "true", "yes")
    debug_dir: str | None = None
    if keep_files:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        debug_dir = os.path.join(os.path.dirname(input_path) or ".", f"{base_name}_debug")
        os.makedirs(debug_dir, exist_ok=True)
        # Sub-directories for frames and audio
        os.makedirs(os.path.join(debug_dir, "frames"), exist_ok=True)
        os.makedirs(os.path.join(debug_dir, "audio"), exist_ok=True)
        print(f"   Debug output     : {debug_dir}")

    print(f"📹 Analyzing video: {input_path}")

    info = get_video_info(input_path)
    duration = float(info["format"]["duration"])
    audio_present = has_audio_stream(info)

    print(f"   Duration          : {duration:.1f}s")
    print(f"   Audio stream      : {'yes' if audio_present else 'no'}")
    print(f"   Scan interval     : {interval}s")
    print(f"   Commentary level  : {commentary_level} → {sorted(allowed)}")
    print(f"   Min comment gap   : {min_gap}s (critical events bypass)")

    # Build scan timestamps (never look ahead, process sequentially)
    timestamps = [ts for ts in range(0, int(duration), interval) if ts < duration - 1]

    print(f"\n🎙️  Scanning {len(timestamps)} checkpoint(s)…\n")

    commentary_clips: list[tuple[float, str]] = []
    previous_comments: list[str] = []
    last_comment_ts: float = -999.0  # timestamp of last spoken comment

    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, ts in enumerate(timestamps):
            ts_f = float(ts)
            label = f"  [{idx + 1}/{len(timestamps)}] {ts}s"
            print(label, end="", flush=True)

            # Extract frame sequence from previous scan point to now
            prev_ts = timestamps[idx - 1] if idx > 0 else max(0, ts - interval)
            frame_paths = extract_frames(
                input_path,
                start_time=prev_ts,
                end_time=ts,
                output_dir=tmpdir,
                prefix=f"seq_{idx:04d}",
            )
            if not frame_paths:
                print(" ⚠️  frame extraction failed – skipping")
                continue

            # Save frames locally if requested
            if debug_dir:
                def _ts_label(s: float) -> str:
                    m, sec = divmod(int(s), 60)
                    return f"{m:02d}m{sec:02d}s"

                range_label = f"{_ts_label(prev_ts)}-{_ts_label(ts)}"
                for i, fp in enumerate(frame_paths, 1):
                    dest = os.path.join(
                        debug_dir, "frames",
                        f"{range_label}_frame{i}of{len(frame_paths)}.jpg",
                    )
                    shutil.copy2(fp, dest)

            # Ask the LLM to classify and comment
            try:
                importance, style, text = get_commentary(
                    frame_paths, ts_f, previous_comments,
                )
            except Exception as exc:
                print(f" ⚠️  LLM failed: {exc}")
                continue

            # LLM decided nothing worth saying
            if importance is None or not text:
                print("  — [SKIP]")
                continue

            # Filter by commentary level
            if importance not in allowed:
                print(f"  — [{importance}] filtered out at level '{commentary_level}'")
                continue

            # Enforce minimum gap between comments (critical events bypass)
            gap = ts_f - last_comment_ts
            if importance != "critical" and gap < min_gap:
                print(f"  — [{importance}] too soon ({gap:.0f}s < {min_gap}s gap)")
                continue

            print(f"\n     🔥 [{importance}] [{style}] {text}")
            previous_comments.append(text)

            # Convert to speech
            audio_path = os.path.join(tmpdir, f"speech_{idx:04d}.mp3")
            try:
                text_to_speech(text, audio_path, voice=voice, style=style)
            except Exception as exc:
                print(f"     ⚠️  TTS failed: {exc}")
                continue

            commentary_clips.append((ts_f, audio_path))
            last_comment_ts = ts_f

            # Save audio clip locally if requested
            if debug_dir:
                m, sec = divmod(int(ts), 60)
                dest = os.path.join(debug_dir, "audio", f"{m:02d}m{sec:02d}s_speech.mp3")
                shutil.copy2(audio_path, dest)

        if not commentary_clips:
            print("\n⚠️  No commentary clips were generated. Copying video as-is.")
            subprocess.run(
                ["ffmpeg", "-i", input_path, "-c", "copy", "-y", output_path],
                check=True,
            )
            return

        print(f"\n🎬 Mixing {len(commentary_clips)} clip(s) into output video…")
        mix_video_audio(
            input_path,
            commentary_clips,
            output_path,
            has_original_audio=audio_present,
        )

    print(f"\n✅ Done!  Output saved to: {output_path}")
