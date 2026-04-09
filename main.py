"""Entry point for Commentator – AI video narrator.

Usage
-----
    python main.py <input_video> [options]

Examples
--------
    python main.py my_video.mp4
    python main.py my_video.mp4 -o narrated.mp4
    python main.py my_video.mp4 --interval 15 --voice nova
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration (can be overridden via environment variables or .env)
# ---------------------------------------------------------------------------
COMMENTARY_INTERVAL: int = int(os.getenv("SCAN_INTERVAL", os.getenv("COMMENTARY_INTERVAL", "3")))
TTS_VOICE: str = os.getenv("TTS_VOICE", "onyx")
COMMENTARY_LEVEL: str = os.getenv("COMMENTARY_LEVEL", "normal")


def main() -> None:
    """Parse CLI arguments and run the video commentary pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="commentator",
        description=(
            "Add AI narrator commentary to a video. "
            "Reads the input video, generates spoken commentary for each scene "
            "using an LLM + TTS, and writes an output video with the mixed audio."
        ),
    )
    parser.add_argument(
        "input",
        help="Path to the input video file (e.g. my_clip.mp4).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="",
        help=(
            "Path for the output video file. "
            "Defaults to <input_name>_commented.mp4 in the same directory."
        ),
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=COMMENTARY_INTERVAL,
        help=f"How often (in seconds) to scan frames for events (default: {COMMENTARY_INTERVAL}).",
    )
    parser.add_argument(
        "--voice",
        default=TTS_VOICE,
        help=(
            f"TTS voice (default: {TTS_VOICE}). "
            "For OpenAI backend: alloy, echo, fable, onyx, nova, shimmer. "
            "For Azure Speech backend: e.g. en-US-JasonNeural (set via AZURE_SPEECH_VOICE)."
        ),
    )
    parser.add_argument(
        "--level",
        default=COMMENTARY_LEVEL,
        choices=["important", "normal", "all"],
        help=(
            f"Commentary verbosity level (default: {COMMENTARY_LEVEL}). "
            "'important' = goals/penalties only. "
            "'normal' = goals + notable plays. "
            "'all' = everything worth mentioning."
        ),
    )

    args = parser.parse_args()

    # Validate input
    if not os.path.isfile(args.input):
        print(f"[ERROR] Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Determine output path
    output = args.output
    if not output:
        base, ext = os.path.splitext(args.input)
        output = f"{base}_commented{ext or '.mp4'}"

    # Validate Azure OpenAI credentials
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    azure_key = os.getenv("AZURE_OPENAI_KEY", "").strip()

    if not (azure_endpoint and azure_key):
        print(
            "[ERROR] Azure OpenAI credentials not found.\n"
            "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY in a .env file\n"
            "(see .env.example).",
            file=sys.stderr,
        )
        sys.exit(1)

    # Run the pipeline
    from video_processor import process_video

    try:
        process_video(
            input_path=args.input,
            output_path=output,
            interval=args.interval,
            voice=args.voice,
            commentary_level=args.level,
        )
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
