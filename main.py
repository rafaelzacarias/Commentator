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
COMMENTARY_INTERVAL: int = int(os.getenv("COMMENTARY_INTERVAL", "10"))
TTS_VOICE: str = os.getenv("TTS_VOICE", "onyx")


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
        help=f"How often (in seconds) to generate a commentary clip (default: {COMMENTARY_INTERVAL}).",
    )
    parser.add_argument(
        "--voice",
        default=TTS_VOICE,
        choices=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
        help=f"TTS voice to use for narration (default: {TTS_VOICE}).",
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

    # Validate API credentials
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    azure_key = os.getenv("AZURE_OPENAI_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()

    if not ((azure_endpoint and azure_key) or openai_key):
        print(
            "[ERROR] No API credentials found.\n"
            "Set OPENAI_API_KEY in a .env file (see .env.example), or\n"
            "set AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_KEY for Azure OpenAI.",
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
        )
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
