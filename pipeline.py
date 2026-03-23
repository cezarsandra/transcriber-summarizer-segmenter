#!/usr/bin/env python3
"""
Sermon Processing Pipeline
==========================

Takes an audio file + two JSON analysis files and produces:
  - Segmented audio files per sermon/speech
  - Text files with title, summary, and corrected transcript

Usage:
    python pipeline.py \\
        --audio slujba.mp3 \\
        --ina ina_output.json \\
        --nemo nemo_output.json \\
        --output ./output \\
        --whisper-model medium \\
        --min-transcribe 60 \\
        --language ro

Steps:
    1. analyze    — Gemini receives both JSONs (text only) → clean speech segments
    2. transcribe — ffmpeg extracts clips, Whisper transcribes each one
    3. summarize  — Gemini receives transcript text → title, summary, correction
    4. segment    — clips converted to MP3, text files written
"""

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _save_segments(segments, path: Path) -> None:
    path.write_text(
        json.dumps([dataclasses.asdict(s) for s in segments], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _load_segments(path: Path):
    from models import SpeechSegment
    raw = json.loads(path.read_text())
    return [SpeechSegment(**s) for s in raw]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sermon transcription, summarization, and segmentation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--audio", required=True, help="Input audio file (mp3, wav, etc.)")
    parser.add_argument("--ina", required=True, help="INA Speech Segmenter JSON output")
    parser.add_argument("--nemo", required=True, help="NeMo diarization JSON output")
    parser.add_argument("--output", default="./output", help="Output directory (default: ./output)")
    parser.add_argument(
        "--whisper-model",
        default="medium",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper model size (default: medium)",
    )
    parser.add_argument(
        "--language",
        default="ro",
        help="Audio language code for Whisper (default: ro)",
    )
    parser.add_argument(
        "--min-transcribe",
        type=float,
        default=60.0,
        help="Minimum segment duration in seconds to transcribe (default: 60)",
    )
    parser.add_argument(
        "--gemini-model",
        default="gemini-2.5-flash",
        help="Gemini model name (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cuda", "cpu"],
        help="Device for Whisper: cuda or cpu (default: auto-detect)",
    )
    parser.add_argument(
        "--analyze-with",
        default="gemini",
        choices=["gemini", "runpod"],
        help="Backend for Step 1 analysis: gemini or runpod (default: gemini)",
    )
    parser.add_argument(
        "--summarize-with",
        default="gemini",
        choices=["gemini", "runpod"],
        help="Backend for Step 3 summarization: gemini or runpod (default: gemini)",
    )
    parser.add_argument(
        "--runpod-url",
        default="",
        help="RunPod endpoint URL, e.g. https://api.runpod.ai/v2/<endpoint-id>/run",
    )
    parser.add_argument(
        "--runpod-api-key",
        default="",
        help="RunPod API key (or set RUNPOD_API_KEY env var)",
    )
    parser.add_argument(
        "--runpod-max-tokens",
        type=int,
        default=16000,
        help="max_tokens for RunPod LLM responses (default: 16000)",
    )
    parser.add_argument("--skip-analyze",    action="store_true", help="Skip Step 1, load from output/segments.json")
    parser.add_argument("--skip-transcribe", action="store_true", help="Skip Step 2, re-use existing transcripts")
    parser.add_argument("--skip-summarize",  action="store_true", help="Skip Step 3, re-use existing summaries")
    return parser.parse_args()


def main():
    args = parse_args()

    audio_path = Path(args.audio)
    ina_path   = Path(args.ina)
    nemo_path  = Path(args.nemo)
    output_dir = Path(args.output)
    clips_dir  = output_dir / "clips"

    for p, name in [(audio_path, "audio"), (ina_path, "ina"), (nemo_path, "nemo")]:
        if not p.exists():
            print(f"Error: {name} file not found: {p}", file=sys.stderr)
            sys.exit(1)

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key and (args.analyze_with == "gemini" or args.summarize_with == "gemini"):
        print("Error: GEMINI_API_KEY not set in environment or .env file", file=sys.stderr)
        sys.exit(1)

    runpod_api_key = args.runpod_api_key or os.environ.get("RUNPOD_API_KEY", "")
    if not runpod_api_key and (args.analyze_with == "runpod" or args.summarize_with == "runpod"):
        print("Error: --runpod-api-key or RUNPOD_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    from steps.analyze import analyze
    from steps.transcribe import transcribe
    from steps.summarize import summarize
    from steps.segment import segment

    print("=" * 60)
    print("SERMON PIPELINE")
    print(f"  Audio:         {audio_path}")
    print(f"  INA JSON:      {ina_path}")
    print(f"  NeMo JSON:     {nemo_path}")
    print(f"  Output:        {output_dir}")
    print(f"  Whisper model: {args.whisper_model}")
    print(f"  Language:      {args.language}")
    print(f"  Min transcribe: {args.min_transcribe}s")
    print(f"  Device:         {args.device or 'auto'}")
    print(f"  Gemini model:  {args.gemini_model}")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)
    segments_json = output_dir / "segments.json"

    # -------------------------------------------------------------------------
    # Step 1 — Gemini merges INA + NeMo (text only, no audio upload)
    # -------------------------------------------------------------------------
    if not args.skip_analyze:
        print("\n[Step 1] Gemini analysis — merging INA + NeMo...")
        segments = analyze(
            ina_json_path=ina_path,
            nemo_json_path=nemo_path,
            api_key=api_key,
            gemini_model=args.gemini_model,
            backend=args.analyze_with,
            runpod_url=args.runpod_url,
            runpod_api_key=runpod_api_key,
            runpod_max_tokens=args.runpod_max_tokens,
        )
        _save_segments(segments, segments_json)
        print(f"  Saved to {segments_json}")
    else:
        print("\n[Step 1] Skipped — loading from", segments_json)
        segments = _load_segments(segments_json)

    if not segments:
        print("No speech segments found. Exiting.")
        sys.exit(0)

    print(f"\n  {len(segments)} speech segment(s) to process:")
    for s in segments:
        print(f"    {s.display_start} → {s.display_end}  ({s.display_duration})  {s.speaker}")

    # -------------------------------------------------------------------------
    # Step 2 — ffmpeg clip extraction + Whisper transcription
    # -------------------------------------------------------------------------
    if not args.skip_transcribe:
        print("\n[Step 2] Whisper transcription...")
        segments = transcribe(
            audio_path=audio_path,
            segments=segments,
            clips_dir=clips_dir,
            model_name=args.whisper_model,
            language=args.language,
            min_duration=args.min_transcribe,
            device=args.device,
        )
        _save_segments(segments, segments_json)
    else:
        print("\n[Step 2] Skipped.")

    # -------------------------------------------------------------------------
    # Step 3 — Gemini summarization (text only)
    # -------------------------------------------------------------------------
    if not args.skip_summarize:
        print("\n[Step 3] Gemini summarization...")
        segments = summarize(
            segments=segments,
            api_key=api_key,
            gemini_model=args.gemini_model,
            backend=args.summarize_with,
            runpod_url=args.runpod_url,
            runpod_api_key=runpod_api_key,
            runpod_max_tokens=args.runpod_max_tokens,
        )
        _save_segments(segments, segments_json)
    else:
        print("\n[Step 3] Skipped.")

    # -------------------------------------------------------------------------
    # Step 4 — Convert clips to MP3 + write text files
    # -------------------------------------------------------------------------
    print("\n[Step 4] Writing output files...")
    segments = segment(
        segments=segments,
        output_dir=output_dir,
    )
    _save_segments(segments, segments_json)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DONE")
    print(f"Output: {output_dir.resolve()}")
    print(f"Manifest: {segments_json.resolve()}")
    for s in segments:
        print(f"\n  [{s.display_start} → {s.display_end}]  ({s.display_duration})")
        print(f"    Speaker : {s.speaker_name or s.speaker}")
        print(f"    Title   : {s.title}")
        print(f"    Audio   : {Path(s.output_audio).name if s.output_audio else '-'}")
        print(f"    Text    : {Path(s.output_text).name if s.output_text else '-'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
