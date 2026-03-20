"""
Step 2 — Whisper transcription.

For each SpeechSegment that passes the minimum duration threshold:
  1. Extract audio clip with ffmpeg
  2. Run Whisper on the clip
  3. Store raw transcript in segment.transcript

Segments shorter than min_duration are skipped (no clip extracted, no transcript).
Music is already excluded by Step 1 (Gemini analysis).
"""

import subprocess
from pathlib import Path
from typing import Optional

import torch
import whisper

from models import SpeechSegment


def _extract_clip(audio_path: Path, start: float, end: float, out_path: Path) -> None:
    duration = end - start
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(audio_path),
        "-t", str(duration),
        "-ar", "16000",
        "-ac", "1",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for clip {out_path.name}:\n{result.stderr}")


def transcribe(
    audio_path: Path,
    segments: list[SpeechSegment],
    clips_dir: Path,
    model_name: str = "medium",
    language: Optional[str] = "ro",
    min_duration: float = 60.0,
) -> list[SpeechSegment]:
    """
    Transcribe speech segments with Whisper.

    Skips segments shorter than min_duration seconds.
    Clips are saved to clips_dir for reuse by Step 4.
    Returns the same list — short segments are kept but have no transcript.
    """
    clips_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading Whisper model '{model_name}' on {device}...")
    model = whisper.load_model(model_name, device=device)

    for i, seg in enumerate(segments):
        prefix = f"  [{i+1}/{len(segments)}] {seg.display_start} → {seg.display_end}  ({seg.display_duration})"

        if seg.duration < min_duration:
            print(f"{prefix}  — skipped (< {int(min_duration)}s)")
            continue

        clip_name = f"clip_{i:03d}_{int(seg.start)}-{int(seg.end)}.wav"
        clip_path = clips_dir / clip_name

        print(f"{prefix}  speaker={seg.speaker}")

        if not clip_path.exists():
            _extract_clip(audio_path, seg.start, seg.end, clip_path)

        seg.audio_clip = str(clip_path)

        result = model.transcribe(
            str(clip_path),
            language=language,
            task="transcribe",
            fp16=(device == "cuda"),
        )
        seg.transcript = result["text"].strip()
        print(f"    → {seg.transcript[:80]}{'...' if len(seg.transcript) > 80 else ''}")

    return segments
