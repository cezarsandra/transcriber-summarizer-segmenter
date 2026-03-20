"""
Step 4 — File output.

The audio clips are already extracted in Step 2 (transcribe.py).
Here we simply convert each WAV clip to MP3 and write the text file.

For each SpeechSegment:
  1. Convert existing WAV clip → {speaker_name}__{title}.mp3
  2. Write {speaker_name}__{title}.txt  (title, summary, transcript)
"""

import re
import subprocess
from pathlib import Path

from models import SpeechSegment


def _sanitize(text: str, max_len: int = 50) -> str:
    """Convert text to a safe filename fragment."""
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", "_", text.strip())
    return text[:max_len]


def _convert_to_mp3(src: Path, dst: Path) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-acodec", "libmp3lame",
        "-q:a", "2",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {dst.name}:\n{result.stderr}")


def _write_text(seg: SpeechSegment, path: Path) -> None:
    speaker_display = seg.speaker_name or seg.speaker
    lines = [
        f"Titlu: {seg.title}",
        f"Speaker: {speaker_display}",
        f"Start: {seg.display_start}",
        f"End: {seg.display_end}",
        f"Durata: {seg.display_duration}",
        "",
        "SUMAR:",
        seg.summary or "",
        "",
        "TRANSCRIERE:",
        seg.corrected_transcript or seg.transcript or "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def segment(
    segments: list[SpeechSegment],
    output_dir: Path,
) -> list[SpeechSegment]:
    """
    Convert existing WAV clips to MP3 and write text files.
    Clips must already exist (set by transcribe step).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, seg in enumerate(segments):
        if not seg.audio_clip:
            print(f"  [{i+1}/{len(segments)}] Skipping (no clip): {seg.display_start}")
            continue

        speaker_part = _sanitize(seg.speaker_name or seg.speaker or f"speaker_{i}")
        title_part = _sanitize(seg.title or f"segment_{i+1:03d}")
        base_name = f"{speaker_part}__{title_part}"

        audio_out = output_dir / f"{base_name}.mp3"
        text_out = output_dir / f"{base_name}.txt"

        print(f"  [{i+1}/{len(segments)}] Saving: {audio_out.name}")

        _convert_to_mp3(Path(seg.audio_clip), audio_out)
        _write_text(seg, text_out)

        seg.output_audio = str(audio_out)
        seg.output_text = str(text_out)

    return segments
