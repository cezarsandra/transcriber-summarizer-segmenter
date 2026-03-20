"""
Shared data models for the transcriber-summarizer-segmenter pipeline.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SpeechSegment:
    """A single speech segment after Gemini analysis (Step 1)."""
    start: float          # seconds
    end: float            # seconds
    duration: float       # seconds
    speaker: str          # e.g. "speaker_0" or a known name
    label: str            # "male" | "female" | "poetry" | "unknown"

    # Filled in Step 2
    audio_clip: Optional[str] = None       # path to extracted audio clip
    transcript: Optional[str] = None       # raw Whisper output

    # Filled in Step 3
    title: Optional[str] = None
    summary: Optional[str] = None
    corrected_transcript: Optional[str] = None
    speaker_name: Optional[str] = None    # name identified by Gemini if known

    # Filled in Step 4
    output_audio: Optional[str] = None    # final saved audio path
    output_text: Optional[str] = None     # final saved text path

    @property
    def display_start(self) -> str:
        return _fmt(self.start)

    @property
    def display_end(self) -> str:
        return _fmt(self.end)

    @property
    def display_duration(self) -> str:
        m = int(self.duration // 60)
        s = int(self.duration % 60)
        return f"{m}m{s:02d}s"


def _fmt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
