"""
Step 1 — Gemini analysis.

Sends only the two JSON files (INA Speech Segmenter + NeMo diarization) as
text to Gemini and asks it to produce a clean, merged list of speech segments
(no music, no very-short segments). No audio is uploaded.

INA JSON schema expected:
  {"segments": [{"start": "hh:mm:ss:ms", "end": "hh:mm:ss:ms", "label": "male|female|music|poetry"}, ...]}

NeMo JSON schema expected:
  [{"start": 45.3, "end": 180.5, "duration": 135.2, "speaker": "speaker_0"}, ...]
"""

import json
from pathlib import Path

from google import genai
from google.genai import types

from models import SpeechSegment


SYSTEM_INSTRUCTION = """You are an expert audio structure analyst for evangelical religious services.

Your role is to correlate two metadata sources from the same recording and produce a precise,
clean table of contents of the service — identifying sermons, songs, and poetry.

=== CLASSIFICATION RULES ===

SERMON:
- INA label is "male" or "female" continuously for more than 5 minutes
- NeMo confirms a dominant speaker (same speaker ID) across that period
- Short interruptions (< 30 seconds) from another speaker within a sermon block
  are part of the same sermon (translation, amen, brief response) — do NOT split them
- Merge consecutive sermon segments from the same speaker if the gap is < 60 seconds

SONG:
- INA label is "music" for more than 30 seconds
- Do NOT include songs in the output — they are excluded

POETRY:
- A short "female" or "male" segment (< 5 minutes) surrounded by music segments
- Include in the output with type "poetry"

TESTIMONY / ANNOUNCEMENT:
- A "male" or "female" segment between 1 and 5 minutes that does not qualify as a sermon
- Include with type "speech"

=== NOISE FILTER (CRITICAL) ===
- Ignore ANY segment shorter than 5 seconds from both sources — these are echo artifacts
- Ignore all "noEnergy" and "noise" labels entirely — treat them as silence

=== OUTPUT FORMAT ===
Return ONLY a valid JSON array. No markdown, no explanation, no commentary.
Times must be in seconds as floats.

[
  {
    "start": 123.4,
    "end": 456.7,
    "duration": 333.3,
    "speaker": "speaker_0",
    "speaker_name": "Preacher",
    "label": "male",
    "type": "sermon"
  }
]

Valid types: "sermon", "speech", "poetry"
"""

USER_PROMPT = """Analyze the two metadata sources below and extract the service structure
according to your system instructions.

=== SOURCE 1: NVIDIA NEMO (SPEAKER DIARIZATION) ===
{nemo_json}

=== SOURCE 2: INASPEECH (AUDIO SEGMENTATION) ===
{ina_json}

=== REQUIREMENT ===
Generate the final list of events (Sermons, Speeches, Poetry).
Timestamps must be in seconds (float). Calculate the real duration for each.
Be aggressive about merging — a sermon with brief pauses is still one sermon.
"""


def analyze(
    ina_json_path: Path,
    nemo_json_path: Path,
    api_key: str,
    gemini_model: str = "gemini-2.5-flash",
) -> list[SpeechSegment]:
    """
    Send both JSON files as text to Gemini and return all speech SpeechSegments.
    Duration filtering happens later in the transcribe step.
    """
    client = genai.Client(api_key=api_key)

    ina_data = json.loads(ina_json_path.read_text())
    nemo_data = json.loads(nemo_json_path.read_text())

    prompt = USER_PROMPT.format(
        nemo_json=json.dumps(nemo_data, ensure_ascii=False, indent=2),
        ina_json=json.dumps(ina_data, ensure_ascii=False, indent=2),
    )

    print(f"  Calling Gemini ({gemini_model}) for segment analysis...")
    response = client.models.generate_content(
        model=gemini_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
        ),
    )

    raw = json.loads(response.text)

    segments = []
    for item in raw:
        dur = float(item.get("duration", item["end"] - item["start"]))
        seg = SpeechSegment(
            start=float(item["start"]),
            end=float(item["end"]),
            duration=dur,
            speaker=item.get("speaker", "unknown"),
            label=item.get("label", "unknown"),
        )
        if item.get("speaker_name"):
            seg.speaker_name = item["speaker_name"]
        segments.append(seg)

    segments.sort(key=lambda s: s.start)
    print(f"  Found {len(segments)} speech segment(s) after filtering.")
    return segments
