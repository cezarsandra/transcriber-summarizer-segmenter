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


def analyze(
    ina_json_path: Path,
    nemo_json_path: Path,
    api_key: str,
    gemini_model: str = "gemini-2.0-flash",
) -> list[SpeechSegment]:
    """
    Send both JSON files as text to Gemini and return all speech SpeechSegments.
    Duration filtering happens later in the transcribe step.
    """
    client = genai.Client(api_key=api_key)

    ina_data = json.loads(ina_json_path.read_text())
    nemo_data = json.loads(nemo_json_path.read_text())

    prompt = f"""You are an assistant that analyzes religious service recordings.

You have two analyses of the same audio file:

1. INA Speech Segmenter output (identifies music vs speech, labels: male/female/music/poetry/noEnergy):
{json.dumps(ina_data, ensure_ascii=False, indent=2)}

2. NeMo Speaker Diarization output (identifies who speaks when, with speaker IDs):
{json.dumps(nemo_data, ensure_ascii=False, indent=2)}

Your task:
- Combine both analyses to identify all SPEECH segments (sermons, speeches, testimonials).
- EXCLUDE: music, songs, noEnergy.
- For each speech segment, try to assign the speaker a meaningful label based on context
  (e.g. "Predicator", "Traducator", "Marturie", or use the NeMo speaker ID if unknown).
- Merge consecutive segments from the same speaker if the gap between them is small (< 10 seconds).
- Times must be in seconds (float), not formatted strings.

Return ONLY a valid JSON array, no markdown, no explanation:
[
  {{
    "start": 123.4,
    "end": 456.7,
    "duration": 333.3,
    "speaker": "speaker_0",
    "speaker_name": "Predicator",
    "label": "male"
  }}
]
"""

    print(f"  Calling Gemini ({gemini_model}) for segment analysis...")
    response = client.models.generate_content(
        model=gemini_model,
        contents=prompt,
        config=types.GenerateContentConfig(
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
