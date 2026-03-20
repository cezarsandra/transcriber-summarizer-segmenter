"""
Step 1 — Gemini analysis.

Sends only the two JSON files (INA Speech Segmenter + NeMo diarization) as
text to Gemini and asks it to produce a clean, merged list of speech segments.
No audio is uploaded.

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
clean table of contents of the service.

Follow these steps exactly:

=== STEP 1: Establish Anchors (INA Speech Segmenter) ===
Identify continuous Male/Female voice blocks from the INA source.
IF a voice block is longer than 300 seconds (5 minutes), it is MANDATORY a SERMON.
Do NOT split it into smaller pieces even if the speaker briefly changes inside it.

=== STEP 2: Overlay with NeMo ===
For each Anchor found in Step 1, look at the NeMo source and identify which speaker
has the most seconds within that time interval.
Assign the entire block duration to that dominant speaker.

=== STEP 3: Noise Filter (Golden Rule) ===
FORBIDDEN: Do not include any row in the final result with duration under 60 seconds,
EXCEPT if it is marked as Poetry.
Ignore all "noise" segments and any pause under 1 second from both sources.

=== STEP 4: Songs ===
Any "music" segment from INA longer than 45 seconds becomes a "song".
Songs are included in the output with type "song".

=== OUTPUT FORMAT ===
Return ONLY a valid JSON array. No markdown, no explanation.
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

Valid types: "sermon", "speech", "poetry", "song"
"""

USER_PROMPT = """[INPUT_DATA]
{{
  "source_inaspeech": {ina_json},
  "source_nemo": {nemo_json}
}}

[TASK]
Correlate the data following your system instructions.
Apply all 4 steps in order.
Return only SERMONS (over 5 min), SONGS (music over 45s), and POETRY.
Ignore all short diarization residues of a few seconds.
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
        ina_json=json.dumps(ina_data, ensure_ascii=False, indent=2),
        nemo_json=json.dumps(nemo_data, ensure_ascii=False, indent=2),
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
            label=item.get("label", item.get("type", "unknown")),
        )
        if item.get("speaker_name"):
            seg.speaker_name = item["speaker_name"]
        segments.append(seg)

    segments.sort(key=lambda s: s.start)
    print(f"  Found {len(segments)} segment(s) after analysis.")
    return segments
