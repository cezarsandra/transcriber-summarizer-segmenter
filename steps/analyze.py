"""
Step 1 — Gemini analysis.

Sends only the two JSON files (INA Speech Segmenter + NeMo diarization) as
text to Gemini and asks it to produce a clean, merged list of speech segments.
Songs are excluded. No audio is uploaded.

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


SYSTEM_INSTRUCTION = """ROLE: You are an expert in audio/video editing for religious content. Your task is to transform raw diarization data (NeMo) and segmentation data (INA Speech Segmenter) into a logical table of contents for a Christian service.

EVENT CONTEXT:
A service is made up of large blocks. People do not speak in 2-second bursts; if a speaker is active, they have a message to deliver (sermon, exhortation, or poetry reading).

LOGIC RULES (Strict):

1. Status Priority: If INA Speech Segmenter indicates "music" on a long segment, ignore any speaker identified by NeMo there. Do NOT include songs or music in the output — they are excluded entirely.

2. Aggregation (Smoothing): If speaker_x speaks, stops for 10 seconds (pause/noise) and continues as the same speaker, merge everything into a single event. Do not cut a speech!

3. Relevance Threshold:
   - Over 5 minutes of voice = Sermon
   - Between 1 and 5 minutes of voice = Exhortation / Poetry
   - Under 1 minute = Background noise / Amens / Reactions -> IGNORE COMPLETELY

4. Overlap Resolution: If two segments overlap in time, keep only one:
   - Prefer the segment from INA Speech Segmenter labeled "male" or "female"
   - If both are from the same source, keep the one with the longer duration
   - The final output must have zero overlapping time intervals

5. Speaker Names (MANDATORY): If the NeMo source already contains a real name for a speaker
   (e.g. "claudiu_sandra", "cezar_sandra"), you MUST use that name in the output.
   Never replace a known name with a generic ID. Ignoring a provided name is a critical error.
   Only use "Brother/Speaker [ID]" if no name is available in the source data.

OUTPUT FORMAT:
Return ONLY a valid JSON array. No markdown, no explanation.
Times must be in seconds as floats.

[
  {
    "start": 192.0,
    "end": 782.0,
    "duration": 590.0,
    "speaker": "speaker_0",
    "speaker_name": "Claudiu Sandra",
    "label": "male",
    "type": "sermon"
  }
]

Valid types: "sermon", "exhortation", "poetry"
Songs and music are NOT included in the output.
"""

USER_PROMPT = """### CONTEXT
Analyze the data below from a religious service recording.
Your goal is to clean the "noise" and give me only the main spoken events (table of contents).

### RAW DATA

[SOURCE 1 - NVIDIA NEMO]:
{nemo_json}

[SOURCE 2 - INASPEECH]:
{ina_json}

### SPECIFIC TASK
1. Ignore "unknown" or unlabeled segments if they are short (under 2 minutes).
2. If you see a known speaker talking for more than 5 minutes, create a "sermon" row.
3. Ignore songs and music entirely — do not include them in the output.
4. Apply the smoothing rule: merge same-speaker blocks separated by less than 60 seconds.
5. Return only spoken events: sermons, exhortations, poetry.
"""


def analyze(
    ina_json_path: Path,
    nemo_json_path: Path,
    api_key: str,
    gemini_model: str = "gemini-2.5-flash",
) -> list[SpeechSegment]:
    """
    Send both JSON files as text to Gemini and return all speech SpeechSegments.
    Songs and music are excluded. Duration filtering happens in the transcribe step.
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
    print(f"  Found {len(segments)} speech segment(s) after analysis.")
    return segments
