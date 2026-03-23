"""
Step 3 — Summarization.

Sends ALL transcripts in a single API call and gets back title, summary,
corrected transcript, and speaker name for each segment.

Supports two backends:
  - gemini  : Google Gemini API (default)
  - runpod  : RunPod serverless worker
"""

import json
from typing import Literal

from models import SpeechSegment
from utils.llm import call_gemini, call_runpod, clean_json_response


SYSTEM_INSTRUCTION = """You are an assistant that processes religious sermon transcripts."""

PROMPT_TEMPLATE = """Below is a JSON array of speech segments from a single service recording.
Each segment has an index, speaker name, duration, and raw transcript.

For EACH segment return a JSON object with:
- "index": same index as input
- "title": short descriptive title (max 8 words)
- "summary": 3-5 sentence summary of the main ideas
- "corrected_transcript": full transcript corrected for grammar, punctuation, and obvious transcription errors — keep it faithful to the original
- "speaker_name": name or role of the speaker if identifiable from context, otherwise null

Return ONLY a valid JSON array with one object per input segment, no markdown, no explanation.

INPUT:
{payload}
"""


def summarize(
    segments: list[SpeechSegment],
    api_key: str,
    gemini_model: str = "gemini-2.5-flash",
    backend: Literal["gemini", "runpod"] = "gemini",
    runpod_url: str = "",
    runpod_api_key: str = "",
) -> list[SpeechSegment]:
    """
    Summarize all transcribed segments in a single API call.

    backend="gemini"  — uses Gemini API
    backend="runpod"  — uses RunPod serverless worker
    """
    to_process = [(i, seg) for i, seg in enumerate(segments) if seg.transcript]

    if not to_process:
        print("  No transcripts to summarize.")
        return segments

    print(f"  Sending {len(to_process)} transcript(s) to {backend} in one call...")

    payload = [
        {
            "index": idx,
            "speaker": seg.speaker_name or seg.speaker,
            "duration": seg.display_duration,
            "transcript": seg.transcript,
        }
        for idx, (_, seg) in enumerate(to_process)
    ]

    prompt = PROMPT_TEMPLATE.format(
        payload=json.dumps(payload, ensure_ascii=False, indent=2)
    )

    if backend == "runpod":
        if not runpod_url or not runpod_api_key:
            raise ValueError("--runpod-url and --runpod-api-key are required when using --summarize-with runpod")
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ]
        text = call_runpod(messages, endpoint_url=runpod_url, api_key=runpod_api_key)
    else:
        text = call_gemini(prompt, system_instruction=SYSTEM_INSTRUCTION, api_key=api_key, model=gemini_model)

    print(f"  Raw response (first 500 chars):\n{text[:500]}")
    results = json.loads(clean_json_response(text))
    # Handle {"segments": [...]} vs plain [...]
    if isinstance(results, dict):
        results = next(iter(results.values()))

    results_by_index = {r["index"]: r for r in results}

    for idx, (i, seg) in enumerate(to_process):
        result = results_by_index.get(idx)
        if not result:
            print(f"  Warning: no result for segment {i+1}, keeping raw transcript.")
            seg.title = f"Segment {i+1}"
            seg.corrected_transcript = seg.transcript
            continue

        seg.title = result.get("title", f"Segment {i+1}")
        seg.summary = result.get("summary", "")
        seg.corrected_transcript = result.get("corrected_transcript", seg.transcript)
        if result.get("speaker_name"):
            seg.speaker_name = result["speaker_name"]

        print(f"  [{i+1}] {seg.title}  ({seg.speaker_name or seg.speaker})")

    return segments
