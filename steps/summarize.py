"""
Step 3 — Gemini summarization.

Sends ALL transcripts in a single API call.
Gemini returns a JSON array with title, summary, corrected transcript
and speaker name for each segment.
"""

import json

from google import genai
from google.genai import types

from models import SpeechSegment


def summarize(
    segments: list[SpeechSegment],
    api_key: str,
    gemini_model: str = "gemini-2.5-flash",
) -> list[SpeechSegment]:
    """
    Summarize all transcribed segments in a single Gemini call.
    Segments without a transcript are skipped and returned unchanged.
    """
    client = genai.Client(api_key=api_key)

    to_process = [(i, seg) for i, seg in enumerate(segments) if seg.transcript]

    if not to_process:
        print("  No transcripts to summarize.")
        return segments

    print(f"  Sending {len(to_process)} transcript(s) to Gemini in one call...")

    segments_payload = []
    for idx, (i, seg) in enumerate(to_process):
        segments_payload.append({
            "index": idx,
            "speaker": seg.speaker_name or seg.speaker,
            "duration": seg.display_duration,
            "transcript": seg.transcript,
        })

    prompt = f"""You are an assistant that processes religious sermon transcripts.

Below is a JSON array of speech segments from a single service recording.
Each segment has an index, speaker name, duration, and raw transcript.

For EACH segment return a JSON object with:
- "index": same index as input
- "title": short descriptive title (max 8 words)
- "summary": 3-5 sentence summary of the main ideas
- "corrected_transcript": full transcript corrected for grammar, punctuation, and obvious transcription errors — keep it faithful to the original
- "speaker_name": name or role of the speaker if identifiable from context, otherwise null

Return ONLY a valid JSON array with one object per input segment, no markdown, no explanation.

INPUT:
{json.dumps(segments_payload, ensure_ascii=False, indent=2)}
"""

    response = client.models.generate_content(
        model=gemini_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )

    if not response.text:
        finish = getattr(response.candidates[0], "finish_reason", "unknown") if response.candidates else "no candidates"
        print(f"  Warning: empty response from Gemini (finish_reason={finish}). Skipping summarization.")
        return segments

    results = json.loads(response.text)

    results_by_index = {r["index"]: r for r in results}

    for idx, (i, seg) in enumerate(to_process):
        result = results_by_index.get(idx)
        if not result:
            print(f"  Warning: no result returned for segment {i+1}, keeping raw transcript.")
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
