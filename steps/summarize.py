"""
Step 3 — Gemini summarization.

For each SpeechSegment that has a transcript:
  - Send the raw transcript to Gemini
  - Get back: title, summary (3-5 sentences), corrected transcript, speaker name
"""

import json

from google import genai
from google.genai import types

from models import SpeechSegment


def summarize(
    segments: list[SpeechSegment],
    api_key: str,
    gemini_model: str = "gemini-2.0-flash",
) -> list[SpeechSegment]:
    """
    Summarize and correct each segment's transcript using Gemini.
    Returns the same list with title/summary/corrected_transcript filled in.
    """
    client = genai.Client(api_key=api_key)

    for i, seg in enumerate(segments):
        if not seg.transcript:
            print(f"  [{i+1}/{len(segments)}] Skipping (no transcript): {seg.display_start}")
            continue

        print(f"  [{i+1}/{len(segments)}] Summarizing: {seg.display_start} → {seg.display_end}  speaker={seg.speaker}")

        speaker_hint = f"The speaker is known as '{seg.speaker_name}'." if seg.speaker_name else ""

        prompt = f"""You are an assistant that processes Romanian religious sermon transcripts.
{speaker_hint}

Below is a raw transcript from a speech segment ({seg.display_duration} long):

---
{seg.transcript}
---

Please return a valid JSON object (no markdown, no explanation):
{{
  "title": "Short descriptive title for this speech (max 8 words)",
  "summary": "3-5 sentence summary of the main ideas",
  "corrected_transcript": "The full transcript corrected for grammar, punctuation, and obvious transcription errors. Keep it faithful to the original.",
  "speaker_name": "Name or role of the speaker if you can identify it from context, otherwise null"
}}
"""

        response = client.models.generate_content(
            model=gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )

        result = json.loads(response.text)
        seg.title = result.get("title", f"Segment {i+1}")
        seg.summary = result.get("summary", "")
        seg.corrected_transcript = result.get("corrected_transcript", seg.transcript)
        if result.get("speaker_name"):
            seg.speaker_name = result["speaker_name"]

        print(f"    Title: {seg.title}")

    return segments
