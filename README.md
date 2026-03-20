# Sermon Transcriber · Summarizer · Segmenter

A pipeline that takes a religious service recording and two pre-computed analysis files, then automatically identifies sermons, transcribes them, summarizes them, and saves each one as a separate audio file with a matching text document.

---

## How it works

```
audio.mp3 + ina.json + nemo.json
          │
          ▼
  [Step 1] Gemini analysis           (JSON text only — no audio upload)
           Merges INA (music vs. speech) with NeMo (who speaks when)
           → list of all speech segments, music excluded
          │
          ▼
  [Step 2] Whisper transcription
           Skips segments shorter than --min-transcribe (default: 60s)
           ffmpeg extracts each qualifying segment as a WAV clip
           Whisper transcribes each clip locally on GPU
           → raw transcript per segment
          │
          ▼
  [Step 3] Gemini summarization      (transcript text only — no audio)
           Receives the raw transcript and returns a title,
           a 3–5 sentence summary, and a corrected transcript
          │
          ▼
  [Step 4] File output
           WAV clips converted to MP3 (no re-cutting from original)
           Each segment saved as:
             {speaker}__{title}.mp3
             {speaker}__{title}.txt
```

---

## Prerequisites

**System dependencies**

```bash
# Debian / Ubuntu
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

**Python 3.10+**

**PyTorch** — install for your platform before everything else:

```bash
# CUDA 12.1 (recommended for RunPod / local GPU)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch
```

**Python dependencies**

```bash
pip install -r requirements.txt
```

---

## Configuration

Copy the example env file and add your Gemini API key:

```bash
cp .env.example .env
```

`.env`:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

---

## Input files

| Argument | Description |
|---|---|
| `--audio` | Audio file — mp3, wav, or any format ffmpeg supports |
| `--ina` | JSON output from [INA Speech Segmenter](https://github.com/ina-foss/inaSpeechSegmenter) |
| `--nemo` | JSON output from [NeMo Diarization](https://github.com/NVIDIA/NeMo) |

**INA JSON format** (segments labeled as `male`, `female`, `music`, `poetry`, `noEnergy`):
```json
{
  "segments": [
    { "start": "00:00:00:000", "end": "00:12:34:500", "label": "male" },
    { "start": "00:12:34:500", "end": "00:15:00:000", "label": "music" }
  ]
}
```

**NeMo JSON format**:
```json
[
  { "start": 0.0, "end": 754.5, "duration": 754.5, "speaker": "speaker_0" },
  { "start": 760.2, "end": 900.0, "duration": 139.8, "speaker": "speaker_1" }
]
```

---

## Usage

```bash
python pipeline.py \
  --audio slujba.mp3 \
  --ina ina_output.json \
  --nemo nemo_output.json \
  --output ./output \
  --whisper-model medium \
  --language ro
```

**All options:**

```
--audio            Audio file path (required)
--ina              INA Speech Segmenter JSON (required)
--nemo             NeMo diarization JSON (required)
--output           Output directory (default: ./output)

--whisper-model    tiny | base | small | medium | large | large-v2 | large-v3
                   (default: medium)
--language         Language code for Whisper, e.g. ro, en, hu
                   (default: ro)
--min-transcribe   Minimum segment duration in seconds to transcribe.
                   Shorter segments are skipped. (default: 60)
--gemini-model     Gemini model name (default: gemini-2.0-flash)

--skip-analyze     Skip Step 1, load existing output/segments.json
--skip-transcribe  Skip Step 2, re-use existing transcripts
--skip-summarize   Skip Step 3, re-use existing summaries
```

---

## Output

```
output/
├── segments.json            intermediate state (saved after each step)
├── clips/                   WAV clips extracted for Whisper, reused in Step 4
│   ├── clip_000_0-754.wav
│   └── clip_001_760-900.wav
├── Predicator__Titlul_predicii.mp3
├── Predicator__Titlul_predicii.txt
├── Traducator__Alt_titlu.mp3
└── Traducator__Alt_titlu.txt
```

Each `.txt` file contains:
```
Titlu: Credinta si rugaciunea
Speaker: Predicator
Start: 00:00:00
End: 00:12:34
Durata: 12m34s

SUMAR:
Predicatorul a vorbit despre importanta rugaciunii...

TRANSCRIERE:
Text complet corectat...
```

---

## Resuming after a failure

Progress is saved to `output/segments.json` after each step. If something fails mid-run, use the skip flags to pick up where you left off:

```bash
# Re-run from Step 2 (transcription) onward
python pipeline.py --audio slujba.mp3 --ina ina.json --nemo nemo.json \
  --skip-analyze

# Re-run from Step 3 (summarization) onward
python pipeline.py --audio slujba.mp3 --ina ina.json --nemo nemo.json \
  --skip-analyze --skip-transcribe

# Re-run only Step 4 (file output)
python pipeline.py --audio slujba.mp3 --ina ina.json --nemo nemo.json \
  --skip-analyze --skip-transcribe --skip-summarize
```

---

## Running on RunPod / GPU

The pipeline auto-detects CUDA. Whisper uses the GPU automatically if available.

**8GB VRAM** — use `medium` (safe) or `large` (may OOM):

```bash
python pipeline.py \
  --audio slujba.mp3 \
  --ina ina.json \
  --nemo nemo.json \
  --whisper-model medium \
  --output ./output
```

**24GB+ VRAM** (A100, A10G) — use `large-v3` for best accuracy:

```bash
python pipeline.py \
  --audio slujba.mp3 \
  --ina ina.json \
  --nemo nemo.json \
  --whisper-model large-v3 \
  --output ./output
```

---

## Project structure

```
.
├── pipeline.py        main entry point and CLI
├── models.py          SpeechSegment dataclass shared across all steps
├── steps/
│   ├── analyze.py     Step 1 — Gemini merges INA + NeMo (JSON text only)
│   ├── transcribe.py  Step 2 — ffmpeg clip extraction + Whisper transcription
│   ├── summarize.py   Step 3 — Gemini title, summary, correction (text only)
│   └── segment.py     Step 4 — WAV → MP3 conversion + text file output
├── requirements.txt
└── .env.example
```
