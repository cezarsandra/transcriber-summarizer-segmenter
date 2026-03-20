AUDIO        ?= audio.mp3
INA          ?= ina.json
NEMO         ?= nemo.json
OUTPUT       ?= ./output
WHISPER      ?= medium
LANGUAGE     ?= ro
MIN_TRANSCRIBE ?= 60
GEMINI_MODEL ?= gemini-2.0-flash

PYTHON := python3

.PHONY: help install run run-large clean clean-clips clean-all \
        skip-analyze skip-transcribe skip-summarize only-segment

help:
	@echo ""
	@echo "Sermon Transcriber · Summarizer · Segmenter"
	@echo "============================================"
	@echo ""
	@echo "Usage:  make <target> [AUDIO=...] [INA=...] [NEMO=...] [OUTPUT=...]"
	@echo ""
	@echo "Setup:"
	@echo "  install          Install Python dependencies"
	@echo ""
	@echo "Run:"
	@echo "  run              Full pipeline (Whisper medium)"
	@echo "  run-large        Full pipeline (Whisper large-v3, for 24GB+ GPU)"
	@echo ""
	@echo "Resume (re-run from a specific step, keeps output/segments.json):"
	@echo "  skip-analyze     Start from Step 2 (transcription)"
	@echo "  skip-transcribe  Start from Step 3 (summarization)"
	@echo "  skip-summarize   Start from Step 4 (file output only)"
	@echo "  only-segment     Step 4 only — convert clips + write text files"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean            Remove final output files (keep clips)"
	@echo "  clean-clips      Remove WAV clips only"
	@echo "  clean-all        Remove entire output directory"
	@echo ""
	@echo "Options (override with make run OPTION=value):"
	@echo "  AUDIO=$(AUDIO)"
	@echo "  INA=$(INA)"
	@echo "  NEMO=$(NEMO)"
	@echo "  OUTPUT=$(OUTPUT)"
	@echo "  WHISPER=$(WHISPER)"
	@echo "  LANGUAGE=$(LANGUAGE)"
	@echo "  MIN_TRANSCRIBE=$(MIN_TRANSCRIBE)"
	@echo "  GEMINI_MODEL=$(GEMINI_MODEL)"
	@echo ""

install:
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) pipeline.py \
		--audio $(AUDIO) \
		--ina $(INA) \
		--nemo $(NEMO) \
		--output $(OUTPUT) \
		--whisper-model $(WHISPER) \
		--language $(LANGUAGE) \
		--min-transcribe $(MIN_TRANSCRIBE) \
		--gemini-model $(GEMINI_MODEL)

run-large:
	$(MAKE) run WHISPER=large-v3

skip-analyze:
	$(PYTHON) pipeline.py \
		--audio $(AUDIO) \
		--ina $(INA) \
		--nemo $(NEMO) \
		--output $(OUTPUT) \
		--whisper-model $(WHISPER) \
		--language $(LANGUAGE) \
		--min-transcribe $(MIN_TRANSCRIBE) \
		--gemini-model $(GEMINI_MODEL) \
		--skip-analyze

skip-transcribe:
	$(PYTHON) pipeline.py \
		--audio $(AUDIO) \
		--ina $(INA) \
		--nemo $(NEMO) \
		--output $(OUTPUT) \
		--gemini-model $(GEMINI_MODEL) \
		--skip-analyze \
		--skip-transcribe

skip-summarize:
	$(PYTHON) pipeline.py \
		--audio $(AUDIO) \
		--ina $(INA) \
		--nemo $(NEMO) \
		--output $(OUTPUT) \
		--skip-analyze \
		--skip-transcribe \
		--skip-summarize

only-segment: skip-summarize

clean:
	find $(OUTPUT) -maxdepth 1 -name "*.mp3" -delete
	find $(OUTPUT) -maxdepth 1 -name "*.txt" -delete

clean-clips:
	rm -rf $(OUTPUT)/clips

clean-all:
	rm -rf $(OUTPUT)
