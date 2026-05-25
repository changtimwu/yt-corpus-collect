# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- Python venv is at `.venv/` — always use `.venv/bin/python` and `.venv/bin/pip`
- `ffmpeg` must be installed system-wide (used by `pack.py` for audio slicing)
- All large files (corpus, parquet, logs) live in this directory on the mounted SSD (`/ssd/devhome/work`). **Never write large files to `/tmp`** — internal storage is limited
- The system has ~7.4GB RAM with limited headroom; avoid accumulating large in-memory buffers

## Scripts

### `download.py` — fetch from YouTube

```bash
# Test with a few videos first
.venv/bin/python download.py --max-videos 5 "<playlist_url>"

# Full playlist, background
nohup .venv/bin/python -u download.py "<playlist_url>" > download.log 2>&1 &
```

- Default lang is `zh-TW` (this corpus is Taiwanese content); change with `--lang`
- Default audio format is `m4a` (no re-encoding); use `--audio-format wav` only when downstream tools require it
- Produces `corpus/<video_id>/` dirs containing `.m4a`, `.zh-TW.vtt`, `.info.json`
- After download, writes `corpus/manifest.csv` with a `needs_whisper` flag for videos with no subtitle

### `align.py` — fix VTT timestamp drift with WhisperX (optional but recommended)

```bash
# Test on a few videos
.venv/bin/python align.py --max-videos 3

# Full corpus, background
nohup .venv/bin/python -u align.py > align.log 2>&1 &
```

- YouTube VTT timestamps drift ~0.5–1s from the actual audio; this stage forced-aligns each cue against the m4a using a wav2vec2 model and writes `corpus/<id>/<id>.zh-TW.aligned.vtt`
- `pack.py` automatically prefers `*.aligned.vtt` over the raw VTT when present, so no flag is needed downstream
- Uses CUDA if available, otherwise CPU (slower). On Jetson, install the matching `torch` wheel before `whisperx`
- Cues with mean alignment score below `--min-score` (default 0.2) are dropped — catches music/ads/mistranscribed cues. Tune lower if too many are dropped on clean speech, higher to be more aggressive
- Re-runs are no-ops unless `--force` is passed; safe to re-invoke after adding new videos

### `pack.py` — build the parquet dataset

```bash
# Test run
.venv/bin/python pack.py --max-videos 5 --output test.parquet

# Full run, background
nohup .venv/bin/python -u pack.py --output dataset.parquet > pack.log 2>&1 &

# Check progress
tail -3 pack.log && grep -c "^\[" pack.log
```

- Reads `corpus/`, slices each VTT segment from the m4a via ffmpeg, embeds audio bytes into parquet
- Writes in batches of 20 segments to keep memory bounded (critical — large videos can have 1000+ segments)
- Output schema matches [ky552/ML2021_ASR_ST](https://huggingface.co/datasets/ky552/ML2021_ASR_ST): `audio` (struct with embedded bytes), `transcription`, `file`, plus `video_id`, `start`, `end`, `title`, `channel`, `upload_date`
- If `GEMINI_API_KEY` is set, uses Gemini (`gemini-3.1-flash-lite`) for sentence-level VTT segmentation; otherwise falls back to rule-based `merge_segments()` with a warning

### `search.py` — find company mentions in transcripts

```bash
.venv/bin/python search.py dataset.parquet -o results.csv
.venv/bin/python search.py dataset.parquet --companies twse_stock_list.csv twse_otc_stocks.csv
```

- Searches every transcript segment for TWSE/OTC equity mentions (plain stocks only, CFICode `ES*`)
- Matches both company names and 4-digit ticker codes using Aho-Corasick multi-pattern search
- Input CSVs: `twse_stock_list.csv` (TWSE) and `twse_otc_stocks.csv` (OTC)
- Output CSV columns: `video_id`, `title`, `channel`, `upload_date`, `start`, `end`, `transcription`, `matched_codes`, `matched_names` (pipe-separated when multiple)

### `segment_vtt.py` — standalone Gemini VTT segmentation example

```bash
source ../common.env
.venv/bin/python segment_vtt.py corpus/VIDEO_ID/VIDEO_ID.zh-TW.vtt
# output: corpus/VIDEO_ID/VIDEO_ID.zh-TW.segmented.vtt
```

- Standalone tool to preview Gemini segmentation output on a single VTT before a full pack run

## Architecture

Two independent scripts connected by the `corpus/` directory:

```
download.py → corpus/<id>/{.m4a, .vtt, .info.json}
                          ↓
                  align.py (optional) → corpus/<id>/<id>.zh-TW.aligned.vtt
                          ↓
                       pack.py → dataset.parquet
                                          ↓
                          search.py + twse_*.csv → results.csv
```

**`download.py`** is a thin wrapper around `yt_dlp.YoutubeDL`. All download behaviour is controlled by the options dict in `build_opts()`. The VTT subtitle lang code must match what YouTube actually provides — for this playlist it's `zh-TW` (manually created CC), not `zh-Hant` or `en`.

**`align.py`** is an optional but recommended stage between download and pack. It reuses `pack.parse_vtt()` to read raw cues, then runs `whisperx.align()` with a wav2vec2 model (lang code `zh`) to get word-level timestamps, and writes a sibling `.aligned.vtt` with corrected per-cue start/end (min/max of word boundaries). `pack.py` picks up `*.aligned.vtt` automatically.

**`pack.py`** has three stages per video: `parse_vtt()` → segmentation → `extract_clip()` (ffmpeg subprocess) → `flush()` (pyarrow batch write). The VTT deduplication in `parse_vtt()` removes consecutive identical cues — YouTube repeats the previous line in each new cue for smooth scrolling. Segmentation uses Gemini if `GEMINI_API_KEY` is set, otherwise `merge_segments()`. The `ParquetWriter` stays open across all videos; `flush()` is called every 20 segments to avoid OOM.

## Loading the dataset

```python
from datasets import load_dataset
ds = load_dataset("parquet", data_files="dataset.parquet", split="train")
# ds[0]["audio"] → {"array": np.ndarray, "sampling_rate": 16000, "path": "..."}
# ds[0]["transcription"] → "今年以來的行情"
```
