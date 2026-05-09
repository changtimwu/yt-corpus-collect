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

### `pack.py` — build the parquet dataset

```bash
# Test run
.venv/bin/python pack.py --max-videos 5 --output test.parquet

# Full run, background
nohup .venv/bin/python -u pack.py --output dataset.parquet > pack.log 2>&1 &

# Check progress
tail -3 pack.log && grep -c "^\[" pack.log
```

- Reads `corpus/`, slices each VTT segment from the m4a via ffmpeg, embeds 16 kHz mono WAV bytes into parquet
- Writes in batches of 20 segments to keep memory bounded (critical — large videos can have 1000+ segments)
- Output schema matches [ky552/ML2021_ASR_ST](https://huggingface.co/datasets/ky552/ML2021_ASR_ST): `audio` (struct with embedded bytes), `transcription`, `file`, plus `video_id`, `start`, `end`, `title`, `channel`, `upload_date`

## Architecture

Two independent scripts connected by the `corpus/` directory:

```
download.py → corpus/<id>/{.m4a, .vtt, .info.json} → pack.py → dataset.parquet
```

**`download.py`** is a thin wrapper around `yt_dlp.YoutubeDL`. All download behaviour is controlled by the options dict in `build_opts()`. The VTT subtitle lang code must match what YouTube actually provides — for this playlist it's `zh-TW` (manually created CC), not `zh-Hant` or `en`.

**`pack.py`** has three stages per video: `parse_vtt()` → `extract_clip()` (ffmpeg subprocess) → `flush()` (pyarrow batch write). The VTT deduplication in `parse_vtt()` removes consecutive identical cues — YouTube repeats the previous line in each new cue for smooth scrolling. The `ParquetWriter` stays open across all videos; `flush()` is called every 20 segments to avoid OOM.

## Loading the dataset

```python
from datasets import load_dataset
ds = load_dataset("parquet", data_files="dataset.parquet", split="train")
# ds[0]["audio"] → {"array": np.ndarray, "sampling_rate": 16000, "path": "..."}
# ds[0]["transcription"] → "今年以來的行情"
```
