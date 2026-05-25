# yt-corpus-collect

A tool for collecting voice and text data from YouTube as training material for ASR (Automatic Speech Recognition) and TTS (Text-to-Speech) models.

## Overview

This project downloads audio and transcripts from YouTube playlists using [yt-dlp](https://github.com/yt-dlp/yt-dlp), then packs them into a [HuggingFace-compatible](https://huggingface.co/docs/datasets) parquet dataset — one row per subtitle segment with embedded audio, ready for fine-tuning.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requirements: Python 3.10+, ffmpeg

## Workflow

### Step 1 — Download

```bash
python download.py [OPTIONS] URL [URL ...]
```

Downloads audio (m4a), subtitles (VTT), and metadata (info.json) for every video in a YouTube playlist into `corpus/<video_id>/`.

| Option | Default | Description |
|---|---|---|
| `-o, --output` | `corpus/` | Output directory |
| `--lang` | `zh-TW` | Subtitle language code |
| `--audio-format` | `m4a` | Audio format (`m4a` keeps the original stream; `wav`/`flac`/`mp3` re-encodes to 16 kHz mono) |
| `--max-videos N` | all | Stop after N videos (useful for testing) |
| `--no-subs` | off | Skip subtitle download |

After downloading, a `corpus/manifest.csv` is written listing every video with its audio path, transcript path, and a `needs_whisper` flag for videos with no subtitle.

**Example:**

```bash
# Download first 50 videos from a playlist
python download.py --max-videos 50 "https://www.youtube.com/playlist?list=..."

# Download everything
python download.py "https://www.youtube.com/playlist?list=..."
```

Corpus layout after download:

```
corpus/
├── manifest.csv
└── <video_id>/
    ├── <video_id>.m4a          # audio (original AAC stream)
    ├── <video_id>.zh-TW.vtt   # subtitles
    └── <video_id>.info.json   # metadata (title, description, channel, upload date, …)
```

---

### Step 2 — Pack

```bash
python pack.py [OPTIONS]
```

Reads the corpus directory, slices each subtitle segment out of the m4a using ffmpeg, and writes a parquet file. Schema matches [ky552/ML2021_ASR_ST](https://huggingface.co/datasets/ky552/ML2021_ASR_ST).

YouTube CC cues are ~1.7 s each, with no punctuation and breaks placed wherever the on-screen text scrolls — too short and too mid-thought for ASR training. Before slicing, `pack.py` runs an **LLM-assisted re-segmentation** step that groups the raw cues into sentence-complete units, and emits one parquet row per group. If `GEMINI_API_KEY` is set, Gemini (`gemini-3.1-flash-lite`) does the grouping by semantic sentence boundaries; otherwise a rule-based fallback (`merge_segments()`) accumulates cues to ~10 s targets.

| Option | Default | Description |
|---|---|---|
| `--corpus` | `corpus/` | Corpus directory |
| `--output` | `dataset.parquet` | Output parquet file |
| `--max-videos N` | all | Pack only N videos (useful for testing) |

**Example:**

```bash
# Test with 5 videos
python pack.py --max-videos 5 --output test.parquet

# Full pack
python pack.py --output dataset.parquet
```

Parquet schema:

| Column | Type | Description |
|---|---|---|
| `video_id` | string | YouTube video ID |
| `file` | string | Segment filename (e.g. `abc123_0001.wav`) |
| `audio` | struct `{bytes, path}` | 16 kHz mono WAV clip embedded as bytes |
| `transcription` | string | Subtitle text for this segment |
| `start` | float32 | Segment start time (seconds) |
| `end` | float32 | Segment end time (seconds) |
| `title` | string | Video title |
| `channel` | string | Channel name |
| `upload_date` | string | Upload date (`YYYYMMDD`) |

---

### Step 3 — Fine-tune

Transfer `dataset.parquet` to the machine where fine-tuning runs:

```bash
# Copy to a remote training server
scp dataset.parquet user@train-server:/data/

# Or push to HuggingFace Hub
huggingface-cli upload <your-org>/<dataset-name> dataset.parquet
```

Load in Python with the HuggingFace `datasets` library:

```python
from datasets import load_dataset

ds = load_dataset("parquet", data_files="dataset.parquet", split="train")
# ds[0]["audio"]          → {"array": np.ndarray, "sampling_rate": 16000, "path": "..."}
# ds[0]["transcription"]  → "今年以來的行情"
```

The `audio` column is automatically decoded to a numpy array at 16 kHz by the `datasets` library, compatible with Whisper, wav2vec2, and most other ASR/TTS fine-tuning frameworks.
