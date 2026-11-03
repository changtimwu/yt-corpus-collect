# yt-corpus-collect

A tool for collecting voice and text data from YouTube as training material for ASR (Automatic Speech Recognition) and TTS (Text-to-Speech) models.

## Overview

This project downloads audio and transcripts from YouTube playlists using [yt-dlp](https://github.com/yt-dlp/yt-dlp), then packs them into a [HuggingFace-compatible](https://huggingface.co/docs/datasets) parquet dataset — one row per subtitle segment with embedded audio, ready for fine-tuning.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --pre -r requirements.txt
```

Requirements: Python 3.10+, ffmpeg, Node.js ≥ 20 (yt-dlp uses it to solve YouTube's n-sig challenge via the `yt-dlp-ejs` package; `--pre` is required because the YouTube counter-fixes ship as yt-dlp nightlies).

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

**Just one or a few videos?** `fetch_youtube.py` writes the same layout using the `yt-dlp` CLI (`brew install yt-dlp`), with no `.venv` needed. It takes uploader subs when they exist and falls back to auto-captions (`--no-auto-subs` to refuse those). It exits non-zero if a video ends up without audio or subtitles. Bare IDs starting with `-` are fine.

```bash
python3 fetch_youtube.py "https://www.youtube.com/watch?v=_4SUBYKiN9E"
python3 fetch_youtube.py --info-json --lang 'zh-TW,zh-Hant' <url> [<url> ...]
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
| `audio` | struct `{bytes, path}` | Clip embedded as bytes: the source **AAC stream copied into MP4** (`-c:a copy -f mp4`), not decoded. Usually 44.1/48 kHz stereo; resample to 16 kHz mono before training |
| `transcription` | string | Subtitle text for this segment |
| `start` | float32 | Segment start time (seconds) |
| `end` | float32 | Segment end time (seconds) |
| `title` | string | Video title |
| `channel` | string | Channel name |
| `upload_date` | string | Upload date (`YYYYMMDD`) |

#### Inspecting a parquet without loading audio

`audio.bytes` is most of the file weight, so a quick text-only sample needs to project the audio column away to stay fast. `peek.sh` does this — prints `video_id  start-end  transcription` for the first N rows of one file or a glob:

```bash
./peek.sh dataset.parquet               # first 20 rows
./peek.sh dataset.parquet 50            # first 50
./peek.sh 'dataset_part_*.parquet' 100  # first 100 across rolling parts
```

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

---

## Adding more videos later (incremental update)

Both the Gemini segmentation and the parquet packing are fully incremental — adding 100 new videos costs roughly **$0.10 in Gemini + ~20 min of ffmpeg**, not a full re-pack. Two caches do the work:

- **Per-video segmentation cache** — `corpus/<id>/<id>.zh-TW.segments.json` is written once per video; subsequent `pack.py` runs read it and skip Gemini entirely.
- **Rolling parquet resume** — when `--snapshot-every N` is set, `pack.py` calls `scan_completed_parts()` at startup, reads `video_id` from every existing `dataset_part_*.parquet`, and removes those IDs from the work queue. Existing part files stay untouched; new rows go into a fresh `dataset_part_NNN.parquet` starting at the next free part number.

### Workflow

```bash
# 1. Download new IDs
.venv/bin/python download.py --cookies cookies.txt \
  --sleep-interval 4 --max-sleep-interval 10 \
  <NEW_VIDEO_URLS>

# 2. Re-run pack — it auto-detects what's already done
nohup bash -c 'set -a && source gemini-key.env && set +a && \
  .venv/bin/python -u pack.py --parallel 32 --workers 4 --snapshot-every 50 \
  --output dataset.parquet' > pack.log 2>&1 &

# pack.log will say:
#   Resume: NNNN videos already packed in existing parts; continuing with part_NNN
#   Packing K videos → dataset.parquet           ← only the new ones
#   Pre-segment: K uncached videos, 32 concurrent Gemini calls.
```

### Retrying failed downloads

Every long run ends with some failed IDs, and they need opposite handling. Rate limits, bot checks, timeouts and 503s clear up later. Private and removed videos never will. `classify_failures.py` reads the yt-dlp logs (oldest first; the last error per ID wins) and writes the two lists. `--corpus` drops IDs that a later pass did download, because a success leaves no log line to override the earlier error:

```bash
python3 classify_failures.py download.log resume.log --corpus corpus \
  --retry pending_retry.txt --skip permanent_skip.txt
.venv/bin/python download.py --cookies cookies.txt --sleep-interval 4 --max-sleep-interval 10 \
  $(sed 's|^|https://www.youtube.com/watch?v=|' pending_retry.txt)
```

On this corpus's three download passes it reproduced the hand-made lists. Of ~90 failed IDs, 33 were permanent and the rest retryable.

### Caveats

- **If you re-download a VTT** (e.g. YouTube fixed a transcript, or you re-ran `align.py`), the segmentation cache is now stale. Delete the matching `<id>.zh-TW.segments.json` to force re-segmentation. The already-packed rows in `dataset_part_*.parquet` are also frozen against the old VTT — delete the parts containing that `video_id` if you want them rebuilt.
- **Resume is keyed by `video_id`, not by segmentation strategy.** If you change strategies (e.g. flip Gemini → rule-based, or switch to the pair-merge approach from #2), you have to delete the existing parts. The segmentation cache still saves the Gemini cost, so even a full strategy change is cheap on the API side.

---

## Download lessons learned

Collected while downloading this ~1,700-video zh-TW corpus (initial run plus two resume passes). Segmentation, cue de-duplication and timestamp drift are covered under *Pack* and in `align.py`.

### Audio

- **Keep the m4a. Don't transcode at download time.** `-f bestaudio[ext=m4a]` saves
  YouTube's AAC stream byte-for-byte. Convert to WAV later, per consumer: each ASR
  stack wants something slightly different (sample rate, channel count, RIFF quirks).
- **The m4a is usually 44.1/48 kHz stereo**, not 16 kHz mono. Two consequences we
  ran into:
  - Azure batch transcription of a stereo file **emits phrases for both channels**,
    so every phrase shows up twice. Downmix or keep one channel (`-ac 1`) first.
  - `pack.py` stream-copies the AAC into each parquet clip, so the parquet's audio
    is **not** 16 kHz mono WAV either (this README used to say it was). Downstream
    consumers must decode and resample themselves.
- **Never pipe ffmpeg's WAV output** (`-f wav pipe:1`). A pipe isn't seekable, so
  ffmpeg can't go back and fill in the RIFF and `data` chunk sizes; both stay
  `0xFFFFFFFF`. ffprobe and HuggingFace don't mind; Azure Custom Speech rejects
  every clip ("Duration in the wave file's header does not match the file size").
  Always write to a real file path.

### Subtitles

- **Pick the language code YouTube actually serves.** Taiwanese channels' manual
  captions are `zh-TW`, not `zh-Hant`/`zh`. Run `yt-dlp --list-subs <url>` when
  unsure. `--lang` accepts a comma list if channels vary.
- **Manual beats auto.** With `--write-subs --write-auto-subs`, yt-dlp takes the
  uploader's track when one exists for that language and only falls back to ASR
  auto-captions otherwise. Use `--no-auto-subs` if machine captions would contaminate
  a reference set; auto-captions are another ASR system's hypothesis, not ground truth.
- **Expect ~13% of videos to have no subtitles at all.** In our corpus, 1,336 of
  1,545 videos had a `zh-TW` track. `download.py` writes `manifest.csv` with a
  `needs_whisper` flag for those.

### Getting blocked (the part that costs the most time)

- **A JS runtime is mandatory now.** YouTube's n-sig challenge is solved through
  `yt-dlp-ejs`, and yt-dlp only enables **deno** by default. With only node installed
  you get thousands of `No supported JavaScript runtime` warnings and missing
  formats. Pass `--js-runtimes node` (`download.py` and `fetch_youtube.py` both do), or install deno.
- **Stay on recent yt-dlp.** YouTube counter-fixes often ship as nightlies first,
  so install with `pip install --pre`.
  When downloads that used to work start failing, upgrade before you debug. Typical
  symptom: subtitles and info.json download fine but the audio fails with
  **`HTTP Error 403: Forbidden`**. A May 2026 nightly did exactly that by Nov 2026;
  upgrading fixed it. That's why `requirements.txt` sets a minimum version, not an exact pin.
- **Throttle long runs.** Unthrottled playlist runs hit *"The current session has been
  rate-limited by YouTube for up to an hour"*. `--sleep-interval 4
  --max-sleep-interval 10` kept 1,500+ videos going. A rate-limited ID is **not**
  dead: retry it after an hour.
- **"Sign in to confirm you're not a bot"** needs cookies: export a Netscape
  `cookies.txt` from a logged-in browser and pass `--cookies`. **Cookies rotate.**
  A resume run produced 112 *"cookies are no longer valid"* warnings followed by
  53 bot-check failures. Export fresh cookies right before each long run, and use a
  throwaway account. **Never commit `cookies.txt`** (it's in `.gitignore`).
- **Keep retryable and permanent failures apart.** Out of ~90 failed IDs over three
  passes: 33 were private/removed (permanent, skip them forever), and the rest were
  rate limits, bot checks, timeouts or 503s that mostly succeeded on a later pass.
  `classify_failures.py` makes the split from the logs. Note that *"Video
  unavailable. This content isn't available, try again later"* is a **rate limit**,
  even though it starts with "Video unavailable".
- **Ignore `no impersonate target is available`.** That warning only means
  `curl_cffi` isn't installed. Downloads work without it.
- **Log everything** (`nohup … > download.log 2>&1 &`). The log is the only record of
  *why* an ID failed; a success leaves no line, so a later success doesn't overwrite
  an earlier error (`classify_failures.py --corpus` handles that).

### Shell gotchas

- YouTube IDs can start with `-` (e.g. `-AbCdEfGhIj`). Put `--` before bare IDs
  (`fetch_youtube.py` does this internally), and quote them in shell loops.
- Always quote URLs: `&list=…` is a shell background operator.
