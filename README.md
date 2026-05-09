# yt-corpus-collect

A tool for collecting voice and text data from YouTube as training material for ASR (Automatic Speech Recognition) and TTS (Text-to-Speech) models.

## Overview

This project downloads audio and transcripts from YouTube videos using [yt-dlp](https://github.com/yt-dlp/yt-dlp), producing aligned speech/text pairs suitable for training speech models.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

_Coming soon._

## Data Pipeline

1. **Download audio** — extract audio tracks from YouTube videos
2. **Download transcripts** — fetch auto-generated or manual subtitles
3. **Align** — pair audio segments with transcript text
4. **Export** — output in a format suitable for ASR/TTS training (e.g., LJSpeech, Common Voice)

## Requirements

- Python 3.10+
- yt-dlp
