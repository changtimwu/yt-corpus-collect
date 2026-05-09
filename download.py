#!/usr/bin/env python3
"""Download YouTube audio and transcripts for ASR/TTS corpus collection."""

import argparse
import csv
import sys
from pathlib import Path

import yt_dlp


def build_opts(output_dir: Path, audio_format: str, lang: str, no_subs: bool) -> dict:
    outtmpl = str(output_dir / "%(id)s" / "%(id)s.%(ext)s")
    postprocessors = [
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": audio_format,
        },
    ]
    # Resample to 16 kHz mono — standard for ASR/TTS training
    if audio_format in ("wav", "flac"):
        postprocessors.append({
            "key": "FFmpegPostProcessor",
            "preferedformat": audio_format,
        })

    return {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": outtmpl,
        "postprocessors": postprocessors,
        "postprocessor_args": {
            "ffmpeg": ["-ar", "16000", "-ac", "1"],
        },
        "writesubtitles": not no_subs,
        "writeautomaticsub": not no_subs,
        "subtitleslangs": [lang],
        "subtitlesformat": "vtt",
        "ignoreerrors": True,
    }


def write_manifest(output_dir: Path, audio_format: str) -> None:
    """Scan output_dir, write manifest.csv, and report missing transcripts."""
    rows = []
    for video_dir in sorted(output_dir.iterdir()):
        if not video_dir.is_dir():
            continue
        audio_file = video_dir / f"{video_dir.name}.{audio_format}"
        if not audio_file.exists():
            continue
        # glob for any .vtt — yt-dlp may append lang variants like .en.vtt or .en-US.vtt
        vtt_files = list(video_dir.glob("*.vtt"))
        vtt_file = vtt_files[0] if vtt_files else None
        rows.append({
            "video_id": video_dir.name,
            "audio": str(audio_file),
            "transcript": str(vtt_file) if vtt_file else "",
            "needs_whisper": vtt_file is None,
        })

    manifest_path = output_dir / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["video_id", "audio", "transcript", "needs_whisper"]
        )
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    with_subs = sum(1 for r in rows if not r["needs_whisper"])
    need_whisper = total - with_subs

    print(f"\nManifest: {manifest_path}")
    print(f"  {total} videos  |  {with_subs} with transcripts  |  {need_whisper} need Whisper")
    if need_whisper:
        print("\nVideos missing transcripts:")
        for r in rows:
            if r["needs_whisper"]:
                print(f"  {r['video_id']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download YouTube audio + transcripts for ASR/TTS corpus collection"
    )
    parser.add_argument("urls", nargs="+", help="YouTube video or playlist URLs")
    parser.add_argument(
        "-o", "--output", default="corpus",
        help="Output directory (default: corpus/)",
    )
    parser.add_argument(
        "--audio-format", default="wav", choices=["wav", "flac", "mp3"],
        help="Audio format (default: wav)",
    )
    parser.add_argument(
        "--lang", default="en",
        help="Subtitle language code (default: en)",
    )
    parser.add_argument(
        "--no-subs", action="store_true",
        help="Skip subtitle/transcript download",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    opts = build_opts(output_dir, args.audio_format, args.lang, args.no_subs)

    with yt_dlp.YoutubeDL(opts) as ydl:
        ret = ydl.download(args.urls)

    if not args.no_subs:
        write_manifest(output_dir, args.audio_format)

    sys.exit(ret)


if __name__ == "__main__":
    main()
