#!/usr/bin/env python3
"""Download one (or more) YouTube videos' audio track as .m4a + subtitles as .vtt.

Writes the same corpus/ layout as download.py, so pack.py / align.py pick it up:

    <output>/<video_id>/<video_id>.m4a            original AAC stream, no re-encode
    <output>/<video_id>/<video_id>.<lang>.vtt     uploader subs, else YouTube auto-subs
    <output>/<video_id>/<video_id>.info.json      (with --info-json)

Uses the `yt-dlp` CLI (e.g. Homebrew) rather than the Python module, so it runs
with any python3 — no .venv needed. YouTube's n-sig challenge is solved with node.

    python3 fetch_youtube.py https://www.youtube.com/watch?v=_4SUBYKiN9E
    python3 fetch_youtube.py <url> -o corpus --lang zh-TW --info-json
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def ytdlp_cmd(url: str, out: Path, lang: str, auto_subs: bool,
              info_json: bool, cookies: str | None) -> list[str]:
    cmd = [
        "yt-dlp",
        "--no-playlist",
        # m4a = YouTube's AAC audio-only stream, saved as-is (no transcoding).
        "-f", "bestaudio[ext=m4a]",
        "-o", str(out / "%(id)s" / "%(id)s.%(ext)s"),
        "--write-subs", "--sub-langs", lang, "--sub-format", "vtt",
        "--js-runtimes", "node",
        "--print", "after_move:id",
    ]
    if auto_subs:
        # Manual subs still win when both exist for the same language.
        cmd.append("--write-auto-subs")
    if info_json:
        cmd.append("--write-info-json")
    if cookies:
        cmd += ["--cookies", cookies]
    # "--" so bare IDs starting with "-" (common on YouTube) aren't read as flags.
    return cmd + ["--", url]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("urls", nargs="+", help="YouTube video URL(s) or ID(s)")
    p.add_argument("-o", "--output", type=Path, default=Path("corpus"),
                   help="output root; each video gets its own <id>/ subdir (default: corpus)")
    p.add_argument("--lang", default="zh-TW",
                   help="subtitle language(s), yt-dlp syntax e.g. 'zh-TW,zh-Hant' (default: zh-TW)")
    p.add_argument("--no-auto-subs", action="store_true",
                   help="only accept uploader-provided subs, never YouTube's auto-generated ones")
    p.add_argument("--info-json", action="store_true", help="also write <id>.info.json")
    p.add_argument("--cookies", metavar="PATH",
                   help="Netscape cookies file (when YouTube triggers its bot check)")
    args = p.parse_args()

    if not shutil.which("yt-dlp"):
        sys.exit("yt-dlp not found on PATH (brew install yt-dlp)")

    failed = 0
    for url in args.urls:
        r = subprocess.run(
            ytdlp_cmd(url, args.output, args.lang, not args.no_auto_subs,
                      args.info_json, args.cookies),
            stdout=subprocess.PIPE, text=True)
        vid = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
        if r.returncode != 0 or not vid:
            print(f"FAIL {url} (yt-dlp exit {r.returncode})", file=sys.stderr)
            failed += 1
            continue
        d = args.output / vid
        m4a = d / f"{vid}.m4a"
        vtts = sorted(d.glob(f"{vid}.*.vtt"))
        print(f"{vid}: {m4a if m4a.exists() else 'NO AUDIO'}")
        for v in vtts:
            print(f"{vid}: {v}")
        if not m4a.exists():
            failed += 1
        if not vtts:
            print(f"{vid}: WARNING no '{args.lang}' subtitles available", file=sys.stderr)
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
