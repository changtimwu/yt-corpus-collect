#!/usr/bin/env python3
"""Split the failed video IDs in a yt-dlp log into retry-later vs give-up lists.

    python3 classify_failures.py download.log [resume.log ...] \
        --retry pending_retry.txt --skip permanent_skip.txt

A large run always ends with some failures, and the two kinds need opposite
handling: rate limits, bot checks and timeouts clear up if you wait (and/or
refresh cookies), while private and removed videos never will. The last error
seen for an ID wins, so pass logs oldest-first; add --corpus to drop IDs a
later run did download (a success leaves no line in the log to override the error).
"""

import argparse
import re
import sys
from pathlib import Path

ERR_RE = re.compile(r"^ERROR: \[youtube\] ([A-Za-z0-9_-]{11}): (.*)$")

PERMANENT = (
    "Private video",
    "This video is private",
    "removed by the uploader",
    "account associated with this video has been terminated",
    "Video unavailable. This video is no longer available",
    "copyright claim",
)
RETRYABLE = (
    "Sign in to confirm you",              # bot check → fresh cookies + slower pace
    "rate-limited by YouTube",             # session throttled ~1h
    "The current session has been",
    "try again later",
    "timed out",
    "HTTP Error 5",
)


def classify(msg: str) -> str:
    # "Video unavailable. This content isn't available, try again later" is a rate
    # limit, not a dead video — so check retryable patterns first.
    if any(p in msg for p in RETRYABLE):
        return "retry"
    if any(p in msg for p in PERMANENT):
        return "skip"
    return "unknown"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("logs", nargs="+", type=Path, help="yt-dlp logs, oldest first")
    p.add_argument("--retry", type=Path, help="write retry-later IDs here")
    p.add_argument("--skip", type=Path, help="write permanent-failure IDs here")
    p.add_argument("--corpus", type=Path,
                   help="drop IDs that already have <corpus>/<id>/<id>.m4a (a later run succeeded)")
    args = p.parse_args()

    last: dict[str, tuple[str, str]] = {}
    for log in args.logs:
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            m = ERR_RE.match(line)
            if m:
                last[m.group(1)] = (classify(m.group(2)), m.group(2))

    if args.corpus:
        done = [v for v in last if (args.corpus / v / f"{v}.m4a").exists()]
        for v in done:
            del last[v]
        print(f"{len(done)} failed IDs were downloaded later; dropped", file=sys.stderr)

    buckets: dict[str, list[str]] = {"retry": [], "skip": [], "unknown": []}
    for vid, (kind, _) in sorted(last.items()):
        buckets[kind].append(vid)

    for kind, path in (("retry", args.retry), ("skip", args.skip)):
        if path:
            path.write_text("".join(f"{v}\n" for v in buckets[kind]))
    print(f"retry={len(buckets['retry'])} skip={len(buckets['skip'])} "
          f"unknown={len(buckets['unknown'])}", file=sys.stderr)
    for vid in buckets["unknown"]:
        print(f"  unknown {vid}: {last[vid][1][:120]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
