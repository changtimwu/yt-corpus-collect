#!/usr/bin/env python3
"""Refresh a playlist's video list, ensure a dir exists per video, and audit
the availability of the three data files (info / audio / subtitle) per video.

This is the re-runnable corpus bookkeeping tool. It does NOT download media —
it only (1) re-enumerates the playlist via yt-dlp --flat-playlist, (2) creates
an empty corpus/<id>/ for any new video, and (3) reports which of the three
data files each video already has on disk.

Usage:
    .venv/bin/python playlist_audit.py [--cookies cookies.txt] [--no-fetch]

    --no-fetch   skip the network enumeration and audit whatever dirs exist
                 (useful when offline or the playlist is unchanged)
    -o CSV       write the per-video availability matrix (default: playlist_audit.csv)
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

PLAYLIST_URL = 'https://www.youtube.com/playlist?list=PLSg6_lakxpXFIWsMoqFIdYtyQb0HSSOVv'
LANG = 'zh-TW'
# yt-dlp writes the playlist-level metadata into a dir named after the playlist
# ID — it is not a video and must be excluded from every count.
PLAYLIST_META_DIR = 'PLSg6_lakxpXFIWsMoqFIdYtyQb0HSSOVv'


def enumerate_playlist(cookies: str | None) -> list[str]:
    """Return the ordered list of video IDs currently in the playlist."""
    cmd = [sys.executable, '-m', 'yt_dlp', '--flat-playlist', '--print', 'id']
    if cookies:
        cmd += ['--cookies', cookies]
    cmd.append(PLAYLIST_URL)
    result = subprocess.run(cmd, capture_output=True, text=True)
    ids = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
    if not ids:
        sys.stderr.write(result.stderr)
        raise SystemExit('Playlist enumeration returned no IDs (auth/cookies?).')
    return ids


def audit_video(video_dir: Path) -> dict:
    """Check which of the three data files exist for one video dir."""
    vid = video_dir.name
    has_info = (video_dir / f'{vid}.info.json').exists()
    has_audio = bool(list(video_dir.glob('*.m4a')))
    has_sub = any(
        p for p in video_dir.glob(f'*.{LANG}.vtt')
        if '.aligned.' not in p.name and '.segmented.' not in p.name
    )
    return {'video_id': vid, 'info': has_info, 'audio': has_audio, 'subtitle': has_sub}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--corpus', default='corpus', help='Corpus directory (default: corpus/)')
    ap.add_argument('--cookies', default='cookies.txt',
                    help='Netscape cookies file for enumeration (default: cookies.txt)')
    ap.add_argument('--no-fetch', action='store_true',
                    help='Skip network enumeration; audit existing dirs only')
    ap.add_argument('-o', '--output', default='playlist_audit.csv',
                    help='Per-video availability CSV (default: playlist_audit.csv)')
    args = ap.parse_args()

    corpus = Path(args.corpus)
    corpus.mkdir(parents=True, exist_ok=True)

    # 1. Refresh playlist + create dirs for new videos.
    if args.no_fetch:
        playlist_ids = None
        created = 0
        print('Skipping playlist fetch (--no-fetch).')
    else:
        cookies = args.cookies if Path(args.cookies).exists() else None
        if not cookies:
            print(f'Warning: {args.cookies} not found — enumerating anonymously '
                  '(YouTube may bot-wall).', file=sys.stderr)
        print('Enumerating playlist...')
        playlist_ids = enumerate_playlist(cookies)
        print(f'Playlist has {len(playlist_ids)} videos.')
        created = 0
        for vid in playlist_ids:
            d = corpus / vid
            if not d.exists():
                d.mkdir()
                created += 1
        print(f'Created {created} new dir(s) for videos not yet on disk.')

    # 2. Audit every video dir on disk (excluding the playlist-meta dir).
    video_dirs = sorted(
        d for d in corpus.iterdir()
        if d.is_dir() and d.name != PLAYLIST_META_DIR
    )
    rows = [audit_video(d) for d in video_dirs]

    # If we have the live playlist, flag any disk dirs no longer in the playlist.
    in_playlist = set(playlist_ids) if playlist_ids else None
    for r in rows:
        r['in_playlist'] = (r['video_id'] in in_playlist) if in_playlist is not None else ''

    # 3. Write per-video matrix.
    out = Path(args.output)
    with out.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['video_id', 'info', 'audio', 'subtitle', 'in_playlist'])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # 4. Summary stats.
    n = len(rows)
    n_info = sum(r['info'] for r in rows)
    n_audio = sum(r['audio'] for r in rows)
    n_sub = sum(r['subtitle'] for r in rows)
    complete = sum(r['info'] and r['audio'] and r['subtitle'] for r in rows)
    audio_no_sub = sum(r['audio'] and not r['subtitle'] for r in rows)
    empty = sum(not (r['info'] or r['audio'] or r['subtitle']) for r in rows)

    def pct(x): return f'{x/n*100:5.1f}%' if n else '   -'

    print(f'\n=== Availability audit ({n} video dirs) ===')
    print(f'  info.json :  {n_info:5d}  ({pct(n_info)})')
    print(f'  audio m4a :  {n_audio:5d}  ({pct(n_audio)})')
    print(f'  {LANG} vtt:  {n_sub:5d}  ({pct(n_sub)})')
    print(f'  -----------------------------------------')
    print(f'  all three :  {complete:5d}  ({pct(complete)})  <- packable')
    print(f'  audio,no sub: {audio_no_sub:4d}  ({pct(audio_no_sub)})  <- needs_whisper')
    print(f'  empty dirs:  {empty:5d}  ({pct(empty)})  <- not downloaded yet')
    if in_playlist is not None:
        stale = sum(1 for r in rows if not r['in_playlist'])
        print(f'  on disk but not in playlist: {stale}')
        not_on_disk = len(in_playlist) - sum(1 for r in rows if r['in_playlist'])
        print(f'  in playlist but no dir:      {not_on_disk}')
    print(f'\nPer-video matrix written to {out}')


if __name__ == '__main__':
    main()
