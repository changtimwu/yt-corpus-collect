#!/usr/bin/env python3
"""Re-time VTT subtitles against real audio using WhisperX forced alignment.

YouTube VTT timestamps drift ~0.5-1s from the actual audio, which produces
miscut training clips in pack.py. This stage runs a wav2vec2 alignment model
over each m4a using the existing VTT text as the reference transcript, then
emits a sibling `<video_id>.zh-TW.aligned.vtt` with corrected per-cue start/end
derived from word-level alignments. pack.py automatically prefers the aligned
file when present.

Reference: https://github.com/changtimwu/yt-corpus-collect/issues/1
"""

import argparse
import re
import sys
from pathlib import Path

# pack.py already implements VTT parsing — reuse it instead of duplicating.
from pack import parse_vtt

_VTT_LANG_RE = re.compile(r'\.([a-zA-Z-]+)\.vtt$')


def fmt_ts(t: float) -> str:
    if t < 0:
        t = 0.0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t - h * 3600 - m * 60
    return f'{h:02d}:{m:02d}:{s:06.3f}'


def write_vtt(cues: list[dict], out_path: Path) -> None:
    lines = ['WEBVTT', '']
    for c in cues:
        lines.append(f'{fmt_ts(c["start"])} --> {fmt_ts(c["end"])}')
        lines.append(c['text'])
        lines.append('')
    out_path.write_text('\n'.join(lines), encoding='utf-8')


def align_video(
    video_dir: Path,
    align_model,
    align_metadata,
    device: str,
    min_score: float,
) -> tuple[int, int]:
    """Align one video's VTT against its audio. Returns (n_cues_in, n_cues_out)."""
    import whisperx

    video_id = video_dir.name
    m4a_files = list(video_dir.glob('*.m4a'))
    if not m4a_files:
        return 0, 0

    # Pick the raw VTT (skip already-aligned and Gemini-segmented variants).
    vtt_candidates = [
        p for p in video_dir.glob('*.vtt')
        if '.aligned' not in p.name and '.segmented' not in p.name
    ]
    if not vtt_candidates:
        return 0, 0
    vtt_path = vtt_candidates[0]

    lang_match = _VTT_LANG_RE.search(vtt_path.name)
    lang_tag = lang_match.group(1) if lang_match else 'zh-TW'
    out_path = video_dir / f'{video_id}.{lang_tag}.aligned.vtt'

    cues = parse_vtt(vtt_path)
    if not cues:
        return 0, 0

    audio = whisperx.load_audio(str(m4a_files[0]))

    # WhisperX align() expects each segment to carry start/end/text; it returns
    # the same segments enriched with `words` (each with start/end/score).
    segments_in = [{'start': c['start'], 'end': c['end'], 'text': c['text']} for c in cues]
    result = whisperx.align(
        segments_in,
        align_model,
        align_metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    aligned: list[dict] = []
    for orig, seg in zip(cues, result['segments']):
        words = seg.get('words') or []
        word_times = [(w.get('start'), w.get('end'), w.get('score', 1.0))
                      for w in words
                      if w.get('start') is not None and w.get('end') is not None]
        if not word_times:
            # Alignment produced no usable words — fall back to original timing.
            aligned.append(orig)
            continue
        starts = [t[0] for t in word_times]
        ends = [t[1] for t in word_times]
        scores = [t[2] for t in word_times]
        mean_score = sum(scores) / len(scores)
        if mean_score < min_score:
            # Low confidence — likely OOV (music, ads, mistranscription). Drop.
            continue
        aligned.append({
            'start': min(starts),
            'end': max(ends),
            'text': orig['text'],
        })

    write_vtt(aligned, out_path)
    return len(cues), len(aligned)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Forced-align VTT timestamps against audio with WhisperX.'
    )
    parser.add_argument('--corpus', default='corpus', help='Corpus directory (default: corpus/)')
    parser.add_argument('--language', default='zh',
                        help='wav2vec2 language code (default: zh — covers both zh-CN and zh-TW)')
    parser.add_argument('--device', default=None,
                        help='cuda or cpu. Default: cuda if available, else cpu.')
    parser.add_argument('--min-score', type=float, default=0.2,
                        help='Drop cues whose mean word-alignment score is below this. '
                             'Default 0.2 (catches ads/music/mistranscriptions; keep low — '
                             'CTC scores for Mandarin are typically 0.3–0.7).')
    parser.add_argument('--max-videos', type=int, default=None, metavar='N',
                        help='Stop after N videos (useful for testing)')
    parser.add_argument('--force', action='store_true',
                        help='Re-align even when *.aligned.vtt already exists')
    args = parser.parse_args()

    try:
        import torch
        import whisperx
    except ImportError as e:
        print(f'error: {e}. Install with: .venv/bin/pip install whisperx', file=sys.stderr)
        sys.exit(2)

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Loading wav2vec2 alignment model (lang={args.language}, device={device})...')
    align_model, metadata = whisperx.load_align_model(language_code=args.language, device=device)

    corpus_dir = Path(args.corpus)
    video_dirs = sorted(
        d for d in corpus_dir.iterdir()
        if d.is_dir() and list(d.glob('*.m4a'))
    )
    if args.max_videos:
        video_dirs = video_dirs[:args.max_videos]

    if not video_dirs:
        print(f'No videos found in {corpus_dir}')
        sys.exit(1)

    total_in = total_out = skipped = 0
    for i, video_dir in enumerate(video_dirs, 1):
        existing = list(video_dir.glob('*.aligned.vtt'))
        if existing and not args.force:
            print(f'  [{i}/{len(video_dirs)}] {video_dir.name}: skip (already aligned)', flush=True)
            skipped += 1
            continue
        try:
            n_in, n_out = align_video(video_dir, align_model, metadata, device, args.min_score)
        except Exception as e:
            print(f'  [{i}/{len(video_dirs)}] {video_dir.name}: error — {e}', file=sys.stderr, flush=True)
            continue
        total_in += n_in
        total_out += n_out
        dropped = n_in - n_out
        print(f'  [{i}/{len(video_dirs)}] {video_dir.name}: {n_out}/{n_in} cues aligned'
              + (f' ({dropped} dropped below min-score)' if dropped else ''), flush=True)

    print(f'\nDone. {total_out}/{total_in} cues aligned across '
          f'{len(video_dirs) - skipped} videos ({skipped} skipped).')


if __name__ == '__main__':
    main()
