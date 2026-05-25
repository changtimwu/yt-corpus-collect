#!/usr/bin/env python3
"""Pack corpus into a HuggingFace-compatible parquet dataset.

Each row is one VTT subtitle segment with its corresponding audio clip
extracted from the m4a, matching the schema of ky552/ML2021_ASR_ST.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from google import genai
from google.genai import types

_GEMINI_MODEL = 'gemini-3.1-flash-lite'

_TS_RE = re.compile(r'(\d+):(\d+):(\d+)[.,](\d+)')
_TAG_RE = re.compile(r'<[^>]+>')

SCHEMA = pa.schema([
    pa.field('video_id', pa.string()),
    pa.field('file', pa.string()),
    pa.field('audio', pa.struct([
        pa.field('bytes', pa.binary()),
        pa.field('path', pa.string()),
    ])),
    pa.field('transcription', pa.string()),
    pa.field('start', pa.float32()),
    pa.field('end', pa.float32()),
    pa.field('title', pa.string()),
    pa.field('channel', pa.string()),
    pa.field('upload_date', pa.string()),
])


def parse_timestamp(ts: str) -> float:
    m = _TS_RE.match(ts.strip())
    if not m:
        return 0.0
    h, mn, s, ms = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    return h * 3600 + mn * 60 + s + ms / 1000


def parse_vtt(vtt_path: Path) -> list[dict]:
    """Parse VTT into deduplicated list of {start, end, text} dicts."""
    segments = []
    lines = vtt_path.read_text(encoding='utf-8').splitlines()

    i = 0
    while i < len(lines):
        if '-->' in lines[i]:
            parts = lines[i].split('-->')
            start = parse_timestamp(parts[0])
            # strip positioning cues after the end timestamp
            end_str = parts[1].split()[0]
            end = parse_timestamp(end_str)
            text_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() and '-->' not in lines[i]:
                cleaned = _TAG_RE.sub('', lines[i]).strip()
                if cleaned:
                    text_lines.append(cleaned)
                i += 1
            text = ' '.join(text_lines)
            if text and end > start:
                segments.append({'start': start, 'end': end, 'text': text})
        else:
            i += 1

    # YouTube VTTs repeat previous lines in each cue for smooth scrolling.
    # Keep only segments whose text wasn't the last seen text.
    deduped = []
    last_text = None
    for seg in segments:
        if seg['text'] != last_text:
            deduped.append(seg)
            last_text = seg['text']

    return deduped


def extract_clip(m4a_path: Path, start: float, end: float) -> bytes:
    """Return m4a bytes for the given time range using AAC stream copy (no decode)."""
    cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-ss', str(start),
        '-i', str(m4a_path),
        '-t', str(end - start),
        '-c:a', 'copy',
        '-f', 'mp4', '-movflags', 'frag_keyframe+empty_moov',
        'pipe:1',
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.stdout if result.returncode == 0 else b''


def segment_with_gemini(cues: list[dict], client: genai.Client) -> list[dict]:
    """Use Gemini to group cues into natural sentences. Returns same {start,end,text} format."""
    numbered = '\n'.join(f'[{i}] {c["text"]}' for i, c in enumerate(cues))
    prompt = f"""\
以下是台灣中文影片的字幕，每行格式為「[編號] 文字」。
請將連續的字幕單元合併成語義完整的自然句子，不要在句子中間切斷。

輸出：JSON 陣列，每個元素是一組連續字幕編號（整數陣列），代表同一句子。
只輸出 JSON，不要其他說明文字。範例：[[0,1,2],[3,4],[5,6,7]]

字幕：
{numbered}"""

    response = client.models.generate_content(
        model=_GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type='application/json'),
    )
    text = re.sub(r'^```[a-z]*\n?|\n?```$', '', response.text.strip())
    groups = json.loads(text)

    n = len(cues)
    groups = [[i for i in g if 0 <= i < n] for g in groups]
    groups = [g for g in groups if g]

    return [
        {
            'start': cues[g[0]]['start'],
            'end': cues[g[-1]]['end'],
            'text': ''.join(cues[i]['text'] for i in g),
        }
        for g in groups
    ]


def merge_segments(
    segments: list[dict],
    target: float = 10.0,
    max_dur: float = 20.0,
    min_dur: float = 1.5,
) -> list[dict]:
    """Merge consecutive short VTT cues into longer, self-contained segments.

    YouTube CC cues average ~1.7s each with no gaps or punctuation, making
    them too short for ASR training. Accumulate until target duration is
    reached, then emit as one merged segment.
    """
    merged = []
    buf_texts: list[str] = []
    buf_start: float = 0.0
    buf_end: float = 0.0

    for seg in segments:
        if not buf_texts:
            buf_start = seg['start']
        buf_end = seg['end']
        buf_texts.append(seg['text'])

        if buf_end - buf_start >= target or buf_end - buf_start >= max_dur:
            merged.append({'start': buf_start, 'end': buf_end, 'text': ' '.join(buf_texts)})
            buf_texts = []

    # flush remainder
    if buf_texts and buf_end - buf_start >= min_dur:
        merged.append({'start': buf_start, 'end': buf_end, 'text': ' '.join(buf_texts)})

    return merged


def pack_video(video_dir: Path, writer: pq.ParquetWriter,
               gemini_client: genai.Client | None = None) -> int:
    """Pack one video dir into the writer. Returns number of segments written."""
    video_id = video_dir.name

    m4a_files = list(video_dir.glob('*.m4a'))
    if not m4a_files:
        return 0

    # Prefer aligned VTT (from align.py) over the raw YouTube one when present.
    # Skip Gemini-segmented VTTs here — segmentation happens in-process below.
    aligned_vtts = list(video_dir.glob('*.aligned.vtt'))
    if aligned_vtts:
        vtt_files = aligned_vtts
    else:
        vtt_files = [p for p in video_dir.glob('*.vtt')
                     if '.aligned' not in p.name and '.segmented' not in p.name]
    if not vtt_files:
        return 0

    info_path = video_dir / f'{video_id}.info.json'
    title = channel = upload_date = ''
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding='utf-8'))
        title = info.get('title', '')
        channel = info.get('channel', '')
        upload_date = info.get('upload_date', '')

    cues = parse_vtt(vtt_files[0])
    if gemini_client is not None:
        try:
            segments = segment_with_gemini(cues, gemini_client)
        except Exception as e:
            print(f'    Gemini error ({e}), falling back to merge_segments', file=sys.stderr)
            segments = merge_segments(cues)
    else:
        segments = merge_segments(cues)
    if not segments:
        return 0

    m4a_path = m4a_files[0]
    BATCH_SIZE = 20  # write every N segments to keep memory bounded
    buf: dict = {k: [] for k in ('video_id', 'file', 'audio', 'transcription',
                                  'start', 'end', 'title', 'channel', 'upload_date')}
    written = 0

    def flush(buf: dict) -> int:
        if not buf['video_id']:
            return 0
        batch = pa.record_batch({
            'video_id': pa.array(buf['video_id'], pa.string()),
            'file': pa.array(buf['file'], pa.string()),
            'audio': pa.array(buf['audio'], pa.struct([
                pa.field('bytes', pa.binary()),
                pa.field('path', pa.string()),
            ])),
            'transcription': pa.array(buf['transcription'], pa.string()),
            'start': pa.array(buf['start'], pa.float32()),
            'end': pa.array(buf['end'], pa.float32()),
            'title': pa.array(buf['title'], pa.string()),
            'channel': pa.array(buf['channel'], pa.string()),
            'upload_date': pa.array(buf['upload_date'], pa.string()),
        }, schema=SCHEMA)
        writer.write_batch(batch)
        n = len(buf['video_id'])
        for v in buf.values():
            v.clear()
        return n

    for idx, seg in enumerate(segments):
        wav = extract_clip(m4a_path, seg['start'], seg['end'])
        if not wav:
            continue
        filename = f'{video_id}_{idx+1:04d}.m4a'
        buf['video_id'].append(video_id)
        buf['file'].append(filename)
        buf['audio'].append({'bytes': wav, 'path': filename})
        buf['transcription'].append(seg['text'])
        buf['start'].append(seg['start'])
        buf['end'].append(seg['end'])
        buf['title'].append(title)
        buf['channel'].append(channel)
        buf['upload_date'].append(upload_date)
        if len(buf['video_id']) >= BATCH_SIZE:
            written += flush(buf)

    written += flush(buf)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Pack downloaded corpus into a HuggingFace-compatible parquet dataset'
    )
    parser.add_argument('--corpus', default='corpus', help='Corpus directory (default: corpus/)')
    parser.add_argument('--output', default='dataset.parquet', help='Output parquet file (default: dataset.parquet)')
    parser.add_argument('--max-videos', type=int, default=None, metavar='N', help='Stop after N videos (useful for testing)')
    args = parser.parse_args()

    corpus_dir = Path(args.corpus)
    video_dirs = sorted(
        d for d in corpus_dir.iterdir()
        if d.is_dir() and list(d.glob('*.m4a'))
    )

    if not video_dirs:
        print(f'No videos found in {corpus_dir}')
        sys.exit(1)

    if args.max_videos:
        video_dirs = video_dirs[:args.max_videos]

    api_key = os.environ.get('GEMINI_API_KEY')
    if api_key:
        gemini_client = genai.Client(api_key=api_key)
        print(f'Gemini segmentation enabled ({_GEMINI_MODEL})')
    else:
        gemini_client = None
        print('Warning: GEMINI_API_KEY not set — using rule-based merge_segments() fallback.',
              file=sys.stderr)

    print(f'Packing {len(video_dirs)} videos → {args.output}')

    total_segments = 0
    with pq.ParquetWriter(args.output, SCHEMA, compression='snappy') as writer:
        for i, video_dir in enumerate(video_dirs, 1):
            n = pack_video(video_dir, writer, gemini_client)
            total_segments += n
            print(f'  [{i}/{len(video_dirs)}] {video_dir.name}: {n} segments', flush=True)

    print(f'\nDone. {total_segments} segments written to {args.output}')


if __name__ == '__main__':
    main()
