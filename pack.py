#!/usr/bin/env python3
"""Pack corpus into a HuggingFace-compatible parquet dataset.

Each row is one VTT subtitle segment with its corresponding audio clip
extracted from the m4a, matching the schema of ky552/ML2021_ASR_ST.
"""

import argparse
import concurrent.futures
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


def presegment_parallel(video_dirs: list[Path], client: genai.Client, workers: int) -> None:
    """Fill the segment cache for every uncached video using a thread pool.

    Each worker runs segment_with_gemini for one video. The sequential pack
    loop afterwards reads the cache files and skips Gemini entirely.
    """
    todo: list[tuple[str, Path, Path]] = []
    for d in video_dirs:
        video_id = d.name
        cache = d / f'{video_id}.zh-TW.segments.json'
        if cache.exists():
            continue
        aligned = list(d.glob('*.aligned.vtt'))
        if aligned:
            vtt = aligned[0]
        else:
            vtts = [p for p in d.glob('*.vtt')
                    if '.aligned' not in p.name and '.segmented' not in p.name]
            if not vtts:
                continue
            vtt = vtts[0]
        todo.append((video_id, vtt, cache))

    if not todo:
        print('Pre-segment: nothing to do (all videos already cached).')
        return

    print(f'Pre-segment: {len(todo)} uncached videos, {workers} concurrent Gemini calls.')

    def work(item: tuple[str, Path, Path]) -> tuple[str, int, str | None]:
        video_id, vtt_path, cache = item
        try:
            cues = parse_vtt(vtt_path)
            segs = segment_with_gemini(cues, client)
            if segs:
                cache.write_text(json.dumps(segs, ensure_ascii=False), encoding='utf-8')
            return video_id, len(segs), None
        except Exception as e:
            return video_id, 0, str(e)

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for vid, n, err in ex.map(work, todo):
            done += 1
            if err:
                print(f'  pre-segment [{done}/{len(todo)}] {vid}: ERROR {err}', flush=True)
            else:
                print(f'  pre-segment [{done}/{len(todo)}] {vid}: {n} segments', flush=True)

    print('Pre-segment phase complete.\n')


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


def extract_records(video_dir: Path, gemini_client: genai.Client | None = None) -> list[dict]:
    """Build the parquet records for one video. Thread-safe — does no writing."""
    video_id = video_dir.name

    m4a_files = list(video_dir.glob('*.m4a'))
    if not m4a_files:
        return []

    # Prefer aligned VTT (from align.py) over the raw YouTube one when present.
    # Skip Gemini-segmented VTTs here — segmentation happens in-process below.
    aligned_vtts = list(video_dir.glob('*.aligned.vtt'))
    if aligned_vtts:
        vtt_files = aligned_vtts
    else:
        vtt_files = [p for p in video_dir.glob('*.vtt')
                     if '.aligned' not in p.name and '.segmented' not in p.name]
    if not vtt_files:
        return []

    info_path = video_dir / f'{video_id}.info.json'
    title = channel = upload_date = ''
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding='utf-8'))
        title = info.get('title', '')
        channel = info.get('channel', '')
        upload_date = info.get('upload_date', '')

    cues = parse_vtt(vtt_files[0])
    # Cache segmented output next to the VTT so re-packs skip the Gemini round-trip.
    # Delete this file if the source VTT is re-generated and you want fresh segmentation.
    seg_cache = video_dir / f'{video_id}.zh-TW.segments.json'
    if seg_cache.exists():
        segments = json.loads(seg_cache.read_text(encoding='utf-8'))
    else:
        if gemini_client is not None:
            try:
                segments = segment_with_gemini(cues, gemini_client)
            except Exception as e:
                print(f'    Gemini error ({e}), falling back to merge_segments', file=sys.stderr)
                segments = merge_segments(cues)
        else:
            segments = merge_segments(cues)
        if segments:
            seg_cache.write_text(json.dumps(segments, ensure_ascii=False), encoding='utf-8')
    if not segments:
        return []

    m4a_path = m4a_files[0]
    records: list[dict] = []
    for idx, seg in enumerate(segments):
        wav = extract_clip(m4a_path, seg['start'], seg['end'])
        if not wav:
            continue
        filename = f'{video_id}_{idx+1:04d}.m4a'
        records.append({
            'video_id': video_id,
            'file': filename,
            'audio': {'bytes': wav, 'path': filename},
            'transcription': seg['text'],
            'start': seg['start'],
            'end': seg['end'],
            'title': title,
            'channel': channel,
            'upload_date': upload_date,
        })
    return records


def scan_completed_parts(output_path: Path) -> tuple[set[str], int]:
    """Look for existing rolling parts and return (already-packed video IDs, next part number).

    Corrupt parts (e.g. from a kill mid-write) are deleted on the spot — their
    videos will be re-packed into a fresh part. Only invoked when --snapshot-every
    is set, so the single-file path is unaffected.
    """
    completed: set[str] = set()
    next_part = 1
    pattern = f'{output_path.stem}_part_*{output_path.suffix}'
    for p in sorted(output_path.parent.glob(pattern)):
        try:
            pf = pq.ParquetFile(p)
            ids = pf.read(columns=['video_id'])['video_id'].to_pylist()
            completed.update(ids)
            num = int(p.stem.rsplit('_', 1)[-1])
            next_part = max(next_part, num + 1)
        except Exception as e:
            print(f'Resume: deleting corrupt part {p.name} ({type(e).__name__})',
                  file=sys.stderr)
            p.unlink()
    return completed, next_part


class RollingWriter:
    """ParquetWriter that rotates to a new file every `snapshot_every` videos.

    Bounds the blast radius if the process is killed mid-run (system reboot,
    OOM, timeout): each closed part file has a valid footer and is readable
    independently. Set snapshot_every=0 to write one monolithic file.
    """

    def __init__(self, output_path: Path, snapshot_every: int, schema: pa.Schema,
                 start_part: int = 1):
        self.output_path = output_path
        self.snapshot_every = snapshot_every
        self.schema = schema
        self.part = start_part
        self.videos_in_part = 0
        self.total_segments = 0
        self.pending: list[dict] = []
        self._open()

    def _part_path(self) -> Path:
        if not self.snapshot_every:
            return self.output_path
        return self.output_path.with_name(
            f'{self.output_path.stem}_part_{self.part:03d}{self.output_path.suffix}'
        )

    def _open(self) -> None:
        self.writer = pq.ParquetWriter(self._part_path(), self.schema, compression='snappy')

    def _flush(self) -> None:
        if self.pending:
            write_records(self.writer, self.pending)
            self.total_segments += len(self.pending)
            self.pending.clear()

    def add(self, records: list[dict]) -> None:
        self.pending.extend(records)
        if len(self.pending) >= 200:
            self._flush()
        self.videos_in_part += 1
        if self.snapshot_every and self.videos_in_part >= self.snapshot_every:
            self._roll()

    def _roll(self) -> None:
        self._flush()
        path = self._part_path()
        self.writer.close()
        print(f'  → snapshot closed: {path.name}', flush=True)
        self.part += 1
        self.videos_in_part = 0
        self._open()

    def close(self) -> None:
        self._flush()
        self.writer.close()


def write_records(writer: pq.ParquetWriter, records: list[dict]) -> None:
    """Write a list of record dicts to the parquet writer as one batch."""
    if not records:
        return
    cols = {k: [r[k] for r in records] for k in
            ('video_id', 'file', 'audio', 'transcription', 'start', 'end',
             'title', 'channel', 'upload_date')}
    batch = pa.record_batch({
        'video_id': pa.array(cols['video_id'], pa.string()),
        'file': pa.array(cols['file'], pa.string()),
        'audio': pa.array(cols['audio'], pa.struct([
            pa.field('bytes', pa.binary()),
            pa.field('path', pa.string()),
        ])),
        'transcription': pa.array(cols['transcription'], pa.string()),
        'start': pa.array(cols['start'], pa.float32()),
        'end': pa.array(cols['end'], pa.float32()),
        'title': pa.array(cols['title'], pa.string()),
        'channel': pa.array(cols['channel'], pa.string()),
        'upload_date': pa.array(cols['upload_date'], pa.string()),
    }, schema=SCHEMA)
    writer.write_batch(batch)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Pack downloaded corpus into a HuggingFace-compatible parquet dataset'
    )
    parser.add_argument('--corpus', default='corpus', help='Corpus directory (default: corpus/)')
    parser.add_argument('--output', default='dataset.parquet', help='Output parquet file (default: dataset.parquet)')
    parser.add_argument('--max-videos', type=int, default=None, metavar='N', help='Stop after N videos (useful for testing)')
    parser.add_argument('--parallel', type=int, default=1, metavar='N',
                        help='Pre-segment uncached videos with N concurrent Gemini calls before the sequential pack pass (default: 1, no parallelism)')
    parser.add_argument('--workers', type=int, default=1, metavar='N',
                        help='Extract audio + build records with N worker threads (default: 1, sequential). ffmpeg subprocess startup dominates per-video time, so 4-8 workers give a big speedup')
    parser.add_argument('--snapshot-every', type=int, default=0, metavar='N',
                        help='Close current parquet and start a new one every N videos. Bounds the blast radius if the process is killed (rolling output: <stem>_part_001.parquet, etc.). Default: 0 (one monolithic file).')
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

    start_part = 1
    if args.snapshot_every:
        completed, start_part = scan_completed_parts(Path(args.output))
        if completed:
            before = len(video_dirs)
            video_dirs = [d for d in video_dirs if d.name not in completed]
            skipped = before - len(video_dirs)
            print(f'Resume: {skipped} videos already packed in existing parts; '
                  f'continuing with part_{start_part:03d}')

    print(f'Packing {len(video_dirs)} videos → {args.output}')

    if args.parallel > 1 and gemini_client is not None:
        presegment_parallel(video_dirs, gemini_client, args.parallel)

    rw = RollingWriter(Path(args.output), args.snapshot_every, SCHEMA, start_part=start_part)
    try:
        done = 0
        if args.workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
                futures = {ex.submit(extract_records, d, gemini_client): d
                           for d in video_dirs}
                for f in concurrent.futures.as_completed(futures):
                    d = futures[f]
                    try:
                        records = f.result()
                    except Exception as e:
                        print(f'  ERROR {d.name}: {e}', file=sys.stderr)
                        records = []
                    done += 1
                    print(f'  [{done}/{len(video_dirs)}] {d.name}: {len(records)} segments', flush=True)
                    rw.add(records)
        else:
            for d in video_dirs:
                records = extract_records(d, gemini_client)
                done += 1
                print(f'  [{done}/{len(video_dirs)}] {d.name}: {len(records)} segments', flush=True)
                rw.add(records)
    finally:
        rw.close()

    print(f'\nDone. {rw.total_segments} segments written to {args.output}')


if __name__ == '__main__':
    main()
