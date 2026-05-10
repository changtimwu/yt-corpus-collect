#!/usr/bin/env python3
"""
segment_vtt.py — resegment a VTT file into natural sentences using Gemini.

Usage:
    source ../common.env
    .venv/bin/python segment_vtt.py corpus/VIDEO_ID/VIDEO_ID.zh-TW.vtt
    .venv/bin/python segment_vtt.py input.vtt -o output.vtt
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from google import genai
from google.genai import types

_TS_RE = re.compile(r'(\d+):(\d+):(\d+)[.,](\d+)')
_TAG_RE = re.compile(r'<[^>]+>')

MODEL = 'gemini-3.1-flash-lite'


def parse_timestamp(ts: str) -> float:
    m = _TS_RE.match(ts.strip())
    if not m:
        return 0.0
    h, mn, s, ms = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    return h * 3600 + mn * 60 + s + ms / 1000


def fmt_ts(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    return f'{h:02d}:{m:02d}:{s:02d}.{ms:03d}'


def parse_vtt(vtt_path: Path) -> list[dict]:
    """Parse VTT into a deduplicated list of {start, end, text} dicts."""
    segments = []
    lines = vtt_path.read_text(encoding='utf-8').splitlines()
    i = 0
    while i < len(lines):
        if '-->' in lines[i]:
            parts = lines[i].split('-->')
            start = parse_timestamp(parts[0])
            end = parse_timestamp(parts[1].split()[0])
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

    deduped, last = [], None
    for seg in segments:
        if seg['text'] != last:
            deduped.append(seg)
            last = seg['text']
    return deduped


def segment_with_gemini(cues: list[dict], client: genai.Client) -> list[list[int]]:
    """
    Ask Gemini to group consecutive cues into complete sentences.
    Returns a list of groups; each group is a list of cue indices.
    """
    numbered = '\n'.join(f'[{i}] {c["text"]}' for i, c in enumerate(cues))
    prompt = f"""\
以下是台灣中文影片的字幕，每行格式為「[編號] 文字」。
請將連續的字幕單元合併成語義完整的自然句子，不要在句子中間切斷。

輸出：JSON 陣列，每個元素是一組連續字幕編號（整數陣列），代表同一句子。
只輸出 JSON，不要其他說明文字。範例：[[0,1,2],[3,4],[5,6,7]]

字幕：
{numbered}"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type='application/json',
        ),
    )
    text = re.sub(r'^```[a-z]*\n?|\n?```$', '', response.text.strip())
    groups = json.loads(text)

    # Sanity: flatten to check all cue indices are covered exactly once
    covered = sorted(idx for g in groups for idx in g)
    expected = list(range(len(cues)))
    if covered != expected:
        print(f'Warning: index mismatch — got {len(covered)} indices, expected {len(cues)}',
              file=sys.stderr)

    # Clamp out-of-range indices Gemini occasionally hallucinates
    n = len(cues)
    groups = [[i for i in g if 0 <= i < n] for g in groups]
    groups = [g for g in groups if g]

    return groups


def write_vtt(groups: list[list[int]], cues: list[dict], out_path: Path) -> None:
    blocks = []
    for g in groups:
        if not g:
            continue
        start = fmt_ts(cues[g[0]]['start'])
        end = fmt_ts(cues[g[-1]]['end'])
        text = ''.join(cues[i]['text'] for i in g)
        blocks.append(f'{start} --> {end}\n{text}')
    out_path.write_text('WEBVTT\n\n' + '\n\n'.join(blocks) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(
        description='Resegment a VTT file into natural sentences using Gemini.'
    )
    parser.add_argument('input', help='Input VTT file')
    parser.add_argument('-o', '--output', help='Output VTT (default: <input>.segmented.vtt)')
    args = parser.parse_args()

    vtt_path = Path(args.input)
    out_path = Path(args.output) if args.output else vtt_path.with_suffix('.segmented.vtt')

    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        sys.exit('Error: GEMINI_API_KEY not set. Run: source ../common.env')

    print(f'Parsing {vtt_path} ...', file=sys.stderr)
    cues = parse_vtt(vtt_path)
    print(f'  {len(cues)} cues after deduplication', file=sys.stderr)

    client = genai.Client(api_key=api_key)

    print(f'Calling {MODEL} ...', file=sys.stderr)
    groups = segment_with_gemini(cues, client)
    print(f'  → {len(groups)} sentences', file=sys.stderr)

    write_vtt(groups, cues, out_path)
    print(f'Written → {out_path}', file=sys.stderr)


if __name__ == '__main__':
    main()
