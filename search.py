#!/usr/bin/env python3
"""
search.py — search parquet transcript segments for TWSE/OTC company mentions.

Two-stage pipeline:
  1. Aho-Corasick exact match (fast pre-filter)
  2. Gemini verification to remove false positives (e.g. "新興" in "新興市場")

Gemini verification runs by default when GEMINI_API_KEY is set; use --no-verify
to skip it. Filters to plain equity shares only (CFICode ES*, ISO 10962).

Usage:
    .venv/bin/python search.py dataset.parquet
    .venv/bin/python search.py test.parquet -o hits.csv
    .venv/bin/python search.py dataset.parquet --no-verify   # exact match only
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import ahocorasick
import pyarrow.parquet as pq
from google import genai
from google.genai import types

_GEMINI_MODEL = 'gemini-3.1-flash-lite'


def parse_companies(csv_paths, min_name_len=2):
    """
    Parse TWSE/OTC CSV files, returning (code, name, market) for plain equity shares.
    Rows are kept only when CFICode starts with 'ES' (equity shares, ISO 10962).
    """
    companies = []
    for path in csv_paths:
        with open(path, encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                first = row[0]
                # Section labels and the header have no ideographic space
                if '　' not in first:
                    continue
                code, name = first.split('　', 1)
                code, name = code.strip(), name.strip()
                cfi = row[5].strip() if len(row) > 5 else ''
                market = row[3].strip() if len(row) > 3 else ''
                if not cfi.startswith('ES'):
                    continue
                if len(name) < min_name_len:
                    continue
                companies.append((code, name, market))
    return companies


def build_automaton(companies):
    """
    Build an Aho-Corasick automaton keyed on both company name and ticker code.
    Each key maps to a list of (code, name, market) tuples in case of duplicates.
    """
    patterns: dict[str, list] = defaultdict(list)
    for code, name, market in companies:
        patterns[name].append((code, name, market))
        patterns[code].append((code, name, market))

    A = ahocorasick.Automaton()
    for pattern, hits in patterns.items():
        A.add_word(pattern, hits)
    A.make_automaton()
    return A


def verify_batch(batch: list[tuple[str, dict]], client: genai.Client) -> list[dict]:
    """
    Ask Gemini which Aho-Corasick candidates are genuine stock mentions.

    batch: list of (transcript_text, seen) where seen is {code: (code, name, market)}
    Returns a parallel list of filtered `seen` dicts with false positives removed.
    """
    lines = []
    for i, (text, seen) in enumerate(batch):
        candidates = '、'.join(f'{name}({code})' for code, (_, name, _) in seen.items())
        lines.append(f'[{i}] 文字：「{text}」\n    候選：{candidates}')

    prompt = """\
以下是多段台灣股市節目的字幕片段，以及每段中偵測到的公司名稱。
請判斷每段字幕中，哪些候選名稱是真正在討論該上市/上櫃公司（股票），
而非剛好出現在一般詞彙中（例如「新興」出現於「新興市場」、「全台」出現於「全台灣」、
「數字」出現於「數字顯示」等）。

輸出格式：JSON 陣列，長度與輸入片段數相同，每個元素是確認為股票提及的代號陣列。
只輸出 JSON，不要其他說明。範例（3個片段）：[["2330"],[],["2605","1303"]]

片段：
""" + '\n\n'.join(lines)

    response = client.models.generate_content(
        model=_GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type='application/json'),
    )
    text = re.sub(r'^```[a-z]*\n?|\n?```$', '', response.text.strip())
    confirmed_per_seg: list[list[str]] = json.loads(text)

    results = []
    for (_, seen), confirmed in zip(batch, confirmed_per_seg):
        results.append({code: seen[code] for code in confirmed if code in seen})
    return results


def search_segments(parquet_path, automaton, output_path,
                    gemini_client: genai.Client | None = None,
                    batch_size: int = 20):
    """
    Scan each transcript segment for company mentions and write matches to CSV.
    If gemini_client is provided, verify candidates in batches before writing.
    """
    table = pq.read_table(parquet_path, columns=[
        'video_id', 'transcription', 'start', 'end',
        'title', 'channel', 'upload_date',
    ])
    rows = table.to_pydict()
    n = len(rows['transcription'])

    # Phase 1: Aho-Corasick pre-filter
    candidates: list[tuple[int, dict]] = []  # (row_index, seen)
    for i in range(n):
        text = rows['transcription'][i] or ''
        seen: dict[str, tuple] = {}
        for _end_idx, hits in automaton.iter(text):
            for code, name, market in hits:
                seen[code] = (code, name, market)
        if seen:
            candidates.append((i, seen))

    print(f"  Pre-filter: {len(candidates)} / {n} segments have candidate matches",
          file=sys.stderr)

    # Phase 2: optional Gemini verification
    if gemini_client is not None:
        print(f"  Verifying with Gemini in batches of {batch_size} ...", file=sys.stderr)
        verified: list[tuple[int, dict]] = []
        for start in range(0, len(candidates), batch_size):
            chunk = candidates[start:start + batch_size]
            batch = [(rows['transcription'][i] or '', seen) for i, seen in chunk]
            try:
                filtered = verify_batch(batch, gemini_client)
            except Exception as e:
                print(f"    Gemini error ({e}), keeping all candidates in this batch",
                      file=sys.stderr)
                filtered = [seen for _, seen in batch]
            for (i, _), seen in zip(chunk, filtered):
                if seen:
                    verified.append((i, seen))
            done = min(start + batch_size, len(candidates))
            print(f"    {done}/{len(candidates)} verified", file=sys.stderr)
        candidates = verified

    # Phase 3: write CSV
    matched = 0
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'video_id', 'title', 'channel', 'upload_date',
            'start', 'end', 'transcription',
            'matched_codes', 'matched_names',
        ])
        for i, seen in candidates:
            matched += 1
            codes = '|'.join(c for c, _, _ in seen.values())
            names = '|'.join(nm for _, nm, _ in seen.values())
            writer.writerow([
                rows['video_id'][i],
                rows['title'][i],
                rows['channel'][i],
                rows['upload_date'][i],
                f"{rows['start'][i]:.2f}",
                f"{rows['end'][i]:.2f}",
                rows['transcription'][i],
                codes,
                names,
            ])

    print(f"  Final: {matched} confirmed matches → {output_path}", file=sys.stderr)
    return matched


def main():
    parser = argparse.ArgumentParser(
        description='Search parquet transcript segments for TWSE/OTC equity mentions.'
    )
    parser.add_argument('input', help='Input parquet file')
    parser.add_argument('-o', '--output', default='search_results.csv', help='Output CSV (default: search_results.csv)')
    parser.add_argument(
        '--companies', nargs='+',
        default=['twse_stock_list.csv', 'twse_otc_stocks.csv'],
        metavar='CSV',
        help='Company list CSVs (default: twse_stock_list.csv twse_otc_stocks.csv)',
    )
    parser.add_argument('--min-name-len', type=int, default=2, metavar='N',
                        help='Minimum company name length to index (default: 2)')
    parser.add_argument('--no-verify', action='store_true',
                        help='Skip Gemini verification, use exact match only')
    parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                        help='Segments per Gemini verification call (default: 20)')
    args = parser.parse_args()

    missing = [p for p in args.companies if not Path(p).exists()]
    if missing:
        sys.exit(f"Error: company CSV not found: {missing}")

    if not Path(args.input).exists():
        sys.exit(f"Error: parquet not found: {args.input}")

    # Set up Gemini client
    api_key = os.environ.get('GEMINI_API_KEY')
    if args.no_verify:
        gemini_client = None
        print("Gemini verification disabled (--no-verify).", file=sys.stderr)
    elif api_key:
        gemini_client = genai.Client(api_key=api_key)
        print(f"Gemini verification enabled ({_GEMINI_MODEL}).", file=sys.stderr)
    else:
        gemini_client = None
        print("Warning: GEMINI_API_KEY not set — skipping Gemini verification, "
              "results may contain false positives.", file=sys.stderr)

    print(f"Loading company lists from: {args.companies}", file=sys.stderr)
    companies = parse_companies(args.companies, args.min_name_len)
    print(f"  {len(companies)} equity-share companies (CFICode ES*)", file=sys.stderr)

    print("Building Aho-Corasick automaton ...", file=sys.stderr)
    automaton = build_automaton(companies)
    print(f"  {len(automaton)} patterns (names + ticker codes)", file=sys.stderr)

    print(f"Searching {args.input} ...", file=sys.stderr)
    search_segments(args.input, automaton, args.output, gemini_client, args.batch_size)


if __name__ == '__main__':
    main()
