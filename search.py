#!/usr/bin/env python3
"""
search.py — search parquet transcript segments for TWSE/OTC company mentions.

Filters to plain equity shares only (CFICode starts with 'ES', per ISO 10962).
Matches both company names (len >= 2) and 4-digit ticker codes.

Usage:
    .venv/bin/python search.py dataset.parquet
    .venv/bin/python search.py test.parquet -o hits.csv
    .venv/bin/python search.py dataset.parquet --companies twse_stock_list.csv twse_otc_stocks.csv
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import ahocorasick
import pyarrow.parquet as pq


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


def search_segments(parquet_path, automaton, output_path):
    """Scan each transcript segment and write rows with at least one match to CSV."""
    table = pq.read_table(parquet_path, columns=[
        'video_id', 'transcription', 'start', 'end',
        'title', 'channel', 'upload_date',
    ])
    rows = table.to_pydict()
    n = len(rows['transcription'])

    matched = 0
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'video_id', 'title', 'channel', 'upload_date',
            'start', 'end', 'transcription',
            'matched_codes', 'matched_names',
        ])

        for i in range(n):
            text = rows['transcription'][i] or ''
            seen: dict[str, tuple] = {}  # code → (code, name, market), deduped per segment
            for _end_idx, hits in automaton.iter(text):
                for code, name, market in hits:
                    seen[code] = (code, name, market)

            if not seen:
                continue

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
                text,
                codes,
                names,
            ])

    print(f"Scanned {n} segments → {matched} matched → {output_path}", file=sys.stderr)
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
    args = parser.parse_args()

    missing = [p for p in args.companies if not Path(p).exists()]
    if missing:
        sys.exit(f"Error: company CSV not found: {missing}")

    if not Path(args.input).exists():
        sys.exit(f"Error: parquet not found: {args.input}")

    print(f"Loading company lists from: {args.companies}", file=sys.stderr)
    companies = parse_companies(args.companies, args.min_name_len)
    print(f"  {len(companies)} equity-share companies (CFICode ES*)", file=sys.stderr)

    print("Building Aho-Corasick automaton ...", file=sys.stderr)
    automaton = build_automaton(companies)
    print(f"  {len(automaton)} patterns (names + ticker codes)", file=sys.stderr)

    print(f"Searching {args.input} ...", file=sys.stderr)
    search_segments(args.input, automaton, args.output)


if __name__ == '__main__':
    main()
