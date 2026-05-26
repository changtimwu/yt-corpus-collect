#!/bin/bash
# Print transcription rows from a parquet (or glob) without loading audio bytes.
# Usage: ./peek.sh <file-or-glob>  [limit]
# Examples:
#   ./peek.sh dataset_part_005.parquet
#   ./peek.sh dataset_part_005.parquet 50
#   ./peek.sh 'dataset_part_*.parquet' 100
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <parquet-file-or-glob> [limit]" >&2
    exit 1
fi

.venv/bin/python -c "
import pyarrow.parquet as pq, glob, sys
files = sorted(glob.glob(sys.argv[1])) or [sys.argv[1]]
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 20
printed = 0
for f in files:
    t = pq.read_table(f, columns=['video_id', 'start', 'end', 'transcription'])
    for r in t.to_pylist():
        if printed >= limit:
            sys.exit(0)
        print(f\"[{r['video_id']}] {r['start']:6.1f}-{r['end']:6.1f}  {r['transcription']}\")
        printed += 1
" "$@"
