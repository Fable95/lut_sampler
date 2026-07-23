#!/usr/bin/env python3
"""Retroactively add a `const N_MAX: u16 = <n>;` stub to exported LUT files.

Newer exports (table_fill/export.rs) emit N_MAX directly; this script patches
tables generated before that. The inserted value is taken from the header
comment every exporter version has always written on line 1:

    // SD(Z, Pi_Z) = ... considered range: [0,<n>]

and placed as a trait const right before the `const LUT_TABLE` line (whose
indentation it copies), satisfying the `N_MAX` item of `tables::{Matrix,Cube}`.

CAVEAT: the header's `n` is the considered-range bound (typically `2^b - 1`),
an UPPER BOUND on the largest value actually stored — which is what N_MAX is
defined as. After running this script, always run

    python3 scripts/verify_n_max.py --fix

to tighten N_MAX to the real table maximum (verified against the data).

Idempotent: files already containing `const N_MAX` are skipped. Files are
streamed line-by-line (some tables are ~100 MB).

Usage: python3 scripts/add_n_max.py [tables_dir]
       (default: src/lut_sampler/tables relative to the repo root)
"""

import re
import sys
from pathlib import Path

RANGE_RE = re.compile(r"considered range: \[0,\s*(\d+)\]")
ANCHOR = "const LUT_TABLE"
N_MAX_MARKER = "const N_MAX"


def patch_file(path: Path) -> str:
    tmp = path.with_suffix(".rs.tmp")
    n = None
    inserted = False
    with path.open("r", encoding="utf-8") as src:
        first = src.readline()
        m = RANGE_RE.search(first)
        if m is None:
            return "SKIP (no 'considered range' header)"
        n = int(m.group(1))
        if not 0 <= n <= 0xFFFF:
            return f"SKIP (n = {n} does not fit u16)"
        with tmp.open("w", encoding="utf-8") as dst:
            dst.write(first)
            for line in src:
                if N_MAX_MARKER in line:
                    tmp.unlink()
                    return "SKIP (already has N_MAX)"
                if not inserted and ANCHOR in line:
                    indent = line[: len(line) - len(line.lstrip())]
                    dst.write(f"{indent}const N_MAX: u16 = {n};\n")
                    inserted = True
                dst.write(line)
    if not inserted:
        tmp.unlink()
        return "SKIP (no 'const LUT_TABLE' anchor)"
    tmp.replace(path)
    return f"PATCHED (N_MAX = {n})"


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    tables_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else repo_root / "src" / "lut_sampler" / "tables"
    if not tables_dir.is_dir():
        print(f"error: {tables_dir} is not a directory", file=sys.stderr)
        return 1
    failures = 0
    for path in sorted(tables_dir.glob("*.rs")):
        if path.name == "mod.rs":
            continue
        result = patch_file(path)
        print(f"{path.name:32} {result}")
        if result.startswith("SKIP (no"):
            failures += 1
    if failures:
        print(f"\n{failures} file(s) could not be patched; fix them manually.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
