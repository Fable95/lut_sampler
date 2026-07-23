#!/usr/bin/env python3
"""Verify (or fix) each exported LUT file's `N_MAX` against its table data.

`N_MAX` is defined as the largest value ACTUALLY stored in the table (the
realized tail cutoff) — this is generally smaller than the considered range
end `n` in the file's header comment (`n` is the representation bound the SD
was computed over, typically `2^b - 1`). Newly exported tables get the
correct value from `table_fill/export.rs` (`checked_table_max`); tables that
were retro-fitted by `add_n_max.py` (which could only lift the header's `n`,
an upper bound) need `--fix` once to tighten `N_MAX` to the data.

The script unpacks every u64 cell into its embedded lanes (u8: 8 lanes,
u16: 4 lanes) and checks

    max(unpacked table values) == N_MAX

Zero-padding lanes are zero and cannot raise the maximum. With `--fix`, a
mismatching `const N_MAX` line is rewritten with the computed maximum.

Streams line-by-line (some tables are ~100 MB). Exits non-zero on any
mismatch (unless fixed) or unparseable file.

Usage: python3 scripts/verify_n_max.py [--fix] [tables_dir]
       (default dir: src/lut_sampler/tables relative to the repo root)
"""

import re
import sys
from pathlib import Path

N_MAX_RE = re.compile(r"const N_MAX: u16 = (\d+);")
EMBEDDED_RE = re.compile(r"type Embedded = (u8|u16|BitShare);")
HEX_RE = re.compile(r"0x[0-9a-fA-F]{16}")
ANCHOR = "const LUT_TABLE"

LANES = {"u8": (8, 8, 0xFF), "u16": (4, 16, 0xFFFF), "BitShare": (8, 8, 0xFF)}


def verify_file(path: Path) -> str:
    n_max = None
    embedded = None
    in_table = False
    table_max = 0
    with path.open("r", encoding="utf-8") as src:
        for line in src:
            if not in_table:
                if embedded is None:
                    m = EMBEDDED_RE.search(line)
                    if m:
                        embedded = m.group(1)
                if n_max is None:
                    m = N_MAX_RE.search(line)
                    if m:
                        n_max = int(m.group(1))
                if ANCHOR in line:
                    if embedded is None:
                        return "FAIL (no 'type Embedded' declaration found)"
                    if n_max is None:
                        return "FAIL (no N_MAX const found — run add_n_max.py first)"
                    in_table = True
                    lanes, width, mask = LANES[embedded]
                continue
            for hex_lit in HEX_RE.findall(line):
                cell = int(hex_lit, 16)
                if cell == 0:
                    continue
                for i in range(lanes):
                    v = (cell >> (i * width)) & mask
                    if v > table_max:
                        table_max = v
    if not in_table:
        return "FAIL (no 'const LUT_TABLE' anchor found)"
    if table_max == n_max:
        return f"OK (N_MAX = {n_max} == max table value)"
    return f"FAIL (N_MAX = {n_max}, but max table value = {table_max})"


def fix_file(path: Path, table_max: int) -> None:
    """Rewrite the `const N_MAX` line with the computed maximum (streamed)."""
    tmp = path.with_suffix(".rs.tmp")
    with path.open("r", encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as dst:
        for line in src:
            m = N_MAX_RE.search(line)
            if m:
                line = line.replace(m.group(0), f"const N_MAX: u16 = {table_max};")
            dst.write(line)
    tmp.replace(path)


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--fix"]
    fix = "--fix" in sys.argv[1:]
    repo_root = Path(__file__).resolve().parent.parent
    tables_dir = Path(args[0]) if args else repo_root / "src" / "lut_sampler" / "tables"
    if not tables_dir.is_dir():
        print(f"error: {tables_dir} is not a directory", file=sys.stderr)
        return 1
    failures = 0
    for path in sorted(tables_dir.glob("*.rs")):
        if path.name == "mod.rs":
            continue
        result = verify_file(path)
        if result.startswith("FAIL (N_MAX") and fix:
            table_max = int(result.rsplit("= ", 1)[1].rstrip(")"))
            fix_file(path, table_max)
            result = f"FIXED (N_MAX rewritten to max table value = {table_max})"
        print(f"{path.name:32} {result}")
        if result.startswith("FAIL"):
            failures += 1
    if failures:
        print(f"\n{failures} file(s) failed verification.", file=sys.stderr)
        return 1
    print("\nAll tables consistent: N_MAX matches the data everywhere.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
