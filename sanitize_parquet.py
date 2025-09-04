#!/usr/bin/env python3
import argparse
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

SUPPORTED_EXTS = {".parquet", ".prq"}


def find_parquet_files(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            yield p


def cast_table_uint64_to_decimal(table: pa.Table):
    """
    Returns (new_table, changed_cols[list of (name, from_type->to_type)])

    Behavior (drop-in compatible with your previous script, but with timestamp support):
      - A column literally named 'timestamp' is assumed to be epoch microseconds and will be cast to TIMESTAMP(us, tz='UTC')
        from either UINT64 or DECIMAL(20,0).
      - All other top-level columns of type UINT64 are cast to DECIMAL(20,0) for Spark/Iceberg compatibility.
      - Everything else is left unchanged.
    """
    changed = []
    cols = []
    schema = table.schema

    new_fields = list(schema)

    for i, field in enumerate(schema):
        col = table.column(i)
        name = field.name
        new_col = col
        new_field = field
        from_type_str = str(field.type)

        # Special handling for a column named 'timestamp': convert epoch micros -> TIMESTAMP(us)
        if name == "timestamp":
            if pa.types.is_uint64(field.type):
                # Arrow doesn't support uint64 -> timestamp directly; hop via int64 first
                new_col = pc.cast(pc.cast(col, pa.int64()), pa.timestamp("us", tz="UTC"))
                new_field = pa.field(name, pa.timestamp("us", tz="UTC"), nullable=field.nullable, metadata=field.metadata)
                changed.append((name, from_type_str, "timestamp(us, tz='UTC')"))
            elif pa.types.is_decimal128(field.type) and field.type.scale == 0 and field.type.precision >= 18:
                # If it was previously sanitized to DECIMAL(20,0), cast to int64 then to timestamp(us)
                new_col = pc.cast(pc.cast(col, pa.int64()), pa.timestamp("us", tz="UTC"))
                new_field = pa.field(name, pa.timestamp("us", tz="UTC"), nullable=field.nullable, metadata=field.metadata)
                changed.append((name, from_type_str, "timestamp(us, tz='UTC')"))
            else:
                # Already a timestamp or another compatible type; leave unchanged
                pass
        else:
            # Non-timestamp UINT64 -> DECIMAL(20,0)
            if pa.types.is_uint64(field.type):
                target = pa.decimal128(20, 0)
                new_col = pc.cast(col, target)
                new_field = pa.field(name, target, nullable=field.nullable, metadata=field.metadata)
                changed.append((name, from_type_str, "decimal128(20,0)"))

        cols.append(new_col)
        new_fields[i] = new_field

    if changed:
        new_schema = pa.schema(new_fields, metadata=schema.metadata)
        new_table = pa.Table.from_arrays(cols, schema=new_schema)
        return new_table, changed
    else:
        return table, changed


def rewrite_one(in_path: Path, out_path: Path, dry_run=False, overwrite=False, compression="snappy"):
    # Read with pyarrow to retain schema
    table = pq.read_table(in_path)
    new_table, changed = cast_table_uint64_to_decimal(table)

    if not changed:
        return {"path": str(in_path), "written": False, "changed": []}

    if dry_run:
        return {"path": str(in_path), "written": False, "changed": changed}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Won't overwrite existing file: {out_path}")

    # Write Parquet with Snappy (default)
    writer = pq.ParquetWriter(
        where=str(out_path),
        schema=new_table.schema,
        compression=compression,
        version="2.6",
        use_dictionary=True,
    )
    try:
        writer.write_table(new_table)
    finally:
        writer.close()

    return {"path": str(in_path), "written": True, "changed": changed}


def main():
    ap = argparse.ArgumentParser(description=(
        "Rewrite Parquet files by casting UINT64 columns to DECIMAL(20,0),\n"
        "and auto-converting a column named 'timestamp' (epoch micros) to TIMESTAMP(us) for Spark/Iceberg."
    ))
    ap.add_argument("--in", dest="in_dir", required=True, help="Input data directory (scanned recursively)")
    ap.add_argument("--out", dest="out_dir", required=True, help="Output directory (mirrors input tree)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    ap.add_argument("--dry-run", action="store_true", help="Only report files/columns that would change")
    ap.add_argument("--compression", default="snappy", help="Parquet compression (default: snappy)")
    args = ap.parse_args()

    in_root = Path(args.in_dir).resolve()
    out_root = Path(args.out_dir).resolve()
    if not in_root.exists() or not in_root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {in_root}")

    total = 0
    changed_files = 0
    changed_cols_total = 0

    for src in find_parquet_files(in_root):
        rel = src.relative_to(in_root)
        dst = out_root / rel

        result = rewrite_one(src, dst, dry_run=args.dry_run, overwrite=args.overwrite, compression=args.compression)
        total += 1
        if result["changed"]:
            changed_files += 1
            changed_cols_total += len(result["changed"])
            print(f"[{'DRY' if args.dry_run else 'FIX'}] {src}")
            for name, from_t, to_t in result["changed"]:
                print(f"    - {name}: {from_t} -> {to_t}")

    print(f"\nScanned {total} file(s).")
    if args.dry_run:
        print(f"{changed_files} file(s) would be rewritten; {changed_cols_total} column cast(s).")
    else:
        print(f"{changed_files} file(s) rewritten; {changed_cols_total} column cast(s).")


if __name__ == "__main__":
    main()
