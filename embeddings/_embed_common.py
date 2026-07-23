"""Shared helpers for make_terramind.py and make_anysat.py."""

from __future__ import annotations

import json
import os
import re
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from pyproj import Transformer

# ── coordinate helpers ────────────────────────────────────────────────────────

_rd_to_wgs84 = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)

_RS_PATTERN = re.compile(r"RS_(E\d+N\d+)\.tif$")


def grid_id_to_lonlat(grid_id: str) -> tuple[float, float]:
    """Convert 'E{XXXX}N{YYYY}' to (centroid_lon, centroid_lat) via RD New."""
    m = re.fullmatch(r"E(\d+)N(\d+)", grid_id)
    if not m:
        raise ValueError(f"Invalid grid_id format: {grid_id!r}")
    lon, lat = _rd_to_wgs84.transform(int(m.group(1)) * 100 + 50, int(m.group(2)) * 100 + 50)
    return float(lon), float(lat)


# ── grid-ID discovery ─────────────────────────────────────────────────────────

def discover_grid_ids_from_dir(rs_dir: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for name in os.listdir(rs_dir):
        m = _RS_PATTERN.match(name)
        if m:
            result[m.group(1)] = 0
    return result


def load_grid_ids_from_jsonl(paths: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                img = rec.get("img", "")
                m = _RS_PATTERN.match(img)
                if m:
                    result[m.group(1)] = int(rec.get("id", 0))
    return result


# ── chunk splitting for SLURM array jobs ─────────────────────────────────────

def slice_chunk(grid_map: dict[str, int], chunk_index: int, num_chunks: int) -> dict[str, int]:
    """Return the subset of grid_map assigned to this array task."""
    keys = sorted(grid_map.keys())
    chunks = [keys[i::num_chunks] for i in range(num_chunks)]
    return {k: grid_map[k] for k in chunks[chunk_index]}


# ── common argument parsing ───────────────────────────────────────────────────

def add_common_args(p) -> None:
    p.add_argument("--input-dir", default="Liva_RS",
                   help="Folder with RS_*.tif files (default: Liva_RS)")
    p.add_argument("--input-json", action="append", default=[], metavar="PATH",
                   help="JSONL file with {id, img} records (repeatable)")
    p.add_argument("--year", type=int, default=2020,
                   help="Year — fetches [year-01-01, year+1-01-01) (default: 2020)")
    p.add_argument("--buffer-m", type=int, default=250,
                   help="Spatial half-width in metres (default: 250)")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel GEE fetch threads (default: 4)")
    p.add_argument("--skip-done", action="store_true", default=True,
                   help="Skip grid IDs whose .npz already exists (default: on)")
    p.add_argument("--no-skip-done", dest="skip_done", action="store_false",
                   help="Re-generate even if the .npz already exists")
    p.add_argument("--grid-ids", default=None,
                   help="Comma-separated list of specific grid IDs to process")


def resolve_grid_map(args, script_dir: Path) -> dict[str, int]:
    if args.grid_ids:
        return {gid.strip(): 0 for gid in args.grid_ids.split(",") if gid.strip()}
    if args.input_json:
        return load_grid_ids_from_jsonl(args.input_json)
    rs_dir = args.input_dir if os.path.isabs(args.input_dir) else str(script_dir / args.input_dir)
    if not os.path.isdir(rs_dir):
        sys.exit(f"Error: --input-dir {rs_dir!r} does not exist.")
    return discover_grid_ids_from_dir(rs_dir)


# ── batch runner ──────────────────────────────────────────────────────────────

def run_batch(
    grid_map: dict[str, int],
    fetch_fn,          # callable(grid_id, feature_id) -> "ok" | "skipped"
    workers: int,
) -> None:
    grid_ids = sorted(grid_map.keys())
    total = len(grid_ids)
    done = ok = skipped = failed = 0

    def _task(gid: str) -> tuple[str, str]:
        try:
            return gid, fetch_fn(gid, grid_map[gid])
        except Exception:
            return gid, f"error: {traceback.format_exc().splitlines()[-1]}"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_task, gid): gid for gid in grid_ids}
        for fut in as_completed(futs):
            gid, status = fut.result()
            done += 1
            if status == "ok":
                ok += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
                print(f"  FAIL [{done}/{total}] {gid}: {status}", flush=True)
                continue
            if done % 100 == 0 or done == total:
                print(f"  [{done}/{total}] ok={ok} skipped={skipped} failed={failed}", flush=True)

    print(f"\nDone. ok={ok}  skipped={skipped}  failed={failed}  total={total}")
