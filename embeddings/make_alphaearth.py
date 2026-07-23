"""
Generate alphaearth/*.npz files using rs-embed's GSE (AlphaEarth) model.

Each file is named  alphaearth_{grid_id}_{year}.npz  and stores a 64-band
spatial embedding fetched from GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL via GEE.

Grid IDs follow the Dutch RD New (EPSG:28992) 100 m grid convention:
    E{XXXX}N{YYYY}  →  RD X = XXXX*100+50 m,  Y = YYYY*100+50 m

Usage
-----
    python make_alphaearth.py [options]

Options
-------
  --input-dir   DIR   Folder with RS_*.tif files to determine grid IDs
                      (default: Liva_RS)
  --input-json  PATH  JSONL file with {"id":..., "img":"RS_*.tif", ...}
                      records.  feature_id is taken from the "id" field.
                      May be supplied multiple times.
  --out-dir     DIR   Output folder (default: alphaearth)
  --year        INT   Year for the annual embedding (default: 2020)
  --buffer-m    INT   Half-width of the spatial window in metres (default: 255)
                      255 m → 510 m square → ~52×52 px at 10 m scale
  --workers     INT   Parallel GEE fetch threads (default: 4)
  --skip-done         Skip grid IDs whose .npz already exists (default: True)
  --no-skip-done      Re-generate even if the .npz already exists
  --grid-ids    IDs   Comma-separated list of specific grid IDs to process
"""

from __future__ import annotations

import argparse
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


def grid_id_to_lonlat(grid_id: str) -> tuple[float, float]:
    """Convert 'E{XXXX}N{YYYY}' to (centroid_lon, centroid_lat) via RD New."""
    m = re.fullmatch(r"E(\d+)N(\d+)", grid_id)
    if not m:
        raise ValueError(f"Invalid grid_id format: {grid_id!r}")
    x_rd = int(m.group(1)) * 100 + 50
    y_rd = int(m.group(2)) * 100 + 50
    lon, lat = _rd_to_wgs84.transform(x_rd, y_rd)
    return float(lon), float(lat)


# ── grid-ID discovery ─────────────────────────────────────────────────────────

_RS_PATTERN = re.compile(r"RS_(E\d+N\d+)\.tif$")


def discover_grid_ids_from_dir(rs_dir: str) -> dict[str, int]:
    """Return {grid_id: feature_id=0} for every RS_*.tif in *rs_dir*."""
    result: dict[str, int] = {}
    for name in os.listdir(rs_dir):
        m = _RS_PATTERN.match(name)
        if m:
            result[m.group(1)] = 0
    return result


def load_grid_ids_from_jsonl(paths: list[str]) -> dict[str, int]:
    """Parse JSONL files; return {grid_id: feature_id} from 'img'/'id' fields."""
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
                    grid_id = m.group(1)
                    result[grid_id] = int(rec.get("id", 0))
    return result


# ── embedding fetch ───────────────────────────────────────────────────────────

def fetch_one(
    grid_id: str,
    feature_id: int,
    year: int,
    buffer_m: int,
    out_dir: str,
    skip_done: bool,
) -> str:
    """Fetch GSE embedding for *grid_id* and save the .npz.  Returns status."""
    out_path = os.path.join(out_dir, f"alphaearth_{grid_id}_{year}.npz")

    if skip_done and os.path.exists(out_path):
        return "skipped"

    from rs_embed import get_embedding
    from rs_embed.core.specs import OutputSpec, PointBuffer, TemporalSpec

    lon, lat = grid_id_to_lonlat(grid_id)
    spatial = PointBuffer(lon=lon, lat=lat, buffer_m=buffer_m)
    temporal = TemporalSpec.year(year)

    emb = get_embedding(
        "gse",
        spatial=spatial,
        temporal=temporal,
        output=OutputSpec.grid(),
        backend="auto",
    )

    # emb.data is an xarray DataArray with dims (d, y, x) and coords d=band_names
    data_arr = np.asarray(emb.data, dtype=np.float64)  # (C, H, W)
    n_bands = data_arr.shape[0]

    band_names = np.array([f"A{i:02d}" for i in range(n_bands)], dtype="<U3")
    num_images = int(emb.meta.get("num_images", 1))

    np.savez_compressed(
        out_path,
        image_data=data_arr,
        feature_id=np.array(feature_id),
        centroid_lon=np.array(lon),
        centroid_lat=np.array(lat),
        year=np.array(year),
        num_images=np.array(num_images),
        band_names=band_names,
        grid_id=np.array(grid_id),
    )
    return "ok"


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate alphaearth GSE embedding .npz files via rs-embed."
    )
    p.add_argument("--input-dir", default="Liva_RS",
                   help="Folder containing RS_*.tif files (default: Liva_RS)")
    p.add_argument("--input-json", action="append", default=[], metavar="PATH",
                   help="JSONL file with {id, img} records (repeatable)")
    p.add_argument("--out-dir", default="alphaearth",
                   help="Output directory (default: alphaearth)")
    p.add_argument("--year", type=int, default=2020,
                   help="Annual embedding year (default: 2020)")
    p.add_argument("--buffer-m", type=int, default=255,
                   help="Spatial half-width in metres (default: 255 → 510 m window)")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel GEE fetch threads (default: 4)")
    p.add_argument("--skip-done", action="store_true", default=True,
                   help="Skip grid IDs whose .npz already exists (default: on)")
    p.add_argument("--no-skip-done", dest="skip_done", action="store_false",
                   help="Re-generate even if the .npz already exists")
    p.add_argument("--grid-ids", default=None,
                   help="Comma-separated list of specific grid IDs to process")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Resolve script dir so relative paths work when called from elsewhere
    script_dir = Path(__file__).parent

    # Add rs-embed to sys.path if not installed as a package
    rs_embed_src = script_dir / "rs-embed" / "src"
    if rs_embed_src.is_dir() and str(rs_embed_src) not in sys.path:
        sys.path.insert(0, str(rs_embed_src))

    # Pre-warm the provider registry before spawning threads.
    # rs_embed's _register_builtin_providers() sets _BUILTINS_LOADED=True before
    # the import completes, causing a race condition: parallel threads see the flag
    # but find an empty registry.  Calling has_provider() once here, sequentially,
    # ensures the registry is fully populated before the thread pool starts.
    from rs_embed.providers import has_provider as _has_provider
    _has_provider("gee")

    # Build grid_id → feature_id mapping
    grid_map: dict[str, int] = {}

    if args.grid_ids:
        for gid in args.grid_ids.split(","):
            gid = gid.strip()
            if gid:
                grid_map[gid] = 0
    elif args.input_json:
        grid_map = load_grid_ids_from_jsonl(args.input_json)
    else:
        rs_dir = args.input_dir if os.path.isabs(args.input_dir) else str(script_dir / args.input_dir)
        if not os.path.isdir(rs_dir):
            sys.exit(f"Error: --input-dir {rs_dir!r} does not exist.")
        grid_map = discover_grid_ids_from_dir(rs_dir)

    if not grid_map:
        sys.exit("No grid IDs found. Check --input-dir or --input-json.")

    out_dir = args.out_dir if os.path.isabs(args.out_dir) else str(script_dir / args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    grid_ids = sorted(grid_map.keys())
    total = len(grid_ids)
    print(f"Grid IDs to process: {total}")
    print(f"Output directory:    {out_dir}")
    print(f"Year: {args.year}  |  buffer_m: {args.buffer_m}  |  workers: {args.workers}")
    print(f"Skip already done:   {args.skip_done}")
    print()

    done = ok = skipped = failed = 0

    def _task(gid: str) -> tuple[str, str]:
        try:
            status = fetch_one(
                grid_id=gid,
                feature_id=grid_map[gid],
                year=args.year,
                buffer_m=args.buffer_m,
                out_dir=out_dir,
                skip_done=args.skip_done,
            )
        except Exception:
            return gid, f"error: {traceback.format_exc().splitlines()[-1]}"
        return gid, status

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
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

    print()
    print(f"Done. ok={ok}  skipped={skipped}  failed={failed}  total={total}")


if __name__ == "__main__":
    main()
