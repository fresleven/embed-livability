"""
Generate anysat/*.npz files using rs-embed's AnySat model.

Each file is named  anysat_{grid_id}_{year}.npz  and stores a grid
embedding produced by AnySat from a Sentinel-2 multi-frame time-series
fetched via GEE.

AnySat uses COPERNICUS/S2_SR_HARMONIZED (10 bands), TemporalSpec.range,
8 temporal frames by default, and outputs a dense spatial grid.

Usage
-----
    python make_anysat.py [options]

Options
-------
  --input-dir    DIR   Folder with RS_*.tif files to determine grid IDs
                       (default: Liva_RS)
  --input-json   PATH  JSONL file with {"id":..., "img":"RS_*.tif", ...}
                       records.  May be supplied multiple times.
  --out-dir      DIR   Output folder (default: anysat)
  --year         INT   Year for the embedding; fetches [year-01-01, year+1-01-01)
                       (default: 2020)
  --buffer-m     INT   Half-width of the spatial window in metres (default: 255)
  --workers      INT   Parallel GEE fetch threads (default: 4)
  --device       STR   Inference device: auto, cpu, cuda (default: auto)
  --chunk-index  INT   0-based index of this task's chunk (for SLURM arrays)
  --num-chunks   INT   Total number of chunks (for SLURM arrays)
  --skip-done          Skip grid IDs whose .npz already exists (default: True)
  --no-skip-done       Re-generate even if the .npz already exists
  --grid-ids     IDs   Comma-separated list of specific grid IDs to process
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# ── bootstrap path ────────────────────────────────────────────────────────────

_SCRIPT_DIR = Path(__file__).parent
_RS_EMBED_SRC = _SCRIPT_DIR / "rs-embed" / "src"
if _RS_EMBED_SRC.is_dir() and str(_RS_EMBED_SRC) not in sys.path:
    sys.path.insert(0, str(_RS_EMBED_SRC))

from _embed_common import (
    add_common_args,
    grid_id_to_lonlat,
    resolve_grid_map,
    run_batch,
    slice_chunk,
)

# ── fetch one grid cell ───────────────────────────────────────────────────────

def fetch_one(
    grid_id: str,
    feature_id: int,
    year: int,
    buffer_m: int,
    out_dir: str,
    skip_done: bool,
    device: str,
) -> str:
    out_path = os.path.join(out_dir, f"anysat_{grid_id}_{year}.npz")
    if skip_done and os.path.exists(out_path):
        try:
            with np.load(out_path) as f:
                required = {"image_data", "feature_id", "centroid_lon", "centroid_lat",
                            "year", "num_images", "band_names", "grid_id"}
                if required.issubset(f.files) and f["image_data"].ndim == 3 and f["image_data"].size > 0:
                    return "skipped"
        except Exception:
            pass  # fall through and regenerate

    from rs_embed import get_embedding
    from rs_embed.core.specs import OutputSpec, PointBuffer, TemporalSpec

    lon, lat = grid_id_to_lonlat(grid_id)
    spatial = PointBuffer(lon=lon, lat=lat, buffer_m=buffer_m)
    temporal = TemporalSpec.range(f"{year}-01-01", f"{year + 1}-01-01")

    emb = get_embedding(
        "anysat",
        spatial=spatial,
        temporal=temporal,
        output=OutputSpec.grid(),
        input_prep="tile",
        backend="auto",
        device=device,
    )

    data_arr = np.asarray(emb.data, dtype=np.float64)  # (C, H, W)
    n_bands = data_arr.shape[0]
    band_names = np.array([f"S{i:03d}" for i in range(n_bands)], dtype="<U4")

    np.savez_compressed(
        out_path,
        image_data=data_arr,
        feature_id=np.array(feature_id),
        centroid_lon=np.array(lon),
        centroid_lat=np.array(lat),
        year=np.array(year),
        num_images=np.array(int(emb.meta.get("num_images", 1))),
        band_names=band_names,
        grid_id=np.array(grid_id),
    )
    return "ok"


# ── main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Generate anysat embedding .npz files via rs-embed."
    )
    add_common_args(p)
    p.add_argument("--out-dir", default="anysat",
                   help="Output directory (default: anysat)")
    p.add_argument("--device", default="auto",
                   help="Inference device: auto, cpu, cuda (default: auto)")
    p.add_argument("--chunk-index", type=int, default=None,
                   help="0-based chunk index for SLURM array jobs")
    p.add_argument("--num-chunks", type=int, default=None,
                   help="Total number of chunks for SLURM array jobs")
    args = p.parse_args(argv)

    # Pre-warm provider registry before spawning threads (avoids race condition)
    from rs_embed.providers import has_provider as _has_provider
    _has_provider("gee")

    grid_map = resolve_grid_map(args, _SCRIPT_DIR)
    if not grid_map:
        sys.exit("No grid IDs found. Check --input-dir or --input-json.")

    # Slice this task's chunk if running as a SLURM array job
    if args.chunk_index is not None and args.num_chunks is not None:
        grid_map = slice_chunk(grid_map, args.chunk_index, args.num_chunks)
        print(f"Chunk {args.chunk_index + 1}/{args.num_chunks}: {len(grid_map)} grid IDs")

    out_dir = args.out_dir if os.path.isabs(args.out_dir) else str(_SCRIPT_DIR / args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Grid IDs to process: {len(grid_map)}")
    print(f"Output directory:    {out_dir}")
    print(f"Year: {args.year}  |  buffer_m: {args.buffer_m}  |  workers: {args.workers}  |  device: {args.device}")
    print(f"Skip already done:   {args.skip_done}")
    print()

    def _fetch(gid: str, fid: int) -> str:
        return fetch_one(
            grid_id=gid,
            feature_id=fid,
            year=args.year,
            buffer_m=args.buffer_m,
            out_dir=out_dir,
            skip_done=args.skip_done,
            device=args.device,
        )

    run_batch(grid_map, _fetch, args.workers)


if __name__ == "__main__":
    main()
