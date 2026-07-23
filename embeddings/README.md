# Foundation-model embedding generation

These scripts generate the geospatial foundation-model embeddings used in the
paper — **AlphaEarth**, **AnySat**, and **TerraMind** — for the Leefbaarometer
(LBM) 100 m × 100 m grid cells. Each script fetches imagery on demand and runs
the corresponding model through the
[`rs-embed`](https://github.com/cybergis/rs-embed) library, saving one
`.npz` per grid cell.

| Script | Model | Source imagery | Output dim (C×H×W) |
|--------|-------|----------------|--------------------|
| `make_alphaearth.py` | AlphaEarth Foundations (GSE) | `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` | 64 × 50 × 50 |
| `make_anysat.py`     | AnySat | Sentinel-2 (`COPERNICUS/S2_SR_HARMONIZED`, S2 time series) | 1536 × 50 × 50 |
| `make_terramind.py`  | TerraMind | Sentinel-2 L2A | 384 × 14 × 14 |

`_embed_common.py` holds shared helpers (grid-ID → lon/lat conversion via the
Dutch RD New / EPSG:28992 grid, input discovery, SLURM-array chunking, and the
threaded batch runner).

## Install

`rs-embed` requires Python ≥ 3.12.

```bash
pip install "rs-embed[terratorch] @ git+https://github.com/cybergis/rs-embed"
pip install pyproj numpy
```

The scripts import `rs_embed` directly. If you keep a local checkout of
`rs-embed` beside these scripts (in a folder named `rs-embed/`), the scripts
also add `rs-embed/src` to `sys.path` automatically as a fallback.

## Authentication

The embeddings are fetched through **Google Earth Engine**, so you need a GEE
account and must authenticate once:

```bash
earthengine authenticate
```

Some `rs-embed` backends also download model weights from the Hugging Face Hub:

```bash
export HF_TOKEN=hf_...   # your own token
```

> ⚠️ Never commit a real token. The submit scripts read `HF_TOKEN` from the
> environment.

## Usage

Grid cells are discovered from a folder of `RS_{grid_id}.tif` files
(`--input-dir`, default `Liva_RS`), from JSONL records (`--input-json`), or
from an explicit `--grid-ids` list. Example:

```bash
# AlphaEarth (CPU-only fetch is fine)
python make_alphaearth.py --input-dir Liva_RS --out-dir alphaearth \
    --year 2020 --buffer-m 255 --workers 8

# AnySat (GPU recommended)
python make_anysat.py --input-dir Liva_RS --out-dir anysat \
    --year 2020 --buffer-m 250 --workers 12 --device cuda

# TerraMind (GPU recommended)
python make_terramind.py --input-dir Liva_RS --out-dir terramind \
    --year 2020 --buffer-m 250 --workers 28 --device cuda
```

Run `python make_<model>.py --help` for the full option list. `--skip-done`
(on by default) skips grid cells whose `.npz` already exists, and
`--chunk-index` / `--num-chunks` split the work across SLURM array tasks.

`submit_anysat.sh`, `submit_terramind.sh`, `anysat.sbatch`, and
`terramind.sbatch` are example SLURM batch scripts — adjust the account,
partition, and environment name for your cluster.

## Output format

Each `.npz` contains:

- `image_data` — the embedding tensor, shape `(C, H, W)`, `float64`
- `grid_id`, `feature_id`, `centroid_lon`, `centroid_lat`, `year`
- `num_images` — number of source scenes composited
- `band_names` — per-channel labels (`A00…`, `S000…`, `T000…`)

The pre-generated AlphaEarth embeddings are published as an archive on
Hugging Face (see the top-level [README](../README.md#data)); AnySat and
TerraMind embeddings can be regenerated with the scripts above.
