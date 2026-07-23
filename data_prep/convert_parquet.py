"""
Convert the published Hugging Face dataset (Parquet) into the on-disk layout
that `train.py` expects.

The Hugging Face dataset
`Vinjou/Multimodal_urban_livability_evaluation_dataset` is distributed as
Parquet files (one row per 100 m grid cell). The training code instead reads a
folder tree of JSONL split files, GeoTIFF images, and `.npz` embeddings:

    <data_root>/
    ├── json/
    │   ├── Livability_train_0320.json     # JSONL: one JSON object per line
    │   ├── Livability_eval_0320.json
    │   └── Livability_test_0320.json
    ├── Liva_RS/      RS_<grid>.tif         # RGB remote-sensing image
    ├── Liva_DSM/     DSM_<grid>.tif        # grayscale digital surface model
    ├── Liva_GIU_RGB/ GIU_<grid>.tif        # RGB nightlight remote sensing (NLRS)
    ├── alphaearth/   alphaearth_<grid>_2020.npz   # optional embeddings
    ├── anysat/       anysat_<grid>_2020.npz       # (only if present in parquet)
    └── terramind/    terramind_<grid>_2020.npz

Each JSONL record has the fields:
    id, img, dsm, giu, text, lbm, fys, onv, soc, vrz, won
(the six labels are lbm=LIV, fys=PHY, onv=NUI, soc=SOC, vrz=AME, won=HOU).

Usage
-----
    # 1) Inspect the parquet schema first (no files written):
    python convert_parquet.py --inspect

    # 2) Convert everything into ./data_livability:
    python convert_parquet.py --out-dir ../data_livability

    # Read from a local parquet folder instead of the Hub:
    python convert_parquet.py --parquet /path/to/*.parquet --out-dir ../data_livability

If the parquet column names differ from the defaults, edit COLUMN_MAP below.
"""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

import numpy as np

DATASET_ID = "Vinjou/Multimodal_urban_livability_evaluation_dataset"

# ── Column mapping ──────────────────────────────────────────────────────────
# LEFT  = field name written to the JSONL / used to name files (do not change)
# RIGHT = the column name in the parquet dataset (adjust to match --inspect output)
COLUMN_MAP = {
    "id":   "id",
    "img":  "img",     # RGB remote-sensing image (or its filename)
    "dsm":  "dsm",     # grayscale DSM image
    "giu":  "giu",     # RGB nightlight (NLRS) image
    "text": "text",    # POI text
    "lbm":  "lbm", "fys": "fys", "onv": "onv",
    "soc":  "soc", "vrz": "vrz", "won": "won",
}

# Embedding columns, if the parquet ships them. Left = output subfolder / prefix,
# right = parquet column holding the (C, H, W) array. Comment out any that are absent.
EMBEDDING_MAP = {
    "alphaearth": "alphaearth",
    # "anysat":   "anysat",
    # "terramind": "terramind",
}

# Map Hugging Face split names to the file suffixes train.py expects.
SPLIT_TO_FILE = {
    "train": "Livability_train_0320.json",
    "validation": "Livability_eval_0320.json",
    "valid": "Livability_eval_0320.json",
    "eval": "Livability_eval_0320.json",
    "test": "Livability_test_0320.json",
}

IMG_SUBDIR = {"img": "Liva_RS", "dsm": "Liva_DSM", "giu": "Liva_GIU_RGB"}


# ── loading ─────────────────────────────────────────────────────────────────

def load_splits(parquet_glob: str | None):
    """Return {split_name: HF Dataset}. Uses local parquet if given, else the Hub."""
    from datasets import load_dataset

    if parquet_glob:
        ds = load_dataset("parquet", data_files=parquet_glob)
    else:
        ds = load_dataset(DATASET_ID)
    return dict(ds)


def inspect(splits) -> None:
    for name, ds in splits.items():
        print(f"\n=== split: {name}  ({len(ds)} rows) ===")
        print("features:")
        for col, feat in ds.features.items():
            print(f"  {col!r}: {feat}")
        row = ds[0]
        print("sample row (truncated):")
        for col, val in row.items():
            preview = repr(val)
            if len(preview) > 90:
                preview = preview[:90] + "…"
            print(f"  {col}: {preview}")


# ── value coercion helpers ────────────────────────────────────────────────

def save_image(value, dest: Path) -> None:
    """Write an image cell (PIL image, {'bytes':..}, ndarray, or filename) to dest."""
    from PIL import Image

    if isinstance(value, Image.Image):
        value.save(dest)
    elif isinstance(value, dict) and value.get("bytes") is not None:
        Image.open(io.BytesIO(value["bytes"])).save(dest)
    elif isinstance(value, np.ndarray):
        Image.fromarray(value).save(dest)
    elif isinstance(value, str) and os.path.exists(value):
        Image.open(value).save(dest)
    else:
        raise TypeError(
            f"Don't know how to save image cell of type {type(value)} to {dest}. "
            f"Inspect the schema and adjust save_image()."
        )


def as_filename(value, prefix: str, grid: str) -> str:
    """Return the '<PREFIX>_<grid>.tif' filename, whether the cell is a name or an image."""
    if isinstance(value, str) and value.endswith(".tif"):
        return os.path.basename(value)
    return f"{prefix}_{grid}.tif"


def grid_from(record_img_name: str) -> str:
    """'RS_E1158N4967.tif' -> 'E1158N4967'."""
    return os.path.basename(record_img_name).split("_", 1)[1].rsplit(".tif", 1)[0]


# ── conversion ──────────────────────────────────────────────────────────────

def convert(splits, out_dir: Path) -> None:
    (out_dir / "json").mkdir(parents=True, exist_ok=True)
    for sub in IMG_SUBDIR.values():
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    for sub in EMBEDDING_MAP:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    for split_name, ds in splits.items():
        json_name = SPLIT_TO_FILE.get(split_name.lower())
        if json_name is None:
            print(f"! skipping unknown split {split_name!r} (add it to SPLIT_TO_FILE)")
            continue

        jsonl_path = out_dir / "json" / json_name
        n = len(ds)
        print(f"converting split {split_name!r} -> {jsonl_path.name}  ({n} rows)")

        with open(jsonl_path, "w") as fout:
            for i in range(n):
                row = ds[i]
                # image filenames first, so we can derive the grid id
                img_name = as_filename(row[COLUMN_MAP["img"]], "RS",
                                       str(row[COLUMN_MAP["id"]]))
                grid = grid_from(img_name)
                rec = {
                    "id": row[COLUMN_MAP["id"]],
                    "img": f"RS_{grid}.tif",
                    "dsm": f"DSM_{grid}.tif",
                    "giu": f"GIU_{grid}.tif",
                }
                for lbl in ("lbm", "fys", "onv", "soc", "vrz", "won"):
                    rec[lbl] = float(row[COLUMN_MAP[lbl]])
                rec["text"] = row[COLUMN_MAP["text"]]
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

                # images
                for field, prefix in (("img", "RS"), ("dsm", "DSM"), ("giu", "GIU")):
                    dest = out_dir / IMG_SUBDIR[field] / f"{prefix}_{grid}.tif"
                    if not dest.exists():
                        save_image(row[COLUMN_MAP[field]], dest)

                # embeddings (optional)
                for sub, col in EMBEDDING_MAP.items():
                    dest = out_dir / sub / f"{sub}_{grid}_2020.npz"
                    if not dest.exists():
                        arr = np.asarray(row[col], dtype=np.float64)
                        np.savez_compressed(dest, image_data=arr, grid_id=np.array(grid),
                                            year=np.array(2020))

                if (i + 1) % 1000 == 0:
                    print(f"  {i + 1}/{n}")

    print("done.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parquet", default=None,
                   help="Local parquet path/glob. Omit to pull from the Hugging Face Hub.")
    p.add_argument("--out-dir", default="data_livability", type=Path,
                   help="Output data root (default: data_livability)")
    p.add_argument("--inspect", action="store_true",
                   help="Print the parquet schema + a sample row, then exit.")
    args = p.parse_args()

    splits = load_splits(args.parquet)
    if args.inspect:
        inspect(splits)
        return
    convert(splits, args.out_dir)


if __name__ == "__main__":
    main()
