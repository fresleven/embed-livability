# Data preparation

The training code (`train.py`) reads a folder tree of **JSONL splits + GeoTIFF
images + `.npz` embeddings**. The published Hugging Face dataset ships as
**Parquet**. This folder bridges the two.

```
<data_root>/                      # default: <repo>/data_livability
├── json/Livability_{train,eval,test}_0320.json   # JSONL, one object per line
├── Liva_RS/      RS_<grid>.tif                    # RGB remote sensing
├── Liva_DSM/     DSM_<grid>.tif                   # grayscale DSM
├── Liva_GIU_RGB/ GIU_<grid>.tif                   # RGB nightlight (NLRS)
├── alphaearth/   alphaearth_<grid>_2020.npz       # AlphaEarth embeddings
├── anysat/       anysat_<grid>_2020.npz           # (regenerate — see ../embeddings)
├── terramind/    terramind_<grid>_2020.npz
└── models/       saved_chexnet.pt                 # optional (see note below)
```

`train.py` looks for `<data_root>` next to the repo by default. Point it
anywhere with an environment variable:

```bash
export LIVABILITY_DATA_DIR=/path/to/data_livability
```

## 1. Convert the Parquet dataset → on-disk format

```bash
pip install datasets pyarrow pillow numpy

# Inspect the parquet schema first (writes nothing):
python convert_parquet.py --inspect

# Convert all splits into ../data_livability:
python convert_parquet.py --out-dir ../data_livability
```

`convert_parquet.py` writes the JSONL split files, the RS/DSM/NLRS GeoTIFFs, and
any embedding columns present in the parquet. If the parquet column names differ
from the defaults, edit `COLUMN_MAP` / `EMBEDDING_MAP` at the top of the script
(the `--inspect` output shows the exact names). JSONL label fields are Dutch
Leefbaarometer names: `lbm`=LIV, `fys`=PHY, `onv`=NUI, `soc`=SOC, `vrz`=AME,
`won`=HOU.

> The base dataset (`Vinjou/Multimodal_urban_livability_evaluation_dataset`) is
> access-gated. Request access on its Hugging Face page and
> `huggingface-cli login` before converting.

## 2. AlphaEarth embeddings (9.2 GB)

The AlphaEarth embeddings (64×50×50 per grid cell, 52,001 cells) are too large
for git, so they are published as a separate Hugging Face dataset:
[**`akhot2/alphaearth-livability-2020`**](https://huggingface.co/datasets/akhot2/alphaearth-livability-2020).

Download and unpack into place:

```bash
pip install -U huggingface_hub
huggingface-cli download akhot2/alphaearth-livability-2020 \
    alphaearth_livability_2020.tar --repo-type dataset --local-dir .
mkdir -p ../data_livability/alphaearth
tar -xf alphaearth_livability_2020.tar -C ../data_livability/alphaearth
```

### Hosting your own copy

If you regenerate the embeddings and want to host them yourself, either of the
following works.

#### Option A — a new Hugging Face dataset (recommended)

```bash
pip install -U "huggingface_hub[hf_transfer]"
export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli login

# Create the repo (once) under your own account/org:
huggingface-cli repo create alphaearth-livability-2020 --repo-type dataset

# Upload the archive produced from the .npz folder:
huggingface-cli upload <your-username>/alphaearth-livability-2020 \
    alphaearth_livability_2020.tar  alphaearth_livability_2020.tar \
    --repo-type dataset
```

To recreate the archive from a folder of `.npz` files:

```bash
tar -cf alphaearth_livability_2020.tar -C /path/to/alphaearth_npz .
```

Prefer browsable per-file access over one tarball? Upload the folder directly
with `huggingface-cli upload-large-folder <repo> /path/to/alphaearth_npz
--repo-type dataset`.

### Option B — GitHub Release assets

GitHub rejects a 9.2 GB file (2 GB/asset limit), so split the archive and
attach the parts to a Release (not the git history):

```bash
split -b 1900M alphaearth_livability_2020.tar alphaearth_part_
gh release create alphaearth-v1 alphaearth_part_* \
    --title "AlphaEarth embeddings (2020)" \
    --notes "cat alphaearth_part_* > alphaearth_livability_2020.tar to reassemble"
```

Reassemble with `cat alphaearth_part_* > alphaearth_livability_2020.tar`.

## Note on DenseNet (CheXNet) weights

The paper's image branch uses CheXNet DenseNet-121 weights at
`<data_root>/models/saved_chexnet.pt`. If that file is absent, the code now
**falls back to torchvision ImageNet DenseNet-121** with a warning, so the model
still runs — results will differ slightly from the paper. Provide the
`saved_chexnet.pt` file to reproduce the exact feature extractor.
