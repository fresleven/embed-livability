# Applying foundation model embeddings towards urban livability evaluation

Code and experiments for the paper:

> **Applying foundation model embeddings towards urban livability evaluation**
> Ayush Khot, Wen Zhou, Shaowen Wang — University of Illinois at Urbana-Champaign
> *Submitted to ACM SIGSPATIAL '26 (soon to be posted on arXiv!)*

We investigate which physical features are encoded within geospatial foundation
model embeddings — **AlphaEarth Foundations**, **AnySat**, and **TerraMind** —
and how they affect urban livability prediction. We extend the transformer-based
multi-task multimodal regression (TMTMR) model of Zhou et al. to fuse these
embeddings with four base modalities (remote sensing, digital surface model,
nightlight remote sensing, and point-of-interest text) and study their marginal
value, robustness to missing inputs, and interpretability via attention entropy
and Full Grad-CAM.

## Model overview

Per 100 m × 100 m Leefbaarometer (LBM) grid cell, the model predicts the overall
livability score plus five domain scores (PHY, HOU, AME, SOC, NUI):

- **RS / DSM / NLRS** images → pretrained **DenseNet** feature extractors
- **POI** text → pretrained **BERT** (`bert-base-multilingual-uncased`)
- **AlphaEarth / AnySat / TerraMind** embeddings → dedicated convolution + pooling stacks
- All features are concatenated and fused by a **Transformer encoder**, then
  decoded by a linear head under a multi-task MAE objective.

Model variants: `base` (RS+DSM+NLRS+POI), `aef`, `as`, `tm` (base + one
embedding), and pairwise / full combinations (`aef+as`, `aef+tm`, `as+tm`,
`aef+as+tm`).

## Repository layout

```
.
├── train.py                      # Training / evaluation entrypoint
├── utils.py                      # Train & evaluate loops
├── cvae_utils.py                 # Conditional-VAE spatial deconfounder utilities
├── textBert_utils.py             # BERT / text helpers
├── tex_utils.py                  # LaTeX results-table generation
├── gradcam.py                    # Full Grad-CAM explainability (Fig. 5)
├── compute_attention_correlation.py  # Attention vs. livability correlation (Sec. 6.1)
├── compute_attention_entropy.py      # Attention entropy analysis (Sec. 6.1)
├── print_cvae_params.py
├── MMBT_liva/                    # Multimodal bitransformer model
│   ├── image_liva.py             #   image / embedding encoders (DenseNet, conv stacks, C-VAE)
│   ├── mmbt_liva.py              #   MMBT model + classification head
│   ├── mmbt_config_liva.py
│   └── mmbt_utils_liva_0318.py   #   dataset, collation, losses
├── embeddings/                   # Foundation-model embedding generation (see its README)
│   ├── make_alphaearth.py  make_anysat.py  make_terramind.py  _embed_common.py
│   └── submit_*.sh  *.sbatch     #   example SLURM scripts
├── slurm/                        # Training-job SLURM scripts (per model / seed / ablation)
├── notebooks/                    # Results, plots, and attention analysis
│   ├── results.ipynb  plots.ipynb  attention.ipynb
│   └── Livability_evaluation_4IM_RM2_0901.ipynb
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

For generating embeddings you additionally need `rs-embed` (Python ≥ 3.12):

```bash
pip install "rs-embed[terratorch] @ git+https://github.com/cybergis/rs-embed"
```

## Data

All data corresponds to the year **2020** over the Netherlands (LBM v3.0,
100 m × 100 m grid). The code reads a single **data root** (default
`<repo>/data_livability`, override with `LIVABILITY_DATA_DIR`) laid out as:

```
data_livability/
├── json/Livability_{train,eval,test}_0320.json   # JSONL splits
├── Liva_RS/  Liva_DSM/  Liva_GIU_RGB/             # RS / DSM / NLRS GeoTIFFs
├── alphaearth/  anysat/  terramind/               # embedding .npz files
└── models/saved_chexnet.pt                        # optional (see below)
```

| Source | Where to get it |
|--------|-----------------|
| **RS, DSM, NLRS, POI** (+ labels / splits) | Hugging Face (Parquet, access-gated): [`Vinjou/Multimodal_urban_livability_evaluation_dataset`](https://huggingface.co/datasets/Vinjou/Multimodal_urban_livability_evaluation_dataset) |
| **AlphaEarth** embeddings (64×50×50, 52,001 cells, 9.2 GB) | Hugging Face: [`akhot2/alphaearth-livability-2020`](https://huggingface.co/datasets/akhot2/alphaearth-livability-2020) — see [`data_prep/README.md`](data_prep/README.md) |
| **AnySat / TerraMind** embeddings | Regenerate with the scripts in [`embeddings/`](embeddings/) |

The Hugging Face dataset ships as Parquet, while `train.py` reads the folder
tree above. **[`data_prep/`](data_prep/) converts one to the other** and also
documents how to host/download the large AlphaEarth archive:

```bash
huggingface-cli login                      # the base dataset is access-gated
cd data_prep
python convert_parquet.py --inspect        # see the parquet schema
python convert_parquet.py --out-dir ../data_livability
```

**DenseNet weights.** The image branch uses CheXNet DenseNet-121 weights at
`<data_root>/models/saved_chexnet.pt`. If that file is missing, the code falls
back to torchvision ImageNet DenseNet-121 (with a warning) so everything still
runs; provide the file to reproduce the paper's exact feature extractor.

## Training

`train.py` selects modalities via flags. Examples:

```bash
# Baseline (RS + DSM + NLRS + POI)
python train.py

# Add AlphaEarth
python train.py --alphaearth

# Add multiple embeddings
python train.py --alphaearth --terramind

# Ablations: drop a modality during training via suffix_index
#   0='' 1=_NULL_RS 2=_NULL_DSM 3=_NULL_GIU(NLRS) 4=_noPOI 5=_NULL
#   6=_only_RS 7=_only_DSM 8=_only_GIU 9=_only_POI
python train.py --alphaearth --suffix_index 1     # train without RS

# Robustness / missing-modality (zero a modality at test time)
python train.py --alphaearth --alphazero
```

Training defaults follow the paper (Appendix A): AdamW, lr `5e-5`, weight decay
`0.1`, batch size 16, 12 epochs, early stopping patience 5, multi-task MAE loss.
`slurm/` contains the exact job scripts used for each model / seed / ablation.
Run `python train.py --help` for all options.

## Analysis & figures

- `notebooks/results.ipynb` — RMSE tables (Tables 1–6)
- `notebooks/plots.ipynb` — geographic prediction maps (Fig. 3)
- `notebooks/attention.ipynb`, `compute_attention_*.py` — attention weights and
  entropy (Fig. 4, Sec. 6.1)
- `gradcam.py` — Full Grad-CAM visualizations (Fig. 5)

## Key findings

- **AlphaEarth** gives the most reliable gains and is the best single-embedding
  choice; `aef` / `aef+tm` offer the best cost-benefit tradeoff in urban areas.
- Embeddings substantially **mitigate missing-modality degradation** (e.g. LIV
  RMSE when RS is zeroed drops from 0.154 → 0.107 with `aef+as+tm`), but do not
  fully replace high-resolution RS imagery.
- In **rural** areas, single embeddings can hurt; only multi-embedding
  combinations (`aef+as+tm`) reliably recover, reflecting an urban-biased
  pretraining distribution.

## Citation

This paper has been **submitted to ACM SIGSPATIAL '26** and will be posted on
arXiv soon. Citation details will be updated once it is available.

```bibtex
@misc{khot2026livability,
  title  = {Applying foundation model embeddings towards urban livability evaluation},
  author = {Khot, Ayush and Zhou, Wen and Wang, Shaowen},
  year   = {2026},
  note   = {Submitted to ACM SIGSPATIAL '26; arXiv preprint forthcoming}
}
```

The embedding library is described in Ye et al., *Any Model, Any Place, Any
Time: Get Remote Sensing Foundation Model Embeddings On Demand* (2026), and is
available at <https://github.com/cybergis/rs-embed>.

## License

MIT — see [LICENSE](LICENSE).
