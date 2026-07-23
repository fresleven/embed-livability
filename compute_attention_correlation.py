import numpy as np
from scipy import stats

ckpt_base = "ckpt_livability"
ckpt_ae   = "ckpt_livability_alphaearth"

modalities_base = ["RS", "DSM", "NLRS", "POI"]
modalities_ae   = ["RS", "DSM", "NLRS", "AE", "POI"]

labels = np.load(f"{ckpt_base}/Livability_test_0320_tgts_test_final.npy")
liv = labels[:, 0]  # (N,)

attn_base = np.load(f"{ckpt_base}/Livability_test_0320_attentions_test_final.npy")
attn_ae   = np.load(f"{ckpt_ae}/Livability_test_0320_attentions_test_final.npy")

print(f"N samples: {len(liv)}")
print(f"LIV scores — mean: {liv.mean():.3f}, std: {liv.std():.3f}, "
      f"min: {liv.min():.3f}, max: {liv.max():.3f}")
print()

print("=" * 60)
print("BASELINE model — correlation of LIV score vs attention weight")
print("=" * 60)
for i, mod in enumerate(modalities_base):
    w = attn_base[:, i]
    r, p = stats.pearsonr(liv, w)
    rho, p_s = stats.spearmanr(liv, w)
    print(f"  {mod:6s}  Pearson r={r:+.4f} (p={p:.2e})  "
          f"Spearman rho={rho:+.4f} (p={p_s:.2e})")

print()
print("=" * 60)
print("AE model — correlation of LIV score vs attention weight")
print("=" * 60)
for i, mod in enumerate(modalities_ae):
    w = attn_ae[:, i]
    r, p = stats.pearsonr(liv, w)
    rho, p_s = stats.spearmanr(liv, w)
    print(f"  {mod:6s}  Pearson r={r:+.4f} (p={p:.2e})  "
          f"Spearman rho={rho:+.4f} (p={p_s:.2e})")
