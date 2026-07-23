import numpy as np
from scipy import stats

ckpt_base = "ckpt_livability"
ckpt_ae   = "ckpt_livability_alphaearth"

modalities_base = ["RS", "DSM", "NLRS", "POI"]
modalities_ae   = ["RS", "DSM", "NLRS", "AE", "POI"]

labels = np.load(f"{ckpt_base}/Livability_test_0320_tgts_test_final.npy")
liv = labels[:, 0]

attn_base = np.load(f"{ckpt_base}/Livability_test_0320_attentions_test_final.npy")
attn_ae   = np.load(f"{ckpt_ae}/Livability_test_0320_attentions_test_final.npy")

def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def entropy(p):
    # Shannon entropy in nats; clip for numerical safety
    p = np.clip(p, 1e-12, None)
    return -(p * np.log(p)).sum(axis=1)

def max_entropy(n_modalities):
    return np.log(n_modalities)

# Normalize attention weights to probabilities
prob_base = softmax(attn_base)
prob_ae   = softmax(attn_ae)

H_base = entropy(prob_base)
H_ae   = entropy(prob_ae)

# Normalized entropy (0=fully concentrated, 1=uniform)
H_base_norm = H_base / max_entropy(len(modalities_base))
H_ae_norm   = H_ae   / max_entropy(len(modalities_ae))

median = np.median(liv)
high = liv >= median
low  = liv <  median

print("=" * 65)
print("ENTROPY ANALYSIS — does high LIV mean reliance or heterogeneity?")
print("=" * 65)
print(f"\nHypothesis A (reliance):     high-LIV → lower entropy (concentrated on AE)")
print(f"Hypothesis B (heterogeneity): high-LIV → higher entropy (spread across modalities)")

for label, H_norm, attn, modalities, ae_idx in [
    ("BASELINE", H_base_norm, prob_base, modalities_base, None),
    ("AE model", H_ae_norm,   prob_ae,   modalities_ae,   3),
]:
    print(f"\n{'─'*65}")
    print(f"{label} model")
    print(f"{'─'*65}")

    # 1. Entropy by LIV group
    h_high = H_norm[high].mean()
    h_low  = H_norm[low].mean()
    t, p   = stats.ttest_ind(H_norm[high], H_norm[low])
    direction = "LOWER (→ reliance)" if h_high < h_low else "HIGHER (→ heterogeneity)"
    print(f"\n  Normalized entropy:")
    print(f"    High-LIV group: {h_high:.4f}")
    print(f"    Low-LIV  group: {h_low:.4f}")
    print(f"    Difference:     {h_high - h_low:+.4f}  ({direction})")
    print(f"    t-test:         t={t:.3f}, p={p:.2e}")

    # 2. Correlation: LIV vs entropy
    r, p_r = stats.pearsonr(liv, H_norm)
    rho, p_rho = stats.spearmanr(liv, H_norm)
    print(f"\n  LIV vs entropy correlation:")
    print(f"    Pearson  r={r:+.4f} (p={p_r:.2e})")
    print(f"    Spearman r={rho:+.4f} (p={p_rho:.2e})")

    # 3. AE-specific: AE attention weight vs entropy (AE model only)
    if ae_idx is not None:
        ae_prob = attn[:, ae_idx]
        r_ae, p_ae = stats.pearsonr(ae_prob, H_norm)
        print(f"\n  AE attention weight vs entropy:")
        print(f"    Pearson  r={r_ae:+.4f} (p={p_ae:.2e})")
        if r_ae < 0:
            print(f"    → higher AE attention = more concentrated = supports RELIANCE")
        else:
            print(f"    → higher AE attention = more spread = supports HETEROGENEITY")

    # 4. Per-modality average attention by LIV group
    print(f"\n  Average attention (normalized) by LIV group:")
    print(f"    {'Modality':8s}  {'High-LIV':>10s}  {'Low-LIV':>10s}  {'Diff':>10s}")
    for j, mod in enumerate(modalities):
        hi = attn[high, j].mean()
        lo = attn[low,  j].mean()
        print(f"    {mod:8s}  {hi:10.4f}  {lo:10.4f}  {hi-lo:+10.4f}")

print()
print("=" * 65)
print("SUMMARY")
print("=" * 65)
r_liv_h, _ = stats.pearsonr(liv, H_ae_norm)
r_ae_h, _  = stats.pearsonr(prob_ae[:, 3], H_ae_norm)
if r_liv_h > 0 and r_ae_h > 0:
    conclusion = "HETEROGENEITY: high-LIV areas spread attention more AND higher AE weight comes with more spread"
elif r_liv_h < 0 and r_ae_h < 0:
    conclusion = "RELIANCE: high-LIV areas concentrate attention AND higher AE weight comes with more concentration"
elif r_liv_h > 0 and r_ae_h < 0:
    conclusion = "MIXED: high-LIV has more spread overall, but AE specifically draws concentrated attention within that"
else:
    conclusion = "MIXED: high-LIV concentrates attention, but higher AE weight itself correlates with spread"
print(f"\n  {conclusion}")
