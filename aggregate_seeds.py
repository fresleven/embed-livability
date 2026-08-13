"""
Aggregate RMSE metrics across multiple training seeds into mean +/- std,
and emit LaTeX table rows ready to paste into the paper.

Each model config in this repo is trained 3x with --seed 42 (default,
no output_dir suffix), --seed 43 (output_dir + "_seed43"), and
--seed 44 (output_dir + "_seed44"). This script reads the
eval_results_test_final.txt JSON written by train.py for each seed
directory and reports mean +/- std per metric.

Usage:
    python aggregate_seeds.py                  # Table 1 (main comparison)
    python aggregate_seeds.py --table ablation  # Table tab:ablation
    python aggregate_seeds.py --table probe     # embedding-only probe baselines

To adapt to a table not predefined here, edit the CONFIGS list for that
table below: each entry is (label, output_dir_base, eval_filename_stem).
"""
import argparse
import json
import os
import numpy as np

SEEDS = [42, 43, 44]

# metric key in eval_results_test_final.txt -> column label
METRIC_MAP = [
    ("rmse_lbm", "LIV"),
    ("rmse_tgts_fys", "PHY"),
    ("rmse_tgts_onv", "NUI"),
    ("rmse_tgts_soc", "SOC"),
    ("rmse_tgts_vrz", "AME"),
    ("rmse_tgts_won", "HOU"),
]

# Table 1 (tab:add): 8 main configs, suffix_index 0 (full data), seed-suffixed dirs.
CONFIGS_MAIN = [
    (r"\textsc{base}", "ckpt_livability"),
    (r"\textsc{aef}", "ckpt_livability_alphaearth"),
    (r"\textsc{as}", "ckpt_livability_anysat"),
    (r"\textsc{tm}", "ckpt_livability_terramind"),
    (r"\textsc{aef+as}", "ckpt_livability_alphaearth_anysat"),
    (r"\textsc{aef+tm}", "ckpt_livability_alphaearth_terramind"),
    (r"\textsc{as+tm}", "ckpt_livability_anysat_terramind"),
    (r"\textsc{aef+as+tm}", "ckpt_livability_alphaearth_anysat_terramind"),
]
EVAL_FILE_MAIN = "Livability_test_0320_eval_results_test_final.txt"

# Table tab:ablation: one modality excluded during training (suffix_index 1-4),
# and all four excluded (suffix_index 5, no base since base needs >=1 modality).
ABLATION_SUFFIXES = [
    ("RS", "_NULL_RS"),
    ("DSM", "_NULL_DSM"),
    ("NLRS", "_NULL_GIU"),
    ("POI", "_noPOI"),
]
CONFIGS_ABLATION = []
for ablation_name, suffix in ABLATION_SUFFIXES:
    for label, base in CONFIGS_MAIN[:4]:  # base, aef, as, tm only
        CONFIGS_ABLATION.append(
            (f"{ablation_name} / {label}", base, f"Livability_test_0320{suffix}_eval_results_test_final.txt")
        )
for label, base in CONFIGS_MAIN[1:]:  # aef, as, tm, and all combos (no base: 0 modalities can't train)
    CONFIGS_ABLATION.append(
        (f"ALL4 / {label}", base, "Livability_test_0320_NULL_eval_results_test_final.txt")
    )

# Embedding-only probe baselines (see probe_baseline.py). Populate output_dir
# bases to match whatever probe_baseline.py writes.
CONFIGS_PROBE = [
    (r"\textsc{probe-aef}", "ckpt_probe_alphaearth"),
    (r"\textsc{probe-as}", "ckpt_probe_anysat"),
    (r"\textsc{probe-tm}", "ckpt_probe_terramind"),
]
EVAL_FILE_PROBE = "Livability_test_0320_eval_results_test_final.txt"


def load_seed_values(output_dir_base, eval_filename, seeds=SEEDS):
    """Return {metric_key: [values across found seeds]}, skipping missing runs."""
    values = {k: [] for k, _ in METRIC_MAP}
    found_seeds = []
    for seed in seeds:
        d = output_dir_base if seed == 42 else f"{output_dir_base}_seed{seed}"
        path = os.path.join(d, eval_filename)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            result = json.load(f)
        for k, _ in METRIC_MAP:
            if k in result:
                values[k].append(result[k])
        found_seeds.append(seed)
    return values, found_seeds


def mean_std(values):
    if len(values) == 0:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return float(np.mean(values)), float(np.std(values, ddof=1))


def format_cell(mean, std, missing_seeds):
    if mean is None:
        return "--"
    if missing_seeds:
        return f"{mean:.3f}$\\pm${std:.3f}*"  # * flags incomplete (fewer than 3 seeds)
    return f"{mean:.3f}$\\pm${std:.3f}"


def run_table(configs, default_eval_file=None, table_label=""):
    print(f"\n=== {table_label} ===")
    rows = []
    for entry in configs:
        if len(entry) == 3:
            label, base, eval_file = entry
        else:
            label, base = entry
            eval_file = default_eval_file
        values, found_seeds = load_seed_values(base, eval_file)
        missing = len(found_seeds) < len(SEEDS)
        cells = []
        for k, colname in METRIC_MAP:
            m, s = mean_std(values[k])
            cells.append(format_cell(m, s, missing))
        status = f"[{len(found_seeds)}/{len(SEEDS)} seeds: {found_seeds}]"
        print(f"{label:30s} {status:30s} " + " ".join(f"{c:18s}" for c in cells))
        rows.append((label, cells))

    print("\n--- LaTeX rows (paste into table body) ---")
    for label, cells in rows:
        print(f"{label} & " + " & ".join(cells) + r" \\")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", choices=["main", "ablation", "probe"], default="main")
    args = parser.parse_args()

    if args.table == "main":
        run_table(CONFIGS_MAIN, EVAL_FILE_MAIN, "Table 1 (tab:add)")
    elif args.table == "ablation":
        run_table(CONFIGS_ABLATION, None, "Table tab:ablation")
    elif args.table == "probe":
        run_table(CONFIGS_PROBE, EVAL_FILE_PROBE, "Embedding-only probe baselines")
