#!/bin/bash
# Generates sbatch scripts for the seeds {43, 44} needed to get 3-seed
# (42 already run, 43/44 new) mean +/- std results for every table in the
# paper, plus embedding-only probe baselines at all 3 seeds.
#
# Table 1 (tab:add) main configs, and the ablation table (tab:ablation)
# configs, are the only ones that need NEW TRAINING RUNS: the city
# breakdown (tab:city), POI stratification (tab:poi), and missing-modality
# zeroing (tab:zeroing_out) tables all re-evaluate the SAME Table-1
# full-data checkpoints (just sliced by city/POI-presence, or with a
# modality zeroed at test time), so once the 8 main configs have 3 seeds
# of checkpoints, all four of those tables can be aggregated for free.
set -e

HEADER() {
  local jobname=$1
  cat <<EOF
#!/bin/bash
#SBATCH --account=bcrm-tgirails
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu,gpu_a100
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --job-name=${jobname}
#SBATCH --output=slurm-%j.out

EOF
}

# --- Table 1 (tab:add) main configs: 8 configs x seeds {43,44} ---
declare -A MAIN_CONFIGS=(
  [base]=""
  [ae]="--alphaearth"
  [as]="--anysat"
  [tm]="--terramind"
  [ae_as]="--alphaearth --anysat"
  [ae_tm]="--alphaearth --terramind"
  [as_tm]="--anysat --terramind"
  [ae_as_tm]="--alphaearth --anysat --terramind"
)

for name in "${!MAIN_CONFIGS[@]}"; do
  flags="${MAIN_CONFIGS[$name]}"
  for seed in 43 44; do
    fname="train_${name}_seed${seed}.sbatch"
    HEADER "${name}_s${seed}" > "$fname"
    echo "python train.py ${flags} --seed ${seed} --resume --no-evaluate_during_training" >> "$fname"
    echo "Wrote $fname"
  done
done

# --- Ablation table (tab:ablation): one modality excluded, base/ae/as/tm ---
declare -A ABLATION_SUFFIX=(
  [noRS]=1
  [noDSM]=2
  [noNLRS]=3
  [noPOI]=4
)

for abl_name in "${!ABLATION_SUFFIX[@]}"; do
  suffix_idx="${ABLATION_SUFFIX[$abl_name]}"
  for name in base ae as tm; do
    flags="${MAIN_CONFIGS[$name]}"
    for seed in 43 44; do
      fname="train_${name}_${abl_name}_seed${seed}.sbatch"
      HEADER "${name}_${abl_name}_s${seed}" > "$fname"
      echo "python train.py ${flags} --suffix_index ${suffix_idx} --seed ${seed} --resume --no-evaluate_during_training" >> "$fname"
      echo "Wrote $fname"
    done
  done
done

# --- Ablation table, ALL4 row (suffix_index 5): every embedding combo, no base ---
for name in ae as tm ae_as ae_tm as_tm ae_as_tm; do
  flags="${MAIN_CONFIGS[$name]}"
  for seed in 43 44; do
    fname="train_${name}_all4_seed${seed}.sbatch"
    HEADER "${name}_all4_s${seed}" > "$fname"
    echo "python train.py ${flags} --suffix_index 5 --seed ${seed} --resume --no-evaluate_during_training" >> "$fname"
    echo "Wrote $fname"
  done
done

# --- Embedding-only probe baselines: all 3 seeds (none run yet) ---
for embedding in alphaearth anysat terramind; do
  for seed in 42 43 44; do
    fname="probe_${embedding}_seed${seed}.sbatch"
    HEADER "probe_${embedding}_s${seed}" > "$fname"
    echo "python probe_baseline.py --embedding ${embedding} --seed ${seed}" >> "$fname"
    echo "Wrote $fname"
  done
done

echo "Done."
