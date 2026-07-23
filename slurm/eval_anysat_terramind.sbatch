#!/bin/bash
#SBATCH --account=bcrm-tgirails
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu,gpu_a100
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --job-name=eval_anysat_terramind
#SBATCH --output=slurm-%j.out

# ckpt_livability_anysat
python train.py --anysat --suffix_index 0 --no-do_train --final_checkpoint
python train.py --anysat --suffix_index 0 --test_suffix_index 1 --no-do_train --final_checkpoint
python train.py --anysat --suffix_index 0 --test_suffix_index 2 --no-do_train --final_checkpoint
python train.py --anysat --suffix_index 0 --test_suffix_index 3 --no-do_train --final_checkpoint
python train.py --anysat --suffix_index 0 --test_suffix_index 4 --no-do_train --final_checkpoint
python train.py --anysat --suffix_index 0 --anyzero --no-do_train --final_checkpoint

# ckpt_livability_anysat_NULL_RS
python train.py --anysat --suffix_index 1 --no-do_train --final_checkpoint

# ckpt_livability_anysat_NULL_DSM
python train.py --anysat --suffix_index 2 --no-do_train --final_checkpoint

# ckpt_livability_anysat_NULL_GIU
python train.py --anysat --suffix_index 3 --no-do_train --final_checkpoint

# ckpt_livability_anysat_noPOI
python train.py --anysat --suffix_index 4 --no-do_train --final_checkpoint

# ckpt_livability_terramind
python train.py --terramind --suffix_index 0 --no-do_train --final_checkpoint
python train.py --terramind --suffix_index 0 --test_suffix_index 1 --no-do_train --final_checkpoint
python train.py --terramind --suffix_index 0 --test_suffix_index 2 --no-do_train --final_checkpoint
python train.py --terramind --suffix_index 0 --test_suffix_index 3 --no-do_train --final_checkpoint
python train.py --terramind --suffix_index 0 --test_suffix_index 4 --no-do_train --final_checkpoint
python train.py --terramind --suffix_index 0 --terrazero --no-do_train --final_checkpoint

# ckpt_livability_terramind_NULL_RS
python train.py --terramind --suffix_index 1 --no-do_train --final_checkpoint

# ckpt_livability_terramind_NULL_DSM
python train.py --terramind --suffix_index 2 --no-do_train --final_checkpoint

# ckpt_livability_terramind_NULL_GIU
python train.py --terramind --suffix_index 3 --no-do_train --final_checkpoint

# ckpt_livability_terramind_noPOI
python train.py --terramind --suffix_index 4 --no-do_train --final_checkpoint

# ckpt_livability_alphaearth_anysat
python train.py --alphaearth --anysat --suffix_index 0 --no-do_train --final_checkpoint
python train.py --alphaearth --anysat --suffix_index 0 --test_suffix_index 1 --no-do_train --final_checkpoint
python train.py --alphaearth --anysat --suffix_index 0 --test_suffix_index 2 --no-do_train --final_checkpoint
python train.py --alphaearth --anysat --suffix_index 0 --test_suffix_index 3 --no-do_train --final_checkpoint
python train.py --alphaearth --anysat --suffix_index 0 --test_suffix_index 4 --no-do_train --final_checkpoint
python train.py --alphaearth --anysat --suffix_index 0 --alphazero --no-do_train --final_checkpoint
python train.py --alphaearth --anysat --suffix_index 0 --anyzero --no-do_train --final_checkpoint
python train.py --alphaearth --anysat --suffix_index 0 --alphazero --anyzero --no-do_train --final_checkpoint

# ckpt_livability_alphaearth_terramind
python train.py --alphaearth --terramind --suffix_index 0 --no-do_train --final_checkpoint
python train.py --alphaearth --terramind --suffix_index 0 --test_suffix_index 1 --no-do_train --final_checkpoint
python train.py --alphaearth --terramind --suffix_index 0 --test_suffix_index 2 --no-do_train --final_checkpoint
python train.py --alphaearth --terramind --suffix_index 0 --test_suffix_index 3 --no-do_train --final_checkpoint
python train.py --alphaearth --terramind --suffix_index 0 --test_suffix_index 4 --no-do_train --final_checkpoint
python train.py --alphaearth --terramind --suffix_index 0 --alphazero --no-do_train --final_checkpoint
python train.py --alphaearth --terramind --suffix_index 0 --terrazero --no-do_train --final_checkpoint
python train.py --alphaearth --terramind --suffix_index 0 --alphazero --terrazero --no-do_train --final_checkpoint

# ckpt_livability_alphaearth_anysat_terramind
python train.py --alphaearth --anysat --terramind --suffix_index 0 --no-do_train --final_checkpoint
python train.py --alphaearth --anysat --terramind --suffix_index 0 --test_suffix_index 1 --no-do_train --final_checkpoint
python train.py --alphaearth --anysat --terramind --suffix_index 0 --test_suffix_index 2 --no-do_train --final_checkpoint
python train.py --alphaearth --anysat --terramind --suffix_index 0 --test_suffix_index 3 --no-do_train --final_checkpoint
python train.py --alphaearth --anysat --terramind --suffix_index 0 --test_suffix_index 4 --no-do_train --final_checkpoint
python train.py --alphaearth --anysat --terramind --suffix_index 0 --alphazero --no-do_train --final_checkpoint
python train.py --alphaearth --anysat --terramind --suffix_index 0 --anyzero --no-do_train --final_checkpoint
python train.py --alphaearth --anysat --terramind --suffix_index 0 --alphazero --anyzero --no-do_train --final_checkpoint
python train.py --alphaearth --anysat --terramind --suffix_index 0 --terrazero --no-do_train --final_checkpoint
python train.py --alphaearth --anysat --terramind --suffix_index 0 --alphazero --terrazero --no-do_train --final_checkpoint
python train.py --alphaearth --anysat --terramind --suffix_index 0 --anyzero --terrazero --no-do_train --final_checkpoint
python train.py --alphaearth --anysat --terramind --suffix_index 0 --alphazero --anyzero --terrazero --no-do_train --final_checkpoint

