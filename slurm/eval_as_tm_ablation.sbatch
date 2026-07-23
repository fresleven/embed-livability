#!/bin/bash
#SBATCH --account=bcrm-tgirails
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu_a100,gpu
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --job-name=eval_as_tm_abl
#SBATCH --output=slurm-%j.out

# ckpt_livability_anysat_terramind -- ablation test splits
# suffix_index 1 = _NULL_RS (noRS)
python train.py --anysat --terramind --suffix_index 0 --test_suffix_index 1 --no-do_train --final_checkpoint

# suffix_index 2 = _NULL_DSM (noDSM)
python train.py --anysat --terramind --suffix_index 0 --test_suffix_index 2 --no-do_train --final_checkpoint

# suffix_index 3 = _NULL_GIU (noGIU)
python train.py --anysat --terramind --suffix_index 0 --test_suffix_index 3 --no-do_train --final_checkpoint

# suffix_index 4 = _noPOI (noPOI)
python train.py --anysat --terramind --suffix_index 0 --test_suffix_index 4 --no-do_train --final_checkpoint
