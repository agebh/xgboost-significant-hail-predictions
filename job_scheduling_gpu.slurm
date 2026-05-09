#!/bin/bash

#SBATCH -n 1
#SBATCH --time=01:00:00
#SBATCH -A es_prein
#SBATCH --gpus=pro_6000:1
#SBATCH --gres=gpumem:20g 

#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=80G

#SBATCH --job-name=analysis_xgbGlobal
#SBATCH --output=analysis_xgbGlobal.out
#SBATCH --error=analysis_xgbGlobal.err

source /cluster/home/agebhardt/env/bin/activate

srun python xgbGlobal_fixedParams_v2.py
