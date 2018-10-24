#!/bin/bash
# NOT TO CALL DIRECTLY, USED BY sbatch_optimization.sh. Request puck3
# and consume one GPU, name the jobs "acqdiv" (puck3 has 2 gpus so the jobs are launched
# 2-by-2)
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=acqdiv

# load the python environment
module load anaconda/3
source activate fair_project

# the python script to be executed
lm_acqdiv="python /scratch2/gloukatou/fair_project/char-lm-code-master/lm-acqdiv.py"

# list of all the arguments, multiples spaces replaced by a single one
args=$(echo "$@" | tr -s ' ')

echo "Start at $(date)"
echo "Python: $(which python)"
echo "Running: $lm_acqdiv"
echo "Arguments: $args"

# run the script
$lm_acqdiv $args

echo "Finish at $(date)"
