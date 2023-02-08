#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH -J dat-1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=8
#SBATCH --mail-type=all
#SBATCH --mem-per-cpu=32GB
#SBATCH --job-name=""
#SBATCH --output="OUTPUT_%j.out" # job standard output file (%j replaced by job id)
#SBATCH --error="ERROR_%j.out" # job standard error file (%j replaced by job id)

ulimit -s unlimited
export OMP_NUM_THREADS=1
#export MODULEPATH=/opt/apps/resif/iris/2019b/broadwell/modules/all/
module load lang/Python
. /home/users/msarkis/git_repositories/qdo/.env/bin/activate
module load toolchain/intel

for MODEL in 21
do
    python src/main.py \
    --layers 8 \
    --cutoff_dim 3 \
    --model $MODEL \
    --atom_list debug debug \
    --active_sd 0.0001 \
    --passive_sd 0.1 \
    --learning_rate 0.01 \
    --epsilon 1e-3 \
    --patience 10 \
    --seed 42 \
    --save_dir ./logs/
done

#python $1