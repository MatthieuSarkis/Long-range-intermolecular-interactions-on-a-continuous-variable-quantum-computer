#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH -J dat-1
#SBATCH --qos=normal
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=8
#SBATCH --job-name="contVars"
#SBATCH --output="OUTPUT.out" # job standard output file (%j replaced by job id)
#SBATCH --error="ERROR.out" # job standard error file (%j replaced by job id)

ulimit -s unlimited
export OMP_NUM_THREADS=1
#export MODULEPATH=/opt/apps/resif/iris/2019b/broadwell/modules/all/
module load lang/Python/3.8.6-GCCcore-10.2.0
. /home/users/msarkis/git_repositories/qdo/.env/bin/activate
module load toolchain/intel

for MODEL in 33
do
    python src/main.py \
    --layers 8 \
    --cutoff_dim 4 \
    --model $MODEL \
    --atom_list Un Un \
    --active_sd 0.0001 \
    --passive_sd 0.1 \
    --learning_rate 0.01 \
    --epsilon 1e-3 \
    --patience 10 \
    --seed 42 \
    --save_dir ./logs/
done

#python $1
