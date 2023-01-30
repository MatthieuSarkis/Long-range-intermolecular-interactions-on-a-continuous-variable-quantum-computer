for MODEL in 21 22 31 32
do
    python src/main.py \
    --layers 10 \
    --cutoff_dim 20 \
    --model $MODEL \
    --atom_list Ar Ar \
    --active_sd 0.0001 \
    --passive_sd 0.1 \
    --learning_rate 0.01 \
    --seed 42 \
    --save_dir ./logs/
done

