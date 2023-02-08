for MODEL in 30
do
    python src/main.py \
    --layers 8 \
    --cutoff_dim 5 \
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