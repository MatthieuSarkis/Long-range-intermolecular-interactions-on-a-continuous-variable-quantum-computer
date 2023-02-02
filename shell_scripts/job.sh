for MODEL in 12
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
    --alpha 0.90 \
    --patience 10 \
    --seed 42 \
    --save_dir ./logs/
done

