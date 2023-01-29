#for DIRECTION in perpendicular parallel
#do
#    python src/main.py \
#    --layers 8 \
#    --cutoff_dim 6 \
#    --order full \
#    --direction $DIRECTION \
#    --dimension 1 \
#    --atom_list Ar Ar \
#    --active_sd 0.0001 \
#    --passive_sd 0.1 \
#    --epochs 500 \
#    --seed 42 \
#    --save_dir ./logs/
#done

python src/main.py \
--layers 8 \
--cutoff_dim 8 \
--model 21 \
--atom_list debug debug \
--active_sd 0.0001 \
--passive_sd 0.1 \
--epochs 500 \
--learning_rate 0.01 \
--seed 42 \
--save_dir ./logs/