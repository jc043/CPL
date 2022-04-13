cd ..
python -u run.py \
    --is_training 1 \
    --isCPL 0 \
    --zdim 128 \
    --is_replay 1 \
    --replay_interval 3\
    --dataset_name action \
    --device cuda:0 \
    --train_data_paths /home/gchen/Dataset/kth/ \
    --valid_data_paths /home/gchen/Dataset/kth/ \
    --train_all_kth_actions_log_save_dir logs/KTH/CPL_base/test/ \
    --save_dir checkpoints/KTH/CPL_base/test/ \
    --gen_frm_dir results/KTH/CPL_base/test/ \
    --tensorboard_dir SummaryDir/KTH/CPL_base/test/ \
    --early_stopping_interval 4 \
    --model_name CPL_base \
    --reverse_input 0 \
    --img_channel 1 \
    --img_width 64 \
    --input_length 10 \
    --total_length 20 \
    --filter_size 5 \
    --stride 1 \
    --num_hidden 64 \
    --num_layers 4 \
    --patch_size 4 \
    --layer_norm True \
    --num_samples 1 \
    --kl_beta 0.0001 \
    --layer_norm 1 \
    --sampling_stop_iter 20000 \
    --sampling_start_value 1.0 \
    --sampling_changing_rate 0.00005 \
    --lr 0.0005 \
    --batch_size 8 \
    --max_iterations 30000 \
    --display_interval 100 \
    --test_interval 2000 \
    --snapshot_interval 5000\
    --is_multi_training True\