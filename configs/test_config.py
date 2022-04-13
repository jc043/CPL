import argparse

def parser():
    parser = argparse.ArgumentParser(description='CPL')

    # training/test
    parser.add_argument('--is_training', type=int, default=1,help='training mode')
    parser.add_argument('--device', type=str, default='cpu:0')
    # Early stopping configuration
    parser.add_argument('--use_optimal_model', type=bool, default=False)
    parser.add_argument('--early_stopping', type=bool, default=False)
    parser.add_argument('--early_stopping_interval', type=int, default=4)

    # Continual training / testing
    parser.add_argument('--is_multi_training', type=bool, default=False,help='if true, train in continual setup; else, do joint training')
    parser.add_argument('--is_multi_testing', type=bool, default=False)
    parser.add_argument('--is_reverse_tasks', type=bool, default=False)

    # multi testing save path
    parser.add_argument('--test_kth_actions_save_path', type=str, default=None)
    parser.add_argument('--train_all_kth_actions_log_save_dir', type=str, default=None)
    parser.add_argument('--tensorboard_dir', type=str, default="SummaryDir/tmp")

    # CPL trainning
    parser.add_argument("--isCPL", type=int, default=0)
    parser.add_argument("--is_TestAdapt", type=int, default=0)
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--zdim', type=int, default=10)
    parser.add_argument("--is_replay", type=int, default=1)
    parser.add_argument("--replay_interval", type=int, default=3)
    parser.add_argument('--kl_beta', type=float, default=0.0001)

    # data
    parser.add_argument('--dataset_name', type=str, default='action',help='dataset, default:action(KTH)')
    parser.add_argument('--train_data_paths', type=str, default='Dataset/kth/')
    parser.add_argument('--valid_data_paths', type=str, default='Dataset/kth/')
    parser.add_argument('--save_dir', type=str, default='checkpoints/CPL')
    parser.add_argument('--gen_frm_dir', type=str, default='results/CPL')
    parser.add_argument('--input_length', type=int, default=10)
    parser.add_argument('--total_length', type=int, default=20)
    parser.add_argument('--img_width', type=int, default=64)
    parser.add_argument('--img_channel', type=int, default=1)

    # model
    parser.add_argument('--model_name', type=str, default='CPL')
    parser.add_argument('--pretrained_model', type=str, default='')
    parser.add_argument('--num_hidden', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--filter_size', type=int, default=5)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--layer_norm', type=bool, default=True)

    # reverse scheduled sampling
    parser.add_argument('--reverse_scheduled_sampling', type=int, default=0)
    parser.add_argument('--r_sampling_step_1', type=float, default=25000)
    parser.add_argument('--r_sampling_step_2', type=int, default=50000)
    parser.add_argument('--r_exp_alpha', type=int, default=5000)
    # scheduled sampling
    parser.add_argument('--num_samples', type=int, default=30)
    parser.add_argument('--scheduled_sampling', type=int, default=1)
    parser.add_argument('--sampling_stop_iter', type=int, default=50000)
    parser.add_argument('--sampling_start_value', type=float, default=1.0)
    parser.add_argument('--sampling_changing_rate', type=float, default=0.00002)

    # optimization
    parser.add_argument('--lr', type=float, default=0.001)
    
    parser.add_argument('--reverse_input', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_iterations', type=int, default=80000)
    parser.add_argument('--display_interval', type=int, default=100)
    parser.add_argument('--test_interval', type=int, default=5000)
    parser.add_argument('--snapshot_interval', type=int, default=5000)
    parser.add_argument('--num_save_samples', type=int, default=10)
    parser.add_argument('--n_gpu', type=int, default=1)

    # visualization of memory decoupling
    parser.add_argument('--visual', type=int, default=0)
    parser.add_argument('--visual_path', type=str, default='./decoupling_visual')
    
    return parser