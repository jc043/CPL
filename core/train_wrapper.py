from unicodedata import category
from core.utils import preprocess
from core.data_provider import datasets_factory
import random
import os
import numpy as np
import core.trainer as trainer
def schedule_sampling(args, eta, itr):
    zeros = np.zeros((args.batch_size,
                      args.total_length - args.input_length - 1,
                      args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    if not args.scheduled_sampling:
        return 0.0, zeros

    if itr < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate
    else:
        eta = 0.0
    random_flip = np.random.random_sample(
        (args.batch_size, args.total_length - args.input_length - 1))
    true_token = (random_flip < eta)
    ones = np.ones((args.img_width // args.patch_size,
                    args.img_width // args.patch_size,
                    args.patch_size ** 2 * args.img_channel))
    zeros = np.zeros((args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.total_length - args.input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                           (args.batch_size,
                            args.total_length - args.input_length - 1,
                            args.img_width // args.patch_size,
                            args.img_width // args.patch_size,
                            args.patch_size ** 2 * args.img_channel))
    return eta, real_input_flag

def train_wrapper(args, model, writer, save_prefix = '', kth_specific_category = None):
    train_log = dict()
    train_log["iter"] = []
    train_log["MSE"] = []
    train_log["SSIM"] = []

    best_model_dir = ""
    best_ave_mse = 1e10
    best_iterations = 0

    if args.pretrained_model:
        model.load(args.pretrained_model)
    # load data
    train_input_handle, test_input_handle = datasets_factory.data_provider(
        args.dataset_name, args.train_data_paths, args.valid_data_paths, args.batch_size, args.img_width,
        seq_length=args.total_length, is_training=True, kth_specific_category = kth_specific_category)

    eta = args.sampling_start_value

    for itr in range(1, args.max_iterations + 1):
        if train_input_handle.no_batch_left():
            train_input_handle.begin(do_shuffle=True)
        ims,train_category = train_input_handle.get_batch()
        ims = preprocess.reshape_patch(ims, args.patch_size)
        eta, real_input_flag = schedule_sampling(args, eta, itr)

        loss = trainer.train(model, ims, real_input_flag, args, itr,train_category)
        if itr % args.display_interval == 0:
            writer.add_scalar("metrics/loss", loss, global_step=itr)

        if itr % args.snapshot_interval == 0:
            print("Model saving")
            model.save(itr, save_prefix)

        if itr % args.test_interval == 0:
            record = trainer.test(model, test_input_handle, args, itr)
            train_log[itr] = record
            train_log["iter"].append(itr)
            train_log["MSE"].append(record["avg_mse_per_seq"])
            train_log["SSIM"].append(record["avg_ssim_per_seq"])
            writer.add_scalar("metrics/MSE", record["avg_mse_per_seq"], global_step=itr)
            writer.add_scalar("metrics/SSIM", record["avg_ssim_per_seq"], global_step=itr)

            # In order to make the training process complete without being constrained by the pretained model
            if itr >= args.sampling_stop_iter - args.early_stopping_interval * args.test_interval / 2:
                if record["avg_mse_per_seq"] < best_ave_mse:
                    best_ave_mse = record["avg_mse_per_seq"]
                    best_iterations = itr
                    best_model_dir = os.path.join(args.save_dir, save_prefix + 'model.ckpt' + '-' + str(itr))

                if args.early_stopping and best_iterations + args.early_stopping_interval * args.test_interval <= itr:
                    print("Early stopping in the iteration:", itr)
                    print("Best average mse is ", best_ave_mse)
                    print("Best model dir is ", best_model_dir)
                    break

        train_input_handle.next()

    if args.use_optimal_model:
        model.load(best_model_dir)
        model.save("Best")

    return train_log
def CPL_train_wrapper(args, model, pre_model,writer, save_prefix = '', kth_specific_category = None,category=None):
    train_log = dict()
    train_log["iter"] = []
    train_log["MSE"] = []
    train_log["SSIM"] = []

    best_model_dir = ""
    best_ave_mse = 1e10
    best_iterations = 0

    if args.pretrained_model:
        model.load(args.pretrained_model)
    # load data
    train_input_handle, test_input_handle = datasets_factory.data_provider(
        args.dataset_name, args.train_data_paths, args.valid_data_paths, args.batch_size, args.img_width,
        seq_length=args.total_length, is_training=True, kth_specific_category = kth_specific_category)

    eta = args.sampling_start_value
    for itr in range(1, args.max_iterations + 1):
        if train_input_handle.no_batch_left():
            train_input_handle.begin(do_shuffle=True)
        ims,train_category = train_input_handle.get_batch()
        ims = preprocess.reshape_patch(ims, args.patch_size)
        eta, real_input_flag = schedule_sampling(args, eta, itr)
        if category>0:
            if args.is_replay:
                if itr%args.replay_interval==0:
                    replay_cat=random.randint(0,category-1)
                    loss = trainer.CPL_train(model,pre_model, ims, real_input_flag, args, itr,replay_cat,is_replay=True)
        loss = trainer.CPL_train(model,pre_model, ims, real_input_flag, args, itr,category)
        if itr % args.display_interval == 0:
            writer.add_scalar("metrics/loss", loss, global_step=itr)

        if itr % args.snapshot_interval == 0:
            print("Model saving")
            model.save(itr, save_prefix)

        if itr % args.test_interval == 0:
            record = trainer.test(model, test_input_handle, args, itr,category)
            train_log[itr] = record
            train_log["iter"].append(itr)
            train_log["MSE"].append(record["avg_mse_per_seq"])
            train_log["SSIM"].append(record["avg_ssim_per_seq"])
            writer.add_scalar("metrics/MSE", record["avg_mse_per_seq"], global_step=itr)
            writer.add_scalar("metrics/SSIM", record["avg_ssim_per_seq"], global_step=itr)

            # In order to make the training process complete without being constrained by the pretained model
            if itr >= args.sampling_stop_iter - args.early_stopping_interval * args.test_interval / 2:
                if record["avg_mse_per_seq"] < best_ave_mse:
                    best_ave_mse = record["avg_mse_per_seq"]
                    best_iterations = itr
                    best_model_dir = os.path.join(args.save_dir, save_prefix + 'model.ckpt' + '-' + str(itr))

                if args.early_stopping and best_iterations + args.early_stopping_interval * args.test_interval <= itr:
                    print("Early stopping in the iteration:", itr)
                    print("Best average mse is ", best_ave_mse)
                    print("Best model dir is ", best_model_dir)
                    break

        train_input_handle.next()

    if args.use_optimal_model:
        model.load(best_model_dir)
        model.save("Best")

    return train_log

