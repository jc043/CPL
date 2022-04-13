import os
from re import M
import shutil
import argparse
from unicodedata import category
import numpy as np
import torch
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle as pkl
from core.data_provider import datasets_factory
from core.model.model_factory import Model
import core.trainer as trainer

from core.train_wrapper import train_wrapper,CPL_train_wrapper
from configs.test_config import parser
from tensorboardX import SummaryWriter



# KTH
args = parser().parse_args()
print(args)



def test_wrapper(model):
    model.load(args.pretrained_model)
    test_input_handle = datasets_factory.data_provider(
        args.dataset_name, args.train_data_paths, args.valid_data_paths, args.batch_size, args.img_width,
        seq_length=args.total_length, is_training=False)
    trainer.test(model, test_input_handle, args, 'test_result')

"""
Record the MSE and SSIM infomation to display the trainning  
"""
def display(log, img_suffix = '', file_suffix = 'train_log.pkl', given_save_dir = None):
    save_dir = args.save_dir
    if given_save_dir != None:
        save_dir = given_save_dir
    itr = log["iter"]
    MSE = log["MSE"]
    SSIM = log["SSIM"]

    plt.plot(itr, MSE, label="MSE")
    plt.title("MSE in different iterations")
    plt.xlabel("iterations")
    plt.ylabel("MSE per sequence")
    plt.savefig(os.path.join(save_dir, 'MSE' + img_suffix + '.jpg' ))

    plt.clf()
    plt.plot(itr, SSIM, label="SSIM")
    plt.title("SSIM in different iterations")
    plt.xlabel("iterations")
    plt.ylabel("SSIM per sequence")
    plt.savefig(os.path.join(save_dir, 'SSIM' + img_suffix + '.jpg'))
    plt.clf()

    file_name = open(os.path.join(save_dir, file_suffix), "wb")
    pkl.dump(log, file_name)


def train_all_kth_actions(model, pre_model=None):

    categories = ['boxing', 'handclapping', 'handwaving', 'walking', 'jogging', 'running']
    # One stable - one move  mode
    # categories = ['boxing', 'walking', 'handclapping', 'jogging', 'handwaving', 'running']
    train_log_list = dict()
    test_log_list = dict()

    num_categories = len(categories)

    for i, item in enumerate(categories):


        tensorboard_dir_path = args.tensorboard_dir + item
        writer = SummaryWriter(tensorboard_dir_path)

        print("---------------Begin Training: Order = ", i + 1 , " ; Categary = ", item, "------------------------")
        if not args.isCPL:
            train_log = train_wrapper(args, model, writer = writer,
                        save_prefix=item + '-', kth_specific_category = [item])
        else:
            train_log = CPL_train_wrapper(args, model, pre_model,writer = writer,
                        save_prefix=item + '-', kth_specific_category = [item],category=i)
            # print("Pre_model no grad")
            pre_model.load_state_dict(copy.deepcopy(model.state_dict()))
            for n, p in pre_model.named_parameters():
                p.requires_grad = False
        train_log_list[item] = train_log

        model.save(categories[i])
        display(train_log, img_suffix='_trainning_' + item + '_process',
                file_suffix='trainning_' + item + '.pkl',
                given_save_dir=args.train_all_kth_actions_log_save_dir)

        test_log = test_all_kth_actions(model, False)
        test_log_list[item] = test_log
        print(test_log)

        display(test_log, img_suffix='_trainning_' + item,
                file_suffix='testing_' + item + '.pkl',
                given_save_dir=args.train_all_kth_actions_log_save_dir)

        for j in range(len(test_log["iter"])):
            writer.add_scalar("categories/MSE", test_log["MSE"][j], global_step=j)
            writer.add_scalar("categories/SSIM", test_log["SSIM"][j], global_step=j)
        writer.export_scalars_to_json(os.path.join(args.tensorboard_dir, "all_datas.json"))
        writer.close()

        print("---------------Finsih Training: Order = ", i + 1 , " ; Categary = " ,item, "-------------------------")
        print(end='\n\n\n\n\n')

    # Show the metrics trend of the different categories
    tensorboard_dir_path = args.tensorboard_dir + "Results"
    writer = SummaryWriter(tensorboard_dir_path)
    for i, item in enumerate(categories):
        test_log = test_log_list[item]
        mse_dict = dict()
        ssim_dict = dict()
        for j, jtem in enumerate(test_log["iter"]):
            mse_dict[jtem] = test_log["MSE"][j]
            ssim_dict[jtem] = test_log["SSIM"][j]
        writer.add_scalars("/MSE_trends", mse_dict, global_step=i)
        writer.add_scalars("/SSIM_trends", ssim_dict, global_step=i)
    writer.export_scalars_to_json(os.path.join(args.tensorboard_dir, "Categries_compare_datas.json"))
    writer.close()

    # Train log save as the train_log_list.pkl 
    log_file_name = "train_log_list.pkl"
    f = open(os.path.join(args.train_all_kth_actions_log_save_dir, log_file_name), "wb")
    pkl.dump(train_log_list, f)
    print("train_log_list : ", train_log_list, end="\n\n\n")

    # Test log save as the test_log_list.pkl 
    log_file_name = "test_log_list.pkl"
    f = open(os.path.join(args.train_all_kth_actions_log_save_dir, log_file_name), "wb")
    pkl.dump(test_log_list, f)
    print("test_log_list : ", test_log_list, end="\n\n\n")


def test_all_kth_actions(model, is_loading_pretrain = True):
    if is_loading_pretrain:
        model.load(args.pretrained_model)


    categories = ['boxing', 'handclapping', 'handwaving', 'walking', 'jogging', 'running']
    # One stable - one move  mode
    # categories = ['boxing', 'walking', 'handclapping', 'jogging', 'handwaving', 'running']
    test_log = dict()
    test_log["iter"] = []
    test_log["MSE"] = []
    test_log["SSIM"] = []
   
    for i, item in enumerate(categories):
        print(f"Testing iteration: {i}, Testing categories: {item}")
        test_input_handle = datasets_factory.data_provider(
            args.dataset_name, args.train_data_paths, args.valid_data_paths, args.batch_size, args.img_width,
            seq_length=args.total_length, is_training=False, kth_specific_category=[item])
        record = trainer.test(model, test_input_handle, args, 'test_result' + str(item),i)

        test_log["iter"].append(categories[i])
        test_log["MSE"].append(record["avg_mse_per_seq"])
        test_log["SSIM"].append(record["avg_ssim_per_seq"])


    test_log["iter"].append("Mean")
    test_log["MSE"].append(np.mean(test_log["MSE"]))
    test_log["SSIM"].append(np.mean(test_log["SSIM"]))
    print(test_log)
    return test_log


if os.path.exists(args.save_dir):
    shutil.rmtree(args.save_dir)
os.makedirs(args.save_dir)

if os.path.exists(args.gen_frm_dir):
    shutil.rmtree(args.gen_frm_dir)
os.makedirs(args.gen_frm_dir)

if args.train_all_kth_actions_log_save_dir != None:
    if os.path.exists(args.train_all_kth_actions_log_save_dir):
        shutil.rmtree(args.train_all_kth_actions_log_save_dir)
    os.makedirs(args.train_all_kth_actions_log_save_dir)



print('Initializing models')

model = Model(args)
pre_model = Model(args)



if args.dataset_name == "action":
    if args.is_training:
        if args.is_multi_training:
            train_all_kth_actions(model, pre_model)
        else:
            writer = SummaryWriter(args.tensorboard_dir)
            train_log = train_wrapper(args, model, writer=writer)
            display(train_log)
            writer.export_scalars_to_json(os.path.join(args.tensorboard_dir, "all_datas.json"))
            writer.close()
    else:
        assert args.pretrained_model != ''
        if args.is_multi_testing:
            test_log = test_all_kth_actions(model)
            display(test_log, file_suffix="multi_testing_log.pkl",
                    given_save_dir=args.test_kth_actions_save_path)
        else:
            test_wrapper(model)