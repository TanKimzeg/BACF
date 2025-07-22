# @Time   : 2022/2/13
# @Author : Hui Yu
# @Email  : ishyu@outlook.com

import os
import torch
import argparse
import numpy as np

from .models import FMLPRecModel
from .trainers import FMLPRecTrainer
from .utils import EarlyStopping, check_path, set_seed, get_local_time, get_seq_dic, get_dataloder, get_rating_matrix

def main(args:argparse.Namespace):
    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    seq_dic, max_item = get_seq_dic(args)

    args.item_size = max_item + 1

    # save model args
    cur_time = get_local_time()
    if args.no_filters:
        args.model_name = "SASRec"
    args_str = f'{args.data_name}'
    args.log_file = os.path.join(args.output_dir, args_str + '.log')
    print(str(args))
    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    # save model
    args.checkpoint_path = os.path.join(args.output_dir, args_str + '.pt')

    train_dataloader, eval_dataloader, test_dataloader = get_dataloder(args,seq_dic)

    model = FMLPRecModel(args=args)
    trainer = FMLPRecTrainer(model, train_dataloader, eval_dataloader,
                              test_dataloader, args)

    if args.full_sort:
        args.valid_rating_matrix, args.test_rating_matrix = get_rating_matrix(args.data_name, seq_dic, max_item)

    if args.do_eval:
        if args.load_model is None:
            print(f"No model input!")
            exit(0)
        else:
            args.checkpoint_path = os.path.join(args.output_dir, args.load_model + '.pt')
            trainer.load(args.checkpoint_path)
            print(f"Load model from {args.checkpoint_path} for test!")
            scores, result_info = trainer.test(0, full_sort=args.full_sort)

    else:
        early_stopping = EarlyStopping(args.checkpoint_path, patience=args.patience, verbose=True)
        for epoch in range(args.epochs):
            trainer.train(epoch)
            scores, _ = trainer.valid(epoch, full_sort=args.full_sort)
            # evaluate on MRR
            early_stopping(np.array(scores[-1:]), trainer.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        print("---------------Sample 99 results---------------")
        # load the best model
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.test(0, full_sort=args.full_sort)
        trainer.iteration(
            epoch=0, 
            output=args.data_dir + args.data_name + '_filter.txt',
            dataloader=test_dataloader)

    print(args_str)
    print(result_info)
    with open(args.log_file, 'a') as f:
        f.write(args_str + '\n')
        f.write(result_info + '\n')
