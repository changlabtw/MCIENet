"""
Author: yen-nan ho
Contact: aaron1aaron2@gmail.com
Create Date: 2023.10.10
Last Update: 2024.07.26
"""
import os
import h5py
import yaml

import argparse

import warnings
warnings.filterwarnings("ignore")

from typing import TextIO
from torchinfo import summary

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from MCIENet.utils.args_paser import build_args
from MCIENet.utils.ana_plot import plot_train_val_loss
from MCIENet.utils.helper_func import *
from MCIENet.utils.torch_utils import *


from MCIENet.trainer import train_model, test_model
from MCIENet.dataset import MyDataset
from MCIENet.loop_model import LoopModel


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # 輸入
    parser.add_argument('--config', type=str, default='conf/base-CNN.yaml', help='config 檔路徑')
    parser.add_argument('--input', type=str, default='data/train/gm12878_ctcf/1000bp.50ms/data.h5')

    # 輸出
    parser.add_argument('--output_folder', type=str, default='output/test_sample')

    # 其他
    parser.add_argument('--device', default='gpu', 
                        help='cpu or cuda')
    parser.add_argument('--eval_freq', type=int, default=1, 
                        help='How often to evaluate during training')

    parser.add_argument('--pin_memory_train', type=str2bool, default=True, 
                        help='argument under torch.utils.data.DataLoader')
    parser.add_argument('--save_epoch_out', type=str2bool, default=False)
    parser.add_argument('--use_state_dict', type=str2bool, default=False)

    parser.add_argument('--retain_defualt_config', type=str2bool, default=True)
    parser.add_argument('--save_pred_result', type=str2bool, default=False)

    args, unknown_args = parser.parse_known_args() # accept unkonw arg

    if os.path.exists(os.path.join(args.output_folder, 'evaluation.json')):
        print(f"task finished {args.output_folder}")
        exit()

    return args, unknown_args

def check_args(args) -> argparse.Namespace:
    with open(os.path.join(args.output_folder, 'configures.yaml'), 'w') as f:
        yaml.dump(args.__dict__, f, sort_keys=False)

    remove_defult = lambda dt: {k:v for k,v in dt.items() if not k.endswith('(defualt)')}
    
    args.data = remove_defult(args.data)
    args.train = remove_defult(args.train)
    args.model = remove_defult(args.model)

    return args

def load_data(args, log:TextIO, eval_stage:bool) -> dict[str, DataLoader]:
    """
    Load data from an h5py file and create data loaders for training, validation, and testing.

    Args:
        args: The command line arguments containing the path to the h5py file.
        log: The logger object used for logging information.
        eval_stage: A flag indicating whether the function is called during the evaluation stage.

    Returns:
        data_loader_dt: A dictionary containing the data loaders for training, validation, and testing.

    """
    f = h5py.File(args.input)

    train_data = MyDataset(f['train']['data'], f['train']['labels'])
    val_data = MyDataset(f['val']['data'], f['val']['labels'])
    test_data = MyDataset(f['test']['data'], f['test']['labels'])

    log_string(log, f"data shape:\n")
    log_string(log, f"\t(train){f['train']['data'].shape}\n")
    log_string(log, f"\t(val){f['val']['data'].shape}\n")
    log_string(log, f"\t(test){f['test']['data'].shape}\n")

    g = set_seed()

    if eval_stage:
        train_loader = DataLoader(train_data, batch_size=args.train['val_batch_size'], shuffle=False, 
                            num_workers=0, pin_memory=args.pin_memory_train,
                            worker_init_fn=seed_worker, generator=g)
    else:
        train_loader = DataLoader(train_data, batch_size=args.train['batch_size'], shuffle=True, 
                            num_workers=0, pin_memory=args.pin_memory_train,
                            worker_init_fn=seed_worker, generator=g)


    data_loader_dt = {
        'train': train_loader, 
        'val': DataLoader(val_data, batch_size=args.train['val_batch_size'], shuffle=False, 
                            num_workers=0), 
        'test': DataLoader(test_data, batch_size=args.train['val_batch_size'], shuffle=False, 
                            num_workers=0), 
    }    


    return data_loader_dt



if __name__ == '__main__':
    # 參數
    args, unknown_args = get_args()

    args, log = build_args(args, unknown_args)
    args = check_args(args)

    log_string(log, '='*20 + '\n[arguments]\n' + f'{str(args)[10: -1]}')
    log_system_info(args, log)

    # load data >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    log_string(log, 'loading data...')

    data = load_data(args, log, eval_stage=False)

    log_string(log, f'data loaded!\n' + '='*20)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # build model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    log_string(log, 'compiling model...')
    
    model = LoopModel(args.data['input_type'], args.model)
    model = model.to(args.device)
    log_string(log, str(model))
    log_string(log, "="*10)


    saveJson(model.get_parm_info(), os.path.join(args.output_folder, 'parms_count.json'))


    if args.data['input_type'] == 'onehot':
        input_shape = (args.train['batch_size'], 2, 4, args.data['anchor_size'])
    else:
        input_shape = (args.train['batch_size'], 2, args.data['anchor_size'])

    log_string(log, str(
        summary(
        model, input_shape,
        verbose=1,
        col_width=16,
        col_names=["input_size", "output_size", "num_params", "params_percent",
                "kernel_size", "mult_adds", "trainable"
                ],
        row_settings=["var_names", "depth", "ascii_only"],
        )
    ))

    # loss function -------------------------
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss(pos_weight=weights[1:])

    # optimizer -----------------------------
    if args.train['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.train['learning_rate'])
    elif args.train['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.train['learning_rate'])
    elif args.train['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.train['learning_rate'])
    else:
        raise ValueError("optimizer only supports SGD, Adam, AdamW")

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                        step_size=args.train['decay_epoch'],
                                        gamma=args.train['decay_rate'],
                                        verbose=True)
    

    log_string(log, 'model loaded!\n' + '='*20)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # train model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    log_string(log, 'training model...')

    result = train_model(
        args, log, model, data, criterion, optimizer, scheduler,
        num_epochs=args.train['max_epoch'], patience=args.train['patience']
        )

    plot_train_val_loss(
        train_total_loss=[i['train']['loss'] for i in result], 
        val_total_loss=[i['val']['loss'] for i in result],
        file_path=os.path.join(args.output_folder, 'train_val_loss.png')
        )

    log_string(log, 'training finish\n')
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # test model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    log_string(log, 'calculating evaluation...')
    data = load_data(args, log, eval_stage=True)
    test_model(args, log, model, data, criterion)
    log_string(log, 'finished!!!\n')
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<