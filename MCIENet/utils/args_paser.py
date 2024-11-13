import re
import os
import yaml
import torch
import argparse

from typing import Tuple, TextIO

from .helper_func import log_string

def build_args(args, unknown_args: list) -> Tuple[argparse.Namespace, TextIO]:

    # build output folder & log
    os.makedirs(args.output_folder, exist_ok=True)
    log = open(os.path.join(args.output_folder, 'log.txt'), 'w', encoding='utf8')

    # 新增參數
    args.device = 'cuda' if torch.cuda.is_available() and args.device in ['gpu', 'cuda'] else 'cpu'

    args.model_file = os.path.join(args.output_folder, 
        'model.pth' if args.use_state_dict else 'model.pkl'
    )

    # args.fig_folder = os.path.join(args.output_folder, 'figure')

    unknown_dt = dict(zip(unknown_args[::2], unknown_args[1::2]))

    # load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    log_string(log, f"[update config]\n")
    unknow_ls = []
    for k,v in unknown_dt.items():
        k = re.sub('^--', '', k)
        re_search = re.findall('^(train|model|data)\.(.+)', k)
        if len(re_search) == 1:
            group, sub_arg = re_search[0]
            config_set = {**config['train'], **config['model'], **config['data']}
            if sub_arg in config_set:
                old_arg = config[group][sub_arg]
                # Convert string to bool (if possible)
                if isinstance(v, str):
                    v = str2bool_soft(v)

                # Convert the passed parameter to the same type as the original parameter
                if isinstance(old_arg, list):
                    new_args = [type(old_arg[0])(i) for i in v.split(',')]
                else:
                    new_args = type(old_arg)(v)

                # Update the parameters if they are different from the default parameters
                if old_arg != new_args:
                    log_string(log, f"{group}.{sub_arg}: {old_arg} -> {new_args}\n")
                    if args.retain_defualt_config:
                        config[group][sub_arg] = new_args
                        config[group][sub_arg + '(defualt)'] = old_arg
            else:
                unknow_ls.append(f'{k}: {v}')
        else:
            unknow_ls.append(f'{k}: {v}')
    
    # Convert all to lowercase
    for args_set, set_dt in config.items():
        if isinstance(set_dt, dict):
            for k, v in set_dt.items():
                if isinstance(v, str):
                    config[args_set][k] = config[args_set][k].lower()

    unknow_ls = ' \n'.join(unknow_ls)
    if len(unknow_ls) != 0:
        log_string(log, f"[unknow config]\n{unknow_ls}\n")

    args.data = config['data']
    args.train = config['train']
    args.model = config['model']

    return args, log

def str2bool_soft(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        return v.lower()