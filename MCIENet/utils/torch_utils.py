import torch
import psutil
import random
import platform

import numpy as np

from .helper_func import log_string

# def save_model(args, model):
#     if args.use_tracedmodule:
#         traced = torch.jit.trace(model.cpu(), torch.rand(1, 2, 4, 3000))
#         traced.save(args.model_file)
#     else:
#         torch.save(model, args.model_file)

# def load_model(args):
#     if args.use_tracedmodule:
#         model = torch.jit.load(args.model_file)
#     else:
#         model = torch.load(args.model_file)
    
#     return model

# 重現實驗用 ===================================================
def set_seed(seed=42):
    random.seed(seed) 
    np.random.seed(seed)  
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True        


    g = torch.Generator()
    g.manual_seed(seed)

    return g

def seed_worker():
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ==============================================================

def log_system_info(args, log):
    message = f"Computer network name: {platform.node()}\n"+ \
                f"Machine type: {platform.machine()}\n" + \
                f"Processor type: {platform.processor()}\n" + \
                f"Platform type: {platform.platform()}\n" + \
                f"Number of physical cores: {psutil.cpu_count(logical=False)}\n" + \
                f"Number of logical cores: {psutil.cpu_count(logical=True)}\n" + \
                f"Max CPU frequency: {psutil.cpu_freq().max if psutil.cpu_freq() else 'unknow'}\n"

    cuda_divice = torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'
    message += f'Train with the {args.device}({cuda_divice})\n'
    log_string(log, '\n[System Info]\n' + message + '='*20)