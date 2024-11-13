"""
Author: 何彥南 (yen-nan ho)
Email: aaron1aaron2@gmail.com
Create Date: 2022.09.05
Last Update: 2022.09.05
Describe: 工具箱
"""
import json
import argparse

import numpy as np

def loadJson(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data
def saveJson(data, path):
    with open(path, 'w', encoding='utf-8') as outfile:  
        json.dump(data, outfile, indent=2, ensure_ascii=False)

def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def match_name(in_arg_dt, name, sup_ls):
    in_arg = in_arg_dt[name]
    if in_arg.lower() not in sup_ls:
        raise ValueError(f"Unrecognized arg: {name}")

    return in_arg.lower()

def pretty_dict(d, decimal=4):
    msg = '' 
    for key, value in d.items():
        value = round(value, decimal)
        msg += f'{key}: {value}\n'

    return msg

def to_prob(nums):
    nums = np.array(nums)
    nums = nums + abs(nums.min())
    nums = nums / nums.sum()
    
    return nums.tolist()

def chrom_to_int(chrom):
    if chrom == "chrX":
        chrom = 23
    else:
        chrom = int(chrom.replace("chr", ""))
    return chrom
