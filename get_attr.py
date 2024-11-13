"""
Author: yen-nan ho
Contact: aaron1aaron2@gmail.com
Create Date: 2024.10.13
Last Update: 2024.10.13
"""
import os
import yaml
import h5py
import pathlib
import argparse
import warnings
warnings.filterwarnings("ignore")

from typing import Tuple
from tqdm.auto import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from captum.attr import IntegratedGradients, Saliency, DeepLift, InputXGradient, GuidedBackprop, Deconvolution


from MCIENet.dataset import MyDataset
from MCIENet.loop_model import LoopModel
from MCIENet.utils.torch_utils import *
from MCIENet.utils.helper_func import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_folder', type=str, 
                        default='output/best/MCIENet-gm12878.ctcf-1kb')
    parser.add_argument('--output_folder', type=str, 
                        default='output/XAI/MCIENet-gm12878.ctcf-1kb')
    parser.add_argument('--data_folder', type=str, 
                        default='data/train/gm12878_ctcf/3000bp.50ms.onehot')
    parser.add_argument('--phases', type=str, nargs='+', 
                        default=['train', 'val', 'test'])
    parser.add_argument('--batch_size', type=int, default=200)

    parser.add_argument('--save_attr', type=str2bool, default=True)
    parser.add_argument('--save_input', type=str2bool, default=True)
    parser.add_argument('--method', type=str, default='DeepLift')

    parser.add_argument('--crop_center', type=int, default=1500)
    parser.add_argument('--crop_size', type=int, default=1000)

    parser.add_argument('--use_cuda', type=str2bool, default=False)

    args = parser.parse_args()

    args.model_path = f'{args.model_folder}/model.pkl'
    args.config_path = f'{args.model_folder}/configures.yaml'

    args.data_path = f'{args.data_folder}/data.h5'

    args.device = 'cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu'
    # build folder
    os.makedirs(args.output_folder, exist_ok=True)

    return args

def concat_anchor(x: torch.Tensor) -> torch.Tensor:
    return torch.concat((x[:, 0, :, :], x[:, 1, :, :]), dim=-1)

def crop_anchor(x: torch.Tensor, clip_center, size) -> torch.Tensor:
    '''[Batch size, 2, 4, bp]'''
    _, _, _, length = x.shape

    if size == length:
        return x
    else:
        half_size = round(size/2)
        start = clip_center - half_size
        end = clip_center + half_size

        assert (start >= 0) and (end < length), 'Crop size exceeds length'
        
        result = x[:, :, :, start:end]

        return result



def load_data(data_path: str, batch_size: int) -> dict[str, DataLoader]:
    """
    Load data from an h5py file and create data loaders for training, validation, and testing.

    Args:
        args: The command line arguments containing the path to the h5py file.
        log: The logger object used for logging information.
        eval_stage: A flag indicating whether the function is called during the evaluation stage.

    Returns:
        data_loader_dt: A dictionary containing the data loaders for training, validation, and testing.

    """
    f = h5py.File(data_path)

    train_data = MyDataset(f['train']['data'], f['train']['labels'])
    val_data = MyDataset(f['val']['data'], f['val']['labels'])
    test_data = MyDataset(f['test']['data'], f['test']['labels'])

    print(f"data shape:\n")
    print(f"\t(train){f['train']['data'].shape}\n")
    print(f"\t(val){f['val']['data'].shape}\n")
    print(f"\t(test){f['test']['data'].shape}\n")

    g = set_seed()

    data_loader_dt = {
        'train': DataLoader(train_data, batch_size=batch_size, shuffle=False, 
                            num_workers=0), 
        'val': DataLoader(val_data, batch_size=batch_size, shuffle=False, 
                            num_workers=0), 
        'test': DataLoader(test_data, batch_size=batch_size, shuffle=False, 
                            num_workers=0), 
    }    


    return data_loader_dt

def load_model(config_path: str, model_path: str, device:str) -> nn.Module:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['model'] = {k:v for k,v in config['model'].items() if k.find('(defualt)') == -1}

    model = LoopModel(config['data']['input_type'], config['model'])

    if model_path[-4:] == '.pth':
        model.load_state_dict(torch.load(model_path)) # state_dict 存取
    else:
        model = torch.load(model_path) # Entire model 存取


    # 假設 model 是你的 PyTorch 模型，已經訓練好
    model.eval()
    model.to(device)

    return model

class output_hander:
    def __init__(self, path: str, group_ls: list, overlap: bool=False):
        super().__init__()
        if overlap:
            pathlib.Path(path).unlink(missing_ok=True)
        
        self.f = self._creat_out_hander(path, group_ls)
        self.groups = group_ls

    def _creat_out_hander(self, path: str, group_ls: list) -> h5py.File:
        f = h5py.File(path, "w")

        for name in group_ls:
            if name not in f:
                f.create_group(name)

        return f

    def add_data(self, group: str, name: str, data: torch.Tensor, data_type: str):
        input_shape = data.shape
        if isinstance(data, torch.Tensor):
            data = data.cpu().detach().numpy()
        if name not in self.f[group]:
            maxshape = list(input_shape)
            maxshape[0] = None

            self.f[group].create_dataset(
                        name,
                        data=data,
                        dtype=data_type,
                        chunks=True,
                        compression="gzip",
                        maxshape=maxshape
                    )

        else:
            tar = self.f[group][name]
            tar.resize((tar.shape[0] + data.shape[0]), axis=0)
            tar[-data.shape[0] :] = data

def get_attr(model: nn.Module, input_tensor: torch.Tensor, device: str,
             method: str='IntegratedGradients') -> Tuple[torch.Tensor, torch.Tensor]:
    # 創建 IntegratedGradients 對象
    if method == 'IntegratedGradients':
        attr_method = IntegratedGradients(model)
    elif method == 'Saliency':
        attr_method = Saliency(model)
    elif method == 'DeepLift':
        attr_method = DeepLift(model)
    elif method == 'InputXGradient':
        attr_method = InputXGradient(model)
    elif method == 'GuidedBackprop':
        attr_method = GuidedBackprop(model)
    elif method == 'Deconvolution':
        attr_method = Deconvolution(model)
    else:
        raise ValueError(f'method - {method} not support')

    # 載入一張影像
    input_tensor = torch.Tensor(input_tensor).to(device)

    # 計算歸因
    attribution = attr_method.attribute(input_tensor, target=1)

    # torch.cuda.empty_cache() # 可能會拖慢速度

    # 合併 anchor 維度
    input_tensor = concat_anchor(input_tensor)
    attribution = concat_anchor(attribution)

    return input_tensor, attribution

def main():
    args = get_args()
    print('='*20 + '\n[arguments]\n' + f'{str(args)[10: -1]}')

    path = os.path.join(args.output_folder, "attribution.h5")

    if os.path.exists(path):
        print(f"task finished {path}")
        exit()

    model = load_model(config_path=args.config_path, model_path=args.model_path, device=args.device)    
    
    data = load_data(data_path=args.data_path, batch_size=args.batch_size)

    out_hander = output_hander(path, args.phases)


    for phase in args.phases:
        dataloader = data[phase]
        for batch in tqdm(dataloader):
            
            inputs = batch[0].to(args.device).float()
            labels = batch[1].to(args.device).long()
            
            # crop anchor
            inputs = crop_anchor(inputs, args.crop_center, args.crop_size)

            # model prediction
            output = model(inputs)
            output = nn.functional.softmax(output, dim=1)

            # add to db
            out_hander.add_data(phase, 'label', labels, 'uint8')
            out_hander.add_data(phase, 'pred', output, 'float32')

            if args.save_input:
                out_hander.add_data(phase, 'input', inputs, 'uint8')
            if args.save_attr:
                inputs, attr = get_attr(model=model,input_tensor=inputs, device=args.device, method=args.method)
                out_hander.add_data(phase, 'attr', attr, 'float32')

    
    out_hander.f.close()


if __name__ == '__main__':
    main()