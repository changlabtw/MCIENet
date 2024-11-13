import torch
import torch.nn as nn

from typing import Any, List
from collections import OrderedDict

from .utils import get_activite_func

class CNNLayer(nn.Module):
    def __init__(self, 
                 in_ch: int,
                 out_ch: int, 
                 conv_kernel_size: int, 
                 conv_stride: int, 
                 mp_kernel_size: int, 
                 mp_stride: int, 
                 pool_type: str, 
                 activite_func:str, 
                 slope: float, 
                 dropout: float, 
                 bn: bool, 
                 bn_eps: float, 
                 bn_momentum: float
                 ):
        super().__init__()

        self.bn = bn
        self.dropout = dropout

        self.conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, 
                              kernel_size=conv_kernel_size, stride=conv_stride)

        if bn:
            self.bn = nn.BatchNorm1d(int(out_ch), eps=bn_eps, momentum=bn_momentum)

        self.activite_func = get_activite_func(activite_func, slope=slope)

        if dropout != 0:
            self.dropout = nn.Dropout(dropout)

        if pool_type == 'max':
            self.pool = nn.MaxPool1d(kernel_size=mp_kernel_size, stride=mp_stride)
        elif pool_type == 'avg':
            self.pool = nn.AvgPool1d(kernel_size=mp_kernel_size, stride=mp_stride)

    def forward(self, x):
        x = self.conv(x)

        if self.bn:
            x = self.bn(x)

        x = self.activite_func(x)

        if self.dropout != 0:
            x = self.dropout(x)

        x = self.pool(x)

        return x

class BasicConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float, 
                 activite_func: str = 'leaky_relu', slope: float = 0.01, **kwargs: Any) -> None:
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, bias=False, **kwargs)
        self.activite_func = get_activite_func(activite_func, slope=slope)
        if dropout != 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = 0

    def forward(self, x):
        out = self.conv(x)
        out = self.activite_func(out)
        if self.dropout != 0:
            x = self.dropout(x)

        return out

class InceptionLayer(nn.Module):
    def __init__(
            self,
            input_lenght: int,
            in_channels: int,
            total_channels: int,
            info_retent_chs: float,
            info_ext_chs_ls: List[float], 
            info_ext_ks_ls: List[int], 
            info_ext_dilation_ls: List[int],
            pool_proj_chs_ls: List[float],
            pool_proj_ks_ls: List[int],
            pool_proj_dilation_ls: List[int],
            pool_proj_type_ls: List[int],
            dropout: float,
            activite_func: str,
            slope: float,
            dim_reduction_pool_ks: int,
            dim_reduction_pool_stride: int,
            allow_unblance_padding: bool = False
            ):
        super(InceptionLayer, self).__init__()

        # check args
        assert len(info_ext_chs_ls) == len(info_ext_ks_ls) == len(info_ext_dilation_ls)
        assert len(pool_proj_chs_ls) == len(pool_proj_ks_ls) == len(pool_proj_dilation_ls) == len(pool_proj_type_ls)
        assert round(sum([info_retent_chs] + info_ext_chs_ls + pool_proj_chs_ls), 2) == 1, \
            'The following channels size related parameters are configured incorrectly, the total probability should be 1\n' + \
            '   info_retent_chs, info_ext_chs_ls, pool_proj_chs_ls'

        # prepare
        basic_parms = {
            'dropout': dropout, 'activite_func': activite_func, 'slope': slope
        }

        self.out_size = int((input_lenght - dim_reduction_pool_ks) / dim_reduction_pool_stride + 1)

        # branch1: Information retention branch
        branch1_chs = int(total_channels*info_retent_chs)
        if branch1_chs != 0:
            self.info_retent_branch = BasicConv1d(in_channels, branch1_chs, kernel_size=1, stride=1, padding=0, **basic_parms)
        else:
            self.info_retent_branch = None

        # branch2: Information extraction branch
        info_ext_block_ls = list(zip(info_ext_chs_ls, info_ext_ks_ls, info_ext_dilation_ls))
        self.info_ext_branchs = nn.ModuleList()
        for chs, ks, dilation in info_ext_block_ls:
            padding = self.__count_padding_size(input_lenght, dilation, ks, 1)
            cur_chs = int(total_channels*chs)
            self.info_ext_branchs.append(
                nn.Sequential(OrderedDict([
                    ('channel_adjust_layer', BasicConv1d(in_channels, cur_chs, kernel_size=1, stride=1, padding=0, **basic_parms)),
                    ('info_extract_layer', BasicConv1d(cur_chs, cur_chs, kernel_size=ks, stride=1, dilation=dilation, padding=padding, **basic_parms))
                    ]))
            )

        # branch3: Pooling projection branch
        pool_proj_block_ls = list(zip(pool_proj_chs_ls, pool_proj_ks_ls, pool_proj_dilation_ls, pool_proj_type_ls))
        self.pool_proj_branchs = nn.ModuleList()

        for chs, ks, dilation, typ in pool_proj_block_ls:
            padding = self.__count_padding_size(input_lenght, dilation, ks, 1)
            if typ == 'maxpool':
                pool = nn.MaxPool1d(kernel_size=ks, stride=1, padding=padding, dilation=dilation, ceil_mode=True)
            elif typ == 'avgpool':
                assert dilation == 1, "Can't use dilation when using avgpool"
                pool = nn.AvgPool1d(kernel_size=ks, stride=1, padding=padding, ceil_mode=True)
            else:
                raise ValueError("Unrecognized pooling map layer type (only `maxpool` or `avgpool` supported)")

            cur_chs = int(total_channels*chs)

            self.pool_proj_branchs.append(
                nn.Sequential(OrderedDict([
                    (f'pool_proj_layer', pool),
                    ('channel_adjust_layer', BasicConv1d(in_channels, cur_chs, kernel_size=1, stride=1, padding=0, **basic_parms))
                ]))
            )

        # Dimension reduction layer
        self.dim_reduction_pool = nn.MaxPool1d(kernel_size=dim_reduction_pool_ks, stride=dim_reduction_pool_stride)

    def __count_padding_size(self, L:int, dilation:int, kernel_size:int, stride:int) -> int:
        assert dilation > 0, f'dilation must be a positive integer'
        padding = ((L - 1) * stride + dilation * (kernel_size - 1) - L + 1) / 2

        if isinstance(padding, float):
            assert padding.is_integer(), f'padding({padding}) cannot be divided'

        return int(padding)

    def forward(self, x):
        if self.info_retent_branch != None:
            branch1 = self.info_retent_branch(x) 
            branch1 = [branch1] 
        else:
            branch1 = []
        
        branch2 = [block(x) for block in self.info_ext_branchs]
        branch3 = [block(x) for block in self.pool_proj_branchs] 

        out = torch.cat(branch1 + branch2 + branch3, 1)
        out = self.dim_reduction_pool(out)

        return out

if __name__ == '__main__':
    # test case
    x = torch.rand(200, 4, 1000) 
    net = InceptionLayer(
        input_lenght=1000,
        in_channels=4,
        total_channels=100,
        info_retent_chs=0.2,
        info_ext_chs_ls=[0.1, 0.2, 0.3], 
        info_ext_ks_ls=[3, 6, 8], 
        info_ext_dilation_ls=[1, 2, 2],
        pool_proj_chs_ls=[0.1, 0.1],
        pool_proj_ks_ls=[3, 7],
        pool_proj_dilation_ls=[1, 1],
        pool_proj_type_ls=['maxpool', 'avgpool'],
        dropout=0.5,
        activite_func='leaky_relu',
        slope=0.01,
        dim_reduction_pool_ks=4,
        dim_reduction_pool_stride=4
    )
    out = net(x)

    # only Information extraction control branch
    x = torch.rand(200, 4, 1000) 
    net = InceptionLayer(
        input_lenght=1000,
        in_channels=4,
        total_channels=100,
        info_retent_chs=0,
        info_ext_chs_ls=[0.3, 0.4, 0.3], 
        info_ext_ks_ls=[3, 6, 8], 
        info_ext_dilation_ls=[1, 2, 2],
        pool_proj_chs_ls=[],
        pool_proj_ks_ls=[],
        pool_proj_dilation_ls=[],
        pool_proj_type_ls=[],
        dropout=0.5,
        activite_func='leaky_relu',
        slope=0.01,
        dim_reduction_pool_ks=4,
        dim_reduction_pool_stride=4
    )
    out = net(x)
