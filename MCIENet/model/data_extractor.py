import torch
import torch.nn as nn


from .layers import CNNLayer, InceptionLayer
from .utils import get_activite_func


class MCIENet(nn.Module):
    """"""
    def __init__(self, input_lenght: int, in_channels: int, output_dim: int, 
                 feature_agg: str = 'fc', feature_agg_rate: float = 0, **kwargs):

        super(MCIENet, self).__init__()

        # model settings
        self.Inception1 = InceptionLayer(
            input_lenght=input_lenght, 
            in_channels=in_channels, 
            **kwargs
            )
        self.Inception2 = InceptionLayer(
            input_lenght=self.Inception1.out_size, 
            in_channels=kwargs['total_channels'], 
            **kwargs
            )

        self.feature_agg_size = round(feature_agg_rate * self.Inception2.out_size) if feature_agg_rate != 0 else 1

        if self.Inception2.out_size > self.feature_agg_size:
            if feature_agg == 'fc':
                self.agg = nn.Sequential(
                    nn.Linear(self.Inception2.out_size, self.feature_agg_size), 
                    get_activite_func(kwargs['activite_func'], slope=kwargs['slope'])
                    )
            elif feature_agg == 'avgpool':
                self.agg = nn.AdaptiveAvgPool1d(self.feature_agg_size)

            elif feature_agg == 'maxpool':
                self.agg = nn.AdaptiveMaxPool1d(self.feature_agg_size)
        else:
            raise ValueError("Parameter `--feature_agg_rate` must be 0 or between 0 and 1")

        if self.feature_agg_size != 1:
            self.fc_ch = nn.Sequential(
                        nn.Linear(self.feature_agg_size, 1), 
                        get_activite_func(kwargs['activite_func'], slope=kwargs['slope'])
            )

        # for channels 
        self.fc_out = nn.Sequential(
                nn.Linear(kwargs['total_channels'], int(output_dim)), 
                get_activite_func(kwargs['activite_func'], slope=kwargs['slope'])
                )

    def forward(self, x):
        # input: [200, 4, 2000]
        x = self.Inception1(x) # [200, 100, 500]
        x = self.Inception2(x) # [200, 100, 125]

        
        if self.Inception2.out_size > self.feature_agg_size:
            x = self.agg(x) # [200, 100, 62]


        if self.feature_agg_size != 1:
            x = self.fc_ch(x) # [200, 100, 1]

        x = torch.squeeze(x, 2) # [200, 100]

        x = self.fc_out(x)

        return x
    

class CNN(nn.Module):
    def __init__(self, in_dim: int, hidden_size: int, output_dim: int, in_ch = 4, flatten: bool = False, 
                 feature_aggregation: str ='fc', feature_agg_rate: float = 0, **kwargs):
        super(CNN, self).__init__()

        self.flatten = flatten
        self.feature_aggregation = feature_aggregation

        self.activite_func = kwargs['activite_func']
        self.slope = kwargs['slope']
        self.conv_kernel_size = kwargs['conv_kernel_size']
        self.conv_stride = kwargs['conv_stride']
        self.mp_kernel_size = kwargs['mp_kernel_size']
        self.mp_stride = kwargs['mp_stride']

        # model settings
        self.conv_out_dim = self.__count_hidden_size(in_dim, self.conv_kernel_size, self.conv_stride, self.mp_kernel_size, self.mp_stride)
        self.feature_agg_size = round(feature_agg_rate * self.conv_out_dim) if feature_agg_rate != 0 else 1

        # build model
        self.conv1 = CNNLayer(in_ch=in_ch, out_ch=hidden_size, **kwargs)
        self.conv2 = CNNLayer(in_ch=hidden_size, out_ch=hidden_size, **kwargs)

        # Reducing the dimensionality of a sequence (last dim)
        if self.conv_out_dim > self.feature_agg_size:
            if feature_aggregation == 'fc':
                self.fc_f_agg = nn.Sequential(
                    nn.Linear(self.conv_out_dim, self.feature_agg_size), 
                    get_activite_func(self.activite_func, slope=self.slope)
                    )
            elif feature_aggregation == 'avgpool':
                self.avgpool = nn.AdaptiveAvgPool1d(self.feature_agg_size)

            elif feature_aggregation == 'maxpool':
                self.maxpool = nn.AdaptiveMaxPool1d(self.feature_agg_size)
        else:
            raise ValueError("Parameter `--feature_agg_rate` must be 0 or between 0 and 1")

        # merge last two dim
        if flatten:
            last_in_dim = hidden_size*self.feature_agg_size
        else:
            if self.feature_agg_size != 1:
                self.fc_ch = nn.Sequential(
                        nn.Linear(self.feature_agg_size, 1), 
                        get_activite_func(self.activite_func, slope=self.slope)
            )
            last_in_dim = hidden_size

        # for channels 
        self.fc_out = nn.Sequential(
                nn.Linear(last_in_dim, int(output_dim)), 
                get_activite_func(self.activite_func, slope=self.slope)
                )
    
    def __count_hidden_size(self, in_dim, conv_kernel_size, conv_stride, mp_kernel_size, mp_stride) -> int:
        count_dim = lambda in_dim, ks, stride: int((in_dim - ks) / stride + 1)

        conv1_out_dim = count_dim(in_dim, conv_kernel_size,  conv_stride)
        pool1_out_dim = count_dim(conv1_out_dim, mp_kernel_size,  mp_stride)
        conv2_out_dim = count_dim(pool1_out_dim, conv_kernel_size,  conv_stride)
        pool2_out_dim = count_dim(conv2_out_dim, mp_kernel_size,  mp_stride)

        return pool2_out_dim

    def forward(self, x):
        x = self.conv1(x) # [batch_size, hidden_dim, 498]
        x = self.conv2(x)  # [batch_size, hidden_dim, 122]
        
        if self.conv_out_dim > self.feature_agg_size:
            if self.feature_aggregation == 'fc':
                x = self.fc_f_agg(x) # [batch_size, hidden_dim, feature_agg_rate*122] 
            elif self.feature_aggregation == 'avgpool':
                x = self.avgpool(x) # [batch_size, hidden_dim, feature_agg_rate*122]
            elif self.feature_aggregation == 'maxpool':
                x = self.maxpool(x) # [batch_size, hidden_dim, feature_agg_rate*122]

        if self.flatten:
            x = torch.flatten(x, 1, -1) # [batch_size, hidden_dim*feature_agg_rate*122] 
        else:
            if self.feature_agg_size != 1:
                x = self.fc_ch(x) # [batch_size, hidden_dim, 1]
            x = torch.squeeze(x, 2) # [batch_size, hidden_dim]

        x = self.fc_out(x) # [batch_size, output_dim]

        return x

