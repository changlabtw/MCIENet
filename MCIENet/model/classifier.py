import torch
import torch.nn as nn

from .utils import get_activite_func
from collections import OrderedDict

class FeedForward(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, hidden_layer, 
                 activite_func, slope=0.01, dropout=0.5, bn=False, bn_eps: float = 0.00001, bn_momentum: float = 0.1):
        super().__init__()

        hidden_layer_ls = []
        for idx, _ in enumerate(range(hidden_layer)):
            dim = input_dim if idx == 0 else hidden_size

            layers = [
                (f'hidden_{idx}', nn.Linear(dim, hidden_size)),
                (f'bn_{idx}', nn.BatchNorm1d(int(hidden_size), eps=bn_eps, momentum=bn_momentum)) if bn else None,
                (f'activite_func_{idx}', get_activite_func(activite_func, slope=slope)),
                (f'dropout_{idx}', nn.Dropout(dropout)) if dropout!=0 else None
                ]
            layers = [i for i in layers if i != None]

            hidden_layer_ls.extend(layers)

        self.hidden_net = nn.Sequential(OrderedDict(hidden_layer_ls))

        self.out = torch.nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = self.hidden_net(x) # [batch_size, hidden_layer] -> ... -> [batch_size, hidden_layer]

        x = self.out(x) # [batch_size, output_dim]

        return x

# =================================
    