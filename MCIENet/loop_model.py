import re

import torch
import torch.nn as nn

from .model.classifier import FeedForward
from .model.data_extractor import CNN, MCIENet, Transformer

from .utils.torch_utils import *

class LoopModel(nn.Module):
    def __init__(self, input_type, model_args):
        super().__init__()
        self.extractor_mode = model_args['extractor_mode'] # args.model['extractor_mode']
        self.input_type = input_type #args.data['input_type']

        ext_params, clf_parms = self.__built__parms(model_args)

        # data extractor
        extractor = self.__get_model(model_args['extractor_model'])
        self.data_extractor = extractor(**ext_params)

        # classifier
        self.classifier = FeedForward(**clf_parms)

    def __built__parms(self, args):
        select = lambda pat, dt: {re.sub(pat, '', k):v for k,v in dt if k.startswith(pat)}

        extractor_params = select('extractor_', args.items())
        classifier_params = select('classifier_', args.items())

        del extractor_params['model']
        del extractor_params['mode']

        return extractor_params, classifier_params

    def __get_model(self, name:str) -> nn.modules:
        models = {
            'cnn': CNN,
            'transformer': Transformer,
            'mcienet': MCIENet
        }

        return models[name]

    def get_parm_info(self) -> dict:
        # parms info
        parms_dt = {
            'total': count_parameters(self),
            'extractor': count_parameters(self.data_extractor),
            'classifier': count_parameters(self.classifier)
        }
        parms_dt['ext_ratio'] = round(parms_dt['extractor'] / parms_dt['total'], 2)
        parms_dt['clf_ratio'] = round(parms_dt['classifier'] / parms_dt['total'], 2)

        return parms_dt

    def forward(self, x):
        if self.input_type == 'onehot':
            anchor1, anchor2 = x[:, 0, :, :], x[:, 1, :, :] # [batch_size, 2, 4, anchor_size] | where 2 is pair of anchors
        elif self.input_type == 'text':
            anchor1, anchor2 = x[:, 0, :],  x[:, 1, :] # [batch_size, 2, anchor_size] | where 2 is pair of anchors

        if self.extractor_mode == 'concat':
            x = torch.concat((anchor1, anchor2), dim=-1) # concat last dimension
            x = self.data_extractor(x) 
    
        elif self.extractor_mode == 'pairs':
            x = self.data_extractor(anchor1, anchor2)

        x = self.classifier(x) # [batch_size, hidden_size] -> [batch_size, 2]

        return x