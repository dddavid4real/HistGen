import numpy as np
import torch
import torch.nn as nn
from modules.visual_extractor import VisualExtractor
from modules.histgen_module import BaseHistGen

class HistGenModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(HistGenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.encoder_decoder = BaseHistGen(args, tokenizer)
        self.wsi_mapping = torch.nn.Linear(768, self.args.d_vf) if "ctranspath" in args.image_dir else torch.nn.Linear(1024, self.args.d_vf)
        self.forward = self.forward_pathology
        self.visual_extractor = VisualExtractor(args)

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_pathology(self, images, targets=None, mode='train', update_opts={}):
        
        att_feats = self.wsi_mapping(images)
        fc_feats = torch.mean(att_feats, dim=1)
        
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            return output
        elif mode == 'sample':
            output, output_probs = self.encoder_decoder(fc_feats, att_feats, mode='sample')
            return output
        else:
            raise ValueError
