import torch
import torch.nn as nn
import numpy as np
import sys

from .ShowTellModel_Modules import ShowTellModel
from modules.wsi_token_select import CrossAttentionTokenReducer, uniform_sampling_batch, kmeans_reduction_batch

from modules.visual_extractor import VisualExtractor

class ShowTell(nn.Module):
    def __init__(self, args, tokenizer):
        super(ShowTell, self).__init__()
        
        self.args = args
        self.tokenizer = tokenizer       
        self.model = ShowTellModel(args = args, tokenizer = tokenizer)
        ### No use ###
        self.visual_extractor = VisualExtractor(args)
        ### WSI Selection Setting ###
        self.token_num = args.token_num
        self.wsi_mapping = torch.nn.Linear(768, 2048) if "ctranspath" in args.image_dir else torch.nn.Linear(1024, 2048)
        self.cross_attn = CrossAttentionTokenReducer(hidden_dim=1024, target_length=self.token_num, num_heads=4)
        # self.cross_attn = CrossAttentionTokenReducer(hidden_dim=1024, target_length=self.args.max_seq_length, num_heads=4)
        self.token_select = args.token_select
        
        ### Dataset Setting ###
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        elif args.dataset_name == 'wsi_report':
            self.forward = self.forward_pathology
        else:
            self.forward = self.forward_mimic_cxr
            
    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images, targets=None, mode='train'):
        raise NotImplementedError

    def forward_mimic_cxr(self, images, targets=None, mode='train'):
        raise NotImplementedError
    
    def forward_pathology(self, images, targets=None, mode='train'):
        # Assuming features is of shape [batch_size, seq_length, 1024]
        if self.token_select == 'cross_attn':
            features = self.cross_attn(images)
        elif self.token_select == 'uniform_sampling':
            features = uniform_sampling_batch(images, self.token_num)
        elif self.token_select == 'k-means':
            features = kmeans_reduction_batch(images, self.token_num)
        
        att_feats = self.wsi_mapping(features)

        ### Global Feature not used in M2Transformer ###
        fc_feats = torch.mean(att_feats, dim=1)
        
        if mode == 'train':
            output = self.model(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':            
            output, _ = self.model(fc_feats, att_feats, mode = 'sample')
        else:
            raise ValueError
        return output
        
        
        
        

