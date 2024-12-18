import torch
import torch.nn as nn
import numpy as np
import sys

from .M2T_modules.transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory, Transformer_new_caption_model
from modules.wsi_token_select import CrossAttentionTokenReducer, uniform_sampling_batch, kmeans_reduction_batch

from modules.visual_extractor import VisualExtractor

class M2Transformer(nn.Module):
    def __init__(self, args, tokenizer):
        super(M2Transformer, self).__init__()
        
        self.args = args
        self.tokenizer = tokenizer
        
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.target_vocab = self.vocab_size + 1 #! notice this one
        
        ### M2Transformer Setting ###
        '''
        bos_idx = 0;
        pad_idx = 0;
        eos_idx = 0
        '''
        self.encoder = MemoryAugmentedEncoder(N = 3, padding_idx = self.tokenizer.token2idx['<pad>'], attention_module=ScaledDotProductAttentionMemory,
                                     attention_module_kwargs={'m': 40})
        self.decoder = MeshedDecoder(vocab_size = self.target_vocab, max_len = self.args.max_seq_length, N_dec = 3, padding_idx = self.tokenizer.token2idx['<pad>'])
        self.model = Transformer(bos_idx = self.tokenizer.token2idx['<bos>'], encoder = self.encoder, decoder = self.decoder, args = args, tokenizer = tokenizer)
        #!
        # self.model = Transformer_new_caption_model(bos_idx = args.bos_idx, encoder = self.encoder, decoder = self.decoder, args = args, tokenizer = tokenizer)
        
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
        
        if mode == 'train':
            output = self.model(att_feats, targets)
            
        elif mode == 'sample':

            output, _ = self.model.beam_search(att_feats, self.args.max_seq_length, self.tokenizer.token2idx['<eos>'], self.args.beam_size + 2, out_size= 1)

        else:
            raise ValueError
        return output
        
        
        
        

