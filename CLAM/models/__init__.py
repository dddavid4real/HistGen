import torch
import numpy as np


__all__ = ['list_models', 'get_model', 'get_custom_transformer']


__implemented_models = {
    'resnet50': 'image-net',
    'resnet101': 'image-net',
    'plip': 'https://huggingface.co/vinid/plip',
    'ctranspath': 'models/ckpts/ctranspath.pth',
    'dinov2_vitl': 'models/ckpts/dinov2_cpath_v1.pth'
}


def list_models():
    print('The following are implemented models:')
    for k, v in __implemented_models.items():
        print('{}: {}'.format(k, v))
    return __implemented_models


def get_model(model_name, device, gpu_num):
    """_summary_

    Args:
        model_name (str): the name of the requried model
        device (torch.device): device, e.g. 'cuda'
        gpu_num (int): the number of GPUs used in extracting features

    Raises:
        NotImplementedError: if the model name does not exist

    Returns:
        nn.Module: model
    """
    if model_name == 'resnet50':
        from models.resnet_custom import resnet50_baseline
        model = resnet50_baseline(pretrained=True).to(device)
    elif model_name == 'resnet101':
        from models.resnet_custom import resnet101_baseline
        model = resnet101_baseline(pretrained=True).to(device)
    elif model_name == 'vit_base_patch16_224_21k':
        from models.transforms_model import vit_base_patch16_224_21k
        model = vit_base_patch16_224_21k(device, gpu_num)
    elif model_name == 'vit_large_patch16_224_21k':
        from models.transforms_model import vit_large_patch16_224_21k
        model = vit_large_patch16_224_21k(device, gpu_num)
    elif model_name == 'vit_huge_patch14_224_21k':
        from models.transforms_model import vit_huge_patch14_224_21k
        model = vit_huge_patch14_224_21k(device, gpu_num)
        
    # our models
    elif model_name in ['mae_vit_large_patch16-1-40000', 'mae_vit_large_patch16-1-140000',
                        'mae_vit_l_1000slides_19epoch', 'mae_vit_l_10000slides_3epoch', 
                        'mae_vit_large_patch16-1epoch-180M',
                        
                        ]:
        from models.mae_endoder import mae_pretrained_model
        model = mae_pretrained_model(device, gpu_num, 'mae_vit_large_patch16',ckpt=__implemented_models[model_name] ,input_size=224)
    
    elif model_name in ['mae_vit_huge_patch14_1000slides_9epoch',
                        'mae_vit_huge_patch14_1000slides_0epoch',
                        'mae_vit_huge_patch14_1000slides_22epoch',
                        ]:
        
        from models.mae_endoder import mae_pretrained_model
        model = mae_pretrained_model(device, gpu_num, 'mae_vit_huge_patch14',ckpt=__implemented_models[model_name] ,input_size=224)

    elif model_name in ['dinov2_vitl']:
        from models.dinov2 import build_model
        model, _ = build_model(device, gpu_num, model_name, __implemented_models[model_name])

    elif model_name == 'ctranspath':
        from models.ctrans import ctranspath
        print('\n!!!! please note that ctranspath requires the modified timm 0.5.4, you can find package at here: models/ckpts/timm-0.5.4.tar , please install if needed ...\n')
        model = ctranspath(ckpt_path=__implemented_models['ctranspath']).to(device)
    elif model_name == 'plip':
        from models.plip import plip
        model = plip(device, gpu_num)
        
    else:
        raise NotImplementedError(f'{model_name} is not implemented')
    
    if model_name in ['resnet50', 'resnet101']:
        if gpu_num > 1:
            model = torch.nn.parallel.DataParallel(model)
        model = model.eval()
    return model


def get_custom_transformer(model_name):
    """_summary_

    Args:
        model_name (str): the name of model

    Raises:
        NotImplementedError: not implementated

    Returns:
        torchvision.transformers: the transformers used to preprocess the image
    """
    if model_name in ['resnet50', 'resnet101']:
        from models.resnet_custom import custom_transforms
        custom_trans = custom_transforms()
    elif model_name in ['vit_base_patch16_224_21k', 'vit_large_patch16_224_21k', 'vit_huge_patch14_224_21k']:
        # Do nothing, let vit process do the image processing
        from torchvision import transforms as tt
        custom_trans = tt.Lambda(lambda x: torch.from_numpy(np.array(x)))
    elif model_name in ['mae_vit_large_patch16-1-40000', 'mae_vit_large_patch16-1-140000',
                        'mae_vit_l_1000slides_19epoch', 'mae_vit_l_10000slides_3epoch',
                        'mae_vit_large_patch16-1epoch-180M',
                        ]:
        from models.mae_endoder import mae_transform
        custom_trans = mae_transform(224)

    elif model_name in ['mae_vit_huge_patch14_1000slides_9epoch',
                        'mae_vit_huge_patch14_1000slides_0epoch',
                        'mae_vit_huge_patch14_1000slides_22epoch',
                        ]:
        from models.mae_endoder import mae_transform
        custom_trans = mae_transform(224)
          
    elif model_name == 'ctranspath':
        from models.ctrans import ctranspath_transformers
        custom_trans = ctranspath_transformers()
    elif model_name == 'hipt':
        from torchvision import transforms as tt
        custom_trans = tt.functional.to_tensor()
    elif model_name == 'plip':
        # Do nothing, let CLIP process do the image processing
        from torchvision import transforms as tt
        custom_trans = tt.Lambda(lambda x: torch.from_numpy(np.array(x)))
    
    elif model_name in ['dinov2_vitl']:
        from models.dinov2 import build_transform
        custom_trans = build_transform()
        
    else:
        raise NotImplementedError('Transformers for {} is not implemented ...'.format(model_name))

    return custom_trans
