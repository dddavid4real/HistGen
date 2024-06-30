from . import vision_transformer as vits
import torch
from torchvision import transforms


def build_model(device, gpu_num, model_name, ckpt_path):
    if model_name == 'dinov2_vitl':
        vit_kwargs = dict(
            img_size=224,
            patch_size=14,
            init_values=1.0e-05,
            ffn_layer='swiglufused',
            block_chunks=4,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True,
        )
        teacher = vits.__dict__['vit_large'](**vit_kwargs)
        ckpt = torch.load(ckpt_path)['teacher']
        # rename keys
        new_ckpt = {}
        for k, v in ckpt.items():
            if 'backbone' in k:
                k = '.'.join(k.split('.')[1:])
                new_ckpt[k] = v
        msg = teacher.load_state_dict(new_ckpt)
        print(msg)
        # cuda setting
        teacher.to(device)
        if gpu_num > 1:
            teacher = torch.nn.parallel.DataParallel(teacher)
        teacher.eval()
        return teacher, teacher.embed_dim
    else:
        raise NotImplementedError(f'{model_name} is not implemented...')


def build_transform():
    # Use timm's names
    # We prefer input size (512, 512), level 0;
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    normalize = transforms.Compose([
        transforms.Resize((224, 224), interpolation=3), 
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])
    return normalize