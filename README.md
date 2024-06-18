# Dataset, model weight, source code for paper "HistGen: Histopathology Report Generation via Local-Global Feature Encoding and Cross-modal Context Interaction"

#### ✨ **We are glad to announce that our paper is accepted by MICCAI2024!!**

This repo contains the dataset, model weight, and source code for paper "HistGen: Histopathology Report Generation via Local-Global Feature Encoding and Cross-modal Context Interaction". We only support PyTorch for now. See our paper for a detailed description of **HistGen**.

**HistGen: Histopathology Report Generation via Local-Global Feature Encoding and Cross-modal Context Interaction**\
*Zhengrui Guo, Jiabo Ma, Yingxue Xu, Yihui Wang, Liansheng Wang, and Hao Chen*\
Paper: <https://arxiv.org/abs/2403.05396>

<!-- Link to our paper: [[arxiv]](https://arxiv.org/abs/2403.05396) -->

### Highlight of our work:
- We introduce **HistGen**, a multiple instance learning-empowered framework for histopathology report generation together with the first benchmark dataset for evaluation. 
- Inspired by diagnostic and report-writing workflows, HistGen features two delicately designed modules, aiming to boost report generation by aligning whole slide images (WSIs) and diagnostic reports from local and global granularity. 
- To achieve this, a local-global hierarchical encoder is developed for efficient visual feature aggregation from a region-to-slide perspective. Meanwhile, a cross-modal context module is proposed to explicitly facilitate alignment and interaction between distinct modalities, effectively bridging the gap between the extensive visual sequences of WSIs and corresponding highly summarized reports. 
- Experimental results on WSI report generation show the proposed model outperforms state-of-the-art (SOTA) models by a large margin. Moreover, the results of fine-tuning our model on cancer subtyping and survival analysis tasks further demonstrate superior performance compared to SOTA methods, showcasing strong transfer learning capability.

### Methodology
![](methodology.png)
Overview of the proposed HistGen framework: (a) local-global hierarchical encoder module, (b) cross-modal context module, (c) decoder module, (d) transfer learning strategy for cancer diagnosis and prognosis.

## Table of Contents
- [Dataset, model weight, source code for paper "HistGen: Histopathology Report Generation via Local-Global Feature Encoding and Cross-modal Context Interaction"](#dataset-model-weight-source-code-for-paper-histgen-histopathology-report-generation-via-local-global-feature-encoding-and-cross-modal-context-interaction)
    - [Highlight of our work:](#highlight-of-our-work)
    - [Methodology](#methodology)
  - [Table of Contents](#table-of-contents)
  - [TO-DO:](#to-do)
  - [Prerequisite](#prerequisite)
  - [HistGen WSI-report dataset](#histgen-wsi-report-dataset)
  - [Pre-trained DINOv2 ViT-L Feature Extractor](#pre-trained-dinov2-vit-l-feature-extractor)
  - [HistGen WSI Report Generation Model](#histgen-wsi-report-generation-model)
    - [Training](#training)
    - [Inference](#inference)
    - [Transfer to Downstream Tasks](#transfer-to-downstream-tasks)
  - [Issues](#issues)
  - [License and Usage](#license-and-usage)

## TO-DO:
- [x] Release the source code for training and testing HistGen
- [x] Release the diagnostic report data
- [x] Release the DINOv2 ViT-L features of WSIs
- [ ] Update checkpoints of HistGen and merge into EasyMIL for cancer diagnosis and survival analysis tasks
- [ ] Release the original WSI data
- [ ] Release model weights of pre-trained DINOv2 ViT-L feature extractor

## Prerequisite
Follow this instruction to create conda environment and install necessary packages:
```
git clone https://github.com/dddavid4real/HistGen.git
cd HistGen
conda env create -f requirements.yml
```
## HistGen WSI-report dataset
Our curated dataset could be downloaded from [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/zguobc_connect_ust_hk/EhmtBBT0n2lKtiCQt97eqcEBWnB6K9-Dwr3wruaLyd_xTQ?e=IdWZmi).


The structure of this fold is shown as follows:
```
HistGen WSI-report dataset/
|-- WSIs
|    |-- slide_1.svs
|    |-- slide_2.svs
|    ╵-- ...
|-- dinov2_vitl
|        |-- slide_1.pt
|        |-- slide_2.pt
|        ╵-- ...
╵-- annotation.json
```
in which **WSIs** denotes the original WSI data from TCGA, **dinov2_vitl** is the features of original WSIs extracted by our pre-trained DINOv2 ViT-L backbone, and **annotation.json** contains the diagnostic reports and case ids of their corresponding WSIs. Concretely, the structure of this file is like this:
```
{
    "train": [
        {
            "id": "TCGA-A7-A6VW-01Z-00-DX1.1BC4790C-DB45-4A3D-9C97-92C92C03FF60",
            "report": "Final Surgical Pathology Report Procedure: Diagnosis A. Sentinel lymph node, left axilla ...",
            "image_path": [
                "/storage/Pathology/wsi-report/wsi/TCGA-A7-A6VW-01Z-00-DX1.1BC4790C-DB45-4A3D-9C97-92C92C03FF60.pt"
            ],
            "split": "train"
        },
        ...
    ],

    "val": [
        {
            "id": "...",
            "report": "...",
            "image_path": ["..."],
            "split": "val"
        },
        ...
    ],

    "test": [
        {
            "id": "...",
            "report": "...",
            "image_path": ["..."],
            "split": "test"
        },
        ...
    ]
}
```
in which we have already split into train/val/test subsets with ratio 8:1:1. Besides, "id" denotes the case id of this report's corresponding WSI, "report" is the full refined text obtained after our proposed report cleaning pipeline, and "image_path" could be just ignored. 

To reproduce our proposed HistGen model, please download the **dinov2_vitl** directory together with **annotation.json**.

<!-- Note that before you use this json file for training, please run the `replace_pt_path.py` we provided to change the "image_path". Usage of `replace_pt_path.py` is written inside the python file. -->

## Pre-trained DINOv2 ViT-L Feature Extractor
We are organizing the training details, dataset used, and other information to release the pre-trained model. Please stay tuned for the update.

## HistGen WSI Report Generation Model
### Training
To try our model for training, validation, and testing, simply run the following commands:
```
cd HistGen
conda activate histgen
sh train_wsi_report.sh
```
Before you run the script, please set the path and other hyperparameters in `train_wsi_report.sh`. Note that **--image_dir** should be the path to the **dinov2_vitl** directory, and **--ann_path** should be the path to the **annotation.json** file.

### Inference
To generate reports for WSIs in test set, you can run the following commands:
```
cd HistGen
conda activate histgen
sh test_wsi_report.sh
```
Similarly, remember to set the path and other hyperparameters in `test_wsi_report.sh`.

### Transfer to Downstream Tasks
In this paper, we consider WSI report generation task as an approach of vision-language pre-training, and we further fine-tune the pre-trained model on cancer subtyping and survival analysis tasks, with the strategy shown in [Methodology](#methodology) subfigure (d). For the implementation of downstream tasks, we recommend to use the [EasyMIL](https://github.com/birkhoffkiki/EasyMIL) repository, which is a flexible and easy-to-use toolbox for multiple instance learning (MIL) tasks developed by our team.

We are currently organizing the pre-trained checkpoints and merging HistGen into EasyMIL. Please stay tuned for the update.

## Issues
- Please open new threads or report issues directly (for urgent blockers) to `zguobc@connect.ust.hk`
- Immediate response to minor issues may not be available.

## License and Usage
If you find our work useful in your research, please consider citing our paper at:
```
@article{guo2024histgen,
  title={HistGen: Histopathology Report Generation via Local-Global Feature Encoding and Cross-modal Context Interaction},
  author={Guo, Zhengrui and Ma, Jiabo and Xu, Yingxue and Wang, Yihui and Wang, Liansheng and Chen, Hao},
  journal={arXiv preprint arXiv:2403.05396},
  year={2024}
}
```
This repo is made available under the Apache-2.0 License. For more details, please refer to the LICENSE file.
