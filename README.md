# Dataset, model weight, source code for paper "HistGen: Histopathology Report Generation via Local-Global Feature Encoding and Cross-modal Context Interaction"

## Prerequisite
Follow this instruction to create conda environment and install necessary packages:
```
git clone https://github.com/dddavid4real/HistGen.git
cd HistGen
conda env create -f requirements.yml
```
## HistGen WSI-report dataset
Our curated dataset could be downloaded from <https://hkustconnect-my.sharepoint.com/:f:/g/personal/zguobc_connect_ust_hk/EhmtBBT0n2lKtiCQt97eqcEBWnB6K9-Dwr3wruaLyd_xTQ?e=IdWZmi>.

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

<!-- Note that before you use this json file for training, please run the `replace_pt_path.py` we provided to change the "image_path". Usage of `replace_pt_path.py` is written inside the python file. -->

## Pre-trained DINOv2 ViT-L Feature Extractor
We are organizing the training details, dataset used, and other information to release the pre-trained model. Please stay tuned for the update.

## HistGen WSI Report Generation Model
To try our model for training, validation, and testing, simply run the following commands:
```
cd HistGen
conda activate histgen
sh train_wsi_report.sh
```
Before you run the script, please set the path and other hyperparameters in `train_wsi_report.sh`.