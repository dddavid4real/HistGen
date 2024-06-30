import pdb
import os
import pandas as pd
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
from datasets.dataset_survival import Generic_WSI_Survival_Dataset, Generic_MIL_Survival_Dataset
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
parser.add_argument('--label_frac', type=float, default= 1.0,
                    help='fraction of labels (default: 1)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=10,
                    help='number of splits (default: 10)')
parser.add_argument('--task', type=str)
parser.add_argument('--val_frac', type=float, default= 0.1,
                    help='fraction of labels for validation (default: 0.1)')
parser.add_argument('--test_frac', type=float, default= 0.1,
                    help='fraction of labels for test (default: 0.1)')
parser.add_argument('--backbone', type=str, default='resnet50')
parser.add_argument('--mode', type = str, choices=['path', 'omic', 'pathomic', 'cluster'], default='path', help='which modalities to use')
parser.add_argument('--apply_sig', action='store_true', default=False, help='Use genomic features as signature embeddings')
parser.add_argument('--apply_sigfeats',  action='store_true', default=False, help='Use genomic features as tabular features.')

args = parser.parse_args()

if args.task == 'LUAD_LUSC':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/LUAD_LUSC.csv',
                            data_dir= None,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'LUAD':0, 'LUSC':1},
                            patient_strat=False,
                            ignore=[])
elif args.task == 'camelyon':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/camelyon.csv',
                            data_dir= None,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'normal':0, 'tumor':1},
                            patient_strat= False,
                            ignore=[])
    
elif args.task == 'RCC':
    args.n_classes=3
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/RCC.csv',
                            data_dir= None,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'KICH':0, 'KIRP':1, 'KIRC':2},
                            patient_strat= False,
                            ignore=[])

elif 'survival' in args.task:
    args.n_classes = 4
    # study = '_'.join(args.task.split('_')[:2])
    # if study == 'tcga_kirc' or study == 'tcga_kirp':
    #     combined_study = 'tcga_kidney'
    # elif study == 'tcga_luad' or study == 'tcga_lusc':
    #     combined_study = 'tcga_lung'
    # else:
    #     combined_study = study
    study = args.task.split('_')[1]
    # study_dir = '%s_20x_features' % combined_study
    # study_dir = 'pt_files/%s' % args.backbone
    dataset = Generic_MIL_Survival_Dataset(csv_path = 'dataset_csv/%s_processed.csv' % study,
                                            mode = args.mode,
                                            apply_sig = args.apply_sig,
                                            data_dir= None,
                                            shuffle = False, 
                                            seed = args.seed, 
                                            print_info = True,
                                            patient_strat= False,
                                            n_bins=4,
                                            label_col = 'survival_months',
                                            ignore=[])
elif args.task == 'PANDA':
    args.n_classes=6
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/PANDA.csv',
                            data_dir= None,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {0:0, 1:1, 2:2, 3:3, 4: 4, 5:5},
                            patient_strat=False,
                            ignore=[])


elif args.task == 'BRACS':
    args.n_classes=7
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/BRACS.csv',
                            data_dir= None,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'PB':0, 'UDH':1, 'IC':2, 'FEA':3, 
                                          'DCIS': 4, 'N':5, 'ADH': 6},
                            patient_strat=False,
                            ignore=[])
    
elif args.task == 'LUAD_LUSC_STAD':
    args.n_classes=3
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/LUAD_LUSC_STAD.csv',
                            data_dir= None,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'LUAD':0, 'LUSC':1, 'STAD':2 },
                            patient_strat=False,
                            ignore=[])
    
elif args.task == 'UBC-OCEAN':
    args.n_classes=5
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/UBC-OCEAN.csv',
                            data_dir= None,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'CC':0, 'EC':1, 'HGSC':2, 'LGSC':3, 'MC':4},
                            patient_strat=False,
                            ignore=[])


else:
    raise NotImplementedError
print('patient cls ids:', dataset.patient_cls_ids)
num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
val_num = np.round(num_slides_cls * args.val_frac).astype(int)
test_num = np.round(num_slides_cls * args.test_frac).astype(int)
print('Val num:', val_num, 'test_num:', test_num)

if __name__ == '__main__':
    if args.label_frac > 0:
        label_fracs = [args.label_frac]
    else:
        label_fracs = [0.1, 0.25, 0.5, 0.75, 1.0]
    
    for lf in label_fracs:
        split_dir = 'splits/'+ str(args.task) + '_{}'.format(int(lf * 100))
        os.makedirs(split_dir, exist_ok=True)
        dataset.create_splits(k = args.k, val_num = val_num, test_num = test_num, label_frac=lf)

        for i in range(args.k):
            dataset.set_splits()
            descriptor_df = dataset.test_split_gen(return_descriptor=True)
            splits = dataset.return_splits(None, from_id=True)
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}.csv'.format(i)))
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)
            descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i)))



