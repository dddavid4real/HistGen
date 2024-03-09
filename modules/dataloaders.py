import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from .datasets import PathologySingleImageDataset

class R2DataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

        if self.dataset_name == 'wsi_report':
            self.dataset = PathologySingleImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)
        else:
            raise ValueError

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers
        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        #* image_ids & report_ids are the same thing
        images_id, images, reports_ids, reports_masks, seq_lengths = zip(*data) #* data is a list of tuples
        images = torch.stack(images, 0)
        max_seq_length = max(seq_lengths) #* Calculate the max_seq_length of the batch

        targets = np.zeros((len(reports_ids), max_seq_length), dtype=int) #* len(reports_ids) is the batch size
        targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)

        for i, report_ids in enumerate(reports_ids):
            targets[i, :len(report_ids)] = report_ids #* Fill the targets with the report_ids

        for i, report_masks in enumerate(reports_masks):
            targets_masks[i, :len(report_masks)] = report_masks #* Fill the targets_masks with the report_masks

        return images_id, images, torch.LongTensor(targets), torch.FloatTensor(targets_masks) #* Now we have the input and the label

