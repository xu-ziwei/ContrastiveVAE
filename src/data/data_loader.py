import h5py
import io
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from data.augmentation import random_transform

class PointCloudDataset(Dataset):
    def __init__(self, point_clouds, labels, augmentations=None):
        self.point_clouds = point_clouds
        self.labels = labels
        self.augmentations = augmentations

    def __len__(self):
        return len(self.point_clouds)

    def __getitem__(self, idx):
        point_cloud = self.point_clouds[idx]
        label = self.labels[idx]

        if self.augmentations:
            augmented_pc1 = self.augmentations(point_cloud)
            augmented_pc2 = self.augmentations(point_cloud)
            return (point_cloud, augmented_pc1, augmented_pc2), label
        else:
            return point_cloud, label

def load_data_from_h5(filename):
    with h5py.File(filename, 'r') as f:
        train_group = f['train']
        test_group = f['test']
        
        train_point_clouds = train_group['point_clouds'][:]
        test_point_clouds = test_group['point_clouds'][:]
        
        train_metadata_str = train_group['metadata'][()].decode('utf-8')
        test_metadata_str = test_group['metadata'][()].decode('utf-8')
        
        train_metadata = pd.read_csv(io.StringIO(train_metadata_str))
        test_metadata = pd.read_csv(io.StringIO(test_metadata_str))
        
        train_labels = train_metadata['cell_type'].values
        test_labels = test_metadata['cell_type'].values
        
    return (train_point_clouds, test_point_clouds, train_labels, test_labels)

def get_data_loaders(train_point_clouds, train_labels, test_point_clouds, test_labels, batch_size=32, augmentations=None):
    train_dataset = PointCloudDataset(train_point_clouds, train_labels, augmentations=augmentations)
    test_dataset = PointCloudDataset(test_point_clouds, test_labels, augmentations=None)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader