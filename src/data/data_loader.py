import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import pandas as pd
import numpy as np
import io

class PointCloudDataset(Dataset):
    def __init__(self, point_clouds, labels, transform=None):
        self.point_clouds = point_clouds
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.point_clouds)

    def __getitem__(self, idx):
        points = self.point_clouds[idx]
        label = self.labels[idx]
        if self.transform:
            points = self.transform(points)
        return {'points': torch.tensor(points, dtype=torch.float32), 'label': torch.tensor(label, dtype=torch.long)}

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

def get_data_loaders(train_point_clouds, train_labels, test_point_clouds, test_labels, batch_size, num_workers=4, shuffle=True):
    train_dataset = PointCloudDataset(train_point_clouds, train_labels)
    test_dataset = PointCloudDataset(test_point_clouds, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader
