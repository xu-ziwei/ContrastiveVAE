import h5py
import io
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from data.augmentation import random_transform, normalize


def load_data_from_h5(filename):
    with h5py.File(filename, 'r') as f:
        train_group = f['train']
        val_group = f['val']
        test_group = f['test']
        
        train_point_clouds = train_group['point_clouds'][:]
        augmented_train_pc1 = train_group['augmented_pc1'][:]
        augmented_train_pc2 = train_group['augmented_pc2'][:]
        val_point_clouds = val_group['point_clouds'][:]
        augmented_val_pc1 = val_group['augmented_pc1'][:]
        augmented_val_pc2 = val_group['augmented_pc2'][:]
        test_point_clouds = test_group['point_clouds'][:]
        augmented_test_pc1 = test_group['augmented_pc1'][:]
        augmented_test_pc2 = test_group['augmented_pc2'][:]
        
        train_metadata_str = train_group['metadata'][()].decode('utf-8')
        val_metadata_str = val_group['metadata'][()].decode('utf-8')
        test_metadata_str = test_group['metadata'][()].decode('utf-8')
        
        train_metadata = pd.read_csv(io.StringIO(train_metadata_str))
        val_metadata = pd.read_csv(io.StringIO(val_metadata_str))
        test_metadata = pd.read_csv(io.StringIO(test_metadata_str))
        
        train_labels = train_metadata['cell_type'].values
        val_labels = val_metadata['cell_type'].values
        test_labels = test_metadata['cell_type'].values
        
    return (train_point_clouds, augmented_train_pc1, augmented_train_pc2, train_labels,
            val_point_clouds, augmented_val_pc1, augmented_val_pc2, val_labels,
            test_point_clouds, augmented_test_pc1, augmented_test_pc2, test_labels)


def save_augmented_data_to_h5(input_filename, output_filename):
    with h5py.File(input_filename, 'r') as f:
        train_group = f['train']
        val_group = f['val']
        test_group = f['test']
        
        train_point_clouds = train_group['point_clouds'][:]
        val_point_clouds = val_group['point_clouds'][:]
        test_point_clouds = test_group['point_clouds'][:]
        
        train_metadata_str = train_group['metadata'][()].decode('utf-8')
        val_metadata_str = val_group['metadata'][()].decode('utf-8')
        test_metadata_str = test_group['metadata'][()].decode('utf-8')
        
        train_metadata = pd.read_csv(io.StringIO(train_metadata_str))
        val_metadata = pd.read_csv(io.StringIO(val_metadata_str))
        test_metadata = pd.read_csv(io.StringIO(test_metadata_str))
        
        train_labels = train_metadata['cell_type'].values
        val_labels = val_metadata['cell_type'].values
        test_labels = test_metadata['cell_type'].values
        
        # Create augmented point clouds
        augmented_train_pc1 = np.array([random_transform(pc) for pc in train_point_clouds])
        augmented_train_pc2 = np.array([random_transform(pc) for pc in train_point_clouds])
        augmented_val_pc1 = np.array([random_transform(pc) for pc in val_point_clouds])
        augmented_val_pc2 = np.array([random_transform(pc) for pc in val_point_clouds])
        augmented_test_pc1 = np.array([random_transform(pc) for pc in test_point_clouds])
        augmented_test_pc2 = np.array([random_transform(pc) for pc in test_point_clouds])
        
    # Save augmented data to a new HDF5 file
    with h5py.File(output_filename, 'w') as f:
        train_group = f.create_group('train')
        val_group = f.create_group('val')
        test_group = f.create_group('test')
        
        train_group.create_dataset('point_clouds', data=train_point_clouds)
        train_group.create_dataset('augmented_pc1', data=augmented_train_pc1)
        train_group.create_dataset('augmented_pc2', data=augmented_train_pc2)
        train_group.create_dataset('metadata', data=train_metadata_str.encode('utf-8'))
        train_group.create_dataset('labels', data=train_labels)
        
        val_group.create_dataset('point_clouds', data=val_point_clouds)
        val_group.create_dataset('augmented_pc1', data=augmented_val_pc1)
        val_group.create_dataset('augmented_pc2', data=augmented_val_pc2)
        val_group.create_dataset('metadata', data=val_metadata_str.encode('utf-8'))
        val_group.create_dataset('labels', data=val_labels)
        
        test_group.create_dataset('point_clouds', data=test_point_clouds)
        test_group.create_dataset('augmented_pc1', data=augmented_test_pc1)
        test_group.create_dataset('augmented_pc2', data=augmented_test_pc2)
        test_group.create_dataset('metadata', data=test_metadata_str.encode('utf-8'))
        test_group.create_dataset('labels', data=test_labels)

def save_nor_data_to_h5(input_filename, output_filename):
    with h5py.File(input_filename, 'r') as f:
        train_point_clouds = f['train/point_clouds'][:]
        augmented_train_pc1 = f['train/augmented_pc1'][:]
        augmented_train_pc2 = f['train/augmented_pc2'][:]
        train_labels = f['train/labels'][:]
        train_metadata_str = f['train/metadata'][()].decode('utf-8')
        
        val_point_clouds = f['val/point_clouds'][:]
        augmented_val_pc1 = f['val/augmented_pc1'][:]
        augmented_val_pc2 = f['val/augmented_pc2'][:]
        val_labels = f['val/labels'][:]
        val_metadata_str = f['val/metadata'][()].decode('utf-8')
        
        test_point_clouds = f['test/point_clouds'][:]
        augmented_test_pc1 = f['test/augmented_pc1'][:]
        augmented_test_pc2 = f['test/augmented_pc2'][:]
        test_labels = f['test/labels'][:]
        test_metadata_str = f['test/metadata'][()].decode('utf-8')
        
        # Normalize the point clouds
        normalized_train_point_clouds = np.array([normalize(pc) for pc in train_point_clouds])
        normalized_augmented_train_pc1 = np.array([normalize(pc) for pc in augmented_train_pc1])
        normalized_augmented_train_pc2 = np.array([normalize(pc) for pc in augmented_train_pc2])
        
        normalized_val_point_clouds = np.array([normalize(pc) for pc in val_point_clouds])
        normalized_augmented_val_pc1 = np.array([normalize(pc) for pc in augmented_val_pc1])
        normalized_augmented_val_pc2 = np.array([normalize(pc) for pc in augmented_val_pc2])
        
        normalized_test_point_clouds = np.array([normalize(pc) for pc in test_point_clouds])
        normalized_augmented_test_pc1 = np.array([normalize(pc) for pc in augmented_test_pc1])
        normalized_augmented_test_pc2 = np.array([normalize(pc) for pc in augmented_test_pc2])
        
    with h5py.File(output_filename, 'w') as f:
        # Save normalized training data
        train_group = f.create_group('train')
        train_group.create_dataset('point_clouds', data=normalized_train_point_clouds)
        train_group.create_dataset('augmented_pc1', data=normalized_augmented_train_pc1)
        train_group.create_dataset('augmented_pc2', data=normalized_augmented_train_pc2)
        train_group.create_dataset('labels', data=train_labels)
        train_group.create_dataset('metadata', data=train_metadata_str.encode('utf-8'))
        
        # Save normalized validation data
        val_group = f.create_group('val')
        val_group.create_dataset('point_clouds', data=normalized_val_point_clouds)
        val_group.create_dataset('augmented_pc1', data=normalized_augmented_val_pc1)
        val_group.create_dataset('augmented_pc2', data=normalized_augmented_val_pc2)
        val_group.create_dataset('labels', data=val_labels)
        val_group.create_dataset('metadata', data=val_metadata_str.encode('utf-8'))
        
        # Save normalized test data
        test_group = f.create_group('test')
        test_group.create_dataset('point_clouds', data=normalized_test_point_clouds)
        test_group.create_dataset('augmented_pc1', data=normalized_augmented_test_pc1)
        test_group.create_dataset('augmented_pc2', data=normalized_augmented_test_pc2)
        test_group.create_dataset('labels', data=test_labels)
        test_group.create_dataset('metadata', data=test_metadata_str.encode('utf-8'))



# only for training data 
class InMemoryPointCloudDataset(Dataset):
    def __init__(self, point_clouds, augmented_pc1, augmented_pc2, labels):
        self.point_clouds = torch.tensor(point_clouds, dtype=torch.float32)
        self.augmented_pc1 = torch.tensor(augmented_pc1, dtype=torch.float32)
        self.augmented_pc2 = torch.tensor(augmented_pc2, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.point_clouds)

    def __getitem__(self, idx):
        return (self.point_clouds[idx], self.augmented_pc1[idx], self.augmented_pc2[idx]), self.labels[idx]

def load_data_from_h5_in_memory(filename):
    with h5py.File(filename, 'r') as f:
        train_point_clouds = f['train/point_clouds'][:]
        augmented_train_pc1 = f['train/augmented_pc1'][:]
        augmented_train_pc2 = f['train/augmented_pc2'][:]
        train_labels = f['train/labels'][:]
        
        val_point_clouds = f['val/point_clouds'][:]
        augmented_val_pc1 = f['val/augmented_pc1'][:]
        augmented_val_pc2 = f['val/augmented_pc2'][:]
        val_labels = f['val/labels'][:]
    
    return (train_point_clouds, augmented_train_pc1, augmented_train_pc2, train_labels, 
            val_point_clouds, augmented_val_pc1, augmented_val_pc2, val_labels)

def get_data_loaders_in_memory(filename, batch_size=32, num_workers=16):
    (train_point_clouds, augmented_train_pc1, augmented_train_pc2, train_labels, 
     val_point_clouds, augmented_val_pc1, augmented_val_pc2, val_labels) = load_data_from_h5_in_memory(filename)
    
    train_dataset = InMemoryPointCloudDataset(train_point_clouds, augmented_train_pc1, augmented_train_pc2, train_labels)
    val_dataset = InMemoryPointCloudDataset(val_point_clouds, augmented_val_pc1, augmented_val_pc2, val_labels)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    
    return train_loader, val_loader

def load_test_from_h5_in_memory(filename):
    with h5py.File(filename, 'r') as f:
        test_point_clouds = f['test/point_clouds'][:]
        test_labels = f['test/labels'][:]
    return (test_point_clouds, test_labels)


class TestDataset(Dataset):
    def __init__(self, point_clouds, labels):
        self.point_clouds = torch.tensor(point_clouds, dtype=torch.float32)

        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.point_clouds)

    def __getitem__(self, idx):
        return (self.point_clouds[idx], self.labels[idx])
    
def get_test_loaders_in_memory(filename, batch_size=32, num_workers=16):
    (test_point_clouds, test_labels) = load_test_from_h5_in_memory(filename)
    
    test_dataset = TestDataset(test_point_clouds, test_labels)

    pin_memory = torch.cuda.is_available()

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

    return test_loader