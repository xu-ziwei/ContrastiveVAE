import numpy as np
import pandas as pd
import tensorflow as tf
from tifffile import imread
from sklearn.model_selection import train_test_split

def load_data(data_dir, metadata_filename, segmentation_filename):
    """
    Load and preprocess data from the specified directory.

    Parameters:
    - data_dir: Directory where the data is stored
    - metadata_filename: Filename of the metadata CSV
    - segmentation_filename: Filename of the segmentation data

    Returns:
    - segmentation_data: The loaded segmentation data
    - labels: One-hot encoded labels
    - input_shape: Shape of the input data
    """
    metadata_path = f"{data_dir}/{metadata_filename}"
    segmentation_path = f"{data_dir}/{segmentation_filename}"

    # Load metadata and segmentation data
    metadata_df = pd.read_csv(metadata_path)
    segmentation_data = imread(segmentation_path)
    segmentation_data = np.expand_dims(segmentation_data, axis=-1)
    input_shape = segmentation_data.shape[1:]

    # Ensure that segmentation_data and metadata_df are aligned
    assert len(segmentation_data) == len(metadata_df), "Data mismatch!"

    # Extract cell types from metadata
    cell_type_mapping = {'goblet': 3, 'intercalate': 2, 'basal': 1, np.nan: 0}
    metadata_df['cell_type'] = metadata_df['cell_type']
    metadata_df['cell_type_int'] = metadata_df['cell_type'].map(cell_type_mapping)
    cell_types = metadata_df['cell_type_int'].values

    # Convert labels to one-hot encoding
    labels = tf.keras.utils.to_categorical(cell_types, num_classes=4)

    return segmentation_data, labels, input_shape


def select_subset(data, labels, subset_ratio=0.05, random_seed=42):
    """
    Select a subset of the data.

    Parameters:
    - data: The full dataset
    - labels: Corresponding labels
    - subset_ratio: Fraction of data to select
    - random_seed: Seed for reproducibility

    Returns:
    - data_subset: Subset of the dataset
    - labels_subset: Subset of the labels
    """
    np.random.seed(random_seed)
    subset_size = int(subset_ratio * len(data))
    indices = np.random.choice(len(data), subset_size, replace=False)
    data_subset = data[indices]
    labels_subset = labels[indices]
    return data_subset, labels_subset

def split_data(data, labels, val_size=0.2, random_seed=42):
    """
    Split the data into training and validation sets.

    Parameters:
    - data: The dataset to split
    - labels: Corresponding labels
    - val_size: Fraction of data to be used as validation set
    - random_seed: Seed for reproducibility

    Returns:
    - x_train: Training data
    - x_val: Validation data
    - y_train: Training labels
    - y_val: Validation labels
    """
    x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=val_size, random_state=random_seed)
    return x_train, x_val, y_train, y_val
