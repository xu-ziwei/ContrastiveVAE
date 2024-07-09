import os
import datetime
import argparse
import yaml
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from src.data.dataloader import load_data, select_subset, split_data
from src.models.vae import LVAE

def train_model(config):
    # Load data
    segmentation_data, labels, input_shape = load_data(
        config['data_dir'],
        config['metadata_file'],
        config['segmentation_file']
    )

    # Select a subset of the data to test
    if config['subset_ratio'] < 1.0 and not None:
        segmentation_data, labels = select_subset(
            segmentation_data, labels, subset_ratio=config['subset_ratio']
        )

    # Split the data into training, validation, and test sets
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(
        segmentation_data, labels, val_size=config['val_size'], test_size=config['test_size']
    )

    # Initialize the model (optimizer is set within the class)
    lvae = LVAE(
        input_dims=input_shape,
        latent_dim=config['latent_dim'],
        beta=config['beta'],
        gamma=config['gamma'],
        num_classes=4
    )

    #print lvae summary
    lvae.model_summary()
    
    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(config['model_dir'], 'best_model'),
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        save_format='tf'
    )
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(
        log_dir=os.path.join(config['log_dir'], current_time),
        histogram_freq=1
    )

    # Train the model with callbacks
    lvae.fit(
        x=x_train,
        y=y_train,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        validation_data=(x_val, y_val),
        callbacks=[checkpoint_callback, tensorboard_callback]
    )

    # Evaluate on the test set
    test_loss = lvae.evaluate(x_test, y_test)
    print(f"Test loss: {test_loss}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train LVAE model.')
    parser.add_argument('-config', type=str, required=True, help='Path to the configuration file')

    args = parser.parse_args()

    # Load the configuration from the YAML file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    train_model(config)
