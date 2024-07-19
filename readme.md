# contrastive_VAE

## Introduction

This repository contains code for training a Labeled Variational Autoencoder (LVAE) model on 3D segmentation data.

## Directory Structure

```plaintext
LVAE/
├── config/
│   ├── config.yaml
├── src/
│   ├── data/
│   │   ├── dataloader.py
│   ├── models/
│   │   ├── vae.py
│   ├── training/
│   │   ├── train.py
├── .gitignore
├── README.md
└── requirements.txt

The data flppws
data.h5
├── train
│   ├── segmentation_data (n,12,100,100) np.array
│   ├── point_clouds (n,1024,3) np.array
│   ├── normals (n,1024,3) np.array
│   └── metadata (n,6) string
└── test
    ├── segmentation_data
    ├── point_clouds
    ├── normals
    └── metadata