import torch
import torch.nn as nn
import torch.nn.functional as F
from .dgcnn import DGCNN
from .ChamferLoss import ChamferLoss
from .decoder import FoldNetDecoder, SimpleTestDecoder
from pytorch_metric_learning import losses 
from .sw_Loss import SWD

# ContrastiveAE class

class ContrastiveAE(nn.Module):
    def __init__(self, latent_dim, projection_dim, k, emb_dims, num_points=2048, std=0.3, temperature=0.5, use_contrastive_loss=True):
        super(ContrastiveAE, self).__init__()
        self.encoder = DGCNN(k=k, emb_dims=emb_dims)
        self.fc_latent = nn.Linear(emb_dims, latent_dim)
        
        self.use_contrastive_loss = use_contrastive_loss
        if use_contrastive_loss:
            self.temperature = temperature
            self.fc_projection = nn.Linear(emb_dims, projection_dim)
            self.contrastive_loss = losses.NTXentLoss(temperature=self.temperature)

        self.swd_loss = SWD(num_projs=100, device="cuda")
        # self.chamfer_loss = ChamferLoss()
        # self.decoder = FoldNetDecoder(num_features=latent_dim, num_points=num_points, std=std)
        # use test decoder 
        self.decoder = SimpleTestDecoder(latent_dim=latent_dim, output_points=num_points)

    def encode(self, x):
        features = self.encoder(x)
        latent = self.fc_latent(features)
        return latent

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        projected = self.fc_projection(self.encoder(x)) if self.use_contrastive_loss else None
        return reconstructed, latent, projected

    def loss_function(self, recon_x, x, latent, projected_concatenated=None, contrastive_labels=None, weights=None):
        Rec_loss = self.swd_loss(recon_x, x)
        w_Rec_loss = weights['Rec_loss'] * Rec_loss
        contrastive_loss = None
        w_contrastive_loss = 0.0
        if self.use_contrastive_loss:
            contrastive_loss = self.contrastive_loss(projected_concatenated, contrastive_labels)
            w_contrastive_loss = weights['contrastive_loss'] * contrastive_loss
        return w_Rec_loss+w_contrastive_loss, w_Rec_loss, w_contrastive_loss
