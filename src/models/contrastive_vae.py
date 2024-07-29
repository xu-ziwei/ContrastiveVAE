import torch
import torch.nn as nn
import torch.nn.functional as F
from .dgcnn import DGCNN
from .ChamferLoss import ChamferLoss
from .decoder import FoldNetDecoder
from pytorch_metric_learning import losses 

# ContrastiveVAE class

class ContrastiveVAE(nn.Module):
    def __init__(self, latent_dim, projection_dim, k, emb_dims, num_points=2048, temperature=0.5, std=0.3):
        super(ContrastiveVAE, self).__init__()
        self.encoder = DGCNN(k=k, emb_dims=emb_dims)
        self.fc_mu = nn.Linear(emb_dims, latent_dim)
        self.fc_logvar = nn.Linear(emb_dims, latent_dim)
        
        self.fc_projection = nn.Linear(emb_dims, projection_dim)
        self.chamfer_loss = ChamferLoss()
        self.contrastive_loss = losses.NTXentLoss(temperature=temperature)
        
        self.decoder = FoldNetDecoder(num_features=latent_dim, num_points=num_points, std=std)

    def encode(self, x):
        features = self.encoder(x)
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        projected = self.fc_projection(self.encoder(x))
        return reconstructed, mu, logvar, projected

    def loss_function(self, recon_x, x, mu, logvar, projected_concatenated, contrastive_labels, weights):
        Rec_loss = self.chamfer_loss(recon_x, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        contrastive_loss = self.contrastive_loss(projected_concatenated, contrastive_labels)
        total_loss = weights['Rec_loss'] * Rec_loss + weights['KLD'] * KLD + weights['contrastive_loss'] * contrastive_loss
        return total_loss, Rec_loss, KLD, contrastive_loss
