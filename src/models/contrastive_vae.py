import torch
import torch.nn as nn
import torch.nn.functional as F
from .dgcnn import DGCNN

class ContrastiveVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, projection_dim, k, emb_dims):
        super(ContrastiveVAE, self).__init__()
        self.encoder = DGCNN(k=k, emb_dims=emb_dims)
        self.fc_mu = nn.Linear(emb_dims, latent_dim)
        self.fc_logvar = nn.Linear(emb_dims, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, emb_dims),
            nn.ReLU(),
            nn.Linear(emb_dims, input_dim),
            nn.Tanh()
        )
        self.fc_projection = nn.Linear(emb_dims, projection_dim)

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
        x = self.decoder(z)
        return x.view(-1, 3, 1024)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        projected = self.fc_projection(self.encoder(x))
        return reconstructed, mu, logvar, projected

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD
