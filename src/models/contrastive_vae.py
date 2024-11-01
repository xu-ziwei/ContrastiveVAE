import torch
import torch.nn as nn
import torch.nn.functional as F
from .dgcnn import DGCNN
from .ChamferLoss import ChamferLoss
from .decoder import FoldNetDecoder, SimpleTestDecoder
from pytorch_metric_learning import losses 
from .sw_Loss import SWD

# ContrastiveVAE class

class ContrastiveVAE(nn.Module):
    def __init__(self, latent_dim, projection_dim, k, emb_dims, num_points=2048, std=0.3, temperature=0.5, use_contrastive_loss=True):
        super(ContrastiveVAE, self).__init__()
        self.encoder = DGCNN(k=k, emb_dims=emb_dims)
        self.fc_mu = nn.Linear(emb_dims, latent_dim)
        self.fc_logvar = nn.Linear(emb_dims, latent_dim)
        
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
        projected = self.fc_projection(self.encoder(x)) if self.use_contrastive_loss else None
        return reconstructed, mu, logvar, z, projected

    def loss_function(self, recon_x, x, mu, logvar, projected_concatenated=None, contrastive_labels=None, weights=None):
        Rec_loss = self.swd_loss(recon_x, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        w_Rec_loss = weights['Rec_loss'] * Rec_loss
        w_KLD = weights['KLD'] * KLD
        total_loss =  w_Rec_loss + w_KLD
        contrastive_loss = None
        w_contrastive_loss = None
        if self.use_contrastive_loss:
            contrastive_loss = self.contrastive_loss(projected_concatenated, contrastive_labels)
            w_contrastive_loss = weights['contrastive_loss'] * contrastive_loss
            total_loss += w_contrastive_loss
        return total_loss, w_Rec_loss, w_KLD, w_contrastive_loss
    

    '''
     # add gamma to balance 
        gamma = torch.sqrt(Rec_loss + 1e-10)

        # Ensure gamma is a tensor for torch.log
        log_gamma = torch.log(gamma)
        log_two_pi = torch.log(torch.tensor(2.0 * torch.pi))
        w_Rec_loss = (Rec_loss / (2 * gamma**2)) + log_gamma + 0.5 * log_two_pi
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # w_Rec_loss = weights['Rec_loss'] * Rec_loss

        w_KLD = weights['KLD'] * KLD
    '''