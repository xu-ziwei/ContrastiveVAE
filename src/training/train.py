import os
import sys
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from models.contrastive_vae import ContrastiveVAE
from data.data_loader import get_data_loaders, load_data_from_h5
from pytorch_metric_learning import losses

class ShapeTrainer:

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device(config['device'])
        self.ckpt_dir = os.path.join(config['experiment_dir'], 'checkpoints')
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=config['tensorboard']['log_dir'])
        self.build_loaders()
        self.reset()

    def reset(self):
        self.build_model()
        self.best_val_loss = None
        self.epoch = 0
        self.step = 0

    def build_loaders(self):
        data_path = self.config['data']['path']
        batch_size = self.config['loader']['batch_size']
        train_point_clouds, test_point_clouds, train_labels, test_labels = load_data_from_h5(data_path)
        self.train_loader, self.val_loader = get_data_loaders(train_point_clouds, train_labels, test_point_clouds, test_labels, batch_size)

    def build_model(self):
        input_dim = self.config['model']['kwargs']['input_dim']
        latent_dim = self.config['model']['kwargs']['latent_dim']
        projection_dim = self.config['model']['kwargs']['projection_dim']
        k = self.config['model']['kwargs']['k']
        emb_dims = self.config['model']['kwargs']['emb_dims']

        self.model = ContrastiveVAE(input_dim=input_dim, latent_dim=latent_dim, projection_dim=projection_dim, k=k, emb_dims=emb_dims)
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=[i for i in range(torch.cuda.device_count())])
            self.model.cuda()
        else:
            self.model = self.model.to(self.device)

        optimizer_class = getattr(torch.optim, self.config['optimizer']['name'])
        self.optimizer = optimizer_class(self.model.parameters(), **self.config['optimizer']['kwargs'])
        self.criterion = losses.NTXentLoss(temperature=self.config['criterion']['temperature'])

    def checkpoint(self, force=True):
        save = force or (self.epoch % self.config['training']['checkpoint_every'] == 0)
        if save:
            info = {
                'epoch': self.epoch,
                'iteration': self.step,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'config/model/name': self.config['model']['name'],
                'config/model/kwargs': self.config['model']['kwargs'],
            }
            ckpt_name = f'best_ckpt_iter_{self.step}.pt' if force else f'ckpt_iter_{self.step}.pt'
            ckpt_path = os.path.join(self.ckpt_dir, ckpt_name)
            torch.save(info, ckpt_path)

    def train_epoch(self):
        self.model.train()
        for iteration, data in enumerate(tqdm(self.train_loader, desc='Iteration')):
            points, labels = data['points'].to(self.device), data['label'].to(self.device)
            points = points.view(points.size(0), -1)  # Flatten the point cloud data
            self.optimizer.zero_grad()
            
            reconstructed, mu, logvar, projected = self.model(points)
            
            # Compute reconstruction loss and KL divergence
            vae_loss = self.model.loss_function(reconstructed, points, mu, logvar)
            
            # Compute contrastive loss
            contrastive_loss = self.criterion(projected, labels)
            
            # Total loss
            loss = vae_loss + contrastive_loss
            loss.backward()
            self.optimizer.step()

            self.writer.add_scalar('training/reconstruction_loss', vae_loss.item(), self.step)
            self.writer.add_scalar('training/contrastive_loss', contrastive_loss.item(), self.step)
            self.writer.add_scalar('training/total_loss', loss.item(), self.step)

            self.step += 1

    def validate_epoch(self):
        if (self.epoch % self.config['training']['validate_every']) != 0:
            return
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for data in tqdm(self.val_loader):
                points, labels = data['points'].to(self.device), data['label'].to(self.device)
                points = points.view(points.size(0), -1)  # Flatten the point cloud data
                reconstructed, mu, logvar, projected = self.model(points)
                
                vae_loss = self.model.loss_function(reconstructed, points, mu, logvar)
                contrastive_loss = self.criterion(projected, labels)
                
                loss = vae_loss + contrastive_loss
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)

        if self.best_val_loss is None or avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.checkpoint(True)

        self.writer.add_scalar('validation/average_loss', avg_loss, self.epoch)

    def train(self):
        for epoch_num in tqdm(range(self.config['training']['epochs']), desc='Epochs'):
            self.train_epoch()
            self.validate_epoch()
            self.epoch += 1
            self.checkpoint(False)

    def run(self):
        self.validate_epoch()
        self.train()

if __name__ == '__main__':
    path_to_config = sys.argv[1]
    with open(path_to_config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    trainer = ShapeTrainer(config)
    trainer.run()
