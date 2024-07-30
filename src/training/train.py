import os
import sys
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.optim as optim
from models.contrastive_vae import ContrastiveVAE
from data.data_loader import get_data_loaders_in_memory


class ShapeTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        num_workers = self.config['loader']['num_workers']
        self.train_loader, self.val_loader = get_data_loaders_in_memory(data_path, batch_size=batch_size, num_workers=num_workers)

    def build_model(self):
        latent_dim = self.config['model']['kwargs']['latent_dim']
        projection_dim = self.config['model']['kwargs']['projection_dim']
        k = self.config['model']['kwargs']['k']
        emb_dims = self.config['model']['kwargs']['emb_dims']
        num_points = self.config['model']['kwargs']['num_points']
        temperature = self.config['criterion']['temperature']
        use_contrastive_loss = self.config['use_contrastive_loss']

        self.model = ContrastiveVAE(latent_dim=latent_dim, projection_dim=projection_dim, k=k, emb_dims=emb_dims, num_points=num_points, temperature=temperature, use_contrastive_loss=use_contrastive_loss)
        self.model = self.model.to(self.device)

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = torch.nn.DataParallel(self.model)

        optimizer_class = getattr(torch.optim, self.config['optimizer']['name'])
        self.optimizer = optimizer_class(self.model.parameters(), **self.config['optimizer']['kwargs'])

        self.loss_weights = self.config['loss_weights']

    def checkpoint(self, is_best=False):
        if is_best:
            info = {
                'epoch': self.epoch,
                'iteration': self.step,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'config/model/name': self.config['model']['name'],
                'config/model/kwargs': self.config['model']['kwargs'],
            }
            ckpt_name = 'best_ckpt.pt'
            ckpt_path = os.path.join(self.ckpt_dir, ckpt_name)
            torch.save(info, ckpt_path)
            print(f"Best model checkpoint saved at {ckpt_path}")

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_rec_loss = 0.0
        total_kld_loss = 0.0
        total_contrastive_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc='Iteration')

        for iteration, (point_clouds, labels) in enumerate(progress_bar):
            try:
                original_pc, augmented_pc1, augmented_pc2 = point_clouds[0].to(self.device, non_blocking=True), point_clouds[1].to(self.device, non_blocking=True), point_clouds[2].to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()

                if self.config['use_contrastive_loss']:
                    concatenated_batch = torch.cat([augmented_pc1, augmented_pc2], dim=0)
                    if torch.cuda.device_count() > 1:
                        projected_concatenated = self.model.module.fc_projection(self.model.module.encoder(concatenated_batch))
                    else:
                        projected_concatenated = self.model.fc_projection(self.model.encoder(concatenated_batch))

                    batch_size = original_pc.size(0)
                    contrastive_labels = torch.arange(batch_size).repeat(2).to(self.device)
                else:
                    projected_concatenated, contrastive_labels = None, None

                reconstructed, mu, logvar, z, _  = self.model(original_pc)
                if torch.cuda.device_count() > 1:
                    loss, rec_loss, kld, contrastive_loss = self.model.module.loss_function(reconstructed, original_pc, mu, logvar, projected_concatenated, contrastive_labels, self.loss_weights)
                else:
                    loss, rec_loss, kld, contrastive_loss = self.model.loss_function(reconstructed, original_pc, mu, logvar, projected_concatenated, contrastive_labels, self.loss_weights)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_rec_loss += rec_loss.item()
                total_kld_loss += kld.item()
                if self.config['use_contrastive_loss']:
                    total_contrastive_loss += contrastive_loss.item()

                progress_bar.set_postfix({
                    'Total Loss': total_loss / (iteration + 1),
                    'Rec Loss': total_rec_loss / (iteration + 1),
                    'KL Loss': total_kld_loss / (iteration + 1),
                    'Contrastive Loss': total_contrastive_loss / (iteration + 1) if self.config['use_contrastive_loss'] else 'N/A'
                })

            except Exception as e:
                print(f"Error at iteration {iteration}: {e}")
                raise

        avg_loss = total_loss / len(self.train_loader)
        avg_rec_loss = total_rec_loss / len(self.train_loader)
        avg_kld_loss = total_kld_loss / len(self.train_loader)
        avg_contrastive_loss = total_contrastive_loss / len(self.train_loader) if self.config['use_contrastive_loss'] else 0

        self.writer.add_scalar('training/total_loss', avg_loss, self.epoch)
        self.writer.add_scalar('training/rec_loss', avg_rec_loss, self.epoch)
        self.writer.add_scalar('training/kld_loss', avg_kld_loss, self.epoch)
        if self.config['use_contrastive_loss']:
            self.writer.add_scalar('training/contrastive_loss', avg_contrastive_loss, self.epoch)

        return avg_loss

    def validate_epoch(self):
        if (self.epoch % self.config['training']['validate_every']) != 0:
            return
        self.model.eval()
        total_loss = 0.0
        total_rec_loss = 0.0
        total_kld_loss = 0.0
        total_contrastive_loss = 0.0
        with torch.no_grad():
            for point_clouds, labels in tqdm(self.val_loader):
                original_pc, augmented_pc1, augmented_pc2 = point_clouds[0].to(self.device), point_clouds[1].to(self.device), point_clouds[2].to(self.device)
                labels = labels.to(self.device)

                if self.config['use_contrastive_loss']:
                    concatenated_batch = torch.cat([augmented_pc1, augmented_pc2], dim=0)
                    if torch.cuda.device_count() > 1:
                        projected_concatenated = self.model.module.fc_projection(self.model.module.encoder(concatenated_batch))
                    else:
                        projected_concatenated = self.model.fc_projection(self.model.encoder(concatenated_batch))

                    batch_size = original_pc.size(0)
                    contrastive_labels = torch.arange(batch_size).repeat(2).to(self.device)
                else:
                    projected_concatenated, contrastive_labels = None, None

                reconstructed, mu, logvar, z, _ = self.model(original_pc)
                if torch.cuda.device_count() > 1:
                    loss, rec_loss, kld, contrastive_loss = self.model.module.loss_function(reconstructed, original_pc, mu, logvar, projected_concatenated, contrastive_labels, self.loss_weights)
                else:
                    loss, rec_loss, kld, contrastive_loss = self.model.loss_function(reconstructed, original_pc, mu, logvar, projected_concatenated, contrastive_labels, self.loss_weights)

                total_loss += loss.item()
                total_rec_loss += rec_loss.item()
                total_kld_loss += kld.item()
                if self.config['use_contrastive_loss']:
                    total_contrastive_loss += contrastive_loss.item()

        avg_loss = total_loss / len(self.val_loader)
        avg_rec_loss = total_rec_loss / len(self.val_loader)
        avg_kld_loss = total_kld_loss / len(self.val_loader)
        avg_contrastive_loss = total_contrastive_loss / len(self.val_loader) if self.config['use_contrastive_loss'] else 0

        is_best = self.best_val_loss is None or avg_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = avg_loss
            self.checkpoint(is_best=True)

        self.writer.add_scalar('validation/total_loss', avg_loss, self.epoch)
        self.writer.add_scalar('validation/rec_loss', avg_rec_loss, self.epoch)
        self.writer.add_scalar('validation/kld_loss', avg_kld_loss, self.epoch)
        if self.config['use_contrastive_loss']:
            self.writer.add_scalar('validation/contrastive_loss', avg_contrastive_loss, self.epoch)

        return avg_loss

    def train(self):
        for epoch_num in tqdm(range(self.config['training']['epochs']), desc='Epochs'):
            try:
                self.epoch = epoch_num
                avg_loss = self.train_epoch()
                self.validate_epoch()
                self.checkpoint(False)
                print(f"Epoch {self.epoch+1} finished with average loss: {avg_loss}")
            except Exception as e:
                print(f"Error in epoch {self.epoch}: {e}")
                raise

    def run(self):
        self.train()

if __name__ == '__main__':
    path_to_config = sys.argv[1]
    with open(path_to_config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    trainer = ShapeTrainer(config)
    trainer.run()