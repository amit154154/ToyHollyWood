import pytorch_lightning as pl
from torch import nn
import torch
from pl_bolts.models.autoencoders.components import (
    resnet18_decoder,
    resnet18_encoder,
)
import torch.nn.functional as F
import piq


class VAE(pl.LightningModule):
    def __init__(self, enc_out_dim=512, latent_dim=256, input_height=80):
        super().__init__()

        self.save_hyperparameters()

        # encoder, decoder
        self.encoder = resnet18_encoder(False, False)
        self.decoder = resnet18_decoder(
            latent_dim=latent_dim,
            input_height=input_height,
            first_conv=False,
            maxpool1=False
        )

        # distribution parameters
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))
        self.kl_coeff = 0.1
        self.brisque_coeff = 0.001
        self.ssim_coeff = 2.0
        self.piq_loss = piq.DISTS(reduction='none')

    #         self.ssim_loss = SSIMLoss(data_range=1.)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def _run_step(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return z, self.decoder(z), p, q

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def training_step(self, batch, batch_idx):
        x, y = batch
        z, x_hat, p, q = self._run_step(x)

        recon_loss = F.mse_loss(x_hat, x, reduction="mean")

        kl = torch.distributions.kl_divergence(q, p)
        kl = kl.mean()
        kl *= self.kl_coeff
        dists_loss = self.piq_loss(x_hat, x).mean()
        loss = kl + recon_loss + dists_loss

        self.log(f"loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log(f"kl", kl, on_epoch=False, prog_bar=True)
        self.log(f"recon_loss", recon_loss, on_epoch=False, prog_bar=True)
        self.log(f"dists", dists_loss)
        if batch_idx % 70 == 0:
            self.logger.log_image(key="samples", images=[x[0:3], x_hat[0:3]])

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z, x_hat, p, q = self._run_step(x)

        recon_loss = F.mse_loss(x_hat, x, reduction="mean")

        kl = torch.distributions.kl_divergence(q, p)
        kl = kl.mean()
        kl *= self.kl_coeff
        dists_loss = self.piq_loss(x_hat, x).mean()
        loss = kl + recon_loss + dists_loss

        self.log(f"loss_val", loss)
        self.log(f"kl_val", kl)
        self.log(f"recon_loss_val", recon_loss)
        self.log(f"dists_val", dists_loss)

        if batch_idx % 70 == 0:
            self.logger.log_image(key="samples_val", images=[x[0:3], x_hat[0:3]])

        return loss


