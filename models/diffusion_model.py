from torch import nn
import torch
import math
import pytorch_lightning as pl
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t, ):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(...,) + (None,) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """

    def __init__(self, down_channels=(64, 128, 256, 512, 1024)
                 , up_channels=(1024, 512, 256, 128, 64)
                 , time_emb_dim=32
                 ):
        super().__init__()
        image_channels = 3
        out_dim = 1

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i + 1], \
                                          time_emb_dim) \
                                    for i in range(len(down_channels) - 1)])
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i + 1], \
                                        time_emb_dim, up=True) \
                                  for i in range(len(up_channels) - 1)])

        self.output = nn.Conv2d(up_channels[-1], 3, out_dim)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        return self.output(x)


class diffusion_sample:
    def __init__(self, T=300):
        self.T = T
        betas = self.linear_beta_schedule(timesteps=T)

        # Pre-calculate different terms for closed form
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        self.posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    def linear_beta_schedule(self, timesteps, start=0.0001, end=0.02):
        return torch.linspace(start, end, timesteps)

    def get_index_from_list(self, vals, t, x_shape):
        """
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def forward_diffusion_sample(self, x_0, t, device="cpu"):
        """
        Takes an image and a timestep as input and
        returns the noisy version of it
        """

        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        # mean + variance
        # return  x_noisy, noise
        return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
               + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


class diffusion_train(pl.LightningModule):
    def __init__(self, model, T=300, lr=0.01, device='cuda'):
        super().__init__()
        self.model = model
        self.sample = diffusion_sample(T)
        self.T = T
        self.lr = lr
        self.d = device

    def linear_beta_schedule(self, timesteps, start=0.0001, end=0.02):
        return torch.linspace(start, end, timesteps)

    def get_index_from_list(self, vals, k, x_shape):
        """
        Retu\rns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = k.shape[0]
        out = vals.gather(-1, k.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).cuda()

    @torch.no_grad()
    def sample_timestep(self, x, k):
        """
        Calls the model to predict the noise in the image and returns
        the denoised image.
        Applies noise to this image, if we are not in the last step yet.
        """
        T = 300
        betas = self.linear_beta_schedule(timesteps=T)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        betas_t = self.get_index_from_list(betas, k, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            sqrt_one_minus_alphas_cumprod, k, x.shape
        )
        sqrt_recip_alphas_t = self.get_index_from_list(sqrt_recip_alphas, k, x.shape)

        # Call model (current image - noise prediction)
        model_mean = sqrt_recip_alphas_t.cuda() * (
                x.cuda() - betas_t.cuda() * self.model(x, k).cuda() / sqrt_one_minus_alphas_cumprod_t.cuda()
        )
        posterior_variance_t = self.get_index_from_list(posterior_variance, k, x.shape)
        if k == 0:
            return model_mean
        else:
            noise = torch.randn_like(x).cuda()
            return model_mean + torch.sqrt(posterior_variance_t).cuda() * noise

    def random_image(self):
        betas = self.linear_beta_schedule(timesteps=self.T)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # Sample noise
        img_size = 80
        img = torch.randn((3, 3, img_size, img_size), device='cuda')

        for i in range(0, self.T)[::-1]:
            r = torch.full((1,), i, device=self.d, dtype=torch.long)
            img = self.sample_timestep(img.cuda(), r)
        return [img[i].detach().cpu() for i in range(len(img))]

    def training_step(self, batch, count, **kwargs):
        images, _ = batch
        BATCH_SIZE = images.size()[0]
        t = torch.randint(0, self.T, (BATCH_SIZE,), device=self.d).long()
        x_noisy, noise = self.sample.forward_diffusion_sample(images, t, self.d)
        noise_pred = self.model(x_noisy, t)
        l1_loss = F.l1_loss(noise, noise_pred)
        self.log('l1_loss', l1_loss)
        loss = l1_loss
        with torch.no_grad():
            if count % 50 == 0:
                print('1')
                images = self.random_image()
                self.logger.log_image(key="samples", images=images)

        return loss

    def validation_step(self, batch, count, **kwargs):
        images, _ = batch
        BATCH_SIZE = images.size()[0]
        t = torch.randint(0, self.T, (BATCH_SIZE,), device=self.d).long()
        x_noisy, noise = self.sample.forward_diffusion_sample(images, t, self.d)
        noise_pred = self.model(x_noisy, t)
        l1_loss = F.l1_loss(noise, noise_pred)
        # self.log('l1_loss_val', l1_loss)
        loss = l1_loss

        if count % 50 == 0:
            real = x_noisy
            # self.logger.log_image(key="samples_val", images=[noise[0:3], fake[0:3]])

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        sch = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.5)
        # learning rate scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": "l1_loss",

            }
        }
