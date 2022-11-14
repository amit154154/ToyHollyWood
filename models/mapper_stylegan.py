import clip
import torch.nn.functional as F

import pytorch_lightning as pl
from torch import nn
import torch
import piq
from torchvision import transforms

class Clip_Moudle:
    def __init__(self):
        self.clip_model, self.preprocess_clip = clip.load("ViT-L/14", device='cuda')
        self.clip_model.eval()
        self.dists_loss_metric = piq.DISTS(reduction='mean')

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self.inv_normalize = transforms.Normalize(
            mean=[-m / s for m, s in zip(mean, std)],
            std=[1 / s for s in std]
        )

    def image_features(self, preproces_image):
        with torch.no_grad():
            encode = self.clip_model.encode_image(preproces_image.cuda())
            preproces_image = preproces_image.cpu()
            return encode

    def image_similarty(self, image_features_1, image_features_2):
        image_features_1 /= image_features_1.norm(dim=-1, keepdim=True)
        image_features_2 /= image_features_2.norm(dim=-1, keepdim=True)
        similarity = image_features_2.cpu().detach().numpy() @ image_features_1.cpu().detach().numpy().T
        image_features_1 = image_features_1.cpu()
        image_features_2 = image_features_2.cpu()
        return torch.tensor(1 - similarity, requires_grad=True)

    def process_images(self, generated_images):
        process_images = []
        transform = transforms.ToPILImage()
        for image in generated_images:
            process_images.append(self.preprocess_clip(transform(image)))

        return torch.stack(process_images)

    def dists_loss(self, imaegs, generated_images):
        return self.dists_loss_metric(imaegs, generated_images)


class mapper_train(pl.LightningModule):
    def __init__(self, decoder ,mapping ,mean_latent_clip):
        super().__init__()
        self.decoder = decoder
        self.mapping = mapping
        self.decoder.eval()
        self.clip_func = Clip_Moudle()
        self.mapper = nn.Sequential(
            nn.Linear(768 ,768), nn.GELU(),
            nn.Linear(768, 6* 512), nn.GELU(),
            nn.Linear(6 * 512, 6 * 512), nn.GELU(),
            nn.Linear(6 * 512, 6 * 512), nn.GELU()
        )
        self.mean_latent_clip = mean_latent_clip.cuda()

    def configure_optimizers(self):
        # params = list(self.mapper.parameters()) + list(self.clip_func.encoder.parameters(),)
        parms = list(self.mapper.parameters())
        return torch.optim.Adam(parms, lr=1e-4)

    def training_step(self, batch, batch_idx):
        random_latents = batch.cuda()
        batch_size = random_latents.shape[0]

        z_plus_random = random_latents.reshape(batch_size * 6, 512)
        w_space_random = self.mapping(z_plus_random).reshape(batch_size, 6, 512)
        noise = torch.FloatTensor(batch_size, 128, 128, 1).uniform_(0., 1.).cuda()
        random_images = self.decoder(w_space_random, noise)

        preprocess_images = self.clip_func.process_images(random_images)
        random_images_features_clip = self.clip_func.image_features(preprocess_images).cuda()

        w_spaces = []
        for i in range(batch_size):
            w_plus = self.mapper(
                (random_images_features_clip[i].reshape(1, 768) - self.mean_latent_clip).float()).reshape(6, 512)
            w_space = self.mapping(w_plus).reshape(1, 6, 512)
            w_spaces.append(w_space)

        if batch_idx % 100 == 0:
            noise = torch.FloatTensor(1, 128, 128, 1).uniform_(0., 1.).cuda()
            generated_image = self.decoder(w_space, noise)
            self.logger.log_image(key="samples", images=[random_images[-1], generated_image])

        w_spaces = torch.stack(w_spaces).reshape(batch_size, 6, 512)
        w_space_mse = F.mse_loss(w_spaces, w_space_random, reduction="mean").cuda()
        # cosin_similarty = F.cosine_similarity(w_spaces, w_space_random).abs().mean()

        total_loss = w_space_mse
        if batch_idx % 10 == 0:
            self.log_dict({'loss': total_loss, 'mse': w_space_mse})
        return total_loss
