import clip
import torch.nn.functional as F
import piq
import torchvision
import lightly.models as models
import lightly.loss as loss
import pytorch_lightning as pl
from torch import nn
import torch


class SimCLR(pl.LightningModule):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = models.modules.SimCLRProjectionHead(
            input_dim=512,
            hidden_dim=512,
            output_dim=512
        )
        self.criterion = loss.NTXentLoss()

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log('loss', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.006)
        return optim

class Clip_Moudle:
    def __init__(self, text='girl', simsclr_path=''):
        self.clip_model, self.preprocess_clip = clip.load("ViT-B/32", device='cuda')
        self.clip_model.eval()
        self.text = text
        self.text_tokenize = clip.tokenize(self.text)
        self.text_features = self.clip_model.encode_text(self.text_tokenize.cuda())
        self.dists_loss_metric = piq.DISTS(reduction='mean')

        resnet = torchvision.models.resnet18()
        backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.SimCLR = SimCLR(backbone).cuda()
        checkpoint = torch.load(simsclr_path)
        self.SimCLR.load_state_dict(checkpoint['state_dict'])
        self.SimCLR.eval()

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

    def clip_similarty(self, image_features):
        similarty = self.text_features.cpu().detach().numpy() @ image_features.cpu().detach().numpy().T
        return torch.tensor(similarty, requires_grad=True)

    def process_images(self, generated_images):
        process_images = []
        transform = transforms.ToPILImage()
        for image in generated_images:
            process_images.append(self.preprocess_clip(transform(image)))

        return torch.stack(process_images)

    def dists_loss(self, imaegs, generated_images):
        return self.dists_loss_metric(imaegs, generated_images)

    def image_features_SimCLR(self, image_1):
        image_1_features = self.SimCLR(image_1.float())
        return image_1_features

    def SimCLR_loss(self, image_1, image_2):
        image_1_features = self.SimCLR(image_1.float())
        image_2_features = self.SimCLR(image_2.float())
        return F.mse_loss(image_1_features, image_2_features, reduction='mean').cuda()


class text_style_gan(pl.LightningModule):
    def __init__(self, style_gan, style, text='', simsclr_path=''):
        super().__init__()
        self.decoder = style_gan
        self.style = style
        self.text = text
        self.decoder.eval()
        self.clip_func = Clip_Moudle(text=text, simsclr_path=simsclr_path)
        self.mapper = nn.Sequential(
            nn.Linear(512, 512), nn.GELU(),
            nn.Linear(512, 512), nn.GELU(),
            nn.Linear(512, 512), nn.GELU(),
            nn.Linear(512, 512), nn.GELU()
        )
        self.use_clip_text = text != ''

    def configure_optimizers(self):
        return torch.optim.Adam(self.mapper.parameters(), lr=1e-4)

    def latent_to_w(self, style_vectorizer, latent_descr):
        return [(style_vectorizer(z), num_layers) for z, num_layers in latent_descr]

    def training_step(self, batch, batch_idx):
        images, y = batch
        batch_size = images.shape[0]

        preprocess_images = self.clip_func.process_images(images)
        image_features_clip = self.clip_func.image_features(preprocess_images)
        image_features_SimCLR = self.clip_func.image_features_SimCLR(images)

        latent_delta = self.mapper(image_features_SimCLR.float())

        w_plus = [(image_features_SimCLR + latent_delta, 6)]

        noise = torch.FloatTensor(batch_size, 128, 128, 1).uniform_(0., 1.).cuda()
        w_space = self.latent_to_w(self.style, w_plus)
        w_styles = torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in w_space], dim=1)
        generated_images = self.decoder(w_styles, noise)
        process_generated_images = self.clip_func.process_images(generated_images)

        generated_images_featuers_clip = self.clip_func.image_features(process_generated_images)
        generated_images_featuers_SimCLR = self.clip_func.image_features_SimCLR(generated_images)

        if self.use_clip_text:
            clip_similarty_generated = self.clip_func.clip_similarty(generated_images_featuers_clip).cuda().mean()
            clip_similarty_images = self.clip_func.clip_similarty(image_features_clip).cuda().mean()
            clip_loss = clip_similarty_images - clip_similarty_generated

        dists_loss = self.clip_func.dists_loss(images, generated_images).cuda()
        clip_encodeing_mse = F.mse_loss(generated_images_featuers_clip, image_features_clip, reduction="mean").cuda()
        imaegs_similarty = self.clip_func.image_similarty(generated_images_featuers_clip,
                                                          image_features_clip).cuda().mean()
        SimCLR_loss = self.clip_func.SimCLR_loss(images, generated_images)
        mse_loss = F.mse_loss(images, generated_images, reduction='mean').cuda()

        total_loss = clip_encodeing_mse + imaegs_similarty + dists_loss + mse_loss + SimCLR_loss

        if self.use_clip_text:
            total_loss = clip_encodeing_mse + clip_loss + imaegs_similarty + dists_loss + mse_loss + SimCLR_loss

        preprocess_images = preprocess_images.cpu()
        image_features_clip = image_features_clip.cpu()
        image_features_SimCLR = image_features_SimCLR.cpu()
        generated_images = generated_images.cpu()
        process_generated_images = process_generated_images.cpu()
        generated_images_featuers_clip = generated_images_featuers_clip.cpu()
        generated_images_featuers_SimCLR = generated_images_featuers_SimCLR.cpu()

        self.log('dists_loss', dists_loss)
        self.log('clip_encodeing_mse', clip_encodeing_mse)
        if self.text != '':
            self.log(f'clip_loss:"{self.text}"', clip_loss)
        self.log('imaegs_similarty', imaegs_similarty)
        self.log('mse_loss', mse_loss)
        self.log('total_loss', total_loss)
        self.log('SimCLR_loss', SimCLR_loss)

        if batch_idx % 70 == 0:
            self.logger.log_image(key="samples", images=[images[0:2], generated_images[0:2]])

        return total_loss


