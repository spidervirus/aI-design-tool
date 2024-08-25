import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from transformers import BertModel, BertTokenizer
import os

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

# Global parameters
IMG_SIZE = 64
BATCH_SIZE = 32
HIDDEN_DIM = 512
LATENT_DIM = 100
IMG_CHANNELS = 3
LEARNING_RATE_G = 0.0002
LEARNING_RATE_D = 0.0001
EPOCHS = 1000
PATIENCE = 10
CHECKPOINT_DIR = "checkpoints"
lambda_gp = 10
GENERATOR_UPDATES = 2

# Define the Generator
class Generator(nn.Module):
    def __init__(self, input_dim, img_shape):
        super(Generator, self).__init__()
        self.init_size = img_shape[1] // 4
        self.l1 = nn.Sequential(nn.Linear(input_dim, 128 * self.init_size ** 2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=False),  # Ensure inplace=False
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=False),  # Ensure inplace=False
            nn.Conv2d(64, img_shape[0], 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, text_embedding):
        gen_input = torch.cat((noise, text_embedding), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self, img_shape, text_embedding_dim):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                nn.utils.spectral_norm(nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)),
                nn.LeakyReLU(0.2, inplace=False),  # Ensure inplace=False
                nn.Dropout2d(0.3)
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

            self.model = nn.Sequential(
            discriminator_block(img_shape[0], 16, bn=False),
            discriminator_block(16, 32),
            discriminator_block(32, 64),
            discriminator_block(64, 128)
        )

        self.init_size = img_shape[1] // 16
        self.feature_size = 128 * self.init_size ** 2

        self.adv_layer = nn.Sequential(nn.Linear(self.feature_size, 1), nn.Sigmoid())
        self.aux_layer = nn.Linear(self.feature_size, text_embedding_dim)

    def forward(self, img, text_embedding):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        return validity, label

# Text Encoder using BERT
class TransformerTextEncoder(nn.Module):
    def __init__(self, pretrained_model_name="bert-base-uncased", hidden_dim=HIDDEN_DIM):
        super(TransformerTextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.fc = nn.Linear(self.bert.config.hidden_size, hidden_dim)

    def forward(self, text_input):
        outputs = self.bert(**text_input)
        cls_output = outputs.last_hidden_state[:, 0, :]
        text_embedding = self.fc(cls_output)
        return text_embedding

# Gradient Penalty
def compute_gradient_penalty(D, real_samples, fake_samples, text_embeddings):
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=real_samples.device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates, _ = D(interpolates, text_embeddings)
    fake = torch.ones((real_samples.size(0), 1), device=real_samples.device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# Data loading
def get_dataloader(batch_size=BATCH_SIZE):
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    return dataloader

# Training loop
def train_gan():
    # Initialize models
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    text_encoder = TransformerTextEncoder()
    generator = Generator(input_dim=LATENT_DIM + HIDDEN_DIM, img_shape=(IMG_CHANNELS, IMG_SIZE, IMG_SIZE))
    discriminator = Discriminator(img_shape=(IMG_CHANNELS, IMG_SIZE, IMG_SIZE), text_embedding_dim=HIDDEN_DIM)

    adversarial_loss = nn.BCELoss()
    auxiliary_loss = nn.MSELoss()

    optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE_G, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE_D, betas=(0.5, 0.999))

    dataloader = get_dataloader()

    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    for epoch in range(EPOCHS):
        for i, (imgs, _) in enumerate(dataloader):
            batch_size = imgs.shape[0]
            valid = torch.full((batch_size, 1), 0.9, device=imgs.device, requires_grad=False)
            fake = torch.full((batch_size, 1), 0.1, device=imgs.device, requires_grad=False)

            real_imgs = imgs.to(imgs.device)
            captions = ["This is a placeholder caption" for _ in range(batch_size)]
            captions = tokenizer(captions, padding=True, truncation=True, return_tensors="pt").to(imgs.device)
            text_embeddings = text_encoder(captions)

            optimizer_D.zero_grad()

            if scaler:
                with torch.amp.autocast(device_type='cuda'):
                    real_pred, real_aux = discriminator(real_imgs, text_embeddings)
                    d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, text_embeddings)) / 2

                    noise = torch.randn(batch_size, LATENT_DIM).to(imgs.device)
                    gen_imgs = generator(noise, text_embeddings)

                    fake_pred, _ = discriminator(gen_imgs.detach(), text_embeddings)
                    d_fake_loss = adversarial_loss(fake_pred, fake)

                    gradient_penalty = compute_gradient_penalty(discriminator, real_imgs, gen_imgs.detach(), text_embeddings)
                    d_loss = (d_real_loss + d_fake_loss) / 2 + lambda_gp * gradient_penalty

                scaler.scale(d_loss).backward(retain_graph=True)
                scaler.step(optimizer_D)
                scaler.update()
            else:
                real_pred, real_aux = discriminator(real_imgs, text_embeddings)
                d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, text_embeddings)) / 2

                noise = torch.randn(batch_size, LATENT_DIM).to(imgs.device)
                gen_imgs = generator(noise, text_embeddings)

                fake_pred, _ = discriminator(gen_imgs.detach(), text_embeddings)
                d_fake_loss = adversarial_loss(fake_pred, fake)

                gradient_penalty = compute_gradient_penalty(discriminator, real_imgs, gen_imgs.detach(), text_embeddings)
                d_loss = (d_real_loss + d_fake_loss) / 2 + lambda_gp * gradient_penalty

                d_loss.backward(retain_graph=True)
                optimizer_D.step()

            for _ in range(GENERATOR_UPDATES):
                optimizer_G.zero_grad()

                if scaler:
                    with torch.amp.autocast(device_type='cuda'):
                        validity, _ = discriminator(gen_imgs, text_embeddings)
                        g_loss = adversarial_loss(validity, valid)
                    scaler.scale(g_loss).backward(retain_graph=True)
                    scaler.step(optimizer_G)
                    scaler.update()
                else:
                    validity, _ = discriminator(gen_imgs, text_embeddings)
                    g_loss = adversarial_loss(validity, valid)
                    g_loss.backward(retain_graph=True)
                    optimizer_G.step()

            print(f"[Epoch {epoch}/{EPOCHS}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

if __name__ == "__main__":
    train_gan()
