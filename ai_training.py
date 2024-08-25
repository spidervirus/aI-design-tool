import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BertModel, BertTokenizer
import matplotlib.pyplot as plt
import numpy as np
import os

# Define global parameters
IMG_SIZE = 64
BATCH_SIZE = 32
EMBED_DIM = 256
HIDDEN_DIM = 512
LATENT_DIM = 100
IMG_CHANNELS = 3
LEARNING_RATE = 0.0002
EPOCHS = 1000
PATIENCE = 10  # Early stopping patience
CHECKPOINT_DIR = "checkpoints"
START_EPOCH = 0  # Default start epoch
lambda_gp = 15  # Weight for the gradient penalty
GENERATOR_UPDATES = 2
LEARNING_RATE_D = 0.0001  # Further lower the discriminator's learning rate
LEARNING_RATE_G = 0.0002  # Generator learning rate remains the same

# Define the Generator with Residual Blocks
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        return x + self.block(x)

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
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, img_shape[0], 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, text_embedding):
        gen_input = torch.cat((noise, text_embedding), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# Define the Discriminator with Spectral Normalization
class Discriminator(nn.Module):
    def __init__(self, img_shape, text_embedding_dim):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                nn.utils.spectral_norm(nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)),  # Spectral Normalization
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.3)  # Added Dropout for regularization
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(img_shape[0], 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128)
        )

        self.init_size = img_shape[1] // 16
        self.feature_size = 128 * self.init_size ** 2

        self.adv_layer = nn.Sequential(
            nn.Linear(self.feature_size, 1),
            nn.Sigmoid()
        )
        self.aux_layer = nn.Sequential(
            nn.Linear(self.feature_size, text_embedding_dim)
        )

    def forward(self, img, text_embedding):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        return validity, label

# Define the Transformer-based Text Encoder
class TransformerTextEncoder(nn.Module):
    def __init__(self, pretrained_model_name="bert-base-uncased", hidden_dim=HIDDEN_DIM):
        super(TransformerTextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(self.bert.config.hidden_size, hidden_dim)

    def forward(self, text_input):
        outputs = self.bert(**text_input)
        last_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)
        cls_output = last_hidden_state[:, 0, :]  # Use [CLS] token output
        text_embedding = self.fc(cls_output)  # Reduce dimensionality if needed
        return text_embedding

# Early Stopping and Checkpointing
class EarlyStopping:
    def __init__(self, patience=PATIENCE, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, epoch)
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch):
        '''Saves model when validation loss decreases.'''
        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
        }, os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth"))
        if self.verbose:
            print(f"Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model...")

# Load checkpoint if available
def load_checkpoint(generator, discriminator, optimizer_G, optimizer_D):
    global START_EPOCH

    if os.path.exists(CHECKPOINT_DIR):
        checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pth')]
        
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[2].split('.')[0]))
            checkpoint_path = os.path.join(CHECKPOINT_DIR, latest_checkpoint)
            
            print(f"Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            
            try:
                generator.load_state_dict(checkpoint['model_state_dict'], strict=False)
                discriminator.load_state_dict(checkpoint['model_state_dict'], strict=False)
                optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
                optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
            except RuntimeError as e:
                print(f"Error loading checkpoint: {e}")

            START_EPOCH = checkpoint['epoch'] + 1
            print(f"Checkpoint loaded, resuming from epoch {START_EPOCH - 1}")
        else:
            print("No checkpoints found, starting from scratch.")
            START_EPOCH = 0
    else:
        print(f"Checkpoint directory {CHECKPOINT_DIR} does not exist, starting from scratch.")
        START_EPOCH = 0

# Compute Gradient Penalty
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

# Data loading and processing
def get_dataloader(batch_size=BATCH_SIZE):
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    return dataloader

# Save and visualize generated images
def save_and_visualize_images(images, epoch, save_dir="generated_images", nrow=5):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    image_grid = make_image_grid(images, nrow=nrow)
    image_path = os.path.join(save_dir, f"epoch_{epoch}.png")
    plt.imsave(image_path, image_grid)

    plt.figure(figsize=(10, 10))
    plt.imshow(image_grid)
    plt.axis('off')
    plt.show()

def make_image_grid(images, nrow=5):
    images = (images + 1) / 2  # Denormalize images from [-1, 1] to [0, 1]
    images = images.permute(0, 2, 3, 1).cpu().detach().numpy()  # Rearrange dimensions for plotting
    ncol = images.shape[0] // nrow
    grid = np.concatenate([np.concatenate(images[i*nrow:(i+1)*nrow], axis=1) for i in range(ncol)], axis=0)
    return grid

# Training loop with increased generator updates
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

    # Initialize learning rate schedulers
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=100, gamma=0.1)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=100, gamma=0.1)

    writer = SummaryWriter(log_dir='runs/test_experiment')
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True)
    dataloader = get_dataloader()

    # Load the last checkpoint if available
    load_checkpoint(generator, discriminator, optimizer_G, optimizer_D)

    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    for epoch in range(START_EPOCH, EPOCHS):
        g_loss_avg = 0.0
        d_loss_avg = 0.0

        for i, (imgs, _) in enumerate(dataloader):
            batch_size = imgs.shape[0]
            valid = torch.full((batch_size, 1), 0.9, device=imgs.device, requires_grad=False)
            fake = torch.full((batch_size, 1), 0.1, device=imgs.device, requires_grad=False)

            real_imgs = imgs.type(torch.FloatTensor).to(imgs.device)
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

            d_loss_avg += d_loss.item()

            for _ in range(GENERATOR_UPDATES):  # Generator updates more frequently
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

                g_loss_avg += g_loss.item()

            print(f"[Epoch {epoch}/{EPOCHS}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

        g_loss_avg /= len(dataloader)
        d_loss_avg /= len(dataloader)

        writer.add_scalar('Loss/Generator', g_loss_avg, epoch)
        writer.add_scalar('Loss/Discriminator', d_loss_avg, epoch)

        early_stopping(g_loss_avg + d_loss_avg, generator, epoch)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        scheduler_G.step()
        scheduler_D.step()

        save_and_visualize_images(gen_imgs.data[:25], epoch)

    writer.close()

# Start training
if __name__ == "__main__":
    img_shape = (3, 64, 64)
    text_embedding_dim = 512
    discriminator = Discriminator(img_shape, text_embedding_dim)
    print(discriminator)
    train_gan()
