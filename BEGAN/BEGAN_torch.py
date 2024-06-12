import argparse
import asyncio
import json
import h5py as h5
import os
import numpy as np

import time
import torchvision.transforms as transforms
from datetime import datetime, timedelta
from torchvision.utils import save_image

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch



def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class HDF5_dataset(Dataset):
  def __init__(self, root, transform=None, target_transform=None,
               load_in_mem=False, train=True,download=False, validate_seed=0,
               val_split=0, **kwargs): # last four are dummies
      
    self.root = root
    self.num_imgs = len(h5.File(root, 'r')['labels'])
    
    # self.transform = transform
    self.target_transform = target_transform   
    
    # Set the transform here
    self.transform = transform
    
    # load the entire dataset into memory? 
    self.load_in_mem = load_in_mem
    
    # If loading into memory, do so now
    if self.load_in_mem:
      print('Loading %s into memory...' % root)
      with h5.File(root,'r') as f:
        self.data = f['imgs'][:]
        self.labels = f['labels'][:]

  def __getitem__(self, index):
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is class_index of the target class.
    """
    # If loaded the entire dataset in RAM, get image from memory
    if self.load_in_mem:
      img = self.data[index]
      target = self.labels[index]
    
    # Else load it from disk
    else:
      with h5.File(self.root,'r') as f:
        img = f['imgs'][index]
        target = f['labels'][index]
    
   
    # if self.transform is not None:
        # img = self.transform(img)
    # Apply my own transform
    img = ((torch.from_numpy(img).float() / 255) - 0.5) * 2
    
    if self.target_transform is not None:
      target = self.target_transform(target)
    
    return img, int(target)

  def __len__(self):
      return self.num_imgs
      # return len(self.f['imgs'])


class Generator(nn.Module):
    def __init__(self, img_size, latent_dim, channels):
        super(Generator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

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
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_size, channels):
        super(Discriminator, self).__init__()
        # Upsampling
        self.down = nn.Sequential(nn.Conv2d(channels, 64, 3, 2, 1), nn.ReLU())
        # Fully-connected layers
        self.down_size = img_size // 2
        down_dim = 64 * (img_size // 2) ** 2
        self.fc = nn.Sequential(
            nn.Linear(down_dim, 32),
            nn.BatchNorm1d(32, 0.8),
            nn.ReLU(inplace=True),
            nn.Linear(32, down_dim),
            nn.BatchNorm1d(down_dim),
            nn.ReLU(inplace=True),
        )
        # Upsampling
        self.up = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(64, channels, 3, 1, 1))

    def forward(self, img):
        out = self.down(img)
        out = self.fc(out.view(out.size(0), -1))
        out = self.up(out.view(out.size(0), 64, self.down_size, self.down_size))
        return out


def run(config, path):
    # Create directory
    os.makedirs(f'{path}/weights', exist_ok=True)
    os.makedirs(f"{path}/images", exist_ok=True)
    
    # Initialize generator and discriminator
    generator = Generator(img_size=config.img_size, channels=config.channels, latent_dim=config.latent_dim)
    discriminator = Discriminator(img_size=config.img_size, channels=config.channels)
    
    # Check if GPU is available
    cuda = torch.cuda.is_available()
    
    if cuda:
        generator.cuda()
        discriminator.cuda()

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Configure data loader USE NEW DATALOADER
    dataset = HDF5_dataset(
        config.dataset,
        transform=transforms.Compose(
                [transforms.Resize(config.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                )
        )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    # dataloader = torch.utils.data.DataLoader(
    #     datasets.MNIST(
    #         "../../data/mnist",
    #         train=True,
    #         download=True,
    #         transform=transforms.Compose(
    #             [transforms.Resize(config.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    #         ),
    #     ),
    #     batch_size=config.batch_size,
    #     shuffle=True,
    # )

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=config.lr, betas=(config.b1, config.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config.lr, betas=(config.b1, config.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    #  Training
    # ----------

    # BEGAN hyper parameters
    gamma = 0.75
    lambda_k = 0.001
    k = 0.0

    start = time.time()
    for epoch in range(config.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], config.latent_dim))))

            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = torch.mean(torch.abs(discriminator(gen_imgs) - gen_imgs))

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            d_real = discriminator(real_imgs)
            d_fake = discriminator(gen_imgs.detach())

            d_loss_real = torch.mean(torch.abs(d_real - real_imgs))
            d_loss_fake = torch.mean(torch.abs(d_fake - gen_imgs.detach()))
            d_loss = d_loss_real - k * d_loss_fake

            d_loss.backward()
            optimizer_D.step()

            # ----------------
            # Update weights
            # ----------------

            diff = torch.mean(gamma * d_loss_real - d_loss_fake)

            # Update weight term for fake samples
            k = k + lambda_k * diff.item()
            k = min(max(k, 0), 1)  # Constraint to interval [0, 1]

            # Update convergence metric
            M = (d_loss_real + torch.abs(diff)).item()

            # --------------
            # Log Progress
            # --------------

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] -- M: %f, k: %f"
                % (epoch, config.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), M, k)
            )

            batches_done = epoch * len(dataloader) + i
            if batches_done % config.sample_interval == 0:
                torch.save(generator, f'{path}/weights/{batches_done}.pt', )
                last_image = f"{path}/images/{batches_done}.png"
                save_image(gen_imgs.data[:25], last_image, nrow=5, normalize=True)


def prepare_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="global path to selected dataset")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=512, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="number of image channels")
    
    return parser

def main():
    parser = prepare_parser()
    config = parser.parse_args()
    today = datetime.today().strftime('%d-%m-%Y')
    os.makedirs(f'runs/{today}', exist_ok=True)
    with open(f'runs/{today}/parameters.json', 'w') as f:
        json.dump(vars(config), f, indent=2)
    print(config)
    run(config, f'runs/{today}')

if __name__ == '__main__':
    main()

