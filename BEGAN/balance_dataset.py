import os

import logging
import torch
from torchvision.utils import save_image
import numpy as np
from torch.autograd import Variable
import torch.nn as nn


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


device = 'cuda'
model_paths = {
    'b': '/home/joseruiz/Desktop/Doctorado/BEGAN/runs/15-04-2024/weights/14400.pt',
    'm': '/home/joseruiz/Desktop/Doctorado/BEGAN/runs/15-04-2024_m/weights/9600.pt'
    }
base_path = '/home/joseruiz/Desktop/Doctorado/custom_dataset/began'
target_number = 2000
Tensor = torch.cuda.FloatTensor
if __name__ == '__main__':
    log_format = logging.Formatter("%(asctime)s: %(message)s")

    logger = logging.getLogger("train-logger")
    logger.setLevel(logging.INFO)
    f_handler = logging.FileHandler('balance_sngan_regan.log')
    f_handler.setFormatter(log_format)

    p_handler = logging.StreamHandler()
    p_handler.setFormatter(log_format)

    logger.addHandler(f_handler)
    logger.addHandler(p_handler)
    
    benign = '/home/joseruiz/Desktop/Doctorado/dataset/b'
    malignant = '/home/joseruiz/Desktop/Doctorado/dataset/m'
    
    benign_imgs = os.listdir(benign)
    malignant_imgs = os.listdir(malignant)

    total_images = len(benign_imgs) + len(malignant_imgs)
    logger.info(f'Total imgs:{total_images}, Benign: {len(benign_imgs)}, Malignant: {len(malignant_imgs)}')
    data_dirs = {
        'b': benign,
        'm': malignant
    }
    imgs_number = {
        'b': len(benign_imgs),
        'm': len(malignant_imgs)
    }
    
    os.makedirs(base_path, exist_ok=True)
    
    for k in model_paths:
        os.makedirs(os.path.join(base_path, k), exist_ok=True)
        
        model_path = model_paths[k]
        
        model = torch.load(model_path)
        # model.load_state_dict(model_dict)
        model.to(device)
        model.eval()
        
        img2create = max(0, target_number - imgs_number[k])
        logger.info(f'{img2create} number of images will be created')
        for i in range(img2create):
            z = Tensor(np.random.normal(0, 1, (1, 128)))
            img = model(z)
            logger.info(f'Generate image shape {img.shape}')
            # logger.info(f'Saving image {base_path}/{k}/{i}.png')
            save_image(img[0], f'{base_path}/{k}/{i}.png')


