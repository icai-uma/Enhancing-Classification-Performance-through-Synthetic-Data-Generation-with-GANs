import os

import logging
import torch
from torchvision.utils import save_image


device = 'cuda'
model_paths = {
    'b': './regan_progan_b.pt',
    'm': './regan_progan_m.pt'
    }
base_path = '/home/joseruiz/Desktop/Doctorado/custom_dataset/regan-progan'
target_number = 2000
if __name__ == '__main__':
    log_format = logging.Formatter("%(asctime)s: %(message)s")
    logger = logging.getLogger("train-logger")
    logger.setLevel(logging.INFO)
    f_handler = logging.FileHandler('balance_progan_regan.log')
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
        # model = ResGenerator32(128)
        model = torch.load(model_path)
        # model.load_state_dict(model_dict)
        model.to(device)
        
        noise = torch.randn(1, 512, 1, 1).to(device)
        
        img2create = max(0, target_number - imgs_number[k])
        logger.info(f'{img2create} number of images will be created')
        for i in range(img2create):
            img = model(noise, 1, 3)
            logger.info(f'Generate image shape {img.shape}')
            # logger.info(f'Saving image {base_path}/{k}/{i}.png')
            save_image(img[0], f'{base_path}/{k}/{i}.png')


