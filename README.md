# Enhancing Histopathological Image Classification Performance through Synthetic Data Generation with Generative Adversarial Networks
This repository contains all the code and resources necessary to reproduce the experiments described in the paper "Enhancing Histopathological Image Classification Performance through Synthetic Data Generation with Generative Adversarial Networks." The included repositories cover the various stages of our research, from data preprocessing and GAN-based image augmentation to training and evaluating Convolutional Neural Networks (CNNs) within Computer-Aided Diagnosis (CAD) systems. By following the provided instructions, one can replicate our results and further explore the potential of GAN-based techniques in medical image analysis.

---

## Set Up

Each GAN implementation in this repository comes with its own `requirements.txt` file, listing all the necessary libraries and dependencies. To ensure proper functionality, please make sure that PyTorch and CUDA are installed on your system.

### Steps:

1. **Install CUDA**: Follow the [CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#installing-cuda-development-tools) to install CUDA on your system.

2. **Install PyTorch**: Once CUDA is installed, follow the [PyTorch installation guide](https://pytorch.org/get-started/locally/) to install the appropriate version of PyTorch for your system.

3. **Install Python Libraries**: Navigate to the directory of each GAN implementation and run the following command to install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

By ensuring these steps are completed, you'll be able to run all the experiments and reproduce the results from our paper.

---
Sure, here's how you can structure the section for ProGAN in your README file:

---

## ProGAN

The `ProGAN` directory contains all the necessary files and scripts to train and evaluate a Progressive GAN model. Below is a description of the contents and instructions on how to use them.

### Directory Structure:

```
ProGAN
│   README.md
│   req.txt
│   train_wrapper.py
│
├───pro_gan_pytorch
│   │   custom_layers.py
│   │   data_tools.py
│   │   gan.py
│   │   losses.py
│   │   modules.py
│   │   networks.py
│   │   utils.py
│   │   __init__.py
│   │
│   └───test
│           conftest.py
│           test_custom_layers.py
│           test_gan.py
│           test_networks.py
│           utils.py
│           __init__.py
│
└───pro_gan_pytorch_scripts
        latent_space_interpolation.py
        train.py
        __init__.py
```

#### Setup Instructions:

1. **Install Requirements**: Navigate to the `ProGAN` directory and install the required Python libraries using the provided `req.txt` file:
   ```bash
   pip install -r req.txt
   ```

2. **Training the ProGAN Model**: Use the `train.py` script located in the `pro_gan_pytorch_scripts` directory to train the ProGAN model. Run the script with appropriate arguments for your dataset and training configuration:
   ```bash
   python pro_gan_pytorch_scripts/train.py --config your_config_file.json
   ```

3. **Computing FID**: To evaluate the trained model using the Frechet Inception Distance (FID) score, use the `compute_fid.py` script:
   ```bash
   python pro_gan_pytorch_scripts/compute_fid.py --model_path path_to_your_model --data_path path_to_dataset
   ```

4. **Latent Space Interpolation**: To perform latent space interpolation and visualize the generated images, use the `latent_space_interpolation.py` script:
   ```bash
   python pro_gan_pytorch_scripts/latent_space_interpolation.py --model_path path_to_your_model --output_path path_to_save_images
   ```

#### File Descriptions:

- `README.md`: This file.
- `req.txt`: Contains the list of required Python libraries.
- `train_wrapper.py`: Wrapper script to facilitate the training process.
- `pro_gan_pytorch`: Directory containing the core implementation of the ProGAN model, including custom layers, GAN architecture, losses, and utility functions.
- `pro_gan_pytorch/test`: Directory containing unit tests for the ProGAN implementation.
- `pro_gan_pytorch_scripts`: Directory containing scripts for training, evaluating, and visualizing the ProGAN model.

By following these steps and utilizing the provided scripts, you will be able to reproduce the experiments and explore the capabilities of ProGAN for generating synthetic histopathological images.