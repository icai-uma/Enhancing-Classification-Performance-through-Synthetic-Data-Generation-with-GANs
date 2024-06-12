import sys

sys.path.append('pro_gan_pytorch')
sys.path.append('pro_gan_pytorch_scripts')

from pro_gan_pytorch_scripts.train import train_progan, parse_arguments


if __name__ == '__main__':
    train_progan(parse_arguments())

