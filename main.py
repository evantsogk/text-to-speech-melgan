from utils.transforms import Audio2Mel
from utils.dataloader import loaders
from utils.models import Generator, Discriminator
from utils.trainer import train
import yaml
import argparse
from pathlib import Path
import torch


def parse_args():
    """
    Configuration.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=Path, required=True)  # path to save logs and checkpoints
    parser.add_argument("--load_path", type=Path, default=None)  # path to checkpoints
    parser.add_argument("--data_path", type=Path, required=True)  # path to data

    parser.add_argument("--n_mel_channels", type=int, default=80)  # number of mel spectrogram channels
    parser.add_argument("--num_D", type=int, default=3)  # number of discriminator blocks
    parser.add_argument("--n_layers_D", type=int, default=4)  # number of strided convolutional layers
    parser.add_argument("--lambda_feat", type=float, default=10)  # lambda value for the feature matching loss

    parser.add_argument("--batch_size", type=int, default=16)  # batch size
    parser.add_argument("--seq_len", type=int, default=8192)  # duration of audio samples for training

    parser.add_argument("--epochs", type=int, default=100000)  # number of epochs to train
    parser.add_argument("--log_interval", type=int, default=100)  # (in steps) prints losses
    parser.add_argument("--checkpoint_interval", type=int, default=10)  # (in epochs) overwrites previous checkpoints
    parser.add_argument("--generate_interval", type=float, default=0.05)  # (in hours) saves samples and generator
    parser.add_argument("--training_time", type=float, default=0.1)  # how many hours to train (optional)
    return parser.parse_args()


if __name__ == "__main__":
    # dump arguments
    args = parse_args()
    args.save_path.mkdir(parents=True, exist_ok=True)  # create save directory if it doesn't exist
    with open(args.save_path / "args.yml", "w") as f:
        yaml.dump(args, f)

    # get default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # initialize models
    generator = Generator(args.n_mel_channels).to(device)
    discriminator = Discriminator().to(device)
    fft = Audio2Mel(n_mel_channels=args.n_mel_channels).to(device)

    # print models
    print(generator)
    print(discriminator)

    # initialize optimizers
    optim_g = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optim_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))

    # continue training from checkpoints if args.load_path is provided
    if args.load_path and args.load_path.exists():
        generator.load_state_dict(torch.load(args.load_path / "generator.pt"))
        optim_g.load_state_dict(torch.load(args.load_path / "optim_g.pt"))
        discriminator.load_state_dict(torch.load(args.load_path / "discriminator.pt"))
        optim_d.load_state_dict(torch.load(args.load_path / "optim_d.pt"))
        print("Successfully loaded checkpoints!")

    # create loaders
    train_loader, test_loader = loaders(batch_size=args.batch_size, path=args.data_path, seq_len=args.seq_len,
                                        num_workers=2)

    # start training
    train(train_loader=train_loader, test_loader=test_loader, fft=fft, generator=generator, discriminator=discriminator,
          optim_g=optim_g, optim_d=optim_d, device=device, args=args)
