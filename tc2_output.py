from fine_tuning.dataloader import loaders
import argparse
from pathlib import Path
import torch
import numpy as np
import os


def parse_args():
    """
    Configuration.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=Path, required=True)  # path to data
    parser.add_argument("--seq_len", type=int, default=8192)  # duration of audio samples for training
    parser.add_argument("--mel_spectogram_size", type=int, default=32)  # to match seq_len

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # get default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create loaders
    train_loader, test_loader = loaders(batch_size=1, path=args.data_path, seq_len=args.seq_len, shuffle=False)

    # load tacotron2
    tacotron2 = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_tacotron2')
    tacotron2 = tacotron2.to(device)
    tacotron2.eval()

    # save tacotron2 output for test set
    for i, batch in enumerate(test_loader):
        _, _, _, normalized_transcript, fileid = batch

        # preprocess text
        sequence = np.array(tacotron2.text_to_sequence(normalized_transcript[0], ['english_cleaners']))[None, :]
        sequence = torch.from_numpy(sequence).to(device=device, dtype=torch.int64)

        # get mel spectogram
        with torch.no_grad():
            _, mel, _, _ = tacotron2.infer(sequence)
            mel = mel.squeeze(0).cpu().numpy()

        # save mel spectogram
        np.save(os.getcwd() + '\\LJSpeech-1.1\\tc2_mels\\' + fileid[0], mel)

    # save tacotron2 output for train set
    for i, batch in enumerate(train_loader):
        _, _, _, normalized_transcript, fileid = batch

        # preprocess text
        sequence = np.array(tacotron2.text_to_sequence(normalized_transcript[0], ['english_cleaners']))[None, :]
        sequence = torch.from_numpy(sequence).to(device=device, dtype=torch.int64)

        # get mel spectogram
        with torch.no_grad():
            _, mel, _, _ = tacotron2.infer(sequence)
            mel = mel.squeeze(0)[:, :args.mel_spectogram_size].cpu().numpy()

        # save mel spectogram
        np.save(os.getcwd() + '\\LJSpeech-1.1\\tc2_mels\\' + fileid[0], mel)
