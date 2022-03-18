"""
Adopted from newer version of torchaudio (with minor changes) + extended to support
transformations inside the __getitem__ method.
"""

import os
import csv
from typing import Tuple, Union
from pathlib import Path
import numpy as np

from torch import Tensor, norm, from_numpy
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.distributions.uniform import Uniform
from scipy.io.wavfile import read


_RELEASE_CONFIGS = {
    "release1": {
        "folder_in_archive": "wavs",
        "url": "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
        "checksum": "be1a30453f28eb8dd26af4101ae40cbf2c50413b1bb21936cbcdc6fae3de8aa5",
    }
}


class LJSPEECH(Dataset):
    """Create a Dataset for LJSpeech-1.1.

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        url (str, optional): The URL to download the dataset from.
            (default: ``"https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"``)
        folder_in_archive (str, optional):
            The top-level directory of the dataset. (default: ``"wavs"``)
        segment (bool, optional): Apply waveform segmentation based on seq_len. (default: ``False``)
        seq_len (int, optional): Maximum sample size - enforce same size for all waveforms, i.e.
            (1, sequence_length). (default: 8192)
        normalize (bool, optional): Normalize the amplitude of the waveform to have range [-1, 1].
            (default: ``False``)
        augment: (bool, optional): Augment the waveform by multiplying it with a random constant
            drawn from a uniform distribution. (default: ``False``)
    """

    def __init__(self,
                 root: Union[str, Path],
                 url: str = _RELEASE_CONFIGS["release1"]["url"],
                 folder_in_archive: str = _RELEASE_CONFIGS["release1"]["folder_in_archive"],
                 segment: bool = False,
                 seq_len: int = 8192,
                 normalize: bool = False,
                 augment: bool = False
                 ) -> None:

        self._parse_filesystem(root, url, folder_in_archive)
        self.seq_len = seq_len
        self.normalize = normalize
        self.augment = augment
        self.segment = segment

    def _parse_filesystem(self, root: str, url: str, folder_in_archive: str) -> None:
        root = Path(root)

        basename = os.path.basename(url)
        basename = Path(basename.split(".tar.bz2")[0])

        folder_in_archive = basename / folder_in_archive

        self._path = root / folder_in_archive
        self._metadata_path = root / basename / 'metadata.csv'

        with open(self._metadata_path, "r", newline='', encoding="utf-8") as metadata:
            flist = csv.reader(metadata, delimiter="|", quoting=csv.QUOTE_NONE)
            self._flist = list(flist)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, Tensor]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform, sample_rate, transcript, normalized_transcript, mel_spectogram)``
        """
        line = self._flist[n]
        fileid, transcript, normalized_transcript = line
        fileid_audio = self._path / (fileid + ".wav")
        fileid_audio = str(fileid_audio)  # fixes TypeError for WindowsPath

        fileid_mel = self._path.parent.absolute() / ('tc2_mels/' + fileid + ".npy")
        fileid_mel = str(fileid_mel)

        # load mel spectogram
        mel_spectogram = np.load(fileid_mel)
        mel_spectogram = from_numpy(mel_spectogram).float()

        # Load audio
        sample_rate, waveform = read(fileid_audio)
        waveform = waveform / 2 ** 15
        waveform = from_numpy(waveform).unsqueeze(0).float()

        # get segment of audio (if necessary)
        if self.segment:
            if waveform.size(1) >= self.seq_len:
                waveform = waveform[:, 0:self.seq_len]
            else:
                waveform = F.pad(
                    waveform, (0, self.seq_len - waveform.size(1)), "constant"
                ).data

        # normalize waveform (if necessary) - same as librosa.util.normalize with default parameters
        if self.normalize:
            waveform = 0.95 * waveform / norm(waveform, float('inf'), dim=1)

        # augment waveform (if necessary)
        if self.augment:
            # initialize uniform distribution (low=0.3, high=1.0)
            u = Uniform(0.3, 1.0)
            waveform = waveform * u.sample().item()

        return (
            waveform,
            sample_rate,
            transcript,
            normalized_transcript,
            mel_spectogram
        )

    def __len__(self) -> int:
        return len(self._flist)
