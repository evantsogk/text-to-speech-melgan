import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel
from utils.slaney_norm import slaney


class Audio2Mel(nn.Module):
    """ Convert raw waveform to logarithmically compressed mel-spectrogram

        Args:
            n_fft: Number of FFT points.
            hop_length: Hop length between successive (STFT) windows.
            win_length: Window length. (default: ``win_length = n_fft``)
            sampling_rate: Sampling rate of audio signal.
            n_mel_channels: Number of mel bins.
            mel_fmin: Minimum frequency for conversion to mel.
            mel_fmax: Maximum frequency for conversion to mel.
                    (default: ``mel_fmax = sampling_rate // 2``)
            """

    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        sampling_rate=22050,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=None,
    ):
        super().__init__()
        mel_fmax = float(sampling_rate // 2) if mel_fmax is None else mel_fmax
        window = torch.hann_window(win_length).float()
        mel_basis = mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
            htk=False,
            norm=None
          )
        mel_basis = torch.from_numpy(mel_basis).float()
        # slaney norm (not available in librosa 0.6.0)
        mel_basis = slaney(mel_basis, n_mel_channels, mel_fmin, mel_fmax)
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels

    def forward(self, audio):
        """
        :param audio: (Torch Tensor) raw waveform of dimension (1, num_samples)
        :return: (Torch Tensor) log-compressed Mel spectrogram of size (1, n_mel_channels, time)
        """

        # custom padding on both sides of the waveform
        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audio, (p, p), "reflect").squeeze(1)
        # short-time Fourier transform
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=False
          )
        # split FFT (4D tensor of size (1, _, _, 2)) to its real and imaginary part
        real_part, imag_part = fft.unbind(-1)
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        mel_output = torch.matmul(self.mel_basis, magnitude)
        # set lower bound to avoid -Inf caused by zeros
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
        return log_mel_spec

