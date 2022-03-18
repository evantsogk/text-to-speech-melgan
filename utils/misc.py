import torch

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.io.wavfile import write


def set_device():
    """ Set proper device """

    return 'cuda' if torch.cuda.is_available() else 'cpu'


def spec_visualization(s, titles=None):
    """ Visualize given spectrogram (expected to be a 3D Tensor)
        If a list is provided, then it creates a subplot for each
        spectrogram in the list (row-wise) and sets the corresponding
        title (if provided). """

    n = 1 if isinstance(s, torch.Tensor) else len(s)
    fig, axs = plt.subplots(n, 1, figsize=(6, 6), squeeze=False)

    for i in range(n):
        spec = s if n == 1 else s[i]
        vis = axs[i, 0].imshow(spec.squeeze(0).numpy(), cmap='viridis', origin='lower')
        axs[i, 0].set_yticks(ticks=[])

        if n == 1 and titles:
            axs[i, 0].set_title(titles)
        elif n != 1 and titles:
            axs[i, 0].set_title(titles[i])

        # create colorbar
        divider = make_axes_locatable(axs[i, 0])
        cax = divider.append_axes("right", size="2.5%", pad=0.15)
        cbar = axs[i, 0].figure.colorbar(vis, cax=cax)
        # cbar.ax.set_ylabel('Magnitude', rotation=-90, va="bottom")

    return axs


def save_sample(path, srate, audio):
    """ Save audio as '.wav' file
    :param path: Path or filename
    :param srate: Audio sampling rate
    :param audio: Torch tensor, contains the audio samples to be stored
    :return: None
    """

    if audio.dtype == torch.int16:
        audio = audio.numpy()
    else:
        audio = (audio.numpy() * int(2**15)).astype('int16')
    # write function expects input to have shape (num_samples, num_channels)
    write(path, srate, audio.T)

    return None


# TODO: implement functions for: 1) CPU/GPU generation speed (test between Melgan, Waveglow)
