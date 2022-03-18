import torch.nn as nn


def weights_init(m):
    """
    Initializes weights from a zero-centered Normal distribution with standard deviation 0.02, according to the
    DCGAN paper.
    """
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
        nn.init.normal_(m.weight, 0.0, 0.02)


def remove_weight_norm(m):
    """
    Removes the weight normalization from all layers.
    Only used for inference.
    """
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
        nn.utils.remove_weight_norm(m)


class ResidualBlock(nn.Module):
    """
    Block with a skip connection based on ResNet.
    Uses dilated convolutions for larger receptive field.
    """
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(dilation),
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation)),
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size=1)),
        )
        # projection shortcut (1x1 convolution) to ensure same dimensions at the addition
        self.shortcut = nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size=1))

    def forward(self, x):
        return self.shortcut(x) + self.block(x)  # skip connection


class Generator(nn.Module):
    """
    The Generator has the following characteristics:

    - upsampling to raw audio scale: Transposed convolutions are used to upsample the input sequence since the
        mel-spectrogram is at a 256× lower temporal resolution.
    - no noise vector: There is little perceptual difference in the generated waveforms when additional noise is fed to
        the generator.
    - long range correlation among audio time-steps: Each transposed convolutional layer is followed by a stack of
        residual blocks with dilated convolutions for larger overlap in the induced receptive field of far apart time-steps.
    - combats checkerboard artifacts of deconvolutions: Kernel-size is a multiple of stride and the dilation grows as a
        power of the kernel size.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.ReflectionPad1d(3),
            nn.utils.weight_norm(nn.Conv1d(in_channels, 512, kernel_size=7)),

            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(512, 256, kernel_size=16, stride=8, padding=4)),
            ResidualBlock(256, dilation=1),
            ResidualBlock(256, dilation=3),
            ResidualBlock(256, dilation=9),

            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(256, 128, kernel_size=16, stride=8, padding=4)),
            ResidualBlock(128, dilation=1),
            ResidualBlock(128, dilation=3),
            ResidualBlock(128, dilation=9),

            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1)),
            ResidualBlock(64, dilation=1),
            ResidualBlock(64, dilation=3),
            ResidualBlock(64, dilation=9),

            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)),
            ResidualBlock(32, dilation=1),
            ResidualBlock(32, dilation=3),
            ResidualBlock(32, dilation=9),

            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            nn.utils.weight_norm(nn.Conv1d(32, 1, kernel_size=7)),
            nn.Tanh(),
        )

        self.apply(weights_init)  # initialize the weights

    def forward(self, x):
        return self.model(x)

    def remove_weight_norm(self):
        self.apply(remove_weight_norm)


class DiscriminatorBlock(nn.Module):    
    """
    Each discriminator block consists of four 4-strided convolution layers for image downsampling, followed by 2 layers
    for Feature Mapping, and Output (T/F), respectively.
    """

    def __init__(self):
        super().__init__()

        # Model Definition
        self.model = nn.ModuleDict({
            'layer_0': nn.Sequential(
                nn.ReflectionPad1d(7),
                nn.utils.weight_norm(nn.Conv1d(1, 16, kernel_size=15)),
                nn.LeakyReLU(0.2, True),
            ),
            'layer_1': nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(16, 64, stride=4, kernel_size=41, padding=20, groups=4)),
                nn.LeakyReLU(0.2, True),
            ),
            'layer_2': nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(64, 256, stride=4, kernel_size=41, padding=20, groups=16)),
                nn.LeakyReLU(0.2, True),
            ),
            'layer_3': nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(256, 1024, stride=4, kernel_size=41, padding=20, groups=64)),
                nn.LeakyReLU(0.2, True),
            ),
            'layer_4': nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(1024, 1024, stride=4, kernel_size=41, padding=20, groups=256)),
                nn.LeakyReLU(0.2, True),
            ),
            "layer_5": nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2)),
                nn.LeakyReLU(0.2, True),
            ),
            'layer_6': nn.utils.weight_norm(nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1))
        })

    def forward(self, x):
        results = []
        for key, layer in self.model.items():
            x = layer(x)
            results.append(x)
        return results


class Discriminator(nn.Module):
    """
    The Discriminator consists of a multi-scale architecture with 3 discriminators (D_1, D_2, D_3) that have identical
    network structure but operate on different audio scales.
    
    More specifically, D_1 operates on (a scale of) raw audio, while D_2 and D_3 operate on raw audio downsampled by a
    factor of 2 and 4, respectively. The downsampling is performed using Average Pooling with *kernel size*, *stride*
    and *padding* equal to 4, 2, and 1, respectivly.
    
    [Kumar et al. 2019] "Multiple discriminators at different scales are motivated from the fact that audio has
    structure at different levels. This structure has an inductive bias that each discriminator learns features for
    different frequency range of the audio."
    """

    def __init__(self):
        super().__init__()
        self.model = nn.ModuleDict({
            f"disc_0": DiscriminatorBlock(),
            f"disc_1": DiscriminatorBlock(),
            f"disc_2": DiscriminatorBlock(),
        })
        
        self.downsample = nn.AvgPool1d(4, stride=2, padding=1, count_include_pad=False)
        self.apply(weights_init)

    def forward(self, x):
        """
            Inputs
            =======
            x: Raw Audio Waveform

            Outputs
            =======
            results: A list of 6 features, discriminator score (Note: we directly predict score without last sigmoid
            function)
        """
        results = []
        for key, disc in self.model.items():
            results.append(disc(x))
            x = self.downsample(x)
        return results
