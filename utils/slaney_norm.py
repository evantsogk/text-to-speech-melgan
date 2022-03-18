import math
import torch


def slaney(fb, n_mels, fmin, fmax):

    # hertz to mel conversion
    m_min = 2595.0 * math.log10(1.0 + (fmin / 700.0))
    m_max = 2595.0 * math.log10(1.0 + (fmax / 700.0))
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    # mel to hertz conversion
    f_pts = 700.0 * (10 ** (m_pts / 2595.0) - 1.0)

    # divide the triangular mel weights by the width of the mel band ('Slaney' style)
    enorm = 2.0 / (f_pts[2:n_mels + 2] - f_pts[:n_mels])
    print(fb.shape, enorm.shape)
    fb *= enorm.unsqueeze(1)

    return fb

