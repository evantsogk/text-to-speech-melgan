import os
import time
import torch
from utils.transforms import Audio2Mel
from utils.load_ljspeech import LJSPEECH
from utils.models import Generator


def generation_speed(model_name, dev, spec):
    """
    Inference (Mel2Audio) speed of given model in kHz (i.e. *10^3 samples per sec)
    :param model_name: Input model (either 'melgan' or 'waveglow')
    :param dev: Selected device (either 'cpu' or 'cuda')
    :param spec: Input spectrogram
    :return: Amount of samples generated per sec
    """

    if dev != 'cpu' and dev != 'cuda':
        raise ValueError('Argument "dev" expects one of the following: "cpu", "cuda".')

    if dev == 'cpu':
        # test on only 1 CPU core
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'

    if model_name == 'melgan':
        model = Generator(80).to(dev)
        model.load_state_dict(torch.load('../checkpoints/generator.pt'))
        model.eval()
    elif model_name == 'waveglow':
        model = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
        model = model.remove_weightnorm(model)
        model = model.to(dev)
        model.eval()
    else:
        raise ValueError('Currently supported models: "melgan" or "waveglow".')

    spec = spec.to(dev)
    start_time = time.time()  # start of experiment
    samples = 0  # samples generated
    overall_time = 0

    # run the experiment for 1 minute max
    while time.time() - start_time <= 60:
        if model_name == 'melgan':
            with torch.no_grad():
                t1 = time.time()
                w = model(spec)
                t2 = time.time()
        else:
            with torch.no_grad():
                t1 = time.time()
                w = model.infer(spec)
                t2 = time.time()
        samples += w.size(-1)
        overall_time += t2 - t1  # time elapsed for an entire mel-spectrogram inversion

    return samples, overall_time


if __name__ == '__main__':
    # create mel-spectrogram from random sample
    '''Alternatively, you can use something like: 
       s = torch.randn(1, 80, 832) 
       as an input mel spectrogram.'''

    name, device = 'melgan', 'cpu'  # set proper model name and device

    t = Audio2Mel().to(device)
    lj = LJSPEECH(root=os.getcwd())
    w = lj[2][0]  # random audio sample (10 sec duration - max for given dataset)
    s = t(w.unsqueeze(0).to(device)).detach()

    res = generation_speed(name, device, s)
    # Speed (in kHz) = 1e-3 * Total generated samples / Total elapsed time
    print('Result for model {} on device {}: {:.4f} * 10^3 samples/sec'.format(name, device, 1e-3 * res[0] / res[1]))
