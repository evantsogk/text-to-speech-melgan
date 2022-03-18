from utils.misc import save_sample
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


def save_original_samples(test_loader, fft, device, args, writer):
    """
    Saves the original test samples in order to track training progress.
    """
    test_mel = []
    test_audio = []
    for i, batch in enumerate(test_loader):
        real_audio, _, _, _ = batch
        real_audio = real_audio.to(device)
        real_mel = fft(real_audio).detach()  # extract mel

        test_mel.append(real_mel)
        test_audio.append(real_audio)

        audio = real_audio.squeeze().cpu()
        save_sample(args.save_path / ("original_{}.wav".format(i)), 22050, audio)
        writer.add_audio("original/sample_{}.wav".format(i), audio, 0, sample_rate=22050)

    return test_mel, test_audio


def train_discriminator(real_audio, fake_audio, discriminator, optim_d):
    """
    Trains the discriminator for one step.
    """
    optim_d.zero_grad()

    d_fake = discriminator(fake_audio.detach())
    d_real = discriminator(real_audio)

    # Hinge loss version of GAN objective (discriminator)
    loss_d = 0
    for scale in d_fake:
        loss_d += F.relu(1 + scale[-1]).mean()
    for scale in d_real:
        loss_d += F.relu(1 - scale[-1]).mean()

    loss_d.backward()
    optim_d.step()

    return d_real, loss_d


def train_generator(fake_audio, d_real, discriminator, optim_g, args):
    """
    Trains the generator for one step.
    """
    optim_g.zero_grad()

    # Hinge loss version of GAN objective (generator)
    d_fake = discriminator(fake_audio)
    loss_g = 0
    for scale in d_fake:
        loss_g += -scale[-1].mean()

    # Feature Matching Loss
    loss_feat = 0
    # feature weights (default: 4 layers per discriminator block)
    feat_weights = 4.0 / (args.n_layers_D + 1)
    # discriminator block weights (default: 3 discriminator blocks)
    d_weights = 1.0 / args.num_D
    wt = d_weights * feat_weights
    for i in range(args.num_D):
        for j in range(len(d_fake[i]) - 1):
            # L1 distance between the
            # discriminator feature maps of real and synthetic audio.
            loss_feat += wt * F.l1_loss(d_fake[i][j], d_real[i][j].detach())

    # Total loss
    (loss_g + args.lambda_feat * loss_feat).backward()
    optim_g.step()

    return loss_g, loss_feat


def train(train_loader, test_loader, fft, generator, discriminator, optim_g, optim_d, device, args):
    writer = SummaryWriter(str(args.save_path))  # create tensorboard writer

    # save the original test samples
    test_mel, test_audio = save_original_samples(test_loader, fft, device, args, writer)

    # enable cudnn autotuner to speed up training
    if device != 'cpu':
        torch.backends.cudnn.benchmark = True

    # start training
    costs = []
    steps = 0
    generator.train()
    discriminator.train()
    start_training_time, start_generate_time, start_batch_time = time.time(), time.time(), time.time()
    for epoch in range(1, args.epochs + 1):
        for i_b, batch in enumerate(train_loader):
            real_audio, _, _, _ = batch
            real_audio = real_audio.to(device)
            real_mel = fft(real_audio).detach()  # extract mel
            fake_audio = generator(real_mel)  # generate samples

            # calculate mel reconstruction error
            with torch.no_grad():
                fake_mel = fft(fake_audio.detach())
                error = F.l1_loss(real_mel, fake_mel).item()

            # train the discriminator
            d_real, loss_d = train_discriminator(real_audio, fake_audio, discriminator, optim_d)

            # train the generator
            loss_g, loss_feat = train_generator(fake_audio, d_real, discriminator, optim_g, args)

            # update tensorboard
            costs.append([loss_d.item(), loss_g.item(), loss_feat.item(), error])
            writer.add_scalar("loss/discriminator", costs[-1][0], steps)
            writer.add_scalar("loss/generator", costs[-1][1], steps)
            writer.add_scalar("loss/feature_matching", costs[-1][2], steps)
            writer.add_scalar("loss/mel_reconstruction", costs[-1][3], steps)
            steps += 1

            # print losses
            if steps % args.log_interval == 0:
                print("Epoch {} | Iters {} / {} | ms/batch {:5.2f} | loss {}".format(
                    epoch, i_b, len(train_loader), 1000 * (time.time() - start_batch_time) / args.log_interval,
                    np.asarray(costs).mean(0)))
                costs = []
                start_batch_time = time.time()

        # save checkpoints
        if epoch % args.checkpoint_interval == 0:
            torch.save(generator.state_dict(), args.save_path / "generator.pt")
            torch.save(optim_g.state_dict(), args.save_path / "optim_g.pt")
            torch.save(discriminator.state_dict(), args.save_path / "discriminator.pt")
            torch.save(optim_d.state_dict(), args.save_path / "optim_d.pt")

        # check if it's time to stop
        stop_training = args.training_time and (time.time() - start_training_time) / 3600 >= args.training_time

        # save generated samples
        if (time.time() - start_generate_time) / 3600 >= args.generate_interval or stop_training:
            with torch.no_grad():
                for i, (mel, _) in enumerate(zip(test_mel, test_audio)):
                    pred_audio = generator(mel)
                    pred_audio = pred_audio.squeeze().cpu()
                    save_sample(args.save_path / ("generated_{}_{}.wav".format(i, epoch)), 22050, pred_audio)
                    writer.add_audio("generated/sample_{}_{}.wav".format(i, epoch), pred_audio, epoch, sample_rate=22050)
            print("Saved generated samples")
            print("-" * 100)
            torch.save(generator.state_dict(), args.save_path / "generator_{}.pt".format(epoch))
            start_generate_time = time.time()

        # stop training
        if stop_training:
            torch.save(generator.state_dict(), args.save_path / "generator_{}.pt".format(epoch))
            torch.save(optim_g.state_dict(), args.save_path / "optim_g_{}.pt".format(epoch))
            torch.save(discriminator.state_dict(), args.save_path / "discriminator_{}.pt".format(epoch))
            torch.save(optim_d.state_dict(), args.save_path / "optim_d_{}.pt".format(epoch))
            print("Finished training")
            break
