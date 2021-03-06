import os

import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

from skimage import io

from torchkit.tools.misc import chk_mkdir
from torchkit.tools.callback import BaseCallback, Logger


# TODO: generalize the class to work with arbitrary model and training structures
class Model:
    def __init__(self, net: nn.Module, loss, optimizer, checkpoint_folder: str,
                 scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                 device: torch.device = torch.device('cpu')):
        """
        Wrapper for PyTorch models.

        Args:
            net: PyTorch model.
            loss: Loss function which you would like to use during training.
            optimizer: Optimizer for the training.
            checkpoint_folder: Folder for saving the results and predictions.
            scheduler: Learning rate scheduler for the optimizer. Optional.
            device: The device on which the model and tensor should be
                located. Optional. The default device is the cpu.

        Attributes:
            net: PyTorch model.
            loss: Loss function which you would like to use during training.
            optimizer: Optimizer for the training.
            checkpoint_folder: Folder for saving the results and predictions.
            scheduler: Learning rate scheduler for the optimizer. Optional.
            device: The device on which the model and tensor should be
                located. Optional.
        """
        self.net = net
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.checkpoint_folder = checkpoint_folder
        chk_mkdir(self.checkpoint_folder)

        # moving net and loss to the selected device
        self.device = device
        self.net.to(device=self.device)
        self.loss.to(device=self.device)

    def fit_batch(self, X_batch, y_batch):
        self.net.train(True)

        X_batch = Variable(X_batch.to(device=self.device))
        y_batch = Variable(y_batch.to(device=self.device))

        # training
        self.optimizer.zero_grad()
        y_out = self.net(X_batch)
        training_loss = self.loss(y_out, y_batch)
        training_loss.backward()
        self.optimizer.step()

        # return the average training loss
        return training_loss.item() / len(X_batch)

    def fit_epoch(self, dataset, n_batch=1, shuffle=False):
        epoch_running_loss = 0
        for X_batch, y_batch, *rest in DataLoader(dataset, batch_size=n_batch, shuffle=shuffle):
            epoch_running_loss += self.fit_batch(X_batch, y_batch)

        # TODO: is this necessary?
        del X_batch, y_batch

        return epoch_running_loss / n_batch

    def fit_dataset(self, dataset: Dataset, n_epochs: int, n_batch: int = 1, shuffle: bool = False,
                    validation_dataset: Dataset = None, save_freq: int = 100, callback: BaseCallback = None,
                    verbose: bool = False):

        # setting up callbacks
        if callback is not None:
            assert isinstance(callback, BaseCallback), 'callback must be inherited from ' \
                                                       'torchkit.tools.callback.BaseCallback'
        else:
            callback = BaseCallback()

        logger = Logger(verbose=verbose)

        self.net.train(True)

        min_loss = np.inf
        for epoch_idx in range(1, n_epochs + 1):
            # doing the epoch
            callback.before_epoch()
            train_loss = self.fit_epoch(dataset, n_batch=n_batch, shuffle=shuffle)
            callback.after_epoch()

            # logging the losses
            logs = {'epoch': epoch_idx,
                    'train_loss': train_loss}

            if self.scheduler is not None:
                self.scheduler.step(train_loss)

            if validation_dataset is not None:
                val_loss = self.validate_dataset(validation_dataset, n_batch=n_batch)
                logs['val_loss'] = val_loss
                if val_loss < min_loss:
                    torch.save(self.net.state_dict(), os.path.join(self.checkpoint_folder, 'model'))
                    min_loss = val_loss
            else:
                if train_loss < min_loss:
                    torch.save(self.net.state_dict(), os.path.join(self.checkpoint_folder, 'model'))
                    min_loss = train_loss

            logger.after_epoch(logs)
            # saving model and logs
            if epoch_idx % save_freq == 0:
                epoch_save_path = os.path.join(self.checkpoint_folder, '%d' % epoch_idx)
                chk_mkdir(epoch_save_path)
                torch.save(self.net.state_dict(), os.path.join(epoch_save_path, 'model'))

        self.net.train(False)

        return logger

    def validate_dataset(self, dataset, n_batch=1):
        self.net.train(False)

        total_running_loss = 0
        for batch_idx, (X_batch, y_batch, *rest) in enumerate(DataLoader(dataset, batch_size=n_batch, shuffle=False)):
            X_batch = Variable(X_batch.to(device=self.device))
            y_batch = Variable(y_batch.to(device=self.device))

            y_out = self.net(X_batch)
            training_loss = self.loss(y_out, y_batch)

            total_running_loss += training_loss.item()

        self.net.train(True)

        del X_batch, y_batch

        return total_running_loss / (batch_idx + 1)

    def predict_dataset(self, dataset, export_path):
        self.net.train(False)
        chk_mkdir(export_path)

        for batch_idx, (X_batch, *rest) in enumerate(DataLoader(dataset, batch_size=1)):
            if isinstance(rest[0][0], str):
                image_filename = rest[0][0]
            else:
                image_filename = '%s.png' % str(batch_idx + 1).zfill(3)

            X_batch = Variable(X_batch.to(device=self.device))
            y_out = self.net(X_batch).cpu().data.numpy()

            io.imsave(os.path.join(export_path, image_filename), y_out[0, :, :, :].transpose((1, 2, 0)))

    def predict_batch(self, X_batch, cpu=False, numpy=False):
        self.net.train(False)

        X_batch = Variable(X_batch.to(device=self.device))
        y_out = self.net(X_batch)

        if numpy or cpu:
            y_out = self.net(X_batch).cpu()
            if numpy:
                y_out = y_out.data.numpy()

        return y_out


class GAN:
    def __init__(self, g: nn.Module, g_opt,
                 d: nn.Module, d_loss, d_opt,
                 noise_shape: tuple,
                 checkpoint_folder: str,
                 device: torch.device = torch.device('cpu')):
        self.g = g
        self.g_opt = g_opt

        if isinstance(noise_shape, tuple):
            self.noise_shape = noise_shape
        elif isinstance(noise_shape, int):
            self.noise_shape = (noise_shape, )

        self.d = d
        self.d_loss = d_loss
        self.d_opt = d_opt

        self.checkpoint_folder = checkpoint_folder
        chk_mkdir(self.checkpoint_folder)

        # moving net and loss to the selected device
        self.device = device
        self.g.to(device=self.device)
        self.d.to(device=self.device)
        self.d_loss.to(device=self.device)

    def save_model(self, save_path):
        chk_mkdir(save_path)
        torch.save(self.g.state_dict(), os.path.join(save_path, 'g'))
        torch.save(self.d.state_dict(), os.path.join(save_path, 'd'))

    def d_fit_batch(self, X_noise, y_real, long_gt: bool = False):
        self.d.train(True)

        # zeroing gradients
        self.d_opt.zero_grad()

        # creating real and fake labels
        d_real_labels = torch.ones(len(y_real)).to(self.device)
        d_fake_labels = torch.zeros(len(y_real)).to(self.device)
        if long_gt:
            d_real_labels = d_real_labels.long()
            d_fake_labels = d_fake_labels.long()

        # real images
        d_real = self.d(y_real)
        d_real_loss = self.d_loss(d_real, d_real_labels)
        # fake images
        y_fake = self.g(X_noise)
        d_fake = self.d(y_fake)
        d_fake_loss = self.d_loss(d_fake, d_fake_labels)
        # gradient step
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        self.d_opt.step()

        self.d.train(False)

        return d_real_loss.item(), d_fake_loss.item()

    def g_fit_batch(self, X_noise, long_gt: bool = False):
        self.g.train(True)

        # generating fake images
        self.g_opt.zero_grad()
        y_fake = self.g(X_noise)

        # creating labels for the discriminator
        d_real_labels = torch.ones(len(X_noise)).to(self.device)
        if long_gt:
            d_real_labels = d_real_labels.long()

        # computing loss
        d_fake = self.d(y_fake)
        g_loss = self.d_loss(d_fake, d_real_labels)
        g_loss.backward()
        self.g_opt.step()

        self.g.train(False)

        return g_loss.item()

    def fit_epoch(self, dataset, n_batch=1, shuffle=False, long_gt: bool = False):
        d_running_real_loss, d_running_fake_loss, g_running_loss = 0, 0, 0

        for batch_idx, (y_real, *rest) in enumerate(DataLoader(dataset, batch_size=n_batch, shuffle=shuffle)):
            # generate noise
            X_noise = Variable(torch.rand(n_batch, *self.noise_shape).to(self.device))
            y_real = Variable(y_real.to(self.device))

            # training the discriminator
            d_real_loss, d_fake_loss = self.d_fit_batch(X_noise, y_real, long_gt=long_gt)
            d_running_real_loss += d_real_loss
            d_running_fake_loss += d_fake_loss

            # training the generator
            g_loss = self.g_fit_batch(X_noise, long_gt=long_gt)
            g_running_loss += g_loss

        return d_running_real_loss/(batch_idx+1), d_running_fake_loss/(batch_idx+1), g_running_loss/(batch_idx+1)

    def fit_dataset(self, dataset: Dataset, n_epochs: int, n_batch: int = 1, shuffle: bool = False,
                    save_freq: int = 100, callback: BaseCallback = None,
                    long_gt: bool = False, verbose: bool = False):

        # setting up callbacks
        if callback is not None:
            assert isinstance(callback, BaseCallback), 'callback must be inherited from ' \
                                                       'torchkit.tools.callback.BaseCallback'
        else:
            callback = BaseCallback()

        logger = Logger(verbose=verbose)

        self.g.train(True)
        self.d.train(True)

        for epoch_idx in range(1, n_epochs + 1):
            callback.before_epoch()
            d_real_loss, d_fake_loss, g_loss = self.fit_epoch(dataset, n_batch=n_batch, shuffle=shuffle,
                                                              long_gt=long_gt)
            callback.after_epoch()

            # logging the losses
            logs = {'epoch': epoch_idx,
                    'd_real_loss': d_real_loss,
                    'd_fake_loss': d_fake_loss,
                    'g_loss': g_loss}

            logger.after_epoch(logs)
            # saving model and logs
            if epoch_idx % save_freq == 0:
                epoch_save_path = os.path.join(self.checkpoint_folder, '%d' % epoch_idx)
                self.save_model(epoch_save_path)

    def predict_batch(self, X_batch, cpu=False, numpy=False):
        self.net.train(False)

        X_batch = Variable(X_batch.to(device=self.device))
        y_out = self.g(X_batch)

        if numpy or cpu:
            y_out = self.net(X_batch).cpu()
            if numpy:
                y_out = y_out.data.numpy()

        return y_out
