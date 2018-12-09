import os
import numpy as np
import torch
import torch.nn as nn

from skimage import io

from torch.autograd import Variable
from torch.utils.data import DataLoader


class Model:
    def __init__(self, net: nn.Module, loss, optimizer, checkpoint_folder: str,
                 scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                 device: torch.device = None):
        """
        Wrapper for PyTorch models.

        Args:
            net: PyTorch model.
            loss: Loss function which you would like to use during training.
            optimizer: Optimizer for the training.
            checkpoint_folder: Folder for saving the results and predictions.
            scheduler: Learning rate scheduler for the optimizer. Optional.
            device: The device on which the model and tensor should be
                located. Optional.

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

        self.device = device
        if self.device:
            self.net.to(device=self.device)
            self.loss.to(device=self.device)

    def train_model(self, dataset, n_epochs, n_batch=1, verbose=False,
                    validation_dataset=None, prediction_dataset=None,
                    save_freq=100):
        self.net.train(True)

        min_loss = np.inf
        total_running_loss = 0
        for epoch_idx in range(n_epochs):

            epoch_running_loss = 0
            for batch_idx, (X_batch, y_batch, name) in enumerate(DataLoader(dataset, batch_size=n_batch, shuffle=True)):

                if self.device:
                    X_batch = Variable(X_batch.to(device=self.device))
                    y_batch = Variable(y_batch.to(device=self.device))
                else:
                    X_batch, y_batch = Variable(X_batch), Variable(y_batch)

                # training
                self.optimizer.zero_grad()
                y_out = self.net(X_batch)
                training_loss = self.loss(y_out, y_batch)
                training_loss.backward()
                self.optimizer.step()

                epoch_running_loss += training_loss.item()

                if verbose:
                    print('(Epoch no. %d, batch no. %d) loss: %f' % (epoch_idx, batch_idx, training_loss.item()))

            total_running_loss += epoch_running_loss/(batch_idx + 1)
            print('(Epoch no. %d) loss: %f' % (epoch_idx, epoch_running_loss/(batch_idx + 1)))
            loss_df.loc[epoch_idx, 'train'] = epoch_running_loss/(batch_idx + 1)

            if validation_dataset is not None:
                validation_error = self.validate(validation_dataset, n_batch=1)
                loss_df.loc[epoch_idx, 'validate'] = validation_error
                if validation_error < min_loss:
                    torch.save(self.net.state_dict(), os.path.join(self.checkpoint_folder, 'model'))
                    print('Validation loss improved from %f to %f, model saved to %s'
                          % (min_loss, validation_error, self.checkpoint_folder))
                    min_loss = validation_error

                if self.scheduler is not None:
                    self.scheduler.step(validation_error)

            else:
                if epoch_running_loss/(batch_idx + 1) < min_loss:
                    torch.save(self.net.state_dict(), os.path.join(self.checkpoint_folder, 'model'))
                    print('Training loss improved from %f to %f, model saved to %s'
                          % (min_loss, epoch_running_loss / (batch_idx + 1), self.checkpoint_folder))
                    min_loss = epoch_running_loss / (batch_idx + 1)

                    if self.scheduler is not None:
                        self.scheduler.step(epoch_running_loss / (batch_idx + 1))

            # saving model and logs
            loss_df.to_csv(os.path.join(self.checkpoint_folder, 'loss.csv'))
            if epoch_idx % save_freq == 0:
                epoch_save_path = os.path.join(self.checkpoint_folder, '%d' % epoch_idx)
                chk_mkdir(epoch_save_path)
                torch.save(self.net.state_dict(), os.path.join(epoch_save_path, 'model'))
                if prediction_dataset:
                    self.predict_large_images(prediction_dataset, epoch_save_path)

        self.net.train(False)

        del X_batch, y_batch

        return total_running_loss/n_batch

    def validate(self, dataset, n_batch=1):
        self.net.train(False)

        total_running_loss = 0
        for batch_idx, (X_batch, y_batch, name) in enumerate(DataLoader(dataset, batch_size=n_batch, shuffle=False)):

            if self.device:
                X_batch = Variable(X_batch.to(device=self.device))
                y_batch = Variable(y_batch.to(device=self.device))
            else:
                X_batch, y_batch = Variable(X_batch), Variable(y_batch)

            y_out = self.net(X_batch)
            training_loss = self.loss(y_out, y_batch)

            total_running_loss += training_loss.item()

        print('Validation loss: %f' % (total_running_loss / (batch_idx + 1)))
        self.net.train(True)

        del X_batch, y_batch

        return total_running_loss/(batch_idx + 1)

    def predict(self, dataset, export_path, channel=None):
        self.net.train(False)
        chk_mkdir(export_path)

        for batch_idx, (X_batch, image_filename) in enumerate(DataLoader(dataset, batch_size=1)):
            if self.device:
                X_batch = Variable(X_batch.to(device=self.device))
                y_out = self.net(X_batch).cpu().data.numpy()
            else:
                X_batch = Variable(X_batch)
                y_out = self.net(X_batch).data.numpy()

                io.imsave(os.path.join(export_path, image_filename[0]), y_out[0, :, :, :].transpose((1, 2, 0)))
