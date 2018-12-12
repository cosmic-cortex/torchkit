import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchkit.utils.model import Model
from torchkit.models.unet import UNet
from torchkit.utils.dataset import ImageToImage, Image, ImageToImageTransform

# getting absolute paths of this script
script_abs_path = os.path.split(os.path.realpath(__file__))[0]

train_dataset_path = os.path.join(script_abs_path, 'data', 'segmentation')
train_tf = ImageToImageTransform(crop=(256, 256), long_mask=True)
train_dataset = ImageToImage(train_dataset_path, joint_transform=train_tf)

n_epochs = 10
n_batch = 1
model_name = 'UNet_example'
checkpoint_folder = os.path.join(script_abs_path, '..', 'checkpoints', 'UNet_example')
predictions_folder = os.path.join(checkpoint_folder, 'predictions')

unet = UNet(3, 2)
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(unet.parameters(), lr=1e-3)
lr_milestones = [int(p*n_epochs) for p in [0.7, 0.9]]
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, lr_milestones)
device = torch.device('cpu')

model = Model(unet, loss, optimizer, scheduler=scheduler,
             checkpoint_folder=checkpoint_folder, device=device)
model.train_model(train_dataset, n_batch=n_batch, n_epochs=n_epochs,
                  verbose=True, save_freq=5)
