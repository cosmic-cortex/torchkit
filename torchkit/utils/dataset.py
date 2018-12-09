import os
import numpy as np
import torch

from skimage import io

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F


def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


def joint_to_long_tensor(image, mask):
    return to_long_tensor(image), to_long_tensor(mask)


def make_joint_transform(
        crop=(256, 256), p_flip=0.5, color_jitter_params=(0.1, 0.1, 0.1, 0.1),
        rotate_range=False, normalize=False, long_mask=False
):

    if color_jitter_params is not None:
        color_tf = T.ColorJitter(*color_jitter_params)
    else:
        color_tf = None

    if normalize:
        tf_normalize = T.Normalize(mean=(0.5, 0.5, 0.5), std=(1, 1, 1))

    def joint_transform(image, mask):
        # transforming to PIL image
        image, mask = F.to_pil_image(image), F.to_pil_image(mask)

        # random crop
        if crop:
            i, j, h, w = T.RandomCrop.get_params(image, crop)
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)
            if np.random.rand() < p_flip:
                image, mask = F.hflip(image), F.hflip(mask)

                    # color transforms || ONLY ON IMAGE

        if color_tf is not None:
            image = color_tf(image)

        # random rotation
        if rotate_range:
            angle = rotate_range * (np.random.rand() - 0.5)
            image, mask = F.rotate(image, angle), F.rotate(mask, angle)

        # transforming to tensor
        image = F.to_tensor(image)
        if not long_mask:
            mask = F.to_tensor(mask)
        else:
            mask = to_long_tensor(mask)

        # normalizing image
        if normalize:
            image = tf_normalize(image)

        return image, mask

    return joint_transform


class ReadTrainDataset(Dataset):
    """
    Structure of the dataset should be:

    dataset_path
      |--images
          |--img001.png
          |--img002.png
      |--masks
          |--img001.png
          |--img002.png

    """

    def __init__(self, dataset_path, transform=None, one_hot_mask=False):
        self.dataset_path = dataset_path
        self.images_path = os.path.join(dataset_path, 'images')
        self.masks_path = os.path.join(dataset_path, 'masks')
        self.images_list = os.listdir(self.images_path)

        self.transform = transform
        self.one_hot_mask = one_hot_mask

    def __len__(self):
        return len(os.listdir(self.images_path))

    def __getitem__(self, idx):
        image_filename = self.images_list[idx]
        image = io.imread(os.path.join(self.images_path, image_filename))
        mask = io.imread(os.path.join(self.masks_path, image_filename))
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=2)

        if self.transform:
            image, mask = self.transform(image, mask)

        if self.one_hot_mask:
            assert self.one_hot_mask >= 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        return image, mask, image_filename


class ReadTestDataset(Dataset):
    """
    Structure of the dataset should be:

    dataset_path
      |--images
          |--img001.png
          |--img002.png
      |--masks
          |--img001.png
          |--img002.png

    """

    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.images_path = os.path.join(dataset_path, 'images')
        self.masks_path = os.path.join(dataset_path, 'masks')
        self.images_list = os.listdir(self.images_path)

        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.images_path))

    def __getitem__(self, idx):
        image_filename = self.images_list[idx]
        image = io.imread(os.path.join(self.images_path, image_filename))

        if self.transform:
            image = self.transform(image)
        else:
            image = F.to_tensor(image)

        return image, image_filename


if __name__ == '__main__':
    pass
