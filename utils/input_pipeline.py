"""ImageNet input pipeline."""

import os
import random
from functools import partial

import jax
import numpy as np
import jax.numpy as jnp
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets.folder import pil_loader

from utils.logging_util import log_for_0
from utils.hsdp_util import pad_and_merge

IMAGE_SIZE = 224
CROP_PADDING = 32
MEAN_RGB = [0.485, 0.456, 0.406]
STDDEV_RGB = [0.229, 0.224, 0.225]

def loader(path: str):
    return pil_loader(path)

def process_image_on_tpu(image, use_flip=True, flip_key=None):
    """
    Process a single image on TPU: convert to float, normalize, flip.
    Center crop is already done on CPU to ensure uniform batch size.

    Args:
        image: uint8 array of shape (image_size, image_size, C)
        use_flip: whether to apply random horizontal flip
        flip_key: JAX random key for flipping (required if use_flip=True)

    Returns:
        Processed image as float32 array of shape (image_size, image_size, C)
        normalized to [-1, 1]
    """
    # Convert to float [0, 1]
    image = image.astype(jnp.float32) / 255.0

    # Random horizontal flip
    if use_flip and flip_key is not None:
        should_flip = jax.random.bernoulli(flip_key, p=0.5)
        image = jnp.where(should_flip, jnp.fliplr(image), image)

    # Normalize to [-1, 1]
    image = (image - 0.5) / 0.5

    return image


def process_batch_on_tpu(batch_dict, use_flip=True, rng_key=None):
    """
    Process a batch of images on TPU (designed to be used with pmap).
    This function processes one device's batch at a time (called by pmap).
    Images are already center-cropped on CPU to uniform size.

    Args:
        batch_dict: dict with 'image' (uint8) and 'label'
                   image shape: (device_batch_size, image_size, image_size, C)
        use_flip: whether to apply random horizontal flip
        rng_key: JAX random key for this device's batch

    Returns:
        Processed batch with images as float32 normalized to [-1, 1]
        image shape: (device_batch_size, image_size, image_size, C)
    """
    images = batch_dict["image"]  # uint8 (device_batch_size, image_size, image_size, C)
    labels = batch_dict["label"]

    # Generate flip keys for each image if needed
    if use_flip and rng_key is not None:
        device_batch_size = images.shape[0]
        flip_keys = jax.random.split(rng_key, device_batch_size)
    else:
        flip_keys = None

    # Process each image in the batch
    def process_single(image, flip_key):
        return process_image_on_tpu(image, use_flip, flip_key)

    if use_flip and flip_keys is not None:
        processed_images = jax.vmap(process_single)(images, flip_keys)
    else:
        processed_images = jax.vmap(lambda img: process_image_on_tpu(img, False, None))(
            images
        )
    
    return {
        "image": processed_images,
        "label": labels,
    }


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


def prepare_batch_data(batch):
    """
    Reformat a input batch from PyTorch Dataloader.

    Args: (torch)
      batch = (image, label)
        image: shape (host_batch_size, H, W, C) - uint8 numpy arrays
        label: shape (host_batch_size)
      batch_size = expected batch_size of this node, for eval's drop_last=False only

    Returns: a dict (numpy)
      image shape (local_devices, device_batch_size, H, W, C) - uint8
    """
    image, label = batch

    image = image.permute(0, 2, 3, 1)

    image = image.numpy()
    label = label.numpy()

    return_dict = {
        "image": image,
        "label": label,
    }
    
    return pad_and_merge(return_dict, image.shape[0])[0] # "mask" is not needed

class ImageNetDataset(torch.utils.data.Dataset):

    def __init__(self, root, img_transform=None):
        self.img_root = root
        
        self.folder_list = sorted([file for file in os.listdir(self.img_root)])
        self.file_list = []
        for folder in self.folder_list:
            folder_path = os.path.join(self.img_root, folder)
            files_in_folder = os.listdir(folder_path)
            files_in_folder = [os.path.join(folder, file) for file in files_in_folder if file.endswith('.JPEG')]
            self.file_list.extend(files_in_folder)
        
        assert len(os.listdir(self.img_root)) == 1000, f'Expect 1000 folders in {self.img_root}, got {len(os.listdir(self.img_root))}'
        assert img_transform is not None, 'transform should be provided'
        self.img_transform = img_transform
        
        # Create synset to label mapping
        self.synset_to_label = {synset: idx for idx, synset in enumerate(self.folder_list)}

    def __len__(self):
        return len(self.file_list)

    def __repr__(self) -> str:
        # copied from some random pytorch code
        head = 'Dataset ' + self.__class__.__name__
        body = [f'Number of datapoints: {self.__len__()}']
        if self.img_root is not None:
            body.append(f'Root location: {self.img_root}')
        lines = [head] + [' ' * 4 + line for line in body]
        return '\n'.join(lines)

    def __getitem__(self, idx):
        basename = self.file_list[idx]
        image_path = os.path.join(self.img_root, basename)
        img = self.img_transform(pil_loader(image_path)) 

        # Extract synset (e.g., 'n02111500') from basename (e.g., 'n02111500/n02111500_8116.JPEG')
        synset = basename.split('/')[0]
        label = self.synset_to_label[synset]
        label = torch.tensor(label, dtype=torch.long)

        return img, label

def worker_init_fn(worker_id, rank):
    seed = worker_id + rank * 1000
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def create_imagenet_split(dataset_cfg, batch_size, split):
    """
    Creates a split from ImageNet using Torchvision Datasets.

    Args:
      dataset_cfg: Configurations for the dataset.
      batch_size: Batch size for the dataloader.
      split: 'train' or 'val'.
    Returns:
      it: A PyTorch Dataloader.
      steps_per_epoch: Number of steps to loop through the DataLoader.
    """
    rank = jax.process_index()
    if dataset_cfg.use_flip:
        transform_to_use = transforms.Compose(
            [
                transforms.Lambda(
                    lambda pil_image: center_crop_arr(pil_image, dataset_cfg.image_size)
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                ),
            ]
        )
    else:
        transform_to_use = transforms.Compose(
            [
                transforms.Lambda(
                    lambda pil_image: center_crop_arr(pil_image, dataset_cfg.image_size)
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                ),
            ]
        )
    ds = ImageNetDataset(
        root=os.path.join(dataset_cfg.root, split),
        img_transform=transform_to_use,
    )
    log_for_0(ds)
    sampler = DistributedSampler(
        ds,
        num_replicas=jax.process_count(),
        rank=rank,
        shuffle=True,
    )
    it = DataLoader(
        ds,
        batch_size=batch_size,
        drop_last=True,
        worker_init_fn=partial(worker_init_fn, rank=rank),
        sampler=sampler,
        num_workers=dataset_cfg.num_workers,
        prefetch_factor=(
            dataset_cfg.prefetch_factor if dataset_cfg.num_workers > 0 else None
        ),
        pin_memory=dataset_cfg.pin_memory,
        persistent_workers=True if dataset_cfg.num_workers > 0 else False,
    )
    steps_per_epoch = len(it)
    return it, steps_per_epoch
