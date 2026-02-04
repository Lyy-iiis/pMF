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
from torchvision import datasets

from utils.logging_util import log_for_0

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


def prepare_batch_data(batch, batch_size=None):
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

    # pad the batch if smaller than batch_size
    if batch_size is not None and batch_size > image.shape[0]:
        image = torch.cat(
            [
                image,
                torch.zeros(
                    (batch_size - image.shape[0],) + image.shape[1:], dtype=image.dtype
                ),
            ],
            axis=0,
        )
        label = torch.cat(
            [label, -torch.ones((batch_size - label.shape[0],), dtype=label.dtype)],
            axis=0,
        )

    # reshape (host_batch_size, height, width, 3) to
    # (local_devices, device_batch_size, height, width, 3)
    local_device_count = jax.local_device_count()
    image = image.reshape((local_device_count, -1) + image.shape[1:])
    label = label.reshape(local_device_count, -1)

    image = image.numpy()
    label = label.numpy()

    return_dict = {
        "image": image,
        "label": label,
    }

    return return_dict


def worker_init_fn(worker_id, rank):
    seed = worker_id + rank * 1000
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

from torchvision.datasets.folder import pil_loader

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
    # Create a loader that applies center crop on CPU
    # This is necessary to ensure all images have uniform size for batching
    def loader_with_crop(path: str):
        img = pil_loader(path)
        img_cropped = center_crop_arr(img, dataset_cfg.image_size)
        return np.array(img_cropped)  # Returns uint8 array (image_size, image_size, C)

    root = os.path.join(dataset_cfg.root, split)
    
    ds = datasets.ImageFolder(
        root,
        transform=None,  # No transforms - crop is done in loader
        loader=loader_with_crop,  # Returns uint8 numpy arrays (image_size, image_size, 3)
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
