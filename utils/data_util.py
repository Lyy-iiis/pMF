import os
from functools import partial

import jax
import numpy as np

NUM_CLASSES = 1000


def create_imagenet_dataloader(
    imagenet_root, split, batch_size, image_size, num_workers=4, for_fid=False
):
    """Create ImageNet dataloader for the specified split."""
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from torchvision import datasets, transforms

    from utils.input_pipeline import center_crop_arr, loader, worker_init_fn
    from utils.logging_util import log_for_0

    if for_fid:
        # For FID: only center crop, return numpy array directly
        def fid_transform(pil_image):
            cropped = center_crop_arr(pil_image, image_size)
            return np.array(cropped)  # PIL -> numpy [0,255] uint8

        transform = fid_transform
    else:
        transform = transforms.Compose(
            [
                transforms.Lambda(
                    lambda pil_image: center_crop_arr(pil_image, image_size)
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                ),
            ]
        )

    dataset = datasets.ImageFolder(
        os.path.join(imagenet_root, split),
        transform=transform,
        loader=loader,
    )

    log_for_0(f"Dataset {split} (FID={for_fid}): {dataset}")

    rank = jax.process_index()
    num_replicas = jax.process_count()
    log_for_0(f"Distributed setup: rank={rank}, num_replicas={num_replicas}")
    log_for_0(f"JAX devices: {jax.devices()}")
    log_for_0(f"JAX local devices: {jax.local_devices()}")

    # Check distributed setup
    if num_replicas == 1:
        log_for_0("WARNING: Only 1 process detected - running in single-worker mode!")

    sampler = DistributedSampler(
        dataset,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=False,
    )

    log_for_0(
        f"DistributedSampler: total_samples={len(dataset)}, samples_per_replica={len(sampler)}"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=False,
        worker_init_fn=partial(worker_init_fn, rank=rank),
        sampler=sampler,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=False,
        persistent_workers=True if num_workers > 0 else False,
    )

    # Return the per-worker dataset size (distributed size) and total dataset size
    return dataloader, len(sampler), len(dataset)
