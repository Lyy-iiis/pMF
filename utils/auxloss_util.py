import jax
import jax.numpy as jnp
from typing import Tuple

from models.convnext import load_convnext_jax_model
from lpips_j.lpips import LPIPS
from utils.logging_util import log_for_0

def _resample_crop_to_square(
    img: jnp.ndarray,
    top: jnp.ndarray,
    left: jnp.ndarray,
    crop_h: jnp.ndarray,
    crop_w: jnp.ndarray,
    out_size: int,
) -> jnp.ndarray:
    out = jnp.asarray(out_size, jnp.float32)

    crop_h_f = jnp.asarray(crop_h, jnp.float32)
    crop_w_f = jnp.asarray(crop_w, jnp.float32)
    top_f = jnp.asarray(top, jnp.float32)
    left_f = jnp.asarray(left, jnp.float32)

    # scale factors
    sy = out / crop_h_f
    sx = out / crop_w_f

    # translations (in output pixel coordinates)
    ty = -top_f * sy
    tx = -left_f * sx

    scale = jnp.stack([sy, sx])          # (2,)
    translation = jnp.stack([ty, tx])    # (2,)

    return jax.image.scale_and_translate(
        img,
        shape=(out_size, out_size, img.shape[-1]),  # must be concrete ints
        spatial_dims=(0, 1),
        scale=scale,
        translation=translation,
        method="cubic",
        antialias=True,
    )

def paired_random_resized_crop(
    rng: jax.Array,
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    out_size: int = 224,
    scale: Tuple[float, float] = (0.08, 1.0),
    ratio: Tuple[float, float] = (3.0/4.0, 4.0/3.0),
):
    assert x1.shape == x2.shape
    B, H, W, C = x1.shape
    keys = jax.random.split(rng, B)

    def sample_params(key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        area = jnp.asarray(H * W, jnp.float32)
        target_area = area * jax.random.uniform(k1, (), minval=scale[0], maxval=scale[1])

        log_ratio = jnp.log(jnp.asarray(ratio, jnp.float32))
        aspect = jnp.exp(jax.random.uniform(k2, (), minval=log_ratio[0], maxval=log_ratio[1]))

        crop_w = jnp.clip(jnp.round(jnp.sqrt(target_area * aspect)).astype(jnp.int32), 1, W)
        crop_h = jnp.clip(jnp.round(jnp.sqrt(target_area / aspect)).astype(jnp.int32), 1, H)

        max_top = H - crop_h
        max_left = W - crop_w
        top = jax.random.randint(k3, (), 0, max_top + 1)
        left = jax.random.randint(k4, (), 0, max_left + 1)
        return top, left, crop_h, crop_w

    tops, lefts, crop_hs, crop_ws = jax.vmap(sample_params)(keys)

    fn = lambda img, t, l, h, w: _resample_crop_to_square(img, t, l, h, w, out_size)
    y1 = jax.vmap(fn)(x1, tops, lefts, crop_hs, crop_ws)
    y2 = jax.vmap(fn)(x2, tops, lefts, crop_hs, crop_ws)
    return y1, y2

def init_auxloss(config):
    def param_count(params):
        return sum([jnp.prod(jnp.array(p.shape)) 
                    for p in jax.tree_util.tree_leaves(params)])
    
    if config.model.lpips:
        log_for_0(f"Loading LPIPS model...")
        lpips_model = LPIPS()
        dummy_input = jnp.zeros((1, 224, 224, 3))  # Example input shape
        lpips_params = lpips_model.init(jax.random.PRNGKey(0), dummy_input, dummy_input)
        log_for_0(f"LPIPS model loaded with param count {param_count(lpips_params)}.")

    if config.model.convnext:
        log_for_0(f"Loading ConvNext classifier...")
        convnext_head_model, convnext_head_params = load_convnext_jax_model()
        log_for_0(f"ConvNext classifier loaded with param count {param_count(convnext_head_params)}.")

    # Implementation following https://arxiv.org/abs/2512.10953
    def auxloss_fn(model_images, gt_images, rng=None):
        """
        input: 
            image batch, shape (B, H, W, 3)
            target batch, shape (B,) with integer class labels
        output:
            loss value, scalar
        """
        bsz = model_images.shape[0]

        model_images, gt_images = paired_random_resized_crop(rng, model_images, gt_images, out_size=224)

        if config.model.lpips:
            lpips_dist = lpips_model.apply(lpips_params, model_images, gt_images).reshape(-1)
        else:
            lpips_dist = jnp.zeros((bsz,), dtype=jnp.float32)


        if config.model.convnext:
            convnext_model_images = convnext_head_model.apply(convnext_head_params, model_images)
            convnext_gt_images = convnext_head_model.apply(convnext_head_params, gt_images)
            class_dist = jnp.sum((convnext_model_images - convnext_gt_images) ** 2, axis=-1)
        else:
            class_dist = jnp.zeros((bsz,), dtype=jnp.float32)
            
        return lpips_dist, class_dist
    
    log_for_0("Auxiliary loss function initialized")
    return auxloss_fn