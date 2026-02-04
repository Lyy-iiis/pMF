from __future__ import annotations

import jax.numpy as jnp

def resize_torch_grid_sample(x, out_h=299, out_w=299):
    assert x.ndim == 4, x.shape
    B, H, W, C = x.shape
    x = x.astype(jnp.float32)

    ys = (jnp.arange(out_h, dtype=jnp.float32) + 0.5) * (H / out_h) - 0.5
    y0 = jnp.floor(ys).astype(jnp.int32)
    y1 = y0 + 1
    wy = (ys - y0.astype(jnp.float32)).reshape(1, out_h, 1, 1)

    row0 = jnp.take(x, y0, axis=1, mode="clip")
    row1 = jnp.take(x, y1, axis=1, mode="clip")
    tmp  = row0 * (1.0 - wy) + row1 * wy

    xs = (jnp.arange(out_w, dtype=jnp.float32) + 0.5) * (W / out_w) - 0.5
    x0 = jnp.floor(xs).astype(jnp.int32)
    x1 = x0 + 1
    wx = (xs - x0.astype(jnp.float32)).reshape(1, 1, out_w, 1)

    col0 = jnp.take(tmp, x0, axis=2, mode="clip")
    col1 = jnp.take(tmp, x1, axis=2, mode="clip")
    out  = col0 * (1.0 - wx) + col1 * wx
    
    return out