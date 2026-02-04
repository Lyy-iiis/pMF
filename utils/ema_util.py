from functools import partial

import jax
import jax.numpy as jnp

def const_schedule(step, ema_value):
    return ema_value

def edm_schedule(step, ema_halflife_kimg):
    ema_halflife_nimg = ema_halflife_kimg * 1000

    ema_rampup_ratio = 0.05
    ema_halflife_nimg = jnp.minimum(ema_halflife_nimg, step * 1024 * ema_rampup_ratio)

    ema_beta = 0.5 ** (1024 / jnp.maximum(ema_halflife_nimg, 1e-8))
    return ema_beta

def ema_schedules(config):
    ema_type = config.training.get("ema_type", "const")

    if ema_type == "const":
        return const_schedule
    elif ema_type == "edm":
        return edm_schedule
    else:
        raise ValueError("Unknown EMA!")


def update_ema(ema_params, params, alpha):
    return jax.tree_map(lambda e, p: alpha * e + (1 - alpha) * p, ema_params, params)
