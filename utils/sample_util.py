import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from utils import fid_util
from utils.logging_util import log_for_0
from utils.hsdp_util import AXIS_NAMES, hsdp_mesh
from jax.sharding import NamedSharding, PartitionSpec as P

def create_global_sample_idx(step):
    mesh = hsdp_mesh()
    
    global_shape = (jax.device_count(),)
    sharding = NamedSharding(mesh, P(AXIS_NAMES))

    def index_callback(index):
        start = index[0].start + step * jax.device_count()
        stop = index[0].stop + step * jax.device_count()
        return jnp.arange(start, stop, dtype=jnp.int32)

    return jax.make_array_from_callback(global_shape, sharding, index_callback)

def run_p_sample_step(
        p_sample_step, state, sample_idx, ema_key=False, **kwargs
):
    """
    Run one p_sample_step to get samples from the model.
    """
    params = state.ema_params[ema_key] if ema_key else state.params

    variable = {"params": params}
    samples = p_sample_step(variable, sample_idx=sample_idx, **kwargs)

    assert not jnp.any(
        jnp.isnan(samples)
    ), f"There is nan in samples!"

    samples = samples.transpose(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
    samples = 127.5 * samples + 128.0
    samples = jnp.clip(samples, 0, 255).astype(jnp.uint8)

    jax.random.normal(random.key(0), ()).block_until_ready()  # dist sync
    return samples


def generate_fid_samples(
    state, config, p_sample_step, run_p_sample_step, ema_key, **kwargs
):
    """
    Generate samples for FID evaluation.
    """
    num_steps = np.ceil(
        config.fid.num_samples / config.fid.device_batch_size / jax.device_count()
    ).astype(int)

    samples_all = []

    log_for_0("Note: the first sample may be significant slower")
    for step in range(num_steps):
        sample_idx = jnp.arange(jax.device_count())
        sample_idx = jax.device_count() * step + sample_idx
        
        assert sample_idx.shape[0] == jax.device_count(), f"sample_idx shape {sample_idx.shape}, device_count {jax.device_count()}"
        log_for_0(f"Sampling step {step} / {num_steps}...")
        samples = run_p_sample_step(
            p_sample_step, state, sample_idx=sample_idx, ema_key=ema_key, **kwargs
        )
        
        local_samples = np.concatenate([
            jax.device_get(shard.data) for shard in samples.addressable_shards
        ], axis=0)
        
        samples_all.append(local_samples)

    samples_all = np.concatenate(samples_all, axis=0)

    return samples_all


def get_fid_evaluator(config, writer):
    """
    Create FID evaluator function.
    """
    inception_net = fid_util.build_jax_inception()
    stats_ref = fid_util.get_reference(config.fid.cache_ref)
    run_p_sample_step_inner = run_p_sample_step

    def _evaluate_one_mode(state, p_sample_step, ema_key, **kwargs):
        log_for_0(f'evaluating using ema_key={ema_key}...')
        # 1) Sampling
        samples_all = generate_fid_samples(
            state, config, p_sample_step, run_p_sample_step_inner, ema_key, **kwargs
        )
        # 2) Stats
        stats = fid_util.compute_stats(samples_all, inception_net)
        # 3) Metrics
        metric = {}

        mode_str = ema_key if ema_key else "online"

        omega = kwargs.get("omega", None)
        t_min = kwargs.get("t_min", None)
        t_max = kwargs.get("t_max", None)
        log_for_0(
            f"Computing FID and Inception Score at omega={omega:.2f}, t_min={t_min:.2f}, t_max={t_max:.2f}, mode={mode_str}..."
        )
        descriptor = f"omega_{omega:.2f}_tmin_{t_min:.2f}_tmax_{t_max:.2f}_{mode_str}"

        fid = fid_util.compute_fid(
            stats_ref["mu"], stats["mu"], stats_ref["sigma"], stats["sigma"]
        )
        is_score, _ = fid_util.compute_inception_score(stats["logits"])

        metric[f"FID_{descriptor}"] = fid
        metric[f"IS_{descriptor}"] = is_score

        log_for_0(f"FID ({descriptor}): {fid:.4f}, IS ({descriptor}): {is_score:.4f}")

        return metric, fid, is_score, ema_key

    def evaluator(state, p_sample_step, step, ema_only=False, **kwargs):
        metric_dict = {}
        
        best_fid = float("inf")
        best_key = None
        
        for k in state.ema_params.keys():
            metric, fid, is_score, ema_key = _evaluate_one_mode(state, p_sample_step, k, **kwargs)
            metric_dict.update(metric)
            best_fid, best_key = min((best_fid, best_key), (fid, ema_key))

        if not ema_only:
            metric, fid, _, _ = _evaluate_one_mode(state, p_sample_step, False, **kwargs)
            ema_key = 'online'
            metric_dict.update(metric)
            # best_fid = min(best_fid, fid) # sometimes, online is better than ema
            best_fid, best_key = min((best_fid, best_key), (fid, ema_key))

        writer.write_scalars(step + 1, metric_dict)
        return best_fid, is_score, best_key

    return evaluator
