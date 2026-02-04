import jax
from flax.training import checkpoints

from utils.logging_util import log_for_0
from jax.experimental import multihost_utils

def restore_checkpoint(state, workdir):
    """
    Restores the model state from a checkpoint located in the specified working directory.
    """
    state = checkpoints.restore_checkpoint(workdir, state)
    log_for_0("Restored from checkpoint at {}".format(workdir))
    return state


def save_checkpoint(state, workdir):
    """
    Saves the model state to a checkpoint in the specified working directory.
    """
    # Save only one copy from device 0.
    jax.block_until_ready(state)
    step = int(jax.device_get(state.step))
    multihost_utils.sync_global_devices(f'ckpt_save_{step}')
    log_for_0("Saving checkpoint step %d.", step)
    checkpoints.save_checkpoint_multiprocess(workdir, state, step, keep=3)
    log_for_0("Checkpoint step %d saved.", step)
