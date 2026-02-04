import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
from flax import linen as nn
from typing import Dict, Any, Sequence, Tuple
import jax
import functools

def get_hsdp_spec(tensor_leaf, axis_names: Tuple[str, str], mode: str):
    """
    Generate PartitionSpec based on tensor dimension and mode.
    """
    if not isinstance(tensor_leaf, (jax.Array, jax.ShapeDtypeStruct)):
        return P()
    
    dp_axis, fsdp_axis = axis_names

    if mode == 'hsdp':
        ndim = tensor_leaf.ndim
        
        if ndim == 0:
            return P()
        if ndim == 1:
            if tensor_leaf.shape[0] % 4 != 0:
                return P()
            return P(fsdp_axis)
        return P(*([None] * (ndim - 1) + [fsdp_axis]))        
    else:
        raise ValueError(f"Unknown mode: {mode}")

@functools.lru_cache(maxsize=None)
def hsdp_mesh():
    mesh_shape = (jax.process_count(), jax.local_device_count())
    axis_names = ['data', 'hsdp']
    devices = mesh_utils.create_device_mesh(mesh_shape, allow_split_physical_axes=True)
    mesh = Mesh(devices, axis_names=axis_names)
    return mesh

def init_model_distributed(
    model: nn.Module,
    dummy_input: Dict[str, Any],
    master_rng: jax.Array = jax.random.PRNGKey(0),
    rng_keys: Sequence[str] = [], 
    mode: str = 'hsdp'
) -> Tuple[Any, Mesh]:
    """
    Args:
        model: Flax Module; 
        dummy_input: input dict passing to the model; 
        master_rng: main seed; 
        rng_keys: extra rng keys needed, e.g. ['dropout', 'gating']
        mode: 'hsdp' or 'ddp'
        
    Returns:
        params: initialized parameters. 
    """
    
    total_keys_needed = 1 + len(rng_keys)
    splitted_keys = jax.random.split(master_rng, total_keys_needed)
    rngs = {'params': splitted_keys[0]}
    for i, key_name in enumerate(rng_keys):
        rngs[key_name] = splitted_keys[i+1]
    
    mesh = hsdp_mesh()
    axis_names = ['data', 'hsdp']
    def _init_wrapper(rngs_inner, inputs_inner):
        return model.init(rngs_inner, **inputs_inner)

    abstract_variables = jax.eval_shape(
        _init_wrapper, rngs, dummy_input
    )

    # build sharding spec
    # pass the axis names of mesh, ensure Spec and Mesh correspond
    sharding_spec_tree = jax.tree_map(
        lambda x: NamedSharding(mesh, get_hsdp_spec(x, axis_names, mode)),
        abstract_variables
    )

    # 5. JIT compile and execute
    def _jit_init_fn(rngs_inner, inputs_inner):
        return model.init(rngs_inner, **inputs_inner)
    
    jit_init_fn = jax.jit(_jit_init_fn, out_shardings=sharding_spec_tree)

    variables = jit_init_fn(rngs, dummy_input)
    
    return variables


def merge_data(data):
    '''
    Args:
        data: pytree; data on current process; 
    Returns:
        data: pytree; data on all devices
    '''
    fsdp_mesh = hsdp_mesh()
    
    def auto_shard(x):
        len_shape = len(x.shape)
        sharding = [('data', 'hsdp')] + [None] * (len_shape - 1)
        sharding = NamedSharding(fsdp_mesh, P(*sharding))
        
        addressable_devices = sharding.addressable_devices
        num_local_devices = len(addressable_devices)
        
        global_shape = (x.shape[0] * jax.process_count(),) + x.shape[1:]
        local_batch_size = x.shape[0]
        shard_per_device = local_batch_size // num_local_devices
        
        local_arrays = []
        for i, device in enumerate(addressable_devices):
            start = i * shard_per_device
            end = (i + 1) * shard_per_device
            device_array = jax.device_put(x[start:end], device)
            local_arrays.append(device_array)
            
        return jax.make_array_from_single_device_arrays(
            global_shape, 
            sharding, 
            local_arrays
        )
        
    return jax.tree.map(auto_shard, data)

def pad_and_merge(data, local_bsz):
    '''
    Args:
        data: pytree; data on current process; 
        local_bsz: int; desired local batch size (e.g. 32)
    Returns:
        data_merged: pytree; global data (JAX Array)
        mask_merged: jax.Array; global mask (1=valid, 0=padding)
    '''
    leaves = jax.tree.leaves(data)
    if not leaves:
        raise ValueError("Data is empty")
    current_len = leaves[0].shape[0]
    
    pad_len = local_bsz - current_len
    assert pad_len >= 0, f"local_bsz: {local_bsz} is less than current_len: {current_len}"
    local_mask = jnp.concatenate([
        jnp.ones(current_len, dtype=jnp.int32),
        jnp.zeros(pad_len, dtype=jnp.int32)
    ])
    
    def pad_leaf(x):
        x = jnp.asarray(x)
        if pad_len == 0:
            return x
        pad_width = [(0, pad_len)] + [(0, 0)] * (x.ndim - 1)
        return jnp.pad(x, pad_width, mode='constant', constant_values=0)
    
    local_data_padded = jax.tree_map(pad_leaf, data)
    return merge_data(local_data_padded), merge_data(local_mask)

def replicate_data(data_to_replicate):
    mesh = hsdp_mesh()

    if isinstance(data_to_replicate, (float, int)):
        data_to_replicate = jnp.array(data_to_replicate)
    
    global_shape = data_to_replicate.shape

    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    
    def callback(index):
        return data_to_replicate
        
    return jax.make_array_from_callback(global_shape, sharding, callback)

def replicate(pytree):
    """Replicate a PyTree across all devices in the mesh."""
    return jax.tree.map(replicate_data, pytree)


# global constants

MESH = hsdp_mesh()
REPLICATED_SHARDING = NamedSharding(MESH, P())
AXIS_NAMES = ('data', 'hsdp')
