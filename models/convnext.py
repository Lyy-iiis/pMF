# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import re
from functools import partial
from typing import Sequence
import jax
import jax.numpy as jnp
import torch
from flax import linen as nn
from utils.logging_util import log_for_0

class ConvNextLayerNorm(nn.Module):
    normalized_shape: int
    eps: float = 1e-6
    
    def setup(self):
        self.weight = self.param('weight', nn.initializers.ones, (self.normalized_shape,))
        self.bias = self.param('bias', nn.initializers.zeros, (self.normalized_shape,))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
        x = (x - mean) / jnp.sqrt(var + self.eps)
        x = self.weight * x + self.bias # broadcast along the last dimension
        return x
    
class ConvNextGRN(nn.Module):
    """ 
    GRN (Global Response Normalization) layer.
    """
    dim: int
    eps: float = 1e-6

    def setup(self):
        self.gamma = self.param('gamma', nn.initializers.zeros, (1, 1, 1, self.dim))
        self.beta = self.param('beta', nn.initializers.zeros, (1, 1, 1, self.dim))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        gamma = self.gamma
        beta = self.beta
        norm = jnp.sum(x ** 2, axis=(1, 2), keepdims=True)
        Gx = jnp.sqrt(norm + self.eps)
        Nx = Gx / (jnp.mean(Gx, axis=-1, keepdims=True) + self.eps)
        return gamma * (x * Nx) + beta + x
    
class ConvNextBlock(nn.Module):
    """ 
    ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
    """
    dim: int
    
    def setup(self):
        # depthwise conv
        self.dwconv = nn.Conv(
            features=self.dim,
            kernel_size=(7, 7),
            padding='SAME',
            feature_group_count=self.dim,
            name='dwconv',
        )
        self.norm = ConvNextLayerNorm(
            self.dim,
            eps=1e-6,
        )
        # pointwise/1x1 convs, implemented with linear layers
        self.pwconv1 = nn.Dense(
            features=4 * self.dim,
            name='pwconv1',
        )
        self.grn = ConvNextGRN(4 * self.dim)
        self.pwconv2 = nn.Dense(
            features=self.dim,
            name='pwconv2',
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)  # pointwise/1x1 convs, implemented with linear layers
        x = jax.nn.gelu(x, approximate=False)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = input + x # residual connection
        return x
    
class ConvNextV2(nn.Module):
    """ ConvNeXt V2
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    in_chans: int = 3
    num_classes: int = 1000
    drop_path_rate: float = 0.0
    head_init_scale: float = 1.0
    depths: Sequence[int] = (3, 3, 9, 3)
    dims: Sequence[int] = (96, 192, 384, 768)
    
    def setup(self):
        # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            layers=[
                nn.Conv(
                    features=self.dims[0],
                    kernel_size=(4, 4),
                    strides=(4, 4),
                ),
                ConvNextLayerNorm(
                    self.dims[0],
                    eps=1e-6,
                ),
            ],
            name='downsample_layers_0',
        )
        layers = [stem]
        for i in range(3):
            downsample_layer = nn.Sequential(
                layers=[
                    ConvNextLayerNorm(
                        self.dims[i],
                        eps=1e-6,
                    ),
                    nn.Conv(
                        features=self.dims[i + 1],
                        kernel_size=(2, 2),
                        strides=(2, 2),
                    ),
                ],
                name=f'downsample_layers_{i + 1}',
            )
            layers.append(downsample_layer)
        self.downsample_layers = layers
        
        # 4 feature resolution stages, each consisting of multiple residual blocks
        stages = []
        for i in range(4):
            stage = nn.Sequential(
                layers=[ConvNextBlock(dim=self.dims[i]) for _ in range(self.depths[i])],
                name=f'stages_{i}',
            )
            stages.append(stage)
        self.stages = stages
        self.norm = nn.LayerNorm(epsilon=1e-6) # final norm layer
        self.head = nn.Dense(features=self.num_classes)

    def forward_features(self, x: jnp.ndarray) -> jnp.ndarray:
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean(axis=(1, 2))) # global average pooling, (N, H, W, C) -> (N, C)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.forward_features(x)
        # x = self.head(x)
        return x
    
##### MODEL CONFIGS #####
ConvNextBase = partial(ConvNextV2, depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])

def convert_weights_to_jax(jax_params: dict, module_pt, hf: bool = False):
    log_for_0("Converting ConvNext weights to jax...")
    jax_params_flat, jax_param_pytree = jax.tree_util.tree_flatten_with_path(jax_params)
    pt_params = {path: param for path, param in module_pt.items()}
    
    if hf: # need additional changes on param name
        # 1. change "classifier." to "head."
        # 2. remove "convnextv2.encoder." prefix
        # 3. change "convnextv2.embeddings.patch_embeddings." to "downsample_layers_0.layers_0."
        # 4. change "convnextv2.embeddings.layernorm." to "downsample_layers_0.layers_1."
        # 5. change "stages.n.downsampling_layer.m" to "downsample_layers_n.layers_m"
        # 6. change "stages.n.layers.m" to "stages_n.layers_m"
        # 7. change "*layernorm*" to "*norm*"
        # 8. remove "convnextv2." prefix
        # 9. change "grn.weight" to "grn.gamma", "grn.bias" to "grn.beta"
        # 10. change "dwconv" to "dwconv.conv"
        new_pt_params = {}
        for path, param in pt_params.items():
            path = re.sub(r"classifier\.", "head.", path)
            path = re.sub(r"convnextv2\.encoder\.", "", path)
            path = re.sub(r"convnextv2\.embeddings\.patch_embeddings\.", "downsample_layers_0.layers_0.", path)
            path = re.sub(r"convnextv2\.embeddings\.layernorm\.", "downsample_layers_0.layers_1.", path)
            path = re.sub(r"stages\.([0-3])\.downsampling_layer\.(\d+)", lambda m: f"downsample_layers_{m.group(1)}.layers_{m.group(2)}", path)
            path = re.sub(r"stages\.([0-3])\.layers\.(\d+)", lambda m: f"stages_{m.group(1)}.layers_{m.group(2)}", path)
            path = re.sub(r"layernorm", "norm", path)
            path = re.sub(r"convnextv2\.", "", path)
            path = re.sub(r"grn\.weight", "grn.gamma", path)
            path = re.sub(r"grn\.bias", "grn.beta", path)
            # path = re.sub(r"dwconv", "dwconv.conv", path)
            new_pt_params[path] = param
        pt_params = new_pt_params
    else:
        # 1. stages.n.m -> stages_n.layers_m
        # 2. downsample_layers.n.m -> downsample_layers_n.layers_m
        new_pt_params = {}
        for path, param in pt_params.items():
            for i in range(4):
                path = re.sub(rf"stages\.{i}\.(\d+)", lambda m: f"stages_{i}.layers_{m.group(1)}", path)
                path = re.sub(rf"downsample_layers\.{i}\.(\d+)", lambda m: f"downsample_layers_{i}.layers_{m.group(1)}", path)
            new_pt_params[path] = param
        pt_params = new_pt_params
        
    # add "param." prefix to PyTorch parameter names
    new_pt_params = {}
    for path, param in pt_params.items():
        new_pt_params[f"params.{path}"] = param
    pt_params = new_pt_params

    direct_copy = ['grn']
    ignore_keys = {}
    pt_params_flat = []
    for path, param in jax_params_flat:
        shape = param.shape
        path = ".".join([p.key for p in path])
        path = re.sub(r"\.scale|.kernel", ".weight", path)
        if path in pt_params:
            pt_param = pt_params[path]
            if any(dc_key in path for dc_key in direct_copy):
                if shape != pt_param.shape:
                    log_for_0(f"[WARNING] shape mismatch for direct copy: expect param '{path}' have shape {shape}, but got {pt_param.shape}")
                pt_params_flat.append(jnp.asarray(pt_param.detach().numpy()))
            else:
                if len(shape) == 4:
                    pt_param = torch.permute(pt_param, (2, 3, 1, 0))
                else:
                    pt_param = torch.permute(
                        pt_param, tuple(reversed(range(len(shape))))
                    )
                if shape != pt_param.shape:
                    log_for_0(f"[WARNING] shape mismatch after transpose: expect param '{path}' have shape {shape}, but got {pt_param.shape}")
                pt_params_flat.append(jnp.asarray(pt_param.detach().numpy()))
            pt_params.pop(path)
        else:
            log_for_0(f"[WARNING] missing param '{path}' with shape {shape} from PyTorch model")
            pt_params_flat.append(None)
            
    for path, param in pt_params.items():
        if path in ignore_keys:
            continue
        log_for_0(f"[WARNING] params not loaded '{path}' with shape {param.shape} from PyTorch model")

    log_for_0("ConvNext conversion done.")

    return jax.tree_util.tree_unflatten(jax_param_pytree, pt_params_flat)


def load_convnext_jax_model():
    model_jax = ConvNextBase()
    dummy_input = jnp.ones((1, 224, 224, 3))
    jax_params = model_jax.init(jax.random.PRNGKey(0), dummy_input)
    
    # using huggingface model weights
    from transformers import ConvNextV2ForImageClassification
    model = ConvNextV2ForImageClassification.from_pretrained("facebook/convnextv2-base-22k-224")
    model_pt = model.state_dict()
    jax_params = convert_weights_to_jax(jax_params, model_pt, hf=True)
    
    # remove huggingface cache
    import shutil
    import os
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache/huggingface")
    if os.path.exists(cache_dir):
        log_for_0("Removing Huggingface cache directory...")
        shutil.rmtree(cache_dir)
    
    return (model_jax, jax_params)