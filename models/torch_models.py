from math import sqrt
import jax
import jax.numpy as jnp
import jax.random as jr
from flax import linen as nn

@jax.custom_jvp
def silu_stable(x):
    sig = jax.nn.sigmoid(x)
    return x * sig

@silu_stable.defjvp
def silu_stable_jvp(primals, tangents):
    (x,), (x_dot,) = primals, tangents
    sig = jax.nn.sigmoid(x)
    inv_sig = jax.nn.sigmoid(-x)
    ydot = x_dot * (sig + x * sig * inv_sig)
    return x * sig, ydot

@jax.custom_jvp
def rms_norm_stable(x):
    eps = jnp.array(1e-6, dtype=jnp.float32)
    mean_square = jnp.mean(jnp.square(x), axis=-1, keepdims=True).astype(jnp.float32)
    return x * jax.lax.rsqrt(mean_square + eps).astype(jnp.float32)

@rms_norm_stable.defjvp
def rms_norm_stable_jvp(primals, tangents):
    (x,), (x_dot,) = primals, tangents
    x = x.astype(jnp.float32)
    x_dot = x_dot.astype(jnp.float32)
    eps = jnp.array(1e-6, dtype=jnp.float32)
    D = x.shape[-1]
    mean_square = jnp.mean(jnp.square(x), axis=-1, keepdims=True).astype(jnp.float32)
    r = jax.lax.rsqrt(mean_square + eps).astype(jnp.float32)
    
    normed = x * r
    normed_g = x_dot * r
    return normed, normed_g - (normed * jnp.mean(normed * normed_g, axis=-1, keepdims=True))


class TorchLinear(nn.Module):
    """A linear layer similar to torch.nn.Linear."""

    in_features: int
    out_features: int
    bias: bool = True
    weight_init: str = "scaled_variance"
    init_constant: float = 1.0
    bias_init: str = "zeros"  # options: 'zeros'

    def setup(self):
        """Setup the linear layer with the specified initialization."""

        if self.weight_init == "scaled_variance":
            std = self.init_constant / sqrt(self.in_features)
            weight_initializer = nn.initializers.normal(stddev=std)
        elif self.weight_init == "zeros":
            weight_initializer = nn.initializers.zeros
        else:
            raise ValueError(f"Invalid weight_init: {self.weight_init}")

        if self.bias_init == "zeros":
            bias_initializer = nn.initializers.zeros
        else:
            raise ValueError(f"Invalid bias_init: {self.bias_init}")

        self._flax_linear = nn.Dense(
            features=self.out_features,
            use_bias=self.bias,
            kernel_init=weight_initializer,
            bias_init=bias_initializer,
        )

    def __call__(self, x):
        return self._flax_linear(x)


class TorchEmbedding(nn.Module):
    """A embedding layer similar to torch.nn.Embedding."""

    num_embeddings: int
    embedding_dim: int
    weight_init: str = "scaled_variance"
    init_constant: float = 1.0

    def setup(self):
        """Setup the embedding layer with the specified initialization."""

        if self.weight_init is None:
            std = 0.02
        elif self.weight_init == "scaled_variance":
            std = self.init_constant / sqrt(self.embedding_dim)
        else:
            raise ValueError(f"Invalid weight_init: {self.weight_init}")

        init = lambda key, shape, dtype: jr.normal(key, shape, dtype) * std
        self._flax_embedding = self.param(
            "embedding", init, (1, self.num_embeddings, self.embedding_dim), dtype=jnp.float32
        )

    def __call__(self, x):
        return self._flax_embedding[:, x, :][0]


class RMSNorm(nn.Module):
    """Root Mean Square Normalization."""

    dim: int
    eps: float = 1e-6

    def setup(self):
        self.weight = self.param("kernel", nn.initializers.ones, (self.dim,))

    def _norm(self, x):
        return rms_norm_stable(x)

    def __call__(self, x):
        output = self._norm(x).astype(x.dtype)
        return output * self.weight


class SwiGLUMlp(nn.Module):
    """Swish-Gated Linear Unit MLP."""

    in_features: int
    hidden_features: int

    weight_init: str = "scaled_variance"
    weight_init_constant: float = 1.0

    def setup(self):
        init_kwargs = dict(
            bias=False,
            weight_init=self.weight_init,
            init_constant=self.weight_init_constant,
        )

        self.w1 = TorchLinear(self.in_features, self.hidden_features, **init_kwargs)
        self.w3 = TorchLinear(self.in_features, self.hidden_features, **init_kwargs)
        self.w2 = TorchLinear(self.hidden_features, self.in_features, **init_kwargs)

    def __call__(self, x):
        return self.w2(silu_stable(self.w1(x)) * self.w3(x))
