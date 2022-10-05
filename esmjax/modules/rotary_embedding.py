import flax.linen as nn
import jax.numpy as jnp
from jaxtyping import Array, Float


class RotaryEmbedding(nn.Module):
    """Applies rotary embeddings (RoPE) to the input sequence tensor,
    as described in https://arxiv.org/abs/2104.09864.

    Attributes:
        dim (int): Dimensionality of the feature vectors
        base_exponent (int): Base exponent to compute embeddings from
    """

    dim: int
    base_exponent: int = 10000

    def setup(self):
        assert self.dim % 2 == 0

    def __call__(
        self,
        x: Float[Array, "batch len heads qk_dim"],
    ) -> Float[Array, "batch len heads qk_dim"]:
        # Compute the per-dimension frequencies
        exponents = jnp.arange(0, self.dim, 2, dtype=x.dtype)
        inv_freq = 1.0 / (self.base_exponent ** (exponents / self.dim))

        # Compute the per element phase (to pass into sin and cos)
        t = jnp.arange(x.shape[1], dtype=x.dtype)
        phase = jnp.einsum("i,j->ij", t, inv_freq)
        phase = jnp.tile(phase, reps=(1, 2))[None, :, None, :]

        x = x * jnp.cos(phase) + self.rotate_half(x) * jnp.sin(phase)

        return x

    @staticmethod
    def rotate_half(
        x: Float[Array, "batch len heads dim"]
    ) -> Float[Array, "batch len heads dim"]:
        "Obtain the rotated counterpart of each feature"
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate((-x2, x1), axis=-1)
