import einops
import jax
import numpy as np
import torch

from esm import rotary_embedding as rot_pyt

from esmjax.modules import rotary_embedding as rot_jax


def test_match():
    rot_jax_layer = rot_jax.RotaryEmbedding(8)
    rot_pyt_layer = rot_pyt.RotaryEmbedding(8)

    key = jax.random.PRNGKey(42)
    q_key, k_key = jax.random.split(key)

    # Create data
    # JAX form is [batch, len, heads, dim]
    q_jax = jax.random.normal(q_key, (3, 8, 2, 8))
    k_jax = jax.random.normal(k_key, (3, 8, 2, 8))
    # Torch form is [batch * heads, seq, dim]
    q_pyt = torch.Tensor(einops.rearrange(np.array(q_jax), "b s H d -> (b H) s d"))
    k_pyt = torch.Tensor(einops.rearrange(np.array(k_jax), "b s H d -> (b H) s d"))

    # Apply the embeddings
    # JAX version
    _, param = rot_jax_layer.init_with_output(key, q_jax)
    q_jax = rot_jax_layer.apply(param, q_jax)
    k_jax = rot_jax_layer.apply(param, k_jax)
    # Torch version
    (q_pyt, k_pyt) = rot_pyt_layer(q_pyt, k_pyt)

    # Check allclose
    assert np.allclose(
        einops.rearrange(q_pyt.numpy(), "(b H) s d -> b s H d", b=3, H=2),
        np.array(q_jax),
    )
    assert np.allclose(
        einops.rearrange(k_pyt.numpy(), "(b H) s d -> b s H d", b=3, H=2),
        np.array(k_jax),
    )
