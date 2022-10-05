import jax
import numpy as np

import pytest
import torch

from esmjax.modules import multihead_attention
from flax.core import frozen_dict

ATOL = 5e-6

layer_nums = [0, 2, 4]


@pytest.mark.parametrize("layer_num", layer_nums)
def test_match(layer_num, torch_model, esm_params):
    rng = jax.random.PRNGKey(0)

    arr = jax.random.normal(rng, (2, 8, 320)) / np.sqrt(320)
    arr_pyt = torch.Tensor(np.array(arr.transpose((1, 0, 2))))

    # Get output of the torch layer.
    torch_layer = torch_model.layers[layer_num].self_attn
    out_pyt = torch_layer(arr_pyt, arr_pyt, arr_pyt)[0]
    out_pyt = out_pyt.detach().numpy().transpose((1, 0, 2))

    # Get output of the JAX layer
    mha = multihead_attention.RoPEMultiHeadDotProductAttention(20)
    params = frozen_dict.freeze({"params": esm_params[f"{layer_num}"]["self_attn"]})
    out_jax = np.array(mha.apply(params, arr, arr))

    assert np.allclose(out_jax, out_pyt, atol=ATOL)
