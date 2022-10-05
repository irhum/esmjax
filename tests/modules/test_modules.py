import jax
import numpy as np
import pytest
import torch
from esmjax import tokenizer as esm_tokenizer
from flax.core import frozen_dict

ATOL = 5e-4


layer_nums = [0, 2, 5]


@pytest.mark.parametrize("layer_num", layer_nums)
def test_layer_match(layer_num, torch_model, encoder_layer_fn, esm_params):
    torch_layer = torch_model.layers[layer_num]
    embed_dim = torch_layer.self_attn.kdim

    # Create seeded random array
    rng = jax.random.PRNGKey(32)
    arr = jax.random.normal(rng, (2, 8, embed_dim))

    # Get torch outputs
    torch_arr = torch.Tensor(np.array(arr.transpose((1, 0, 2))))
    out_pyt = torch_layer(torch_arr)[0].detach().numpy().transpose((1, 0, 2))

    # Get JAX outputs
    block = encoder_layer_fn()
    params = frozen_dict.freeze({"params": esm_params[f"{layer_num}"]})
    out_jax = np.array(block.apply(params, arr))

    assert np.allclose(out_pyt, out_jax, atol=ATOL)


def torch_inference(torch_model, batch):
    with torch.no_grad():
        torch_arr = torch.LongTensor(batch)
        torch_outs = torch_model(torch_arr, repr_layers=[6], return_contacts=True)
        torch_out = torch_outs["representations"][6].numpy()

    return torch_out


def jax_inference(encoder, esm_params, batch):
    esm_params = frozen_dict.freeze({"params": esm_params})
    jax_out = encoder.apply(esm_params, batch)

    return np.array(jax_out)


SEQ_INSULIN = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN"

test_batches = [
    np.array([[0, 5, 4, 3, 4, 6, 7, 2], [0, 9, 6, 8, 5, 4, 8, 2]]),
    np.array([[0, 5, 4, 3, 4, 6, 7, 2], [0, 32, 32, 6, 2, 1, 1, 1]]),
    np.array([esm_tokenizer.protein_tokenizer().encode(SEQ_INSULIN).ids]),
]


@pytest.mark.parametrize("batch", test_batches)
def test_model_match(batch, torch_model, encoder, esm_params):
    torch_out = torch_inference(torch_model, batch)
    jax_out = jax_inference(encoder, esm_params, batch)

    mask = batch[:, :, None] != 1
    mask = np.repeat(mask, repeats=jax_out.shape[-1], axis=-1).reshape(-1)
    assert np.allclose(
        torch_out.reshape(-1)[mask], jax_out.reshape(-1)[mask], atol=ATOL
    )
