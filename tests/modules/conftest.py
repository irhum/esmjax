import functools

import flax.linen as nn
import pytest
from esm import pretrained
from esmjax import io
from esmjax.modules import modules

MODEL_NAME = "esm2_t6_8M_UR50D"


@pytest.fixture
def torch_state():
    return io.get_torch_state(MODEL_NAME)


# PyTorch fixtures
@pytest.fixture
def torch_model(torch_state):
    model, _ = pretrained.load_model_and_alphabet_core(MODEL_NAME, torch_state)

    return model


# JAX fixtures
@pytest.fixture
def encoder_layer_fn(torch_state):
    embed_dim = torch_state["cfg"]["model"].encoder_embed_dim
    num_heads = torch_state["cfg"]["model"].encoder_attention_heads
    return functools.partial(modules.EncoderLayer, num_heads, embed_dim, embed_dim * 4)


@pytest.fixture
def encoder(torch_state, encoder_layer_fn):
    embed_dim = torch_state["cfg"]["model"].encoder_embed_dim
    num_layers = torch_state["cfg"]["model"].encoder_layers

    embedding = nn.Embed(33, embed_dim)
    jax_model = modules.ESM2(embedding, encoder_layer_fn, num_layers)
    return jax_model


@pytest.fixture
def esm_params(torch_state):
    esm_params = io.convert_encoder(torch_state["model"], torch_state["cfg"])
    return esm_params
