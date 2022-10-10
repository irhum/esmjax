import functools
from typing import Mapping

import numpy as np

import torch


def get_torch_state(model_name: str):
    """Downloads the model weights corresponding to model_name"""

    # Download the core model weights and config.
    fname_weights = f"https://dl.fbaipublicfiles.com/fair-esm/models/{model_name}.pt"
    torch_state = torch.hub.load_state_dict_from_url(fname_weights, map_location="cpu")

    return torch_state


def extract(
    key: str, torch_params: Mapping[str, torch.Tensor], delete: bool = False
) -> np.ndarray:
    """Extract the param accessed by key from the torch_params Mapping.

    Args:
        torch_params (Mapping[str, torch.Tensor]): Mapping with original torch params.
        key (str): Name of the parameter to extract.
        delete (bool, optional): If True, delete the original torch param, to
        avoid duplicate memory use. Defaults to False.

    Returns:
        param: The extracted param as a numpy array.
    """
    param = torch_params[key].numpy()

    # Delete torch param to avoid double use of memory
    # by both torch and numpy version of the param.
    if delete:
        del torch_params[key]

    return param


def convert_self_attn(torch_params, cfg, layer_num: int, delete: bool = False):
    """Returns nested dictionary with params for the self-attention module of
    the specified layer."""

    # Obtain the self-attention layer weight specs.
    embed_dim = cfg["model"].encoder_embed_dim
    num_heads = cfg["model"].encoder_attention_heads
    head_dim = embed_dim // num_heads

    # Specify prefix key and initialize nested dict.
    self_attn_key = f"encoder.sentence_encoder.layers.{layer_num}.self_attn"
    proj_names = ["k_proj", "q_proj", "v_proj", "out_proj"]
    params = {key: {} for key in proj_names}

    # Begin extraction.
    extract_fn = functools.partial(extract, torch_params=torch_params, delete=delete)

    for proj_name in proj_names:
        # The out projection, and query/key/value projections have different param shapes.
        if proj_name == "out_proj":
            weight_shape = (num_heads, head_dim, embed_dim)
            bias_shape = (embed_dim,)

        else:
            weight_shape = (embed_dim, num_heads, head_dim)
            bias_shape = (num_heads, head_dim)

        # Get the key, and extract the weight and bias for the projection.
        proj_key = f"{self_attn_key}.{proj_name}"
        weight = extract_fn(f"{proj_key}.weight").T.reshape(weight_shape)
        bias = extract_fn(f"{proj_key}.bias").reshape(bias_shape)

        params[proj_name]["kernel"] = weight
        params[proj_name]["bias"] = bias

    return params


def convert_encoder_layer(torch_params, cfg, layer_num: int, delete: bool = False):
    """Returns nested dictionary with params for the specified layer."""

    # Specify prefix key and initialize nested dict.
    layer_key = f"encoder.sentence_encoder.layers.{layer_num}"
    sublayer_names = ["fc1", "fc2", "self_attn_layer_norm", "final_layer_norm"]
    params = {key: {} for key in sublayer_names}

    # Begin extraction.
    extract_fn = functools.partial(extract, torch_params=torch_params, delete=delete)

    for sublayer_name in sublayer_names:
        sublayer_key = f"{layer_key}.{sublayer_name}"
        weight = extract_fn(f"{sublayer_key}.weight")
        bias = extract_fn(f"{sublayer_key}.bias")

        # If LayerNorm, the weight is a vector to be renamed `scale`.
        if "norm" in sublayer_name:
            params[sublayer_name]["scale"] = weight

        # Else, its a matrix requiring a transpose.
        else:
            weight = weight.T
            params[sublayer_name]["kernel"] = weight
        params[sublayer_name]["bias"] = bias

    # Extract the params for the self attention layer.
    params["self_attn"] = convert_self_attn(torch_params, cfg, layer_num=layer_num)

    return params


def convert_lm_head(torch_params: Mapping[str, torch.Tensor], delete: bool = False):
    """Returns nested dictionary of params needed for the language model head."""
    params = {"lm_head_fc": {}, "lm_head_layer_norm": {}}

    # Begin extraction.
    extract_fn = functools.partial(extract, torch_params=torch_params, delete=delete)

    params["lm_head_fc"]["kernel"] = extract_fn("encoder.lm_head.dense.weight").T
    params["lm_head_fc"]["bias"] = extract_fn("encoder.lm_head.dense.bias")
    params["lm_head_layer_norm"]["scale"] = extract_fn(
        "encoder.lm_head.layer_norm.weight"
    )
    params["lm_head_layer_norm"]["bias"] = extract_fn("encoder.lm_head.layer_norm.bias")
    params["logit_bias"] = extract_fn("encoder.lm_head.bias")

    return params


def convert_encoder(
    torch_params: Mapping[str, torch.Tensor],
    cfg,
    delete: bool = False,
    lm_head: bool = False,
):
    """Returns nested dictionary with params for the full encoder network.

    Args:
        torch_params (Mapping[str, torch.Tensor]): Mapping containing the torch params.
        cfg (dict): Config dict obtained when loading the torch state.
        delete (bool, optional): If True, will remove loaded torch weight from memory once
            converted, to alleviate memory pressure when loading large models. Defaults to False.
        lm_head (bool, optional): If True, `torch_params` will also have params for the language
            model head. Defaults to False.

    Returns:
        params: Nested dictionary of np.ndarrays containing the converted weights.
    """
    # Specify prefix key and initialize nested dict.
    num_layers = cfg["model"].encoder_layers
    params = {"embedding": {}, "post_norm": {}}
    prefix = "encoder.sentence_encoder"

    # Begin extraction.
    extract_fn = functools.partial(extract, torch_params=torch_params, delete=delete)

    # Extract the initial embedding vectors.
    emb_prefix = f"{prefix}.embed_tokens"
    params["embedding"]["embedding"] = extract_fn(f"{emb_prefix}.weight")

    # Extract the weights of the encoder layers.
    for idx in range(num_layers):
        params[f"{idx}"] = convert_encoder_layer(torch_params, cfg, layer_num=idx)

    # Extract the params for the final layer LayerNorm.
    norm_prefix = f"{prefix}.emb_layer_norm_after"
    params["post_norm"]["scale"] = extract_fn(f"{norm_prefix}.weight")
    params["post_norm"]["bias"] = extract_fn(f"{norm_prefix}.bias")

    if lm_head:
        params.update(convert_lm_head(torch_params, delete=delete))

    return params
