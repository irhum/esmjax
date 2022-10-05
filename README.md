# esmjax

This repository provides a JAX/Flax reimplementation of the 15B parameter ESM-2 protein language model initially introduced in [Lin et al. (2022)](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1). The original implementation was written in PyTorch, which you can find [here](https://github.com/facebookresearch/esm). 

### Current Features:

* `io.py` - Weight porting of all ESM-2 models (8M to 15B) to JAX from original PyTorch weights.
* `tokenizer.py` - A protein tokenizer matching the output of the original, but re-written with HuggingFace's `tokenizers` library.
* `modules` - Pure Flax definitions of all the network layers needed to create an ESM2 model.
    * The network definition uses sharding constraints (as introduced in [GSPMD](https://arxiv.org/abs/2105.04663), Table 1 ("2D finalized")) on both the weights and activations, enabling scaling to multi-device setups.
* `modules/partitioning.py` - Implements a mix-in class that can add sharding constraints to any pre-existing Flax layer (and enable use of `pjit`).

A sample notebook, running inference for embeddings of the 15B model with model parallelism on a TPUv2-8 can be found in `examples/inference_15B.ipynb`

#### Note: numerical precision
* bfloat16 matmul precision: Work to validate the model perplexity on TPUs (and identify potential degradation) is WIP. Detailed results + plots coming soon and will be updated here.

### Remarks
This repository exists independently of that of the original authors; I just found the model fascinating and wanted to understand it better. I figured it may be of interest to others too!

Access to TPUs was provided through the [TPU Research Cloud](https://sites.research.google/trc/about/).