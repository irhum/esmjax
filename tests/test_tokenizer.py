import pytest
from esm import data
from esmjax import tokenizer as esm_tokenizer


seqs = [
    "MALWMRLLPLLALLALWGPDPAAAFVNQHL",
    "MALWMR  LLPLLALLA LWGPDPAAAF  VNQHL",
    "MALWMR  LLPLLALLA LWGPDPAAAF <mask> VNQHL",
]


def get_original_tokens(alphabet, seq):
    batch_converter = alphabet.get_batch_converter()
    _, _, tokens_orig = batch_converter([(None, seq)])
    return list(tokens_orig[0].numpy())


@pytest.mark.parametrize("seq", seqs)
def test_tokenization(seq):
    alphabet = data.Alphabet.from_architecture("ESM-1b")
    tokens_orig = get_original_tokens(alphabet, seq)

    tokenizer = esm_tokenizer.protein_tokenizer()
    tokens_ours = tokenizer.encode(seq).ids

    assert tokens_orig == tokens_ours
