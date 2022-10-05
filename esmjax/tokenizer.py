from typing import Optional

import tokenizers

# toks from https://github.com/facebookresearch/esm/blob/main/esm/constants.py#L7
# prepend/append toks from https://github.com/facebookresearch/esm/blob/main/esm/data.py#L153
PROTEINSEQ_TOKS = {
    "toks": [
        "L",
        "A",
        "G",
        "V",
        "S",
        "E",
        "R",
        "T",
        "I",
        "D",
        "P",
        "K",
        "Q",
        "N",
        "F",
        "Y",
        "M",
        "H",
        "W",
        "C",
        "X",
        "B",
        "U",
        "Z",
        "O",
        ".",
        "-",
    ],
    "prepend_toks": ["<cls>", "<pad>", "<eos>", "<unk>"],
    "append_toks": ["<null_1>", "<mask>"],
}


def protein_tokenizer(
    pad_to_multiple_of: Optional[int] = None, max_length: Optional[int] = 1024
) -> tokenizers.Tokenizer:
    """Returns the default protein tokenizer.

    Args:
        pad_to_multiple_of (Optional[int]): If specified, pads all sequences to
            nearest (higher) multiple of value. Defaults to None.
        max_length (Optional[int]): Maximum length to truncate to. Defaults to 1024.

    Returns:
        tokenizers.Tokenizer: HuggingFace Tokenizer to tokenize proteins.
    """

    # Just repurpose the WordLevel tokenizer, since the vocabulary is fully known.
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {
                key: idx
                for (idx, key) in enumerate(
                    PROTEINSEQ_TOKS["prepend_toks"]
                    + PROTEINSEQ_TOKS["toks"]
                    + PROTEINSEQ_TOKS["append_toks"]
                )
            },
            unk_token="<unk>",
        )
    )

    # Use regex to strip out the capital letters + <special> tokens.
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Split(
        tokenizers.Regex("[A-Z]|[<][a-z]*[>]"), behavior="removed", invert=True
    )

    # Specify the special tokens; makes decoding + printing neater.
    tokenizer.add_special_tokens(
        PROTEINSEQ_TOKS["prepend_toks"] + PROTEINSEQ_TOKS["append_toks"]
    )

    # Add template, to add <cls> to start and <eos> to end of all sequences.
    tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
        single="<cls> $A <eos>",
        pair=None,
        special_tokens=[("<cls>", 0), ("<eos>", 2)],
    )

    # If padding is requested, enable padding.
    if pad_to_multiple_of:
        tokenizer.enable_padding(
            pad_id=1, pad_token="<pad>", pad_to_multiple_of=pad_to_multiple_of
        )

    # Enable truncation to specified max length.
    tokenizer.enable_truncation(max_length=max_length)

    return tokenizer
