from transformers import PreTrainedTokenizer, AddedToken, PreTrainedTokenizerBase


class BooleanTokenizer(PreTrainedTokenizer):
    def __init__(self, pad_token="[PAD]",
                 mask_token="[MASK]", unk_token="[UNK]", sep_token="[SEP]",
                 cls_token="[CLS]", **kwargs):
        self.pad_token = AddedToken(pad_token, rstrip=True, lstrip=True)
        self.mask_token = AddedToken(mask_token, rstrip=True, lstrip=True)
        self.unk_token = AddedToken(unk_token, rstrip=True, lstrip=True)
        self.sep_token = AddedToken(sep_token, rstrip=True, lstrip=True)
        self.cls_token = AddedToken(cls_token, rstrip=True, lstrip=True)
        self.vocab = [pad_token, mask_token, unk_token, sep_token, cls_token] + ['-1', '1']
        self.idx2token = {i: token for i, token in enumerate(self.vocab)}
        self.token2idx = {token: i for i, token in enumerate(self.vocab)}

        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            sep_token=sep_token,
            cls_token=cls_token,
            **kwargs,
        )

    def get_vocab(self):
        vocab = dict(self.token2idx, **self.added_tokens_encoder)
        return vocab

    @property
    def pad_token_id(self):
        return self.token2idx[self.pad_token]

    @property
    def mask_token_id(self):
        return self.token2idx[self.mask_token]

    @property
    def unk_token_id(self):
        return self.token2idx[self.unk_token]

    @property
    def sep_token_id(self):
        return self.token2idx[self.sep_token]

    @property
    def cls_token_id(self):
        return self.token2idx[self.cls_token]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, bits, add_special_tokens=True, **kwargs):
        encoded_bits = [self.token2idx[str(bit)] for bit in bits]
        if add_special_tokens:
            encoded_bits = [self.cls_token_id, *encoded_bits, self.sep_token_id]
        else:
            encoded_bits = [*encoded_bits]
        return encoded_bits