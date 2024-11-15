from abc import ABC, abstractmethod
import itertools
import random
from typing import Callable, Iterator, List, Optional, Tuple
from attr import define
import regex as re

import numpy as np
from torch.utils.data import IterableDataset

from lizrd.text.datasets import AbstractDataset
from lizrd.text.packers import AbstractPacker, take_circular
from lizrd.text.tokenizers import AbstractTokenizer
from research.mole.utils.data import LLMMetaExample

EOT_TAG = "EOT"

def encode_with_meta(sentence, tokenizer, spacy_nlp):
    spacy_tokens = spacy_nlp(sentence)

    pretokenized = []
    pretokenized_pos = []
    leading_ws = ""
    for t in spacy_tokens:
        pretokenized.append(leading_ws+t.text)
        leading_ws = t.whitespace_
        pretokenized_pos.append(t.pos_)
    
    ids = []
    poss = []
    for pt, pos in zip(pretokenized, pretokenized_pos):
        n_ids = tokenizer.encode(pt)
        ids.extend(n_ids)
        poss.extend([pos for _ in range(len(n_ids))])
    
    assert len(ids) == len(poss)
    return ids, poss


class GPTMetaPacker(
    AbstractPacker,
):
    def __init__(
        self,
        sequence_length: int,
        dataset_maker: AbstractDataset,
        tokenizer_maker: Callable[[], AbstractTokenizer],
        seed: Optional[int] = None,
    ):
        super().__init__(
            sequence_length,
            dataset_maker,
            tokenizer_maker,
            seed=seed,
        )


    def get_sample(self) -> LLMMetaExample:
        """
        Sample examples from the dataset until we reach the desired sequence length.
        """
        eot_id = self.tokenizer.eot_id
        assert eot_id is not None

        buffer: List[int] = []
        t_metadata_buffer: List[str] = []
        calculate_loss: List[int] = []
        document_lengths: List[int] = []

        while True:
            document = self.dataset.get_document()
            # tokens = self.tokenizer.text_to_ids(document)
            tokens, t_metadata = encode_with_meta(document, self.tokenizer.tokenizer, )
            buffer.extend(tokens + [eot_id])
            t_metadata_buffer.extend(t_metadata + [EOT_TAG])

            document_lengths.append(len(tokens) + 1)
            if (sum(document_lengths) - max(document_lengths)) > self.sequence_length:
                break

        sample_start = self.py_rng.randint(0, len(buffer) - 1)
        sample_end = sample_start + self.sequence_length

        input_ids = list(take_circular(buffer, sample_start, sample_end))
        target_ids = list(take_circular(buffer, sample_start + 1, sample_end + 1))
        calculate_loss = [1] * len(target_ids)

        return LLMMetaExample(input_ids, target_ids, calculate_loss)
