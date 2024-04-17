from abc import abstractmethod
import random
from typing import Optional

from datasets import load_dataset, load_from_disk
import numpy as np


class AbstractDataset:
    def __init__(self, seed: Optional[int] = None):
        self.set_rng(seed)

    def set_rng(self, seed: Optional[int] = None):
        np_rng = np.random.default_rng(seed)
        py_rng = random.Random(seed)

        self.np_rng = np_rng
        self.py_rng = py_rng

    @abstractmethod
    def get_document(self) -> str:
        raise NotImplementedError()


class WikiBookDataset(AbstractDataset):
    def __init__(
        self,
        seed: Optional[int] = None,
        use_dummy_dataset: bool = False,
        split: str = "train",
    ):
        super().__init__(seed=seed)
        assert split in ["train", "eval"]
        self.split = split

        self.dataset_wiki = load_dataset(
            "wikipedia", f"20220301.{'simple' if use_dummy_dataset else 'en'}"
        )["train"]
        self.dataset_book = (
            load_dataset("bookcorpus")["train"]
            if not use_dummy_dataset
            else self.dataset_wiki
        )

        self.bookcorpus_chance = len(self.dataset_book) / len(self.dataset_wiki)

    def get_document(self) -> str:
        selector = self.py_rng.random()
        if selector < self.bookcorpus_chance:
            return self._get_random_book_example()
        else:
            return self._get_random_wiki_example()

    def _belongs_to_split(self, document_id: int) -> bool:
        eval_percentage = 5

        if self.split == "train":
            return hash(document_id) % 100 >= eval_percentage
        elif self.split == "eval":
            return hash(document_id) % 100 < eval_percentage
        else:
            raise ValueError("split must be either 'train' or 'eval'")

    def _get_random_book_example(self) -> str:
        doc_id = None
        while doc_id is None or not self._belongs_to_split(doc_id):
            doc_id = self.py_rng.randint(0, len(self.dataset_book) - 1)
        document = self.dataset_book[doc_id]
        return document["text"]

    def _get_random_wiki_example(self) -> str:
        doc_id = None
        while doc_id is None or not self._belongs_to_split(doc_id):
            doc_id = self.py_rng.randint(0, len(self.dataset_book) - 1)
        document = self.dataset_wiki[doc_id]
        return document["text"]


class C4Dataset(AbstractDataset):
    total_gpt2_tokens = 173_648_052_806  # number of tokens in the C4 dataset when using GPT2TokenizerFast

    def __init__(
        self,
        seed: Optional[int] = None,
        split: str = "train",
        use_dummy_dataset: bool = False,
        dataset_path: Optional[str] = None,
    ):
        super().__init__(seed=seed)
        assert split in ["train", "validation"]
        if dataset_path is not None:
            # self.dataset = load_from_disk(dataset_path)
            pass
        elif use_dummy_dataset:
            if split != "train":
                raise NameError(
                    "Dummy dataset only supports train split for C4 dataset"
                )
            self.dataset = load_dataset("stas/c4-en-10k", split=split)
        else:
            self.dataset = load_dataset("c4", "en", split=split)

    def get_document(self) -> str:
        # return self.dataset[self.py_rng.randint(0, len(self.dataset) - 1)]["text"]
        # copilot generated example text
        return """The C4 dataset is a large-scale dataset of web pages collected by Google. It is designed to be used for training large language models. The dataset contains over 750GB of text data, which is equivalent to over 40 billion words. The C4 dataset is available in multiple languages, including English, Spanish, French, German, and Chinese. The dataset is preprocessed and tokenized, making it easy to use for training language models. The C4 dataset is commonly used for training large language models, such as GPT-3 and BERT. The dataset is available for download on the Google Research website."""
