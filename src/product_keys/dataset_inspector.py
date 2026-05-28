import torch
import os

from torch.utils.data import DataLoader

from src.product_keys.datasets import GlueDataset, glue_2_sentences_tokenize_fn, \
    glue_tokenize_fn, glue_collate_wrapper

def inspect_dataset(dataloader: DataLoader, num_samples=10):
    dataloader_iter = iter(dataloader)
    tokens_sum = 0
    exceed = 0
    max_tokens = 0

    buckets = [0, 0, 0]

    for i in range(num_samples):
        sample = next(dataloader_iter)
        text, label, attention_mask = sample
        tokens_num = torch.sum(attention_mask).item()
        tokens_sum += tokens_num
        max_tokens = max(max_tokens, tokens_num)
        if tokens_num == 1024:
            exceed += 1
        
        if tokens_num < 256:
            buckets[0] += 1
        elif tokens_num < 512:
            buckets[1] += 1
        else:
            buckets[2] += 1
        # print(f"Tokens number: {tokens_num}")

    print(f"Average tokens number: {tokens_sum / num_samples:.2f}")
    print(f"Exceeding samples: {exceed/num_samples * 100:.2f}%")
    print(f"Max tokens number: {max_tokens}")
    print(f"<256: {buckets[0] / num_samples * 100:.2f}%")
    print(f"[256, 512): {buckets[1] / num_samples * 100:.2f}%")
    print(f">512: {buckets[2] / num_samples * 100:.2f}%")

def get_mnli_dataset(seq_len: int, split: str = "validation") -> DataLoader:
    dataset = GlueDataset(
        task_name="mnli",
        sequence_length=seq_len,
        path="data/ft_dataset/mnli",
        split=split,
        seed=123,
        use_new_sampling_method=True,
        shuffle=True,
        world_size_independent=False,
        tokenize_fn=glue_2_sentences_tokenize_fn(seq_len=seq_len)
    )

    return dataset


def get_sst2_dataset(seq_len: int, split: str = "validation") -> DataLoader:
    dataset = GlueDataset(
        task_name="sst2",
        sequence_length=seq_len,
        path="data/ft_dataset/sst2/test",
        split=split,
        seed=123,
        use_new_sampling_method=True,
        shuffle=True,
        world_size_independent=False,
        tokenize_fn=glue_tokenize_fn(seq_len=seq_len)
    )

    return dataset

def get_imdb_dataset(seq_len: int, split: str = "test") -> DataLoader:
    dataset = GlueDataset(
        task_name="imdb",
        sequence_length=seq_len,
        path="data/ft_dataset/imdb/test",
        split=split,
        seed=123,
        use_new_sampling_method=True,
        shuffle=True,
        world_size_independent=False,
        tokenize_fn=glue_tokenize_fn(seq_len=seq_len)
    )

    return dataset

if __name__ == "__main__":
    seq_len = 1024
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    num_samples = 25000
    
    # dataset = get_mnli_dataset(seq_len)
    dataset = get_mnli_dataset(seq_len)

    # dataset = get_imdb_dataset(seq_len)
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=glue_collate_wrapper
    )
    inspect_dataset(dataloader, num_samples=num_samples)
