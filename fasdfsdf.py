from datasets import load_from_disk
from transformers import GPT2TokenizerFast
import numpy as np
import pickle as pkl

path = "/net/tscratch/people/plgkciebiera/datasets/c4/train"
VOCAB_SIZE = 50257
print("Load dataset!!! pls")
dataset = load_from_disk(path)
print("Loaded")
bigram_matrix = np.empty((VOCAB_SIZE, VOCAB_SIZE), dtype=np.int32)

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
print(tokenizer.eos_token_id)
tokenizer.model_max_length = 100_000
sum_ids = 0
for i, document in enumerate(dataset):

    ids = tokenizer.encode(document["text"])
    ids.append(tokenizer.eos_token_id)
    sum_ids += len(ids)
    prev = -1
    for id in ids:
        if prev != -1:
            bigram_matrix[prev, id] += 1
        prev = id

    if i % 100000 == 0:
        print(f"Krok taki: {i} :) ")


fileName = "bigram_matrix.pkl"
fileObject = open(fileName, "wb")

pkl.dump(bigram_matrix, fileObject)
fileObject.close()
