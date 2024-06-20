import torch
from typing import Literal

from lizrd.core.initialization import get_init_weight


class SubtokenEmbedding(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        max_n_bytes: int,
        init_type: Literal["kaiming_uniform", "truncated_normal"],
        init_scale: float,
    ):
        super().__init__()
        self.byte_embedding = torch.nn.Embedding(256, embedding_dim)

        self.positional_embedding = torch.nn.Parameter(
            get_init_weight(
                shape=(embedding_dim, embedding_dim),
                fan_in=1,
                init_type=init_type,
                scale=init_scale,
                dtype=self.byte_embedding.weight.dtype,
            )
        )

        self.max_n_bytes = max_n_bytes

        # default_weight = self.positional_embedding.data
        # self.positional_embedding.weight.data = get_init_weight(
        #     shape=default_weight.shape,
        #     fan_in=1,
        #     init_type=init_type,
        #     scale=init_scale,
        #     dtype=default_weight.dtype,
        # )

    def forward(self, bytes_ids):
        # positional_embeddings = [self.positional_embedding]
        # for _ in range(self.max_n_bytes - 1):
        #     positional_embeddings.append(
        #         self.positional_embedding @ positional_embeddings[-1]
        #     )

        # positional_embeddings = torch.stack(positional_embeddings)
        is_missing_byte = bytes_ids.eq(-1)
        bytes_ids = bytes_ids.clone()
        bytes_ids[is_missing_byte] = 0
        # print(bytes_ids.device)
        # print(self.byte_embedding.weight.device)
        bytes_embeddings = self.byte_embedding(bytes_ids).unsqueeze(-1)
        # positional_embeddings.unsqueeze_(0).unsqueeze_(0)
        # embeddings = torch.matmul(positional_embeddings, bytes_embeddings).squeeze(-1)
        embeddings = bytes_embeddings * ~is_missing_byte.unsqueeze(-1)
        embeddings = embeddings.sum(dim=-2)

        return embeddings


# # %%
# max_n_bytes = 11
# ste = SubtokenEmbedding(10, max_n_bytes, "kaiming_uniform", 0.1)
# batch_size = 7
# seqlen = 13
# ste(
#     torch.randint(-1, 256, (batch_size, seqlen, max_n_bytes)),
# ).shape
# # %%
# torch.matmul(
#     torch.zeros([1, 1, 11, 10, 10]),
#     torch.zeros([7, 13, 11, 10, 1]),
# ).shape
