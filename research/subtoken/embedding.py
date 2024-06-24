# %%
import torch
from typing import Literal

# from lizrd.core.initialization import get_init_weight


class SubtokenEmbedding(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        max_n_bytes: int,
        init_type: Literal["kaiming_uniform", "truncated_normal"],
        init_scale: float,
        lowrank_ratio,
        normalization,
    ):
        super().__init__()
        assert normalization in [
            "layernorm",
            "max_n_bytes",
            "actual_n_bytes",
            "none",
            None,
        ]
        if normalization == "none":
            normalization = None

        lowrank_dim = int(lowrank_ratio * embedding_dim)
        self.byte_embedding = torch.nn.Embedding(256, lowrank_dim)
        self.upscaler = torch.nn.Linear(lowrank_dim, embedding_dim)
        # self.positional_embedding = torch.nn.Embedding(
        #     max_n_bytes, (lowrank_dim * lowrank_dim)
        # )
        self.positional_embedding = torch.nn.Parameter(
            torch.randn(
                (max_n_bytes, lowrank_dim, lowrank_dim),
                dtype=self.byte_embedding.weight.dtype,
            )
        )
        self.positional_embedding.requires_grad = True
        # self.positional_embedding = torch.nn.Parameter(

        # )
        self.max_n_bytes = max_n_bytes
        self.lowrank_dim = lowrank_dim

        self.normalization = normalization
        if normalization == "layernorm":
            self.normalizer = torch.nn.LayerNorm(embedding_dim)
        elif normalization == "max_n_bytes":
            self.normalizer = lambda x: x / self.max_n_bytes
        elif normalization == None:
            self.normalizer = torch.nn.Identity()
        elif normalization == "actual_n_bytes":
            self.normalizer = None
        else:
            raise ValueError(f"Unknown normalization type: {normalization}")

        # self.layer = nn.Embedding(max_length, embedding_dim)
        # default_weight = self.layer.weight.data
        # self.layer.weight.data = get_init_weight(
        #     shape=default_weight.shape,
        #     fan_in=1,
        #     init_type=init_type,
        #     scale=init_scale,
        #     dtype=default_weight.dtype,
        # )
        # TODO(jaszczur): add initialization as positional encoding

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
        bytes_embeddings = self.byte_embedding(bytes_ids)
        embeddings = torch.einsum(
            "pij,...pi->...pj", self.positional_embedding, bytes_embeddings
        )
        embeddings = embeddings * ~is_missing_byte.unsqueeze(-1)
        embeddings = embeddings.sum(dim=-2)

        # positions = positions * torch.ones_like(x)
        # embeddings = self.layer(positions)
        embeddings = self.upscaler(embeddings)

        if self.normalization in ["layernorm", "max_n_bytes", None]:
            embeddings = self.normalizer(embeddings)
        elif self.normalization == "actual_n_bytes":
            normalization_value = (~is_missing_byte).sum(-1, keepdim=True)
            embeddings /= normalization_value + (1e-5)
        else:
            raise ValueError(f"Unknown normalization type: {self.normalization}")

        return embeddings


# %%
# max_n_bytes = 11
# ste = SubtokenEmbedding(
#     10,
#     max_n_bytes,
#     "kaiming_uniform",
#     0.1,
#     lowrank_ratio=0.5,
#     normalization="actual_n_bytes",
# )
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
