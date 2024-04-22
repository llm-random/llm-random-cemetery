import torch
import torch.nn as nn

from lizrd.core.misc import Linear


class TokenReductionLayer(nn.Module):
    """
    This function randomly selects a `result_seq_len` subset of tokens from the input
    """

    def __init__(
        self, result_seq_len, dm=None, init_type="kaiming_uniform", init_scale=1.0
    ):
        super(TokenReductionLayer, self).__init__()
        self.result_seq_len = result_seq_len

        self.transform_to_merge = Linear(
            dm, dm, init_type=init_type, init_scale=init_scale, bias=False
        )

    def _random_indeces(self, batch_size, seq_len):

        random_perms = [torch.randperm(seq_len) for _ in range(batch_size)]

        pairs = [
            (
                torch.sort(permutation[: self.result_seq_len])[0],
                permutation[self.result_seq_len :],
            )
            for permutation in random_perms
        ]
        indices_to_keep, indices_to_reduct = zip(*pairs)

        indices_to_keep, indices_to_reduct = torch.stack(indices_to_keep), torch.stack(
            indices_to_reduct
        )

        return indices_to_keep, indices_to_reduct

    def _random_indeces2(self, batch_size, seq_len):

        random_perms = [torch.randperm(seq_len-1) for _ in range(batch_size)]

        pairs = [
            (
                torch.cat((torch.sort(permutation[: self.result_seq_len-1])[0], torch.tensor([seq_len-1]))),
                permutation[self.result_seq_len-1 :],
            )
            for permutation in random_perms
        ]

        for i, (indices_to_keep, indices_to_reduct) in enumerate(pairs):
            indices_to_keep += i * seq_len
            indices_to_reduct += i * seq_len

        indices_to_keep, indices_to_reduct = zip(*pairs)

        indices_to_keep, indices_to_reduct = torch.stack(indices_to_keep), torch.stack(
            indices_to_reduct
        )
        indices_to_keep = torch.flatten(indices_to_keep)
        indices_to_reduct = torch.flatten(indices_to_reduct)
        return indices_to_keep, indices_to_reduct

    def _batched_index_select(self, input, dim, index):
        """
        origin: https://discuss.pytorch.org/t/batched-index-select/9115/8
        input: B x * x ... x *
        dim: 0 < scalar
        index: B x M
        """
        views = [input.shape[0]] + [
            1 if i != dim else -1 for i in range(1, len(input.shape))
        ]
        expanse = list(input.shape)
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        assert self.result_seq_len <= seq_len
        x = x.view(-1, x.shape[-1])

        indices_to_keep, indices_to_reduce = self._random_indeces2(batch_size, seq_len)
        indices_to_keep, indices_to_reduce = indices_to_keep.to(
            x.device
        ), indices_to_reduce.to(x.device)
        self.indices_to_reduce = indices_to_reduce
        self.indices_to_keep = indices_to_keep

        # kept_tokens = self._batched_index_select(x, 1, indices_to_keep)
        # reduced_tokens = self._batched_index_select(x, 1, indices_to_reduce)
        reduced_tokens = torch.index_select(x, 0, indices_to_reduce)
        transformed_reduced_tokens = self.transform_to_merge(reduced_tokens)

        x.index_add_(0, indices_to_reduce+1, transformed_reduced_tokens)
        kept_tokens = torch.index_select(x, 0, indices_to_keep)

        return kept_tokens.view(batch_size, self.result_seq_len, -1)


class TokenReductionLLM(nn.Module):

    def __init__(self, embedding_layer, encoder_tower, head, reduced_number_of_tokens):
        super(TokenReductionLLM, self).__init__()
        self.embedding_layer = embedding_layer
        self.encoder = encoder_tower
        self.head = head
        self.reduced_number_of_tokens = reduced_number_of_tokens

    def forward(self, x):
        if self.training:
            x = self.embedding_layer(x)
        else:
            x = self.embedding_layer.normal(x)

        x = self.encoder(x)
        x = self.head(x)
        return x

    def get_chosen_indices(self):
        return self.embedding_layer.token_reduction.indices_to_keep
