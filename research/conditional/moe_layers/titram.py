from typing import Literal

import mamba_ssm
import torch

from lizrd.core import nn
from lizrd.core.initialization import get_init_weight
from lizrd.support.logging import make_histogram
from research.conditional.utils.layer_manager import LoggingLayer


class TiTraMamba(LoggingLayer):
    def __init__(
        self,
        dmodel: int,
        init_type: Literal["kaiming_uniform", "truncated_normal"],
        init_scale: float,
    ):
        super().__init__()
        self.dmodel = dmodel
        self.mamba = mamba_ssm.Mamba(d_model=self.dmodel)
        self.regression = nn.Parameter(
            get_init_weight(
                self.dmodel,
                fan_in=self.dmodel,
                init_type=init_type,
                scale=init_scale,
            )
        )
        self.weight = nn.Parameter(
            get_init_weight(
                self.dmodel,
                fan_in=self.dmodel,
                init_type=init_type,
                scale=init_scale,
            )
        )
        self.non_neg = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        Returns: same shape as x
        """
        seq_len = x.shape[1]
        lookback_limit = torch.arange(seq_len, device=x.device)

        mamba_output = self.mamba.forward(x)
        lookback_weight = torch.matmul(mamba_output, self.weight).view(-1, seq_len, 1)
        lookback_regression = torch.add(torch.matmul(mamba_output, self.regression), 1)
        lookback_regression = torch.round(self.non_neg(lookback_regression)).type(
            torch.int64
        )
        self.update_cache_for_logging("regression", lookback_regression)
        self.update_cache_for_logging("weight", lookback_weight)
        lookback_regression = torch.cumsum(lookback_regression, dim=1)
        lookback_regression = torch.clamp(lookback_regression, max=lookback_limit)
        lookback_regression = lookback_regression.view(-1, seq_len, 1).expand(
            -1, -1, self.dmodel
        )
        lookback = torch.gather(mamba_output, 1, lookback_regression)
        return mamba_output + lookback_weight * lookback

    def log_heavy(self):
        return {
            "lookback_regression": make_histogram(
                self.logging_cache["regression"].flatten()
            ),
            "lookback_weight": make_histogram(self.logging_cache["weight"].flatten()),
        }
