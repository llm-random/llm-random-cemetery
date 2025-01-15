from collections import OrderedDict
from typing import Literal, Callable, Optional
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from lizrd.core import misc
from lizrd.core.misc import default, Aggregate
from lizrd.core.initialization import get_init_weight, ValidInitType
from lizrd.core.misc import Linear, LoggingLayer


def decode_bias_string(bias):
    assert bias in ["both", "first", "second", "none"]
    if bias == "both":
        bias_first = bias_second = True
    elif bias == "first":
        bias_first = True
        bias_second = False
    elif bias == "second":
        bias_first = False
        bias_second = True
    else:
        bias_first = bias_second = False
    return bias_first, bias_second


def ProjectedFeedForward( #dev
    dmodel,
    dff,
    projected_dmodel,
    projected_dff,
    init_type: ValidInitType,
    init_scale: float,
    bias: Literal["both", "first", "second", "none"] = "both",
):
    """
    P1 = torch.rand(xs, xb)
    W = torch.rand(xb, yb)
    P2 = torch.rand(yb, ys)
    P1@W@P2 = (xs, ys)

    :param _type_ dmodel: _description_ #xb
    :param _type_ dff: _description_ #yb
    :param _type_ projected_dmodel: _description_ #xs
    :param _type_ projected_dff: _description_ #ys
    :param ValidInitType init_type: _description_
    :param float init_scale: _description_
    :param Literal[&quot;both&quot;, &quot;first&quot;, &quot;second&quot;, &quot;none&quot;] bias: _description_, defaults to "both"
    :return _type_: _description_
    """

    bias_first, bias_second = decode_bias_string(bias)
    return nn.Sequential(
        OrderedDict(
            [

                (
                    "logging_ff_pre_relu_p11",
                    Linear(
                        projected_dmodel, #xs
                        dmodel, #xb
                        bias=bias_first,
                        init_type=init_type,
                        init_scale=init_scale,
                    ),
                ),
                (
                    "logging_ff_pre_relu",
                    Linear(
                        dmodel, #xb
                        dff, #yb
                        bias=bias_first,
                        init_type=init_type,
                        init_scale=init_scale,
                    ),
                ),
                (
                    "logging_ff_pre_relu_p12",
                    Linear(
                        dff, #yb
                        projected_dff, #ys
                        bias=bias_first,
                        init_type=init_type,
                        init_scale=init_scale,
                    ),
                ),
                ("relu", nn.ReLU()),
                (
                    "logging_ff_post_relu_p21",
                    Linear(
                        projected_dff, #ys
                        dff, #yb
                        bias=bias_second,
                        init_type=init_type,
                        init_scale=init_scale,
                    ),
                ),
                (
                    "logging_ff_post_relu",
                    Linear(
                        dff, #yb
                        dmodel, #xb
                        bias=bias_second,
                        init_type=init_type,
                        init_scale=init_scale,
                    ),
                ),
                (
                    "logging_ff_post_relu_p22",
                    Linear(
                        dmodel, #xb
                        projected_dmodel, #xs
                        bias=bias_second,
                        init_type=init_type,
                        init_scale=init_scale,
                    ),
                ),
            ]
        )
    )



class Attention(LoggingLayer):
    def __init__(
        self,
        dmodel,
        heads,
        causal,
        init_type: str,
        init_scale: float,
        dhead=None,
        flash=False,
    ):
        super(Attention, self).__init__()
        if dhead is None:
            assert dmodel % heads == 0
            dhead = dmodel // heads

        self.heads = heads
        self.dhead = dhead
        self.causal = causal
        self.flash = flash

        self.input_projection = Linear(
            dmodel,
            3 * heads * dhead,
            bias=False,
            init_type=init_type,
            init_scale=init_scale,
        )
        self.output_projection = Linear(
            heads * dhead,
            dmodel,
            bias=False,
            init_type=init_type,
            init_scale=init_scale,
        )
        self.attention_mechanism = AttentionMechanism(use_flash_attention=flash)

    def forward(self, x):
        projected = self.input_projection(x)

        batch, seq_len = x.shape[:-1]
        projected = projected.view(
            batch, seq_len, self.heads, 3 * self.dhead
        ).transpose(1, 2)
        q, k, v = torch.chunk(projected, chunks=3, dim=-1)

        attention_output = self.attention_mechanism(
            query=q, key=k, value=v, dhead=self.dhead, causal=self.causal
        )

        output = self.output_projection(attention_output.transpose(1, 2).flatten(-2))

        return output