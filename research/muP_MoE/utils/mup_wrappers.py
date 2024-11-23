import torch
import torch.nn as nn
import torch.nn.functional as F

from lizrd.core.misc import Linear, LoggingLayer
from lizrd.core.llm import Attention

# class AttentionMuP(Attention):
#     def __init__(
#         self,
#         dmodel,
#         heads,
#         causal,
#         init_type: str,
#         init_scale: float,
#         dhead=None,
#         flash=False,
#     ):
#         super(Attention, self).__init__()
        
# class Attention(LoggingLayer):
#     def __init__(
#         self,
#         dmodel,
#         heads,
#         causal,
#         init_type: str,
#         init_scale: float,
#         dhead=None,
#         flash=False,
#     ):
#         super(Attention, self).__init__()
#         if dhead is None:
#             assert dmodel % heads == 0
#             dhead = dmodel // heads

#         self.heads = heads
#         self.dhead = dhead
#         self.causal = causal
#         self.flash = flash

#         self.input_projection = Linear(
#             dmodel,
#             3 * heads * dhead,
#             bias=False,
#             init_type=init_type,
#             init_scale=init_scale,
#         )
#         self.output_projection = Linear(
#             heads * dhead,
#             dmodel,
#             bias=False,
#             init_type=init_type,
#             init_scale=init_scale,
#         )
#         self.attention_mechanism = AttentionMechanism(use_flash_attention=flash)