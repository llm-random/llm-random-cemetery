from collections import OrderedDict

import torch
from torch import nn

from lizrd.core.llm import Residual
from lizrd.support.test_utils import GeneralTestCase
from research.conditional.moe_layers.cont_moe_designs.learnable_temperature import (
    ContinuousMoEAdaTemp,
)
from research.conditional.moe_layers.cont_moe_designs.learnable_temperature_positive import (
    ContinuousMoEAdaTempPositive,
)
from research.conditional.utils.layer_manager import LayerManager


class TestLearningStartAdatemp(GeneralTestCase):
    layers = []
    for i in range(10):
        mot = ContinuousMoEAdaTemp(dm=8, dff=32, n_experts=4, group_size=2, sparsity_dim=0, temperature=1.0, init_type="kaiming_uniform", init_scale=1.0, expert_size=None,flop_matched=True)
        residual = Residual(
            nn.Sequential(
                OrderedDict(
                    [
                        ("pre_norm", nn.LayerNorm(8)),
                        ("feedforward", mot),
                    ]
                )
            )
        )
        layers.append((f"block_{i}", residual))
    model = nn.Sequential(OrderedDict(layers))

    batch = torch.randn(2, 4, 8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    steps_until_start_temperature_learn = 3

    layer_manager = LayerManager(
        model,
        0,
        0,
        steps_until_start_temperature_learn=steps_until_start_temperature_learn,
    )
    layer_manager.manage_learnable_temperature(0)
    for step in range(6):
        layer_manager.manage_learnable_temperature(step)
        x = model(batch)
        loss = x.sum()
        loss.backward()
        optimizer.step()
        checked = False
        for name, module in model.named_modules():
            if isinstance(module, ContinuousMoEAdaTemp):
                checked = True
                if step < steps_until_start_temperature_learn:
                    assert not module.temperature_emit.requires_grad
                    assert not module.temperature_merge.requires_grad
                    assert module.temperature_emit.data == 1.0
                else:
                    assert module.temperature_emit.requires_grad
                    assert module.temperature_merge.requires_grad
                    assert module.temperature_emit.data != 1.0
        assert checked


class TestLearningStartAdatempPositive(GeneralTestCase):
    layers = []
    for i in range(10):
        mot = ContinuousMoEAdaTempPositive(dm=8, dff=32, n_experts=4, group_size=2, sparsity_dim=0, temperature=1.0, init_type="kaiming_uniform", init_scale=1.0, expert_size=None,flop_matched=True)
        residual = Residual(
            nn.Sequential(
                OrderedDict(
                    [
                        ("pre_norm", nn.LayerNorm(8)),
                        ("feedforward", mot),
                    ]
                )
            )
        )
        layers.append((f"block_{i}", residual))
    model = nn.Sequential(OrderedDict(layers))

    batch = torch.randn(2, 4, 8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    steps_until_start_temperature_learn = 3

    layer_manager = LayerManager(
        model,
        0,
        0,
        steps_until_start_temperature_learn=steps_until_start_temperature_learn,
    )
    layer_manager.manage_learnable_temperature(0)
    for step in range(6):
        layer_manager.manage_learnable_temperature(step)
        x = model(batch)
        loss = x.sum()
        loss.backward()
        optimizer.step()
        checked = False
        for name, module in model.named_modules():
            if isinstance(module, ContinuousMoEAdaTempPositive):
                checked = True
                if step < steps_until_start_temperature_learn:
                    assert not module.temperature_emit.requires_grad
                    assert not module.temperature_merge.requires_grad
                    assert module.temperature_emit.data == 1.0
                else:
                    assert module.temperature_emit.requires_grad
                    assert module.temperature_merge.requires_grad
                    assert module.temperature_emit.data != 1.0
        assert checked
