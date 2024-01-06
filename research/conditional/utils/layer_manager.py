import re
import time
from contextlib import contextmanager
from typing import Union
from plotly import express as px

import torch
import random
from lizrd.core import nn
from lizrd.support.logging import get_current_logger
import math


class ProbabilityScheduler:
    def __init__(
        self,
        warmup_constant_steps: int,
        start_value: float,
        final_value: float,
        final_schedule_step: int,
    ):
        self.warmup_constant_steps = warmup_constant_steps
        self.start_value = start_value
        self.final_schedule_step = final_schedule_step
        self.final_value = final_value

    def get_value(self, step: int):
        if step < self.warmup_constant_steps:
            return self.start_value
        elif step < self.final_schedule_step:
            return self.final_value + 0.5 * (self.start_value - self.final_value) * (
                1
                + math.cos(
                    math.pi
                    * (step - self.warmup_constant_steps)
                    / (self.final_schedule_step - self.warmup_constant_steps)
                )
            )
        else:
            return self.final_value


class LayerManager:
    """
    This class is used to manage the feedforward layers of a model.
    It is used to log everything from weights and activations to gradients and your mum's phone number. [citation needed][this an unfiltered Codex suggestion I had to leave this in im sorry]
    """

    def __init__(
        self,
        model,
        logging_interval_light,
        logging_interval_heavy,
        steps_until_start_temperature_learn,
        chimera_option: str = None,
        first_mode: str = None,
        second_mode: str = None,
        warmup_constant_steps: int = None,
        final_schedule_step: int = None,
        start_prob: float = None,
        end_prob: float = None,
    ):
        self._layers = []
        self._register_layers(model)
        self.logger = get_current_logger()
        self.logging_interval_light = logging_interval_light
        self.logging_interval_heavy = logging_interval_heavy
        self.steps_until_start_temperature_learn = steps_until_start_temperature_learn

        assert first_mode in ["mot", "ec", "switch"] or first_mode == None
        assert second_mode in ["mot", "ec", "switch"] or second_mode == None
        self.chimera_option = chimera_option
        self.first_mode = first_mode
        self.second_mode = second_mode
        if (
            warmup_constant_steps is not None
            and start_prob is not None
            and end_prob is not None
            and final_schedule_step is not None
        ):
            self.modes_probabiltiy_scheduler = ProbabilityScheduler(
                warmup_constant_steps, start_prob, end_prob, final_schedule_step
            )

    def _register_layers(self, model):
        """
        Iterates over all submodules and finds the ones that are of interest.
        Currently, those are only the feedforward and residual blocks.
        During model creation in LLM [llm.py], the feedforward layers are expected to be named "feedforward" and the residual layers "residual" (hardcoded in the repo as of 14.11.2023).
        """
        for name, layer in model.named_modules():
            registered_name = None
            suffix = name.split(".")[-1]

            if suffix in [
                "residual_feedforward",
                "residual_attention",
                "feedforward",
                "expert_gating",
                "router",
            ]:
                block_name = self.extract_block_name(name)
                registered_name = f"{block_name}/{suffix}"
            if registered_name is not None:
                self._layers.append((registered_name, layer))

    def extract_block_name(self, name):
        pattern = r"block_(\d+)"
        match = re.search(pattern, name)
        if match:
            block_name = match.group()
        else:
            raise Exception(
                f"The expected pattern {pattern} was not found in name: {name}. The naming convention of model layers is not as expected. Every TransformerBlock [llm.py] should be named 'block_[block_number]'"
            )
        return block_name

    def prepare_for_logging(self, step):
        if (
            self.logging_interval_light > 0
            and step % self.logging_interval_light == 0
            or self.logging_interval_heavy > 0
            and step % self.logging_interval_heavy == 0
        ):
            for block_name, layer in self._layers:
                if hasattr(layer, "prepare_for_logging"):
                    layer.prepare_for_logging()

    def log(self, step):
        verbosity_levels = []
        if self.logging_interval_heavy > 0 and step % self.logging_interval_heavy == 0:
            verbosity_levels = [2, 1, 0]
        elif (
            self.logging_interval_light > 0 and step % self.logging_interval_light == 0
        ):
            verbosity_levels = [1, 0]

        should_clean_up = len(verbosity_levels) > 0

        for verbosity_level in verbosity_levels:
            for block_name, layer in self._layers:
                if isinstance(layer, LoggingLayer) and not hasattr(
                    layer, "chimera_layer"
                ):
                    info = layer.log(verbosity_level)
                    for name, data in info.items():
                        logging_name = block_name + "/" + name
                        self.logger.report_generic_info(
                            title=logging_name, iteration=step, data=data
                        )
        if should_clean_up:
            for _, layer in self._layers:
                if isinstance(layer, LoggingLayer):
                    layer.clean_up_after_logging()

        if self.modes_probabiltiy_scheduler is not None:
            self.logger.report_scalar(
                title="chimera_prob",
                value=self.modes_probabiltiy_scheduler.get_value(step),
                iteration=step,
            )

    def manage_learnable_temperature(self, step):
        is_learning_temperature = step >= self.steps_until_start_temperature_learn
        for block_name, layer in self._layers:
            for name, param in layer.named_parameters():
                if name in ["temperature_merge", "temperature_emit"]:
                    param.requires_grad = is_learning_temperature

    def change_chimera_mode_step_independent(self, step):
        mode = self._draw_next_mode(step)
        for _, l in self._layers:
            if hasattr(l, "current_mode"):
                l.set_mode(mode)

    def change_chimera_mode_layer_independent(self, step):
        for _, l in self._layers:
            if hasattr(l, "current_mode"):
                mode = self._draw_next_mode(step)
                l.set_mode(mode)

    def _draw_next_mode(self, step):
        probability = self.modes_probabiltiy_scheduler.get_value(step)
        mode = self.first_mode if random.random() < probability else self.second_mode
        return mode

    def change_chimera_mode(self, step):  # , schedule_type_id):
        if self.chimera_option == "step_independent":
            self.change_chimera_mode_step_independent(step)
        elif self.chimera_option == "layer_independent":
            self.change_chimera_mode_layer_independent(step)
        else:
            raise ValueError("Unknown chimera mode")


class LoggingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # info about position in model
        self.layer_type: Union[str, None] = None
        self.block_number: Union[int, None] = None

        # whether to log
        self.logging_switch = False

        # caches for logging and propagation
        self.logging_cache = {}
        self.forward_pass_cache: Union[dict, None] = None

    def clean_up_after_logging(self):
        assert self.logging_switch
        self.logging_switch = False
        self.logging_cache.clear()

    def prepare_for_logging(self):
        self.logging_switch = True

    def update_cache_for_logging(self, key, value):
        if self.logging_switch:
            if isinstance(value, dict):
                if key in self.logging_cache:
                    self.logging_cache[key].update(value)
                else:
                    self.logging_cache[key] = value
            elif isinstance(value, torch.Tensor):
                self.logging_cache[key] = value.clone().detach().cpu()
            elif isinstance(value, float) or isinstance(value, int):
                self.logging_cache[key] = value
            else:
                raise NotImplementedError

    def _combine_to_dict_key(self, key, layer_type, block_number):
        return f"block_{block_number}_{layer_type}_{key}"

    def update_forward_pass_cache(self, key, value):
        combined_key = self._combine_to_dict_key(
            key, self.layer_type, self.block_number
        )
        self.forward_pass_cache[combined_key] = value

    def get_from_forward_pass_cache(self, key, block_number, layer_type):
        combined_key = self._combine_to_dict_key(key, layer_type, block_number)
        return self.forward_pass_cache[combined_key]

    def log(self, verbosity_level):
        if verbosity_level == 0:
            return self.log_time()
        elif verbosity_level == 1:
            return self.log_light()
        elif verbosity_level == 2:
            return self.log_heavy()
        else:
            raise Exception("Invalid verbosity level")

    def log_light(self):
        return {}

    def log_heavy(self):
        return {}

    def log_time(self):
        log = {}
        if "time" in self.logging_cache:
            instr_names = list(self.logging_cache["time"].keys())
            instr_times = list(self.logging_cache["time"].values())
            times_fig = px.bar(x=instr_names, y=instr_times)
            log["time"] = times_fig
        return log


@contextmanager
def measure_time(layer: LoggingLayer, instruction_name: str):
    """
    This simple context manager is used to measure the time of a block of code.
    Args:
        layer: The LoggingLayer object that will be used to cache the time.
        instruction_name: The name of the instruction that is being measured.
    """
    if layer.logging_switch:
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            start = time.time()
    yield
    if layer.logging_switch:
        if torch.cuda.is_available():
            end.record()
            torch.cuda.synchronize()
            layer.update_cache_for_logging(
                "time", {instruction_name: start.elapsed_time(end)}
            )
        else:
            end = time.time()
            layer.update_cache_for_logging("time", {instruction_name: end - start})
