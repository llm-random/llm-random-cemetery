"""Internals of mfu_calculator.ipynb: parameter count, FLOPs and the MFU / time / steps solver."""

from dataclasses import dataclass

PEAK_BF16 = {"h100": 989e12, "a100": 312e12}  # dense bf16, as in src/core/mfu.py


@dataclass
class Setup:
    n_layers: int
    dmodel: int
    n_experts: int
    vocab: int
    sl: int
    bs: int  # global batch, all GPUs together
    gpu: str  # key of PEAK_BF16
    n_gpus: int
    include_attention_scores: bool  # False reproduces the old 6 * N notebooks

    @property
    def total_params(self):
        dff = self.n_experts * 4 * self.dmodel
        per_layer = 4 * self.dmodel**2 + 2 * self.dmodel * dff
        return self.n_layers * per_layer + 2 * self.vocab * self.dmodel

    @property
    def active_params(self):
        # one expert of size 4 * dmodel; the head counts, the embedding lookup doesn't
        per_layer = 4 * self.dmodel**2 + 8 * self.dmodel**2
        return self.n_layers * per_layer + self.vocab * self.dmodel

    @property
    def score_flops(self):
        if not self.include_attention_scores:
            return 0
        return 12 * self.n_layers * self.sl * self.dmodel

    @property
    def flops_per_token(self):
        return 6 * self.active_params + self.score_flops  # convention of src/core/mfu.py

    @property
    def peak(self):
        return PEAK_BF16[self.gpu] * self.n_gpus

    def describe(self):
        print(
            f"{self.active_params / 1e6:.1f}M active / {self.total_params / 1e6:.1f}M total params | "
            f"{self.flops_per_token / 1e9:.3f} GFLOPs/token "
            f"({self.score_flops / self.flops_per_token:.1%} attention) | "
            f"{self.n_gpus}x {self.gpu} = {self.peak / 1e15:.2f} PFLOP/s peak"
        )

    def solve(self, mfu=None, time_h=None, steps=None):
        """Pass exactly two of mfu (0-1), time_h (hours), steps; prints the third first."""
        given = dict(mfu=mfu, time_h=time_h, steps=steps)
        missing = [name for name, value in given.items() if value is None]
        if len(missing) != 1:
            raise ValueError("pass exactly two of mfu, time_h, steps")

        flops_per_step = self.flops_per_token * self.sl * self.bs
        if mfu is None:
            mfu = flops_per_step * steps / (time_h * 3600 * self.peak)
        elif time_h is None:
            time_h = flops_per_step * steps / (mfu * self.peak) / 3600
        else:
            steps = mfu * self.peak * time_h * 3600 / flops_per_step

        tokens = steps * self.sl * self.bs
        parts = {
            "mfu": f"MFU {mfu:.2%}",
            "time_h": f"time {time_h:.2f} h = {time_h / 24:.2f} d",
            "steps": f"{steps:,.0f} steps",
        }
        inputs = ", ".join(text for name, text in parts.items() if name != missing[0])
        print(
            f"{parts[missing[0]]:<22} <- {inputs} | "
            f"{tokens / 1e9:.2f}B tokens ({tokens / self.active_params:.1f}TTP), "
            f"{tokens / (time_h * 3600):,.0f} tok/s"
        )
