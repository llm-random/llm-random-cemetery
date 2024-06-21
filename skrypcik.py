layers = ["layer.feedforward.logging_ff", "layer.attention"]
relative_lrs = [0.2, 0.5, 2, 5]

for lr in relative_lrs:
    print(f"[5, 0.2, {lr}], ", end="")
