"""Compare attention weight distribution with different temperatures."""
import math
import torch
import torch.nn.functional as F

torch.manual_seed(42)

# Simulate PK selection scores (top-K from a grid)
# These are sums of two sub-scores from normalized vectors
DHEAD = 64
DHEAD_HALF = 32
TOP_K = 16
S = 256
M = 16

q1 = F.rms_norm(torch.randn(1, 1, 1, DHEAD_HALF), (DHEAD_HALF,))
q2 = F.rms_norm(torch.randn(1, 1, 1, DHEAD_HALF), (DHEAD_HALF,))
k1 = F.rms_norm(torch.randn(1, 1, M, DHEAD_HALF), (DHEAD_HALF,))
k2 = F.rms_norm(torch.randn(1, 1, M, DHEAD_HALF), (DHEAD_HALF,))

s1 = torch.matmul(q1, k1.transpose(-2, -1)).squeeze()  # (M,)
s2 = torch.matmul(q2, k2.transpose(-2, -1)).squeeze()  # (M,)
grid = (s1.unsqueeze(-1) + s2.unsqueeze(-2)).flatten()  # (S,)
top_scores = grid.topk(TOP_K).values

old_temp = 1.0 / math.sqrt(DHEAD)
new_temp = 1.0

for name, temp in [("old (1/sqrt(d)=0.125)", old_temp), ("new (1.0)", new_temp)]:
    scaled = top_scores * temp
    weights = F.softmax(scaled, dim=-1)
    entropy = -(weights * weights.log()).sum()
    max_ent = math.log(TOP_K)
    print(f"temp={name}")
    print(f"  score range: {scaled.max() - scaled.min():.3f}")
    print(f"  entropy: {entropy:.3f}/{max_ent:.3f} ({entropy/max_ent*100:.0f}% uniform)")
    print(f"  max weight: {weights.max():.4f}, min weight: {weights.min():.4f}")
    print(f"  top-1 / top-16 weight ratio: {weights[0]/weights[-1]:.1f}x")

    # Check gradient magnitude through softmax
    scores_grad = top_scores.clone().requires_grad_(True)
    w = F.softmax(scores_grad * temp, dim=-1)
    loss = (w * torch.randn_like(w)).sum()
    loss.backward()
    print(f"  grad norm through softmax: {scores_grad.grad.norm():.4f}")
    print()
