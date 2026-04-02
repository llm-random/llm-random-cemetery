"""Check gradient flow through the PK attention mechanism."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

B, S, D, H = 4, 64, 128, 4
DHEAD = D // H        # 32
DHEAD_HALF = DHEAD // 2  # 16
M = int(math.sqrt(S))  # 8
TOP_K = 8

# Shared projections
q_proj = nn.Linear(D, D, bias=False).to(device)
k_proj = nn.Linear(D, D, bias=False).to(device)
v_proj = nn.Linear(D, D, bias=False).to(device)
o_proj = nn.Linear(D, D, bias=False).to(device)

q_norm1 = nn.RMSNorm(DHEAD_HALF).to(device)
q_norm2 = nn.RMSNorm(DHEAD_HALF).to(device)
k_norm1 = nn.RMSNorm(DHEAD_HALF).to(device)
k_norm2 = nn.RMSNorm(DHEAD_HALF).to(device)
attn_temp = nn.Parameter(torch.full((1, H, 1, 1, 1), 1.0 / math.sqrt(DHEAD), device=device))

x = torch.randn(B, S, D, device=device, requires_grad=True)


def pk_forward(x):
    """PK attention forward with hooks to inspect intermediates."""
    q = q_proj(x).view(B, S, H, DHEAD).transpose(1, 2)
    k = k_proj(x).view(B, S, H, DHEAD).transpose(1, 2)
    v = v_proj(x).view(B, S, H, DHEAD).transpose(1, 2)

    k_grid = k.view(B, H, M, M, DHEAD)
    k1_unnorm = k_grid[..., :DHEAD_HALF].sum(-2)
    k2_unnorm = k_grid[..., DHEAD_HALF:].sum(-3)
    k1 = k_norm1(k1_unnorm)
    k2 = k_norm2(k2_unnorm)
    q1 = q_norm1(q[..., :DHEAD_HALF])
    q2 = q_norm2(q[..., DHEAD_HALF:])
    q_normed = torch.cat([q1, q2], dim=-1)

    s1 = torch.matmul(q1, k1.transpose(-2, -1))
    s2 = torch.matmul(q2, k2.transpose(-2, -1))
    grid_scores = s1.unsqueeze(-1) + s2.unsqueeze(-2)
    flat_scores = grid_scores.flatten(-2, -1)
    _, sel_idx = flat_scores.topk(TOP_K, dim=-1)

    idx1 = sel_idx // M
    idx2 = sel_idx % M

    k1_exp = k1.unsqueeze(2).expand(-1, -1, S, -1, -1)
    k2_exp = k2.unsqueeze(2).expand(-1, -1, S, -1, -1)
    k1_sel = torch.gather(k1_exp, 3, idx1.unsqueeze(-1).expand(-1, -1, -1, -1, DHEAD_HALF))
    k2_sel = torch.gather(k2_exp, 3, idx2.unsqueeze(-1).expand(-1, -1, -1, -1, DHEAD_HALF))
    final_k = torch.cat([k1_sel, k2_sel], dim=-1)

    v_indices = idx1 * M + idx2
    v_exp = v.unsqueeze(2).expand(-1, -1, S, -1, -1)
    final_v = torch.gather(v_exp, 3, v_indices.unsqueeze(-1).expand(-1, -1, -1, -1, DHEAD))

    attn_scores = torch.matmul(q_normed.unsqueeze(-2), final_k.transpose(-2, -1))
    attn_scores = attn_scores * attn_temp
    attn_weights = F.softmax(attn_scores, dim=-1)
    out = torch.matmul(attn_weights, final_v).squeeze(-2)
    out = o_proj(out.transpose(1, 2).contiguous().view(B, S, D))

    return out, {
        "q": q, "k": k, "v": v,
        "k1_unnorm": k1_unnorm, "k2_unnorm": k2_unnorm,
        "k1": k1, "k2": k2,
        "q1": q1, "q2": q2,
        "s1": s1, "s2": s2,
        "flat_scores": flat_scores,
        "final_k": final_k, "final_v": final_v,
        "attn_scores": attn_scores,
        "attn_weights": attn_weights,
    }


out, intermediates = pk_forward(x)
loss = out.sum()
loss.backward()

print("=== Gradient Norms Through PK Attention ===")
print(f"{'Parameter/Tensor':<25} {'Grad Norm':>12} {'Value Norm':>12} {'Ratio':>10}")
print("-" * 62)

for name, param in [
    ("q_proj.weight", q_proj.weight),
    ("k_proj.weight", k_proj.weight),
    ("v_proj.weight", v_proj.weight),
    ("o_proj.weight", o_proj.weight),
    ("attn_temp", attn_temp),
]:
    if param.grad is not None:
        gn = param.grad.norm().item()
        vn = param.data.norm().item()
        print(f"{name:<25} {gn:>12.4f} {vn:>12.4f} {gn/vn:>10.4f}")
    else:
        print(f"{name:<25} {'NO GRAD':>12}")

# Check which intermediates got gradients
print()
print("=== Intermediate Tensor Stats ===")
print(f"{'Tensor':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
print("-" * 68)
for name, tensor in intermediates.items():
    t = tensor.detach().float()
    print(f"{name:<25} {t.mean():>10.4f} {t.std():>10.4f} {t.min():>10.4f} {t.max():>10.4f}")

# Key question: how peaked/uniform are the attention weights?
print()
print("=== Attention Weight Analysis ===")
aw = intermediates["attn_weights"].detach()
entropy = -(aw * (aw + 1e-10).log()).sum(-1).mean()
max_ent = math.log(TOP_K)
print(f"Attention entropy: {entropy:.4f} / {max_ent:.4f} ({entropy/max_ent*100:.0f}% of uniform)")
print(f"Attention weight stats: mean={aw.mean():.4f}, max={aw.max():.4f}, min={aw.min():.4f}")

# Key question: are the attention scores all similar (making softmax uniform)?
print()
print("=== Attention Score Analysis ===")
as_ = intermediates["attn_scores"].detach()
print(f"Score stats: mean={as_.mean():.4f}, std={as_.std():.4f}")
print(f"Score range per query (mean): {(as_.max(-1).values - as_.min(-1).values).mean():.4f}")
print(f"  → small range = uniform softmax = attention can't differentiate positions")

# Check: do the PK selection scores also have small range?
print()
print("=== PK Selection Score Analysis ===")
fs = intermediates["flat_scores"].detach()
top_scores = fs.topk(TOP_K, dim=-1).values
score_range = (top_scores[..., 0] - top_scores[..., -1]).mean()
print(f"Top-{TOP_K} score range (mean): {score_range:.4f}")
print(f"Full score range (mean): {(fs.max(-1).values - fs.min(-1).values).mean():.4f}")

# CRITICAL: What fraction of k_proj gradient comes from the aggregation path?
# The gradient to k_proj flows through:
#   k -> k_grid -> sum -> k1/k2 (aggregated) -> selected keys -> attn_scores -> loss
# The sum(-2) and sum(-3) distribute gradient equally to all positions in a row/col
print()
print("=== Gradient Distribution Analysis ===")
# Check if k_proj gets meaningful gradients
k_grad = k_proj.weight.grad
q_grad = q_proj.weight.grad
v_grad = v_proj.weight.grad
print(f"q_proj grad norm: {q_grad.norm():.4f}")
print(f"k_proj grad norm: {k_grad.norm():.4f}")
print(f"v_proj grad norm: {v_grad.norm():.4f}")
print(f"k/q grad ratio:   {k_grad.norm() / q_grad.norm():.4f}")
print(f"k/v grad ratio:   {k_grad.norm() / v_grad.norm():.4f}")

# Check per-position gradient magnitude in v
# Only TOP_K positions per query get gradients through the gather
print()
x_grad = x.grad
print(f"Input gradient norm: {x_grad.norm():.4f}")
per_pos_grad = x_grad.norm(dim=-1)  # (B, S)
print(f"Per-position grad norm: mean={per_pos_grad.mean():.4f}, std={per_pos_grad.std():.4f}")
print(f"  min={per_pos_grad.min():.4f}, max={per_pos_grad.max():.4f}")
print(f"  → large std means some positions get much more gradient signal than others")
