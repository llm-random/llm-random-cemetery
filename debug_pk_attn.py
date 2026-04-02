"""Quick diagnostic for product key attention across sequence lengths."""
import math
import torch
import torch.nn.functional as F

torch.manual_seed(42)

D = 1024
H = 16
TOP_K = 16
DHEAD = D // H          # 64
DHEAD_HALF = DHEAD // 2  # 32

def evaluate_pk_quality(S, B=2, n_query_samples=50):
    M = int(math.sqrt(S))

    q = torch.randn(B, H, S, DHEAD)
    k = torch.randn(B, H, S, DHEAD)

    q1 = F.rms_norm(q[..., :DHEAD_HALF], (DHEAD_HALF,))
    q2 = F.rms_norm(q[..., DHEAD_HALF:], (DHEAD_HALF,))
    q_normed = torch.cat([q1, q2], dim=-1)

    k_grid = k.view(B, H, M, M, DHEAD)
    k1_unnorm = k_grid[..., :DHEAD_HALF].sum(-2)
    k2_unnorm = k_grid[..., DHEAD_HALF:].sum(-3)
    k1 = F.rms_norm(k1_unnorm, (DHEAD_HALF,))
    k2 = F.rms_norm(k2_unnorm, (DHEAD_HALF,))

    overlaps = []
    corrs = []
    for _ in range(n_query_samples):
        qpos = torch.randint(0, S, (1,)).item()
        b, h = 0, 0

        true_scores = torch.matmul(q_normed[b, h, qpos], k[b, h].T)
        pk_s1 = torch.matmul(q1[b, h, qpos], k1[b, h].T)
        pk_s2 = torch.matmul(q2[b, h, qpos], k2[b, h].T)
        pk_scores = (pk_s1.unsqueeze(-1) + pk_s2.unsqueeze(-2)).flatten()

        true_top = set(true_scores.topk(TOP_K).indices.tolist())
        pk_top = set(pk_scores.topk(TOP_K).indices.tolist())
        overlaps.append(len(true_top & pk_top))
        corrs.append(torch.corrcoef(torch.stack([true_scores, pk_scores]))[0, 1].item())

    avg_overlap = sum(overlaps) / len(overlaps)
    avg_corr = sum(corrs) / len(corrs)
    random_baseline = TOP_K * TOP_K / S

    return M, avg_overlap, avg_corr, random_baseline


print(f"Product Key Attention Quality vs Sequence Length (top_k={TOP_K}, dhead={DHEAD})")
print(f"{'S':>6} {'M':>4} {'1st filter':>12} {'Avg Overlap':>14} {'Random Base':>14} {'Correlation':>14}")
print("-" * 72)

for S in [64, 256, 1024, 4096, 16384]:
    M, overlap, corr, rand_base = evaluate_pk_quality(S)
    first_filter = f"{TOP_K}/{M}"
    print(f"{S:>6} {M:>4} {first_filter:>12} {overlap:>10.1f}/{TOP_K:<2} {rand_base:>10.1f}/{TOP_K:<2} {corr:>12.4f}")
