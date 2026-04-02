"""Train a tiny PK attention model vs full attention baseline on a simple task."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

# === Tiny config ===
B = 32
S = 64  # seq_len, m=8
D = 128
H = 4
DHEAD = D // H  # 32
DHEAD_HALF = DHEAD // 2  # 16
M = int(math.sqrt(S))  # 8
TOP_K = 8
VOCAB = 512
N_STEPS = 2000
LR = 3e-4
MASK_PCT = 0.2
MASK_ID = VOCAB  # use vocab+1 as mask token
TOTAL_VOCAB = VOCAB + 1


class FullAttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.RMSNorm(D)
        self.norm2 = nn.RMSNorm(D)
        self.q_proj = nn.Linear(D, D, bias=False)
        self.k_proj = nn.Linear(D, D, bias=False)
        self.v_proj = nn.Linear(D, D, bias=False)
        self.o_proj = nn.Linear(D, D, bias=False)
        self.ff1 = nn.Linear(D, D * 2, bias=False)
        self.ff2 = nn.Linear(D * 2, D, bias=False)

    def forward(self, x):
        # Attention
        h = self.norm1(x)
        q = self.q_proj(h).view(B, S, H, DHEAD).transpose(1, 2)
        k = self.k_proj(h).view(B, S, H, DHEAD).transpose(1, 2)
        v = self.v_proj(h).view(B, S, H, DHEAD).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(DHEAD)
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)
        out = self.o_proj(out.transpose(1, 2).contiguous().view(B, S, D))
        x = x + out
        # FF
        h = self.norm2(x)
        x = x + self.ff2(F.silu(self.ff1(h)))
        return x


class PKAttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.RMSNorm(D)
        self.norm2 = nn.RMSNorm(D)
        self.q_proj = nn.Linear(D, D, bias=False)
        self.k_proj = nn.Linear(D, D, bias=False)
        self.v_proj = nn.Linear(D, D, bias=False)
        self.o_proj = nn.Linear(D, D, bias=False)
        self.ff1 = nn.Linear(D, D * 2, bias=False)
        self.ff2 = nn.Linear(D * 2, D, bias=False)

        self.q_norm1 = nn.RMSNorm(DHEAD_HALF)
        self.q_norm2 = nn.RMSNorm(DHEAD_HALF)
        self.k_norm1 = nn.RMSNorm(DHEAD_HALF)
        self.k_norm2 = nn.RMSNorm(DHEAD_HALF)

        self.attn_temp = nn.Parameter(torch.full((1, H, 1, 1, 1), 1.0 / math.sqrt(DHEAD)))

    def forward(self, x):
        h = self.norm1(x)
        q = self.q_proj(h).view(B, S, H, DHEAD).transpose(1, 2)
        k = self.k_proj(h).view(B, S, H, DHEAD).transpose(1, 2)
        v = self.v_proj(h).view(B, S, H, DHEAD).transpose(1, 2)

        # PK decomposition
        k_grid = k.view(B, H, M, M, DHEAD)
        k1 = self.k_norm1(k_grid[..., :DHEAD_HALF].sum(-2))
        k2 = self.k_norm2(k_grid[..., DHEAD_HALF:].sum(-3))
        q1 = self.q_norm1(q[..., :DHEAD_HALF])
        q2 = self.q_norm2(q[..., DHEAD_HALF:])
        q_normed = torch.cat([q1, q2], dim=-1)

        # Scores
        s1 = torch.matmul(q1, k1.transpose(-2, -1))  # (B, H, S, M)
        s2 = torch.matmul(q2, k2.transpose(-2, -1))
        grid_scores = s1.unsqueeze(-1) + s2.unsqueeze(-2)  # (B, H, S, M, M)
        flat_scores = grid_scores.flatten(-2, -1)

        _, sel_idx = flat_scores.topk(TOP_K, dim=-1)
        idx1 = sel_idx // M
        idx2 = sel_idx % M

        # Gather marginal keys
        k1_exp = k1.unsqueeze(2).expand(-1, -1, S, -1, -1)
        k2_exp = k2.unsqueeze(2).expand(-1, -1, S, -1, -1)
        k1_sel = torch.gather(k1_exp, 3, idx1.unsqueeze(-1).expand(-1, -1, -1, -1, DHEAD_HALF))
        k2_sel = torch.gather(k2_exp, 3, idx2.unsqueeze(-1).expand(-1, -1, -1, -1, DHEAD_HALF))
        final_k = torch.cat([k1_sel, k2_sel], dim=-1)

        # Gather values
        v_indices = idx1 * M + idx2
        v_exp = v.unsqueeze(2).expand(-1, -1, S, -1, -1)
        final_v = torch.gather(v_exp, 3, v_indices.unsqueeze(-1).expand(-1, -1, -1, -1, DHEAD))

        # Attention over selected
        attn_scores = torch.matmul(q_normed.unsqueeze(-2), final_k.transpose(-2, -1))
        attn_scores = attn_scores * self.attn_temp
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, final_v).squeeze(-2)
        out = self.o_proj(out.transpose(1, 2).contiguous().view(B, S, D))
        x = x + out

        # FF
        h = self.norm2(x)
        x = x + self.ff2(F.silu(self.ff1(h)))
        return x


class TinyMLM(nn.Module):
    def __init__(self, block_cls, n_layers=4):
        super().__init__()
        self.embed = nn.Embedding(TOTAL_VOCAB, D)
        self.blocks = nn.ModuleList([block_cls() for _ in range(n_layers)])
        self.norm = nn.RMSNorm(D)
        self.head = nn.Linear(D, TOTAL_VOCAB, bias=False)

    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


def make_mlm_batch():
    """Random token sequences with MLM masking."""
    tokens = torch.randint(0, VOCAB, (B, S), device=device)
    labels = tokens.clone()
    mask = torch.bernoulli(torch.full((B, S), MASK_PCT, device=device)).bool()
    labels[~mask] = -100
    tokens[mask] = MASK_ID
    return tokens, labels


def train(name, model):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    losses = []

    for step in range(N_STEPS):
        tokens, labels = make_mlm_batch()
        logits = model(tokens)
        loss = F.cross_entropy(logits.view(-1, TOTAL_VOCAB), labels.view(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())

        if (step + 1) % 200 == 0:
            avg = sum(losses[-200:]) / 200
            print(f"  [{name}] step {step+1:>5}: loss={avg:.3f}")

    return losses


def pk_overlap_on_trained(model):
    """Check PK selection quality after training."""
    model.eval()
    tokens, _ = make_mlm_batch()
    with torch.no_grad():
        x = model.embed(tokens)
        h = model.blocks[0].norm1(x)
        q = model.blocks[0].q_proj(h).view(B, S, H, DHEAD).transpose(1, 2)
        k = model.blocks[0].k_proj(h).view(B, S, H, DHEAD).transpose(1, 2)

        q1 = F.rms_norm(q[..., :DHEAD_HALF], (DHEAD_HALF,))
        q2 = F.rms_norm(q[..., DHEAD_HALF:], (DHEAD_HALF,))
        q_normed = torch.cat([q1, q2], dim=-1)

        k_grid = k.view(B, H, M, M, DHEAD)
        k1 = F.rms_norm(k_grid[..., :DHEAD_HALF].sum(-2), (DHEAD_HALF,))
        k2 = F.rms_norm(k_grid[..., DHEAD_HALF:].sum(-3), (DHEAD_HALF,))

        overlaps = []
        for _ in range(100):
            bi = torch.randint(0, B, (1,)).item()
            hi = torch.randint(0, H, (1,)).item()
            qi = torch.randint(0, S, (1,)).item()

            true_scores = torch.matmul(q_normed[bi, hi, qi], k[bi, hi].T)
            pk_s1 = torch.matmul(q1[bi, hi, qi], k1[bi, hi].T)
            pk_s2 = torch.matmul(q2[bi, hi, qi], k2[bi, hi].T)
            pk_scores = (pk_s1.unsqueeze(-1) + pk_s2.unsqueeze(-2)).flatten()

            true_top = set(true_scores.topk(TOP_K).indices.tolist())
            pk_top = set(pk_scores.topk(TOP_K).indices.tolist())
            overlaps.append(len(true_top & pk_top))

        return sum(overlaps) / len(overlaps)


# === Run ===
print(f"Config: S={S}, M={M}, top_k={TOP_K}, D={D}, H={H}, layers=4, vocab={VOCAB}")
print(f"Random baseline overlap: {TOP_K * TOP_K / S:.1f}/{TOP_K}")
print(f"Random loss: {math.log(VOCAB):.2f}")
print()

print("Training Full Attention:")
full_model = TinyMLM(FullAttentionBlock)
full_losses = train("Full", full_model)

print()
print("Training PK Attention:")
pk_model = TinyMLM(PKAttentionBlock)
pk_losses = train("PK", pk_model)

print()
print("=== Final Results ===")
full_final = sum(full_losses[-200:]) / 200
pk_final = sum(pk_losses[-200:]) / 200
print(f"Full attention final loss: {full_final:.3f}")
print(f"PK attention final loss:   {pk_final:.3f}")
print(f"Random guessing loss:      {math.log(VOCAB):.3f}")

overlap = pk_overlap_on_trained(pk_model)
print(f"\nPK top-{TOP_K} overlap after training: {overlap:.1f}/{TOP_K}")
print(f"(random baseline: {TOP_K * TOP_K / S:.1f}/{TOP_K})")
