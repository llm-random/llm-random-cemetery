import numpy as np
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
import argparse
from tqdm import tqdm
from itertools import product

parser = argparse.ArgumentParser(description="Scaling laws for MoE")
parser.add_argument("--train_data_path", type=str)
parser.add_argument("--test_data_path", type=str)
parser.add_argument("--huber_delta", type=float)

args = parser.parse_args()

# read the data
table = pd.read_csv(args.train_data_path)

test_table = pd.read_csv(args.test_data_path)
N_test = torch.Tensor(test_table["params"].tolist())
D_test = torch.Tensor((test_table["step"] * 256 * 2048).tolist())
L_test = torch.Tensor((test_table["loss_interval/100"]).tolist())

N = torch.Tensor(table["params"].tolist()).requires_grad_(False)
D = torch.Tensor((table["step"] * 256 * 2048).tolist()).requires_grad_(False)
G = torch.Tensor((table["args/granularity"]).tolist()).requires_grad_(False)
L = torch.Tensor(table["loss_interval/100"].tolist()).requires_grad_(False)


# helper funcs
def predict_loss(N, a, alpha, D, b, beta, e):
    return a / N**alpha + b / D**beta + e


def rmse(L, L_pred):
    return torch.sqrt(torch.mean((L - L_pred) ** 2))


# scaling law computation
def compute_scaling_law(
    N,
    a,
    alpha,
    D,
    b,
    beta,
    e,
    L,
    G,
    gamma,
    c,
    huber_delta,
    weight_decay=0,
    optimize_e=False,
):
    """
    params:
    N, a, alpha, D, b, beta, e: initial values for the parameters of the scaling law
    huber_delta: delta parameter for the Huber loss
    weight_decay: weight decay parameter for the L1 regularization
    """
    alpha = torch.Tensor([alpha]).requires_grad_(True)
    a = torch.Tensor([a]).requires_grad_(True)
    beta = torch.Tensor([beta]).requires_grad_(True)
    b = torch.Tensor([b]).requires_grad_(True)
    e = torch.Tensor([e]).requires_grad_(optimize_e)
    gamma = torch.Tensor([gamma]).requires_grad_(True)
    c = torch.Tensor([c]).requires_grad_(True)

    def objective(N, a, alpha, D, b, beta, G, gamma, c, e, L, delta, weight_decay):
        inp = torch.logsumexp(
            torch.stack(
                [
                    torch.logsumexp(
                        torch.stack([a - gamma * torch.log(G), c.repeat(D.shape[0])]), 0
                    )
                    - alpha * torch.log(N),
                    b - beta * torch.log(D),
                    e.repeat(D.shape[0]),
                ]
            ),
            0,
        )
        target = torch.log(L)

        loss = F.huber_loss(inp, target, delta=delta, reduction="sum")
        # Adding L1 regularization
        l1_reg = 0.0
        for param in [a, alpha, b, beta, e]:
            l1_reg += torch.norm(param, 2)
        loss += weight_decay * l1_reg

        return loss

    def closure():
        lbfgs.zero_grad()
        loss = objective(
            N, a, alpha, D, b, beta, G, gamma, c, e, L, huber_delta, weight_decay
        )
        loss.backward()
        return loss

    lbfgs = optim.LBFGS(
        [a, alpha, b, beta, c, gamma, e],  # apparently no e here!!!
        history_size=10000,
        lr=0.1,
        max_iter=100000,
        line_search_fn="strong_wolfe",
    )

    lbfgs.step(closure)

    A = torch.exp(a)
    B = torch.exp(b)
    E = torch.exp(e)
    C = torch.exp(c)

    return A, alpha, B, beta, E, gamma, C


alpha_inits = [0.0, 0.5, 1.0, 1.5, 2.0]
beta_inits = [0.0, 0.5, 1.0, 1.5, 2.0]
gamma_inits = [0.0, 0.5, 1.0, 1.5, 2.0]
a_inits = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
b_inits = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
c_inits = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
e_inits = [-1.0, -0.5, 0.0, 0.5, 1.0]

# create progress bar
total_iterations = (
    len(alpha_inits)
    * len(beta_inits)
    * len(a_inits)
    * len(b_inits)
    * len(c_inits)
    * len(gamma_inits)
    * len(e_inits)
)
pbar = tqdm(total=total_iterations, dynamic_ncols=True)

df = pd.DataFrame(
    columns=[
        "huber_delta",
        "a_init",
        "alpha_init",
        "b_init",
        "beta_init",
        "gamma_init",
        "c_init",
        "e_init",
        "train_score",
        "test_score",
        "A",
        "alpha",
        "B",
        "beta",
        "E",
    ]
)

for alpha_init, beta_init, gamma_init, a_init, b_init, c_init, e_init in product(
    alpha_inits, beta_inits, gamma_inits, a_inits, b_inits, c_inits, e_inits
):
    train_scores = []
    valid_scores = []
    test_scores = []

    A, alpha, B, beta, E, gamma, c = compute_scaling_law(
        N,
        a_init,
        alpha_init,
        D,
        b_init,
        beta_init,
        e_init,
        L,
        G,
        gamma_init,
        c_init,
        huber_delta=args.huber_delta,
        weight_decay=0.0,
        optimize_e=True,
    )
    predicted_loss = predict_loss(N, A, alpha, D, B, beta, E)
    train_scores.append(rmse(L, predicted_loss).item())
    test_scores.append(
        rmse(L_test, predict_loss(N_test, A, alpha, D_test, B, beta, E)).item()
    )
    temp_df = pd.DataFrame(
        [
            {
                "huber_delta": args.huber_delta,
                "a_init": a_init,
                "alpha_init": alpha_init,
                "b_init": b_init,
                "beta_init": beta_init,
                "e_init": e_init,
                "gamma_init": gamma_init,
                "c_init": c_init,
                "train_score": np.mean(train_scores),
                "test_score": np.mean(test_scores),
                "A": A.item(),
                "alpha": alpha.item(),
                "B": B.item(),
                "beta": beta.item(),
                "E": E.item(),
                "gamma": gamma.item(),
                "c": c.item(),
            }
        ]
    )
    # Append the temporary data frame to the main one
    df = pd.concat([df, temp_df], ignore_index=True)
    pbar.update(1)
    df.to_csv(f"scaling_law_results_huber_{args.huber_delta}.csv", index=False)