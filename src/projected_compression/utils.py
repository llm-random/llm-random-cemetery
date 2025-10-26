# 1) Using the SVD (orthogonal projectors)
import torch


def svd_op(a):
    A = a
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    r = (S > 1e-12).sum().item()
    Ur = U[:, :r]
    Vr = Vh[:r, :].T

    P1 = Ur @ Ur.T          # m x m
    P2 = Vr @ Vr.T          # n x n
    return P1, P2


# 2) Using the Moore–Penrose pseudoinverse (canonical projectors)
def mpp(a):
    A = a
    Ap = torch.linalg.pinv(A)

    P1 = A @ Ap   # m x m
    P2 = Ap @ A   # n x n
    return P1, P2

def svd_g(a):
    A = a
    U, S, Vh = torch.linalg.svd(A, full_matrices=True)
    r = (S > 1e-12).sum().item()
    # print(r)
    U_perp = U[:, r:]
    V_perp = Vh[r:, :].T
    # U_perp = U
    # V_perp = Vh.T

    L = torch.randn(A.shape[0], U_perp.shape[1])
    K = torch.randn(V_perp.shape[1], A.shape[1])

    P1 = torch.eye(A.shape[0]) + L @ U_perp.T
    P2 = torch.eye(A.shape[1]) + V_perp @ K
    return P1, P2

def smart_projections(t, iy, ix, fun=svd_g):
    al = t[iy]
    ar = t[:, ix]
    # p1l, p2l = fun(al)
    p1r, p2r = fun(ar)
    p1ll, p2ll = fun(p1r[iy]@t)
    err = torch.norm(p1r[iy]@t@p2ll[:, ix] - t[iy][:, ix])
    assert err < 0.01
    return p1r[iy], p2ll[:, ix]
