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

def _norm_index(idx, dim_size: int, device: torch.device):
    # None → full range
    if idx is None:
        # return torch.arange(dim_size, device=device)
        return None
    # Slices are device-agnostic
    if isinstance(idx, slice):
        return idx
    # Python lists/tuples/NumPy arrays → LongTensor on device
    if isinstance(idx, (list, tuple)):
        return torch.as_tensor(idx, dtype=torch.long, device=device)
    if torch.is_tensor(idx):
        if idx.dtype == torch.bool:
            # boolean masks must already be correct shape; just move device
            return idx.to(device)
        # integer/long indices
        return idx.to(device=device, dtype=torch.long)
    raise TypeError(f"Unsupported index type: {type(idx)}")

def smart_projections(t, iy, ix, fun=svd_g):
    iy = _norm_index(iy, t.shape[0], t.device)
    ix = _norm_index(ix, t.shape[1], t.device)
    assert not iy is None and ix is None
    
    if iy is None:
        print("iy is None")#dev
        ar = t[:, ix]
        _, p2ll = fun(t)
        _ = None
        err = torch.norm(t@p2ll[:, ix] - t[:, ix])
        assert err < 0.01
        return None, p2ll[:, ix]
    elif ix is None:
        print("ix is None")#dev
        p1r, _ = fun(t)
        _, p2ll = fun(p1r[iy]@t)
        _ = None
        err = torch.norm(p1r[iy]@t - t[iy])
        assert err < 0.01
        return p1r[iy], None
    else:
        ar = t[:, ix]
        print("before p1r") #dev
        p1r, _ = fun(ar)
        print("before p2ll") #dev
        _, p2ll = fun(p1r[iy]@t)
        _ = None
        print("before err") #dev
        err = torch.norm(p1r[iy]@t@p2ll[:, ix] - t[iy][:, ix])
        assert err < 0.01
        return p1r[iy], p2ll[:, ix]
