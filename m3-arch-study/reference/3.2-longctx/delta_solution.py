"""参考答案：delta rule 的递推与块内三角求解。"""

import torch
import torch.nn.functional as F


def _validate(q, k, v, beta, alpha):
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
        raise ValueError("q, k and v must share shape (B,H,T,D)")
    if beta.shape != q.shape[:-1]:
        raise ValueError("beta must have shape (B,H,T)")
    if alpha is not None and alpha.shape != beta.shape:
        raise ValueError("alpha must have the same shape as beta")


def delta_rule_recurrent(q, k, v, beta, alpha=None) -> torch.Tensor:
    """直接实现冻结公式；状态和所有运算提升为 fp32。

    ``S @ (I-beta kkT)`` 改写为 rank-1 更新
    ``S - beta (S k) kT``，数学等价且避免显式构造单位阵。
    """
    _validate(q, k, v, beta, alpha)
    out_dtype = q.dtype
    q, k, v, beta = (z.float() for z in (q, k, v, beta))
    k = F.normalize(k, dim=-1)
    alpha = torch.ones_like(beta) if alpha is None else alpha.float()
    b, h, t, d = q.shape
    state = torch.zeros(b, h, d, d, device=q.device, dtype=torch.float32)
    outputs = []
    for i in range(t):
        ki, vi, qi = k[:, :, i], v[:, :, i], q[:, :, i]
        bi, ai = beta[:, :, i], alpha[:, :, i]
        old_value = torch.einsum("bhvk,bhk->bhv", state, ki)
        state = ai[..., None, None] * state
        correction = bi[..., None] * (vi - ai[..., None] * old_value)
        state = state + correction.unsqueeze(-1) * ki.unsqueeze(-2)
        outputs.append(torch.einsum("bhvk,bhk->bhv", state, qi))
    return torch.stack(outputs, dim=2).to(out_dtype)


def delta_rule_chunked(q, k, v, beta, alpha=None, chunk_size: int = 16) -> torch.Tensor:
    """用一个块内下三角系统同时求出全部 rank-1 修正向量。

    令 ``S_t = alpha_t S_(t-1) + u_t k_t^T``。代回冻结公式后，块内
    ``u`` 的因果依赖组成单位下三角系统。一次 ``solve_triangular`` 得到全部
    ``u``；随后用因果系数矩阵同时产生整个块的输出。块间只携带 D×D 的 S。
    """
    _validate(q, k, v, beta, alpha)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    out_dtype = q.dtype
    q, k, v, beta = (z.float() for z in (q, k, v, beta))
    k = F.normalize(k, dim=-1)
    alpha = torch.ones_like(beta) if alpha is None else alpha.float()
    b, h, total, d = q.shape
    n = b * h
    state = torch.zeros(n, d, d, device=q.device, dtype=torch.float32)
    blocks = []
    for start in range(0, total, chunk_size):
        stop = min(total, start + chunk_size)
        c = stop - start
        qc = q[:, :, start:stop].reshape(n, c, d)
        kc = k[:, :, start:stop].reshape(n, c, d)
        vc = v[:, :, start:stop].reshape(n, c, d)
        bc = beta[:, :, start:stop].reshape(n, c)
        ac = alpha[:, :, start:stop].reshape(n, c)

        # prefix[t+1] / prefix[j+1] is decay from just after j through t.
        prefix = torch.cat([torch.ones(n, 1, device=q.device), ac.cumprod(-1)], -1)
        end_prefix = prefix[:, 1:]
        ratio = end_prefix.unsqueeze(-1) / prefix[:, 1:].unsqueeze(-2)
        causal = torch.ones(c, c, dtype=torch.bool, device=q.device).tril()
        decay = torch.where(causal, ratio, torch.zeros_like(ratio))

        gram = torch.einsum("ntd,nsd->nts", kc, kc)
        lower = bc.unsqueeze(-1) * gram * decay
        system = torch.eye(c, device=q.device).expand(n, c, c) + lower.tril(-1)
        initial_read = torch.einsum("nvd,ntd->ntv", state, kc)
        rhs = bc.unsqueeze(-1) * (
            vc - end_prefix.unsqueeze(-1) * initial_read
        )
        updates = torch.linalg.solve_triangular(system, rhs, upper=False)

        kq = torch.einsum("nsd,ntd->nts", kc, qc)
        weights = decay * kq
        base = end_prefix.unsqueeze(-1) * torch.einsum("nvd,ntd->ntv", state, qc)
        block_out = base + torch.einsum("nts,nsv->ntv", weights, updates)
        blocks.append(block_out.reshape(b, h, c, d))

        tail_decay = prefix[:, -1:].expand(-1, c) / prefix[:, 1:]
        state = prefix[:, -1, None, None] * state + torch.einsum(
            "ns,nsv,nsd->nvd", tail_decay, updates, kc
        )
    return torch.cat(blocks, dim=2).to(out_dtype)
