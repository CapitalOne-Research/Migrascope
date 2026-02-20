import numpy as np
from .utils import softmax, zscore, logit

def standardize_scores(S: np.ndarray) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    out = np.zeros_like(S)
    for j in range(S.shape[1]):
        out[:,j],_,_ = zscore(S[:,j])
    return out

def fusion_linear_z(S: np.ndarray, w: np.ndarray) -> np.ndarray:
    Z = standardize_scores(S)
    w = np.asarray(w, dtype=float).reshape(-1)
    w = w / (w.sum() + 1e-12)
    return Z @ w

def fusion_prob_logop(S: np.ndarray, w: np.ndarray, tau: float=1.0) -> np.ndarray:
    P = []
    for j in range(S.shape[1]):
        P.append(softmax(S[:,j], tau=tau))
    P = np.stack(P, axis=1)
    w = w / (w.sum() + 1e-12)
    logp = (np.log(P + 1e-12) * w.reshape(1,-1)).sum(axis=1)
    return logp

def fusion_logit_pool(P_hat: np.ndarray, w: np.ndarray) -> np.ndarray:
    w = w / (w.sum() + 1e-12)
    logits = (logit(P_hat) * w.reshape(1,-1)).sum(axis=1)
    return logits

def fusion_noisy_or(P_hat: np.ndarray, w: np.ndarray) -> np.ndarray:
    w = w / (w.sum() + 1e-12)
    prod = np.ones(P_hat.shape[0], dtype=float)
    for j in range(P_hat.shape[1]):
        prod *= np.power(1.0 - np.clip(P_hat[:,j], 0.0, 1.0), w[j])
    return 1.0 - prod

def ranks_from_scores(S: np.ndarray) -> np.ndarray:
    order = np.argsort(-S, axis=0)
    ranks = np.zeros_like(order, dtype=float)
    for j in range(S.shape[1]):
        r = np.empty(S.shape[0], dtype=float)
        r[order[:,j]] = np.arange(1, S.shape[0]+1, dtype=float)
        ranks[:,j] = r
    return ranks

def rrf_from_ranks(ranks: np.ndarray, k: float=60.0) -> np.ndarray:
    return np.sum(1.0 / (k + ranks), axis=1)

def fusion_rrf(S: np.ndarray, k: float=60.0) -> np.ndarray:
    ranks = ranks_from_scores(S)
    return rrf_from_ranks(ranks, k=k)

def fusion_borda(S: np.ndarray, w: np.ndarray) -> np.ndarray:
    ranks = ranks_from_scores(S)
    K = S.shape[0]
    score = (K + 1 - ranks)
    w = w / (w.sum() + 1e-12)
    return (score * w.reshape(1,-1)).sum(axis=1)

def robust_rank_aggregation(S: np.ndarray) -> np.ndarray:
    ranks = ranks_from_scores(S)
    K, m = ranks.shape
    p = ranks / (K + 1.0)
    stat = -np.sum(np.log(np.clip(p, 1e-12, 1.0)), axis=1)
    return stat

def rank_centrality(S: np.ndarray, iters: int=100, lr: float=0.85) -> np.ndarray:
    Z = standardize_scores(S)
    K = Z.shape[0]
    M = np.zeros((K, K), dtype=float)
    for j in range(Z.shape[1]):
        diff = Z[:,j].reshape(-1,1) - Z[:,j].reshape(1,-1)
        P = 1.0 / (1.0 + np.exp(-diff))
        M += P
    np.fill_diagonal(M, 0.0)
    row_sum = M.sum(axis=1, keepdims=True) + 1e-12
    Pmat = M / row_sum
    pi = np.ones(K, dtype=float) / K
    for _ in range(iters):
        pi = lr * (Pmat.T @ pi) + (1.0 - lr) * (np.ones(K)/K)
    return pi
