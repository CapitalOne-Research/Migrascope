import numpy as np

def softmax(x, tau=1.0):
    x = np.asarray(x, dtype=float) / float(tau)
    x = x - np.max(x)
    ex = np.exp(x)
    s = ex / (np.sum(ex) + 1e-12)
    return s

def zscore(x, eps=1e-8):
    x = np.asarray(x, dtype=float)
    mu = np.mean(x)
    sd = np.std(x)
    return (x - mu) / (sd + eps), mu, sd

def logit(p, eps=1e-12):
    p = np.clip(p, eps, 1.0-eps)
    return np.log(p) - np.log(1.0 - p)

def jsd(p, q, eps=1e-12):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    m = 0.5*(p+q)
    def kl(a,b):
        a = np.clip(a, eps, 1.0)
        b = np.clip(b, eps, 1.0)
        return np.sum(a*np.log(a/b))
    return 0.5*kl(p,m)+0.5*kl(q,m)