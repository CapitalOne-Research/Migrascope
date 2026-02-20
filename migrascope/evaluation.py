import numpy as np
from .utils import softmax, jsd

def avg_jsd_per_query(fused_scores_by_qid, cp_by_qid, tau=1.0):
    vals = []
    for qid in fused_scores_by_qid:
        s = fused_scores_by_qid[qid]
        cp = cp_by_qid[qid]
        p = softmax(s, tau=tau)
        q = np.asarray(cp, dtype=float)
        q = q / (q.sum() + 1e-12)
        vals.append(jsd(p, q))
    if not vals:
        return float('nan')
    return float(np.mean(vals))

