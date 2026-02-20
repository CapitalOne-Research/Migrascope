import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from .estimators import EstimatorConfig, InfoEstimators

@dataclass
class AttributionResult:
    F_all: float
    marginals: List[float]
    shapley: List[float]
    interaction_matrix: List[List[float]]  # II(Y; X_i; X_j)
    total_mi: List[float]

class AttributionEngine:
    def __init__(self, cfg: EstimatorConfig):
        self.cfg = cfg
        self.est = InfoEstimators(cfg)

    def build_XY(self, ps_gt: Dict[str, Tuple[List[str], List[float]]], retrievers: Dict[str, Dict[str, Tuple[List[str], List[float]]]]):
        Y_by_qid = {}
        X_by_qid = {}
        retr_names = list(retrievers.keys())
        for qid, (chunks, cp) in ps_gt.items():
            Y_by_qid[qid] = np.asarray(cp, dtype=float)
            cols = []
            valid = True
            for r in retr_names:
                r_chunks, r_scores = retrievers[r].get(qid, ([], []))
                if r_chunks != chunks or len(r_scores) != len(cp):
                    valid = False
                    break
                cols.append(np.asarray(r_scores, dtype=float).reshape(-1,1))
            if not valid or not cols:
                continue
            X_by_qid[qid] = np.hstack(cols)
        return Y_by_qid, X_by_qid, retr_names

    def compute(self, ps_gt, retrievers) -> AttributionResult:
        Y_by_qid, X_by_qid, _ = self.build_XY(ps_gt, retrievers)
        X, y = self.est.build_samples(Y_by_qid, X_by_qid)
        m = X.shape[1] if X.size else 0
        if m == 0:
            return AttributionResult(0.0, [], [], [], [])
        all_set = list(range(m))
        F_all = self.est.MI(y, X, all_set)
        marginals = []
        for i in range(m):
            others = [j for j in all_set if j != i]
            Ci = self.est.CMI(y, X, [i], others)
            marginals.append(Ci)
        shap = self.est.shapley(y, X)

        total_mi = []
        for i in range(m):
            total_mi.append(float(self.est.MI(y, X, [i])))
        
        II = np.zeros((m,m), dtype=float)
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                I_y_xi = self.est.MI(y, X, [i])
                I_y_xi_given_xj = self.est.CMI(y, X, [i], [j])
                II[i,j] = I_y_xi - I_y_xi_given_xj
        return AttributionResult(
            F_all=float(F_all), 
            marginals=[float(x) for x in marginals], 
            shapley=[float(x) for x in shap], 
            interaction_matrix=II.tolist(),
            total_mi=total_mi,
        )
