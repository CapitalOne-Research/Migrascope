import math
import numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

from sklearn.ensemble import RandomForestRegressor

@dataclass
class EstimatorConfig:
    gaussian: bool = True
    shapley_exact_cutoff: int = 8          # exact when m <= cutoff; else MC permutations
    shapley_mc_permutations: int = 512
    random_state: int = 42

class InfoEstimators:
    def __init__(self, cfg: EstimatorConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.random_state)

    # Assemble flattened samples: X in R^{N x m}, y in R^{N}
    def build_samples(self, Y_by_qid: Dict[str, np.ndarray], X_by_qid: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        X_list, y_list = [], []
        for qid in Y_by_qid:
            y = np.asarray(Y_by_qid[qid]).reshape(-1)
            X = np.asarray(X_by_qid[qid])
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if len(y) != X.shape[0]:
                continue
            X_list.append(X)
            y_list.append(y)
        if not X_list:
            return np.zeros((0,1)), np.zeros((0,))
        X_all = np.vstack(X_list)
        y_all = np.concatenate(y_list)
        return X_all, y_all

    # Gaussian-residual MI via linear regression variance ratio
    def mutual_info_gaussian(self, y: np.ndarray, X: np.ndarray, S: List[int]) -> float:
        if len(S) == 0:
            return 0.0
        Xs = X[:, S]
        y_c = y - y.mean()
        Xs_c = Xs - Xs.mean(axis=0, keepdims=True)
        reg = LinearRegression().fit(Xs_c, y_c)
        yhat = reg.predict(Xs_c)
        res = y_c - yhat
        var_y = np.var(y_c) + 1e-12
        var_res = np.var(res) + 1e-12
        mi = 0.5 * np.log(var_y / var_res)
        return float(max(0.0, mi))

    def conditional_mi_gaussian(self, y: np.ndarray, X: np.ndarray, S: List[int], given: List[int]) -> float:
        if len(S) == 0:
            return 0.0
        mi_joint = self.mutual_info_gaussian(y, X, list(set(S).union(set(given))))
        mi_given = self.mutual_info_gaussian(y, X, given)
        return float(max(0.0, mi_joint - mi_given))

    # Non-Gaussian: plug-in via regression residual variance (XGB if available, else RF)
    def _fit_regressor(self, X: np.ndarray, y: np.ndarray):
        if HAS_XGB:
            model = XGBRegressor(n_estimators=100, max_depth=4, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=self.cfg.random_state)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=self.cfg.random_state)
        model.fit(X, y)
        return model

    def mutual_info_plugin(self, y: np.ndarray, X: np.ndarray, S: List[int]) -> float:
        if len(S) == 0:
            return 0.0
        Xs = X[:, S]
        y_c = y - y.mean()
        var_y = np.var(y_c) + 1e-12
        model = self._fit_regressor(Xs, y_c)
        yhat = model.predict(Xs)
        res = y_c - yhat
        var_res = np.var(res) + 1e-12
        mi = 0.5 * np.log(var_y / var_res)
        return float(max(0.0, mi))

    def conditional_mi_plugin(self, y: np.ndarray, X: np.ndarray, S: List[int], given: List[int]) -> float:
        if len(S) == 0:
            return 0.0
        mi_joint = self.mutual_info_plugin(y, X, list(set(S).union(set(given))))
        mi_given = self.mutual_info_plugin(y, X, given)
        return float(max(0.0, mi_joint - mi_given))

    def MI(self, y: np.ndarray, X: np.ndarray, S: List[int]) -> float:
        if self.cfg.gaussian:
            return self.mutual_info_gaussian(y, X, S)
        return self.mutual_info_plugin(y, X, S)

    def CMI(self, y: np.ndarray, X: np.ndarray, S: List[int], given: List[int]) -> float:
        if self.cfg.gaussian:
            return self.conditional_mi_gaussian(y, X, S, given)
        return self.conditional_mi_plugin(y, X, S, given)

    # Shapley values over F(S) = I(Y; X_S)
    def shapley(self, y: np.ndarray, X: np.ndarray) -> np.ndarray:
        m = X.shape[1]
        idxs = list(range(m))
        if m <= self.cfg.shapley_exact_cutoff:
            vals = np.zeros(m, dtype=float)
            F_cache = {}
            def F(S):
                key = tuple(sorted(S))
                if key in F_cache:
                    return F_cache[key]
                val = self.MI(y, X, list(S))
                F_cache[key] = val
                return val
            for i in idxs:
                s = 0.0
                others = [j for j in idxs if j != i]
                for k in range(0, m):
                    from itertools import combinations
                    for S in combinations(others, k):
                        S = set(S)
                        s += math.factorial(k)*math.factorial(m-k-1)/math.factorial(m) * (F(S | {i}) - F(S))
                vals[i] = s
            return vals
        else:
            P = self.cfg.shapley_mc_permutations
            vals = np.zeros(m, dtype=float)
            for _ in range(P):
                perm = self.rng.permutation(m).tolist()
                S = set()
                prev = 0.0
                for i in perm:
                    cur = self.MI(y, X, list(S | {i}))
                    vals[i] += (cur - prev)
                    prev = cur
                    S.add(i)
            vals /= float(P)
            return vals
