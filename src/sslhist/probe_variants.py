"""Probe robustness: another linear probe trained on the same saved features.

The conference protocol (``probe.train_probe``: AdamW lr 1e-4, 10 epochs, batch 256)
fixes the number of epochs, so the number of updates grows with the number of labels:
20 updates at 1% of PCam (352 images), 10 at 1% of PANDA (16 images), against
initial weights of order 1/sqrt(d). At low label fractions the probe stays close to
its random initialisation.

``logreg_cv`` is trained to convergence, using only the labelled subset (no validation
labels), on the same standardised features: L2-regularised logistic regression (L-BFGS),
C chosen by stratified k-fold cross-validation on the labelled subset by log-loss
(k = 5, fewer if a class has fewer images), the standard linear evaluation of SSL papers.
(A fixed budget of Adam updates is not a converged probe: on small subsets Adam moves
every weight at the same speed, uninformative ones included, and overfits.)
"""
import warnings
from typing import Callable, Dict

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold


VARIANTS = ["legacy", "logreg_cv"]
CS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]


def standardizer(x: np.ndarray):
    """Fitted on the labelled subset, as in the protocol (std clamped at 1e-6)."""
    x = np.asarray(x, dtype=np.float32)
    mu = x.mean(axis=0, keepdims=True)
    sd = np.maximum(x.std(axis=0, ddof=1, keepdims=True), 1e-6)  # torch.std is unbiased
    return lambda z: np.clip(np.nan_to_num((np.asarray(z, dtype=np.float32) - mu) / sd), -1e4, 1e4)


def fit_logreg_cv(x: np.ndarray, y: np.ndarray, seed: int) -> Dict:
    std = standardizer(x)
    xs, y = std(x).astype(np.float64), np.asarray(y)
    k = int(min(5, np.bincount(y, minlength=2).min()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", FutureWarning)  # attribute/parameter changes across sklearn versions
        if k >= 2:
            clf = LogisticRegressionCV(Cs=CS, cv=StratifiedKFold(k, shuffle=True, random_state=seed),
                                       scoring="neg_log_loss", max_iter=5000, n_jobs=1).fit(xs, y)
            c = float(np.ravel(clf.C_)[0])
        else:  # a class with a single image: no cross-validation possible
            c = 1.0
            clf = LogisticRegression(C=c, max_iter=5000).fit(xs, y)
    w, b = clf.coef_.ravel(), float(clf.intercept_[0])
    return {"logits": lambda z: std(z).astype(np.float64) @ w + b, "C": c, "cv_folds": k}


FITTERS: Dict[str, Callable] = {"logreg_cv": fit_logreg_cv}
