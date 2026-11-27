"""Stage 1 (PROTOCOL §6): training-release CV diagnostic. Touches NO holdout labels.

Usage:  .venv/Scripts/python -m src.run_cv
Output: results/cv_diagnostic.json
"""
import json
import time

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from . import rankers as R
from .catalogs import CURVATURE_FEATURE
from .config import CV_FOLDS, RESULTS, SEED
from .stats import auc
from .windows import BUILDERS


def cv_window(w):
    X, y = w.train_X, w.train_y
    cols = R.feature_sets(X.columns, CURVATURE_FEATURE[w.train_release])
    skf = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)
    out = {k: [] for k in ['Curvature', *R.FITTED]}
    for tr, te in skf.split(X, y):
        yte = y[te].astype(bool)
        c = X[CURVATURE_FEATURE[w.train_release]].values[te]       # monotone in raw significance
        out['Curvature'].append(auc(c[yte], c[~yte]))
        for name, make in R.FITTED.items():
            m = clone(make()).fit(X.iloc[tr][cols[name]], y[tr])
            s = m.predict_proba(X.iloc[te][cols[name]])[:, 1]
            out[name].append(auc(s[yte], s[~yte]))
    return {k: {'mean': float(np.mean(v)), 'std': float(np.std(v)), 'folds': [round(x, 4) for x in v]}
            for k, v in out.items()}


def main():
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()
    res = {}
    for name, build in BUILDERS.items():
        w = build()
        res[name] = {'train_release': w.train_release, **w.meta, 'cv_auc': cv_window(w)}
        print(name, json.dumps(res[name]['cv_auc'], indent=1), flush=True)
    res['runtime_s'] = round(time.time() - t0, 1)
    (RESULTS / 'cv_diagnostic.json').write_text(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
