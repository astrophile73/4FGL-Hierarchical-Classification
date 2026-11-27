"""Stage 2 (PROTOCOL §7): the one-shot temporal-holdout evaluation.

Usage:  .venv/Scripts/python -m src.run_holdout
Outputs: results/holdout_results.json, results/holdout_scores_<W>.csv, results/recovery_curves.png
Run ONCE after tests and the CV diagnostic pass. Any re-run after a code change is a deviation (PROTOCOL §9).
"""
import json
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import rankers as R
from .catalogs import CURVATURE_FEATURE
from .config import N_BOOT, RECALL_FRACS, RESULTS, SEED
from .stats import auc, ci, paired_bootstrap, recall_at_frac, recovery_curve
from .windows import BUILDERS


def score_window(w):
    cols = R.feature_sets(w.train_X.columns, CURVATURE_FEATURE[w.train_release])
    scores = {'Curvature': w.pool_curv.astype(float)}
    for name, make in R.FITTED.items():
        m = make().fit(w.train_X[cols[name]], w.train_y)
        scores[name] = m.predict_proba(w.pool_X[cols[name]])[:, 1]
    return scores


def evaluate(w, scores):
    pos = w.pos
    point = {k: {'auc': auc(s[pos], s[~pos]),
                 **{f'recall@{int(f * 100)}%': recall_at_frac(s, pos, f) for f in RECALL_FRACS}}
             for k, s in scores.items()}
    point['Random'] = {'auc': 0.5, **{f'recall@{int(f * 100)}%': f for f in RECALL_FRACS}}
    aucs, recs = paired_bootstrap(scores, pos, N_BOOT, SEED, RECALL_FRACS)
    for k in scores:
        point[k]['auc_ci95'] = ci(aucs[k])
        for f in RECALL_FRACS:
            point[k][f'recall@{int(f * 100)}%_ci95'] = ci(recs[(k, f)])
    d = aucs[R.PRIMARY_ML] - aucs[R.PRIMARY_BASELINE]
    primary = {'delta_auc': point[R.PRIMARY_ML]['auc'] - point[R.PRIMARY_BASELINE]['auc'],
               'delta_auc_ci95': ci(d), 'p_one_sided_boot': float(np.mean(d <= 0))}
    secondary_deltas = {f'{k}_minus_Curvature': {'delta_auc': point[k]['auc'] - point['Curvature']['auc'],
                                                  'ci95': ci(aucs[k] - aucs['Curvature'])}
                        for k in scores if k not in ('Curvature', R.PRIMARY_ML)}
    # Exploratory strata (PROTOCOL §7; no multiplicity correction)
    expl = {}
    lowb = np.abs(w.pool_glat) < 10
    for label, mask in [('abs_b_lt_10', lowb), ('abs_b_ge_10', ~lowb)]:
        p = pos & mask
        if p.sum() >= 3:
            expl[label] = {'n_pos': int(p.sum()), 'n_pool': int(mask.sum()),
                           **{k: auc(s[p], s[mask & ~pos]) for k, s in scores.items()}}
    for cl in ('msp', 'psr'):
        p = w.pos_class == cl
        if p.sum() >= 3:
            expl[f'pos_{cl}_only'] = {'n_pos': int(p.sum()), **{k: auc(s[p], s[~pos]) for k, s in scores.items()}}
    return {'n_pool': int(len(pos)), 'n_pos': int(pos.sum()), 'n_train': w.meta['n_train'],
            'n_train_pos': w.meta['n_train_pos'], 'point': point, 'primary': primary,
            'secondary_deltas': secondary_deltas, 'exploratory': expl}


def verdict(res):
    sig = {w: res[w]['primary']['delta_auc_ci95'][0] > 0 for w in ('W1', 'W2')}
    neg = {w: res[w]['primary']['delta_auc_ci95'][1] < 0 for w in ('W1', 'W2')}
    if all(sig.values()):
        return 'SUPPORTED'
    if all(neg.values()):
        return 'REVERSED (curvature better)'
    if sig['W1']:
        return 'PARTIALLY SUPPORTED (W1 only)'
    if sig['W2']:
        return 'INCONCLUSIVE (W2 only significant; outcome not pre-registered, report as such)'
    return 'NOT SUPPORTED (curvature suffices)'


def plot(curves, res, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    style = {'RF': ('#1f77b4', '-'), 'Curvature': ('#d62728', '-'), 'LogReg_curv_var': ('#2ca02c', '--'),
             'RF_noPos': ('#9467bd', ':')}
    for ax, wn in zip(axes, ('W1', 'W2')):
        grid, cv = curves[wn]
        ax.plot(grid, grid, color='0.6', lw=1, label='Random')
        for k, (c, ls) in style.items():
            ax.plot(grid, cv[k], color=c, ls=ls, lw=1.6,
                    label=f"{k} (AUC {res[wn]['point'][k]['auc']:.2f})")
        r = res[wn]
        ax.set_title(f"{wn}: {r['train']} → {r['later']}  (n_pos={r['n_pos']}, pool={r['n_pool']})", fontsize=9)
        ax.set_xlabel('Fraction of unassociated pool inspected')
        ax.legend(fontsize=7, loc='lower right')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    axes[0].set_ylabel('Fraction of later-identified pulsars recovered')
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def main():
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()
    res, curves = {}, {}
    grid = np.linspace(0.005, 1, 200)
    for name, build in BUILDERS.items():
        w = build()
        scores = score_window(w)
        res[name] = {'train': w.train_release, 'later': w.later_release, **evaluate(w, scores)}
        curves[name] = (grid, {k: recovery_curve(s, w.pos, grid) for k, s in scores.items()})
        pd.DataFrame({'Source_Name': w.pool_names, 'later_pulsar': w.pos, 'later_class': w.pos_class,
                      **{f'score_{k}': s for k, s in scores.items()}}).to_csv(
            RESULTS / f'holdout_scores_{name}.csv', index=False)
        print(name, json.dumps(res[name]['primary']), flush=True)
    res['verdict'] = verdict(res)
    res['config'] = {'seed': SEED, 'n_boot': N_BOOT, 'protocol': 'PROTOCOL.md v1 (tag protocol-v1)'}
    res['runtime_s'] = round(time.time() - t0, 1)
    (RESULTS / 'holdout_results.json').write_text(json.dumps(res, indent=2))
    plot(curves, res, RESULTS / 'recovery_curves.png')
    print('VERDICT:', res['verdict'])


if __name__ == '__main__':
    main()
