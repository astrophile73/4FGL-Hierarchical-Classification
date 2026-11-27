"""PROTOCOL §10: one-shot W3 evaluation (DR4 -> FL16Y). Run ONCE, after results/w3_positives.csv is committed.

Usage:  .venv/Scripts/python -m src.run_w3
Outputs: results/w3_results.json, results/holdout_scores_W3.csv, results/w3_recovery_msp.png
"""
import json
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import N_BOOT, RESULTS, SEED
from .run_holdout import score_window
from .stats import auc, ci, paired_bootstrap, recovery_curve
from .windows import build_w3


def compare(scores, keep, pos, a, b, label):
    """Paired stratified bootstrap of AUC(a) - AUC(b) on the sub-pool `keep` with positives `pos` (both masks)."""
    sub = {k: np.asarray(v)[keep] for k, v in scores.items() if k in (a, b)}
    p = pos[keep]
    aucs, _ = paired_bootstrap(sub, p, N_BOOT, SEED, fracs=(0.10,))
    d = aucs[a] - aucs[b]
    pa, pb = auc(sub[a][p], sub[a][~p]), auc(sub[b][p], sub[b][~p])
    lo, hi = ci(d)
    return {'label': label, 'n_pos': int(p.sum()), 'n_unlabelled': int((~p).sum()),
            f'auc_{a}': pa, f'auc_{b}': pb, 'delta_auc': pa - pb, 'delta_ci95': (lo, hi),
            'p_one_sided_boot': float(np.mean(d <= 0)), 'ci_excludes_zero_positive': lo > 0}


def main():
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()
    w = build_w3()
    cur = pd.read_csv(RESULTS / 'w3_positives.csv')
    assert sorted(cur.dr4_name) == sorted(w.pool_names[w.pos]), 'w3_positives.csv does not match the window'

    scores = score_window(w)
    beta = np.asarray(w.meta['pool_lp_beta'], float)
    assert not np.isnan(beta).any(), 'NaN LP_beta in pool: Shape-ranker NaN rule not pre-registered'
    scores['Shape_LP_beta'] = beta

    unl = ~w.pos
    msp = w.pos_class == 'msp'
    flags = cur.set_index('dr4_name')
    s1 = np.array([bool(flags.s1_rf_targeted.get(n, False)) for n in w.pool_names])
    s2 = np.array([bool(flags.s2_identified.get(n, False)) for n in w.pool_names])
    lowb = np.abs(w.pool_glat) < 10

    res = {'window': 'W3 DR4->FL16Y', 'n_pool': int(len(w.pos)), 'n_pos': int(w.pos.sum()),
           'n_msp': int(msp.sum()), 'n_train': w.meta['n_train'], 'n_train_pos': w.meta['n_train_pos']}
    res['point_auc_msp'] = {k: auc(np.asarray(s)[msp], np.asarray(s)[unl]) for k, s in scores.items()}
    res['point_auc_all'] = {k: auc(np.asarray(s)[w.pos], np.asarray(s)[unl]) for k, s in scores.items()}
    res['H2_primary'] = compare(scores, msp | unl, msp, 'RF', 'Curvature', 'MSP positives: RF - Curvature')
    res['H2b_mechanism'] = compare(scores, msp | unl, msp, 'Shape_LP_beta', 'Curvature',
                                   'MSP positives: Shape(LP_beta) - Curvature')
    res['H1_replication'] = compare(scores, w.pos | unl, w.pos, 'RF', 'Curvature', 'All positives: RF - Curvature')
    m1 = msp & ~s1
    res['S1_excl_RF_targeted'] = compare(scores, m1 | unl, m1, 'RF', 'Curvature', 'MSP excl. TRAPUM: RF - Curvature')
    m2 = msp & s2
    res['S2_identified_only'] = compare(scores, m2 | unl, m2, 'RF', 'Curvature', 'MSP identified only: RF - Curvature')
    res['S3_latitude_msp'] = {lab: {'n_pos': int((msp & m).sum()),
                                    **{k: auc(np.asarray(s)[msp & m], np.asarray(s)[unl & m]) for k, s in scores.items()}}
                              for lab, m in (('abs_b_lt_10', lowb), ('abs_b_ge_10', ~lowb))}
    res['verdict_H2'] = ('CONFIRMED' if res['H2_primary']['ci_excludes_zero_positive']
                         else 'NOT CONFIRMED (CI includes 0)')
    res['config'] = {'seed': SEED, 'n_boot': N_BOOT, 'protocol': 'PROTOCOL.md §10 (tag protocol-v2)'}
    res['runtime_s'] = round(time.time() - t0, 1)
    (RESULTS / 'w3_results.json').write_text(json.dumps(res, indent=2))

    pd.DataFrame({'Source_Name': w.pool_names, 'later_pulsar': w.pos, 'later_class': w.pos_class,
                  **{f'score_{k}': s for k, s in scores.items()}}).to_csv(RESULTS / 'holdout_scores_W3.csv', index=False)

    grid = np.linspace(0.005, 1, 200)
    keep = msp | unl
    fig, ax = plt.subplots(figsize=(4.8, 4))
    ax.plot(grid, grid, color='0.6', lw=1, label='Random')
    for k, c, ls in (('RF', '#1f77b4', '-'), ('Curvature', '#d62728', '-'), ('Shape_LP_beta', '#ff7f0e', '--')):
        ax.plot(grid, recovery_curve(np.asarray(scores[k])[keep], msp[keep], grid), color=c, ls=ls, lw=1.6,
                label=f"{k} (AUC {res['point_auc_msp'][k]:.2f})")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Fraction of unassociated pool inspected',
           ylabel='Fraction of later-identified MSPs recovered',
           title=f"W3: DR4 → FL16Y MSPs (n={int(msp.sum())}, pool={int(keep.sum())})")
    ax.title.set_fontsize(9)
    ax.legend(fontsize=7, loc='lower right')
    fig.tight_layout()
    fig.savefig(RESULTS / 'w3_recovery_msp.png', dpi=200)

    for key in ('H2_primary', 'H2b_mechanism', 'H1_replication', 'S1_excl_RF_targeted', 'S2_identified_only'):
        r = res[key]
        print(f"{key:22s} n_pos={r['n_pos']:3d}  dAUC={r['delta_auc']:+.3f}  CI=[{r['delta_ci95'][0]:+.3f}, "
              f"{r['delta_ci95'][1]:+.3f}]  p1={r['p_one_sided_boot']:.3f}")
    print('VERDICT H2:', res['verdict_H2'])


if __name__ == '__main__':
    main()
