"""Figure 1 for the research note, built ONLY from saved one-shot outputs (no refitting, no new analysis).

Inputs:  results/holdout_scores_W{1,2,3}.csv, results/holdout_results.json, results/w3_results.json
Outputs: results/figure1_recovery.png, results/figure1_recovery.pdf
Usage:   .venv/Scripts/python -m src.make_figure
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import RESULTS
from .stats import recovery_curve


def main():
    h = json.loads((RESULTS / 'holdout_results.json').read_text())
    w3 = json.loads((RESULTS / 'w3_results.json').read_text())
    delta = {'W1': (h['W1']['primary']['delta_auc'], h['W1']['primary']['delta_auc_ci95']),
             'W2': (h['W2']['primary']['delta_auc'], h['W2']['primary']['delta_auc_ci95']),
             'W3': (w3['H1_replication']['delta_auc'], w3['H1_replication']['delta_ci95'])}
    titles = {'W1': '3FGL → 4FGL-DR3', 'W2': '4FGL-DR3 → DR4', 'W3': '4FGL-DR4 → FL16Y'}
    grid = np.linspace(0.002, 1, 400)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)
    for ax, wn in zip(axes, ('W1', 'W2', 'W3')):
        df = pd.read_csv(RESULTS / f'holdout_scores_{wn}.csv')
        pos = df['later_pulsar'].astype(bool).values
        ax.plot(grid, grid, color='0.65', lw=1, label='Random')
        aucs = {}
        for col, lab, c in (('score_RF', 'Random forest', '#1f77b4'),
                            ('score_Curvature', 'Curvature significance', '#d62728')):
            s = df[col].values
            aucs[lab] = np.mean([np.mean(s[~pos] < x) + 0.5 * np.mean(s[~pos] == x) for x in s[pos]])
            ax.plot(grid, recovery_curve(s, pos, grid), color=c, lw=1.7, label=lab)
        d, (lo, hi) = delta[wn]
        ax.text(0.97, 0.05, f"AUC: RF {aucs['Random forest']:.3f}, curvature {aucs['Curvature significance']:.3f}\n"
                f"ΔAUC = {d:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}]", transform=ax.transAxes,
                ha='right', va='bottom', fontsize=7.5, bbox=dict(fc='white', ec='0.8', lw=0.5))
        ax.set_title(f'{wn}: {titles[wn]}\n{int(pos.sum())} later pulsars / {len(pos)} unassociated', fontsize=9)
        ax.set_xlabel('Fraction of unassociated sources inspected')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    axes[0].set_ylabel('Fraction of later-identified pulsars recovered')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8.5, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ext in ('png', 'pdf'):
        fig.savefig(RESULTS / f'figure1_recovery.{ext}', dpi=300)


if __name__ == '__main__':
    main()
