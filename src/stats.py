"""Ranking metrics and the paired stratified bootstrap (PROTOCOL.md §7)."""
import numpy as np


def auc(pos_scores, neg_scores):
    """Mann-Whitney AUC, ties counted as 1/2. Equals sklearn roc_auc_score."""
    neg = np.sort(np.asarray(neg_scores, float))
    pos = np.asarray(pos_scores, float)
    below = np.searchsorted(neg, pos, side='left')
    at_or_below = np.searchsorted(neg, pos, side='right')
    return float((below + 0.5 * (at_or_below - below)).sum() / (len(pos) * len(neg)))


def recall_at_frac(scores, pos, frac):
    """Expected recall in the top ceil(frac*N) of the pool, with random tie-breaking at the cut."""
    scores = np.asarray(scores, float)
    k = int(np.ceil(frac * len(scores)))
    thr = np.sort(scores)[::-1][k - 1]
    above = scores > thr
    tied = scores == thr
    slots = k - above.sum()                       # how many tied items make the cut
    p_tied_in = slots / tied.sum()
    hits = (pos & above).sum() + p_tied_in * (pos & tied).sum()
    return float(hits / pos.sum())


def recovery_curve(scores, pos, grid):
    """Fraction of positives recovered vs fraction of pool inspected (expected under random tie-breaking)."""
    return np.array([recall_at_frac(scores, pos, f) for f in grid])


def paired_bootstrap(score_dict, pos, n_boot, seed, fracs=(0.05, 0.10)):
    """Stratified (positives and unlabelled resampled separately), paired (same draw for all rankers).

    Returns {ranker: array of AUC draws}, plus recall draws.
    """
    rng = np.random.default_rng(seed)
    ip, iu = np.flatnonzero(pos), np.flatnonzero(~pos)
    aucs = {k: np.empty(n_boot) for k in score_dict}
    recs = {(k, f): np.empty(n_boot) for k in score_dict for f in fracs}
    for b in range(n_boot):
        bp = rng.choice(ip, size=len(ip), replace=True)
        bu = rng.choice(iu, size=len(iu), replace=True)
        idx = np.concatenate([bp, bu])
        bpos = np.zeros(len(idx), bool)
        bpos[:len(bp)] = True
        for k, s in score_dict.items():
            s = np.asarray(s)
            aucs[k][b] = auc(s[bp], s[bu])
            for f in fracs:
                recs[(k, f)][b] = recall_at_frac(s[idx], bpos, f)
    return aucs, recs


def ci(draws, level=0.95):
    lo, hi = np.percentile(draws, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)
