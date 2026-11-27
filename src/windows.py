"""Construct the two disjoint temporal windows (PROTOCOL.md §3–4).

Each window holds:
  train_X, train_y : labelled sources of the TRAINING release (features from that release only)
  pool_X           : unassociated sources of the training release (features from that release only)
  pool_names, pos  : pool source names; boolean mask of pool sources identified as pulsars in the LATER release
No information from the later release enters anything except `pos`.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import catalogs as C
from .config import EXCLUDED_FROM_TRAINING, PULSAR_CLASSES


@dataclass
class Window:
    name: str
    train_release: str
    later_release: str
    train_X: pd.DataFrame
    train_y: np.ndarray
    pool_X: pd.DataFrame
    pool_names: np.ndarray
    pool_curv: np.ndarray
    pos: np.ndarray
    pos_class: np.ndarray            # 'msp'/'psr' for positives, '' otherwise
    pool_glat: np.ndarray
    meta: dict = field(default_factory=dict)


def _split(t, feat_fn, pool_classes):
    X, curv = feat_fn(t)
    cls = C.classes(t)
    nm = C.names(t)
    in_pool = np.isin(cls, list(pool_classes))
    in_train = ~np.isin(cls, list(EXCLUDED_FROM_TRAINING))
    y = np.isin(cls, list(PULSAR_CLASSES)).astype(int)
    return X, curv, cls, nm, in_pool, in_train, y


def build_w1():
    """W1: train on 3FGL; pool = 3FGL unassociated; positives = pulsar in DR3 (via DR3 ASSOC_FGL)."""
    t3, tdr3 = C.load('3FGL'), C.load('DR3')
    X, curv, cls, nm, in_pool, in_train, y = _split(t3, C.features_3fgl, {''})
    later_cls = C.classes(tdr3)
    assoc_fgl = C._str(tdr3['ASSOC_FGL'])
    psr_map = {}
    for a, c in zip(assoc_fgl, later_cls):
        if a and c in PULSAR_CLASSES:
            psr_map.setdefault(a, c)
    pool_names = nm[in_pool]
    pos_class = np.array([psr_map.get(n, '') for n in pool_names])
    return Window('W1', '3FGL', 'DR3',
                  X[in_train].reset_index(drop=True), y[in_train],
                  X[in_pool].reset_index(drop=True), pool_names, curv[in_pool],
                  pos_class != '', pos_class, X['GLAT'].values[in_pool],
                  meta={'n_train': int(in_train.sum()), 'n_train_pos': int(y[in_train].sum()),
                        'n_pool': int(in_pool.sum())})


def build_w2():
    """W2: train on DR3; pool = DR3 unassociated + unk; positives = pulsar in DR4 (by Source_Name)."""
    tdr3, tdr4 = C.load('DR3'), C.load('DR4')
    X, curv, cls, nm, in_pool, in_train, y = _split(tdr3, C.features_4fgl, {'', 'unk'})
    later = dict(zip(C.names(tdr4), C.classes(tdr4)))
    pool_names = nm[in_pool]
    pos_class = np.array([later.get(n, '') if later.get(n, '') in PULSAR_CLASSES else '' for n in pool_names])
    return Window('W2', 'DR3', 'DR4',
                  X[in_train].reset_index(drop=True), y[in_train],
                  X[in_pool].reset_index(drop=True), pool_names, curv[in_pool],
                  pos_class != '', pos_class, X['GLAT'].values[in_pool],
                  meta={'n_train': int(in_train.sum()), 'n_train_pos': int(y[in_train].sum()),
                        'n_pool': int(in_pool.sum())})


def w3_later_classes():
    """DR4 Source_Name -> list of (FL16Y name, CLASS1 case-preserved, ASSOC1) via FL16Y ASSOC_FGL."""
    tf = C.load('FL16Y')
    out = {}
    for a, n, c, a1 in zip(C._str(tf['ASSOC_FGL']), C.names(tf), C._str(tf['CLASS1']), C._str(tf['ASSOC1'])):
        if a.startswith('4FGL'):
            out.setdefault(a, []).append((n, c, a1))
    return out


def build_w3():
    """W3 (PROTOCOL §10): train on DR4; pool = DR4 unassociated + unk; positives = pulsar in FL16Y."""
    tdr4 = C.load('DR4')
    X, curv, cls, nm, in_pool, in_train, y = _split(tdr4, C.features_4fgl, {'', 'unk'})
    later = w3_later_classes()
    pool_names = nm[in_pool]

    def pulsar_class(n):
        found = sorted({c.lower() for _, c, _ in later.get(n, [])} & PULSAR_CLASSES)
        return found[0] if found else ''

    pos_class = np.array([pulsar_class(n) for n in pool_names])
    w = Window('W3', 'DR4', 'FL16Y',
               X[in_train].reset_index(drop=True), y[in_train],
               X[in_pool].reset_index(drop=True), pool_names, curv[in_pool],
               pos_class != '', pos_class, X['GLAT'].values[in_pool],
               meta={'n_train': int(in_train.sum()), 'n_train_pos': int(y[in_train].sum()),
                     'n_pool': int(in_pool.sum())})
    w.meta['pool_lp_beta'] = X['LP_beta'].values[in_pool]       # raw LP_beta for the Shape ranker (§10.3 H2b)
    return w


BUILDERS = {'W1': build_w1, 'W2': build_w2}       # W1/W2 are complete; W3 is run separately (src/run_w3.py)
