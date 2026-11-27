"""Census of the DR3 -> DR4 temporal holdout (sources whose association status changed between releases).

Inputs (public FSSC catalogs; pin by SHA256):
  data/dr3/gll_psc_v31.fit  4FGL-DR3  sha256 57b0923893630b8b4a739134927dabc35cc7d751d2ce56554c9c95e0d0cbf951
  gll_psc_v35.fit           4FGL-DR4  sha256 e3b3ea278412b7bda4c259ab558f00b76605be59eb84e44086a862a179ffc3e6
  data/3fgl/gll_psc_v16.fit 3FGL      sha256 527fc8e21a1eb26a7775125430dfe6a6b06eaec36eb957cf93bf2338e32a6396
Run from repo root:  .venv/Scripts/python audit/holdout_census.py
"""
import collections
import numpy as np
from astropy.table import Table

MACRO = {'bll': 'AGN', 'fsrq': 'AGN', 'bcu': 'AGN', 'agn': 'AGN', 'rdg': 'AGN', 'sey': 'AGN', 'nlsy1': 'AGN',
         'css': 'AGN', 'ssrq': 'AGN', 'msp': 'Pulsar', 'psr': 'Pulsar', 'bin': 'Binary', 'hmb': 'Binary',
         'lmb': 'Binary', 'nov': 'Binary', 'snr': 'Galactic', 'pwn': 'Galactic', 'spp': 'Galactic',
         'glc': 'Galactic', 'gc': 'Galactic', 'mc': 'Galactic', 'sfr': 'Galactic', 'gal': 'Galaxy', 'sbg': 'Galaxy'}
UNAS = {'', 'unk'}


def load(path):
    t = Table.read(path, hdu=1)
    cls = np.char.strip(np.char.lower(np.asarray(t['CLASS1']).astype('U10')))  # U10 avoids U5 truncation
    names = np.char.strip(np.asarray(t['Source_Name']).astype(str))
    return {n: (c, float(b)) for n, c, b in zip(names, cls, np.asarray(t['GLAT']))}


def main():
    d3, d4 = load('data/dr3/gll_psc_v31.fit'), load('gll_psc_v35.fit')
    common = set(d3) & set(d4)
    print(f'DR3 {len(d3)}  DR4 {len(d4)}  common-by-name {len(common)}  dropped {len(set(d3) - set(d4))}')
    newly = sorted(n for n in common if d3[n][0] in UNAS and d4[n][0] not in UNAS)
    print(f'DR3 unassociated/unk -> DR4 associated: {len(newly)}')
    print('  by DR4 class:', collections.Counter(d4[n][0] for n in newly).most_common())
    print('  by macro    :', collections.Counter(MACRO.get(d4[n][0], '?') for n in newly).most_common())
    b = np.array([abs(d4[n][1]) for n in newly])
    print(f'  |b|<10: {(b < 10).sum()}   |b|>=10: {(b >= 10).sum()}')
    bcu = [n for n in common if d3[n][0] == 'bcu' and d4[n][0] in ('bll', 'fsrq')]
    print(f'DR3 bcu -> DR4 bll/fsrq: {len(bcu)}', collections.Counter(d4[n][0] for n in bcu))
    psr = [n for n in common if d3[n][0] in UNAS and d4[n][0] in ('msp', 'psr')]
    print(f'DR3 unassoc/unk -> DR4 msp/psr (primary positives): {len(psr)}', collections.Counter(d4[n][0] for n in psr))
    lost = [n for n in common if d3[n][0] not in UNAS and d4[n][0] in UNAS]
    print(f'DR3 associated -> DR4 unassociated: {len(lost)}')

    # ---- Window 1: 3FGL (2015) -> 4FGL-DR3 (2022), disjoint from window 2 (DR3 -> DR4) ----
    # 3FGL has no 'unk' class; unassociated = blank CLASS1. Crosswalk via DR3's ASSOC_FGL column.
    t3 = Table.read('data/3fgl/gll_psc_v16.fit', hdu=1)
    tdr3 = Table.read('data/dr3/gll_psc_v31.fit', hdu=1)
    s = lambda a: np.char.strip(np.asarray(a).astype('U40'))
    c3 = dict(zip(s(t3['Source_Name']), np.char.lower(s(t3['CLASS1']))))
    pool1 = [n for n, c in c3.items() if c == '']
    xw = list(zip(s(tdr3['ASSOC_FGL']), s(tdr3['Source_Name']), np.char.lower(s(tdr3['CLASS1']))))
    w1 = sorted({a for a, _, c in xw if a in c3 and c3[a] == '' and c in ('msp', 'psr')})
    w1cls = collections.Counter(c for a, _, c in xw if a in w1 and c in ('msp', 'psr'))
    print(f'\nWINDOW 1  3FGL-unassociated pool: {len(pool1)};  -> pulsar by DR3 (positives): {len(w1)}', dict(w1cls))
    print(f'WINDOW 2  DR3-unassociated/unk pool: {sum(v[0] in UNAS for v in d3.values())};'
          f'  -> pulsar by DR4 (positives): {len(psr)}')
    w1_dr3names = {n for a, n, c in xw if a in w1}
    print(f'  overlap of window-1 positives with window-2 positives: {len(w1_dr3names & set(psr))} (must be 0)')


if __name__ == '__main__':
    main()
