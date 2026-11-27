"""Catalog loading and feature construction (PROTOCOL.md §4–5)."""
import hashlib

import numpy as np
import pandas as pd
from astropy.table import Table

from .config import CATALOGS, LOG_FLOOR


def _str(col):
    """Full-width, stripped, lower-cased strings. Avoids the fixed-width truncation bug seen in data_extraction.ipynb."""
    return np.char.strip(np.asarray(col).astype(str))


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load(key, verify=True):
    """Load a catalog's point-source HDU, verifying its SHA256 against the pinned value."""
    path, expected = CATALOGS[key]
    if not path.exists():
        raise FileNotFoundError(f'{key}: {path} missing (see src/config.py SOURCE_URLS)')
    if verify and sha256(path) != expected:
        raise ValueError(f'{key}: checksum mismatch for {path}')
    return Table.read(path, hdu=1)


def names(t):
    return _str(t['Source_Name'])


def classes(t):
    return np.char.lower(_str(t['CLASS1']))


def ln_sig(x):
    """ln of a significance-type quantity, floored (zeros occur)."""
    return np.log(np.maximum(np.asarray(x, float), LOG_FLOOR))


def ln_flux(x):
    """ln of a flux-type quantity; non-positive -> NaN (imputed later)."""
    x = np.asarray(x, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(x > 0, np.log(np.where(x > 0, x, 1.0)), np.nan)


def hardness_ratios(nufnu, is_ul):
    """Adjacent-band HR = (F_{i+1} - F_i) / (F_{i+1} + F_i); NaN if either band is an upper limit or F <= 0.

    nufnu, is_ul: arrays of shape (n_sources, n_bands).
    """
    f = np.where(is_ul | ~(nufnu > 0), np.nan, nufnu.astype(float))
    lo, hi = f[:, :-1], f[:, 1:]
    return (hi - lo) / (hi + lo)


def finite_or_nan(df):
    """Non-finite values (±inf) are treated as missing and imputed (PROTOCOL §9, deviation D1)."""
    return df.replace([np.inf, -np.inf], np.nan)


def features_4fgl(t):
    """19 features for 4FGL releases (used for W2, trained on DR3)."""
    unc = np.asarray(t['Unc_Flux_Band'], float)          # (n, 8, 2): [lower, upper]
    is_ul = np.isnan(unc[..., 0])
    hr = hardness_ratios(np.asarray(t['nuFnu_Band'], float), is_ul)
    df = pd.DataFrame({
        'GLAT': np.asarray(t['GLAT'], float),
        'cosGLON': np.cos(np.radians(np.asarray(t['GLON'], float))),
        'ln_EF100': ln_flux(t['Energy_Flux100']),
        'ln_Unc_EF100': ln_flux(t['Unc_Energy_Flux100']),
        'ln_Pivot': ln_flux(t['Pivot_Energy']),
        'ln_VarIdx': ln_sig(t['Variability_Index']),
        'Frac_Variability': np.asarray(t['Frac_Variability'], float),
        'LP_Index': np.asarray(t['LP_Index'], float),
        'LP_beta': np.asarray(t['LP_beta'], float),
        'ln_LP_SigCurv': ln_sig(t['LP_SigCurv']),
        'PLEC_IndexS': np.asarray(t['PLEC_IndexS'], float),
        'PLEC_ExpfactorS': np.asarray(t['PLEC_ExpfactorS'], float),
    })
    for i in range(hr.shape[1]):
        df[f'HR_{i + 1}{i + 2}'] = hr[:, i]
    curvature = np.asarray(t['LP_SigCurv'], float)       # raw, for the curvature ranker
    return finite_or_nan(df), curvature


BANDS_3FGL = ['100_300', '300_1000', '1000_3000', '3000_10000', '10000_100000']


def features_3fgl(t):
    """13 features for 3FGL (used for W1)."""
    nufnu = np.column_stack([np.asarray(t['nuFnu' + b], float) for b in BANDS_3FGL])
    is_ul = np.column_stack([np.isnan(np.asarray(t['Unc_Flux' + b], float)[:, 0]) for b in BANDS_3FGL])
    hr = hardness_ratios(nufnu, is_ul)
    df = pd.DataFrame({
        'GLAT': np.asarray(t['GLAT'], float),
        'cosGLON': np.cos(np.radians(np.asarray(t['GLON'], float))),
        'ln_EF100': ln_flux(t['Energy_Flux100']),
        'ln_Unc_EF100': ln_flux(t['Unc_Energy_Flux100']),
        'ln_Pivot': ln_flux(t['Pivot_Energy']),
        'ln_VarIdx': ln_sig(t['Variability_Index']),
        'Spectral_Index': np.asarray(t['Spectral_Index'], float),
        'PowerLaw_Index': np.asarray(t['PowerLaw_Index'], float),
        'ln_Signif_Curve': ln_sig(t['Signif_Curve']),
    })
    for i in range(hr.shape[1]):
        df[f'HR_{i + 1}{i + 2}'] = hr[:, i]
    curvature = np.asarray(t['Signif_Curve'], float)
    return finite_or_nan(df), curvature


POSITION_FEATURES = ['GLAT', 'cosGLON']
CURVATURE_FEATURE = {'3FGL': 'ln_Signif_Curve', 'DR3': 'ln_LP_SigCurv'}
