"""Fixed configuration. Every value here is set by PROTOCOL.md v1; do not change without logging a deviation."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
SEED = 42
N_BOOT = 10_000
CV_FOLDS = 5
RECALL_FRACS = (0.05, 0.10)
LOG_FLOOR = 1e-3          # ln(max(x, 1e-3)) for significance-type quantities (PROTOCOL §5 clarification)

CATALOGS = {
    '3FGL': (ROOT / 'data/3fgl/gll_psc_v16.fit',
             '527fc8e21a1eb26a7775125430dfe6a6b06eaec36eb957cf93bf2338e32a6396'),
    'DR3': (ROOT / 'data/dr3/gll_psc_v31.fit',
            '57b0923893630b8b4a739134927dabc35cc7d751d2ce56554c9c95e0d0cbf951'),
    'DR4': (ROOT / 'gll_psc_v35.fit',
            'e3b3ea278412b7bda4c259ab558f00b76605be59eb84e44086a862a179ffc3e6'),
    'FL16Y': (ROOT / 'data/fl16y/gll_psc_v41.fit',
              '057fc22a17c20d7e05500da78d93566f98b59b0648d86dea0f625b8e71c281de'),
}
SOURCE_URLS = {
    '3FGL': 'https://fermi.gsfc.nasa.gov/ssc/data/access/lat/4yr_catalog/gll_psc_v16.fit',
    'DR3': 'https://fermi.gsfc.nasa.gov/ssc/data/access/lat/12yr_catalog/gll_psc_v31.fit',
    'DR4': 'https://fermi.gsfc.nasa.gov/ssc/data/access/lat/14yr_catalog/gll_psc_v35.fit',
    'FL16Y': 'https://fermi.gsfc.nasa.gov/ssc/data/access/lat/fl16y/gll_psc_v41.fit',
}

# PROTOCOL §10.4 S1: pulsars discovered by RF-targeted TRAPUM surveys (names verbatim from the papers).
TRAPUM_PULSARS = {
    'Clark+2023 L-band (arXiv:2212.08528)': [
        'J1036-4353', 'J1526-2744', 'J1623-6939', 'J1709-0333', 'J1757-6032',
        'J1803-6707', 'J1823-3543', 'J1858-5422', 'J1906-1754'],
    'TRAPUM UHF (arXiv:2602.20811)': [
        'J0657-4657', 'J1259-8148', 'J1346-2610', 'J1356+0230', 'J1712-1920',
        'J1823+1208', 'J1831-6503', 'J1910-5320', 'J2029-4239'],
}

PULSAR_CLASSES = {'psr', 'msp'}
EXCLUDED_FROM_TRAINING = {'glc', '', 'unk'}   # glc: MSP-powered, ambiguous label; '', unk: unassociated
