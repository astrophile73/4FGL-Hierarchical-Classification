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
}
SOURCE_URLS = {
    '3FGL': 'https://fermi.gsfc.nasa.gov/ssc/data/access/lat/4yr_catalog/gll_psc_v16.fit',
    'DR3': 'https://fermi.gsfc.nasa.gov/ssc/data/access/lat/12yr_catalog/gll_psc_v31.fit',
    'DR4': 'https://fermi.gsfc.nasa.gov/ssc/data/access/lat/14yr_catalog/gll_psc_v35.fit',
}

PULSAR_CLASSES = {'psr', 'msp'}
EXCLUDED_FROM_TRAINING = {'glc', '', 'unk'}   # glc: MSP-powered, ambiguous label; '', unk: unassociated
