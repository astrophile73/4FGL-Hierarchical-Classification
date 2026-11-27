"""PROTOCOL §10.4–10.5: curate W3 positives and S1/S2 flags from catalogs ONLY (no model scores).

Usage:  .venv/Scripts/python -m src.curate_w3      -> results/w3_positives.csv (commit BEFORE src.run_w3)

S1 rule: a positive is RF-targeted if its FL16Y ASSOC1 pulsar name matches a TRAPUM discovery:
         same RA (hhmm) and declination within 1 arcmin (tolerates the ±1' naming differences seen in the papers).
S2 rule: identified = upper-case CLASS1 ('MSP'/'PSR') in FL16Y.
"""
import re

import pandas as pd

from .config import RESULTS, TRAPUM_PULSARS
from .windows import build_w3, w3_later_classes

_PAT = re.compile(r'J(\d{2})(\d{2})([+-])(\d{2})(\d{2})?')


def parse(name):
    """'PSR J1709-0333' (any dash) -> (hhmm, signed dec arcmin) or None."""
    m = _PAT.search(name.replace('−', '-').replace('–', '-'))
    if not m:
        return None
    hh, mm, sgn, dd, dm = m.groups()
    arcmin = int(dd) * 60 + (int(dm) if dm else 0)
    return hh + mm, (-1 if sgn == '-' else 1) * arcmin


def trapum_match(assoc1):
    p = parse(assoc1) if assoc1 else None
    if p is None:
        return ''
    for survey, names in TRAPUM_PULSARS.items():
        for n in names:
            q = parse(n)
            if q and q[0] == p[0] and abs(q[1] - p[1]) <= 1:
                return survey
    return ''


def main():
    RESULTS.mkdir(exist_ok=True)
    w = build_w3()
    later = w3_later_classes()
    rows = []
    for name, cls, glat in zip(w.pool_names[w.pos], w.pos_class[w.pos], w.pool_glat[w.pos]):
        hits = [(n, c, a1) for n, c, a1 in later[name] if c.lower() == cls]
        fl_name, fl_class, assoc1 = hits[0]
        survey = trapum_match(assoc1)
        rows.append({'dr4_name': name, 'fl16y_name': fl_name, 'fl16y_class1': fl_class, 'pulsar_type': cls,
                     'assoc1': assoc1, 'glat': round(float(glat), 3),
                     's1_rf_targeted': bool(survey), 's1_survey': survey,
                     's2_identified': fl_class.isupper()})
    df = pd.DataFrame(rows).sort_values('dr4_name')
    df.to_csv(RESULTS / 'w3_positives.csv', index=False)
    print(df.to_string(index=False))
    print(f"\nn_pos={len(df)}  MSP={int((df.pulsar_type == 'msp').sum())}  PSR={int((df.pulsar_type == 'psr').sum())}  "
          f"S1 RF-targeted={int(df.s1_rf_targeted.sum())}  S2 identified={int(df.s2_identified.sum())}")


if __name__ == '__main__':
    main()
