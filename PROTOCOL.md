# Pre-registered analysis protocol
**Status:** FROZEN v1, 2026-11-27 (git tag `protocol-v1`). Approved before any holdout result was computed.
Any later change is a dated, justified deviation logged in §9.

## 1. Question
Fermi-LAT pulsar searches prioritise unassociated sources, using either machine-learning (ML) "pulsar-likeness"
scores or the physical cue those scores mostly encode: spectral curvature. **Does an ML ranking recover pulsars that
were identified *later* better than ranking by curvature significance alone?** No published study we found makes this head-to-head
comparison with ranking metrics (literature_notes.md).

## 2. Hypotheses
- **H1 (primary):** AUC(ML) − AUC(curvature) > 0 for later-identified pulsars, in each temporal window.
- **H0:** ΔAUC ≤ 0, meaning curvature alone is at least as good. This is a legitimate, publishable outcome.
- **Interpretation is fixed in advance:**

| Outcome | Interpretation |
|---|---|
| 95% CI of ΔAUC > 0 in both windows | Supported |
| Window 1 only | Partially supported (window 2 underpowered) |
| Neither window | Not supported; curvature suffices |
| CI < 0 in both | Curvature is better |

## 3. Temporal windows (disjoint; `audit/holdout_census.py`)
| | Train on (associations + features of) | Scored pool | Positives (identified as msp/psr in) | n_pos |
|---|---|---|---|---|
| W1 | 3FGL (gll_psc_v16) | 1,010 3FGL unassociated | 4FGL-DR3, via DR3 `ASSOC_FGL` | 55 (31 MSP, 24 PSR) |
| W2 | 4FGL-DR3 (gll_psc_v31) | 2,291 DR3 unassociated + unk | 4FGL-DR4 (v35), by Source_Name | 23 (20 MSP, 3 PSR) |
- **Pool rules.** The pool includes sources that later vanished or were never associated; these count as unlabelled, not negatives (a PU setting).
  The AUC is therefore a lower bound, equally for every method.
- **No lookahead.** Features and labels come only from the training-release catalog, and nothing from the later release enters training.

## 4. Labels (training release only)
- **Positive:** `psr`, `msp` (case-insensitive; 3FGL has no separate `msp`).
- **Negative:** every other associated class, except those excluded below.
- **Excluded from training:** `glc` (MSP-powered; ambiguous), unassociated, `unk`.

## 5. Features (fixed lists; transforms fixed in advance)
- **4FGL (W2):**
  - GLAT, cos(GLON)
  - ln Energy_Flux100, ln Unc_Energy_Flux100, ln Pivot_Energy, ln Variability_Index, Frac_Variability
  - LP_Index, LP_beta, ln LP_SigCurv, PLEC_IndexS, PLEC_ExpfactorS
  - Band hardness ratios computed from `nuFnu_Band`. A band whose `Unc_Flux_Band` lower bound is NaN (an upper limit) is set to NaN before the ratio.
- **3FGL (W1):**
  - GLAT, cos(GLON)
  - ln Energy_Flux100, ln Unc_Energy_Flux100, ln Pivot_Energy, ln Variability_Index
  - Spectral_Index, PowerLaw_Index, ln Signif_Curve
  - Hardness ratios from the five 100 MeV–100 GeV `nuFnu*` bands. Upper limits are detected via NaN `Unc_*` lower bounds where present, or else Sqrt_TS < 1.
- **Missing values:** median imputation, fitted on the training data only (inside each CV fold for the training diagnostics).
- **Pre-freeze clarifications** (settled from a data inspection of feature distributions only; no labels or holdout involved):
  - **Upper limits:** a band is an upper limit iff the lower bound of its `Unc_*` is NaN. This was verified to coincide with Sqrt_TS ≤ 1
    in DR3, DR4 and 3FGL, so the 3FGL Sqrt_TS fallback is never needed. Bands with nuFnu ≤ 0 are also NaN.
  - **"ln" of significance-type quantities** (LP_SigCurv, Signif_Curve, Variability_Index) means ln(max(x, 1e-3)).
    Zeros occur: 68 LP_SigCurv values in DR3 and 13 Signif_Curve values in 3FGL.
  - **"ln" of flux-type quantities** (Energy_Flux100, Unc_Energy_Flux100, Pivot_Energy): values ≤ 0 become NaN, then are imputed.
    For example, 5 3FGL Unc_Energy_Flux100 values are negative.
  - **The curvature ranker** uses the raw (untransformed) significance. Ties get average rank in the AUC.

## 6. Rankers (fixed; no tuning on any holdout)
- **ML (primary):** RandomForest(n_estimators=500, max_features='sqrt', min_samples_leaf=2,
  class_weight='balanced_subsample', random_state=42). No resampling, since AUC is insensitive to class balance.
- **Curvature (primary comparator):** rank by ln LP_SigCurv (W2) or ln Signif_Curve (W1).
- **Secondary:** a balanced logistic regression on [curvature, ln Variability_Index]; ML without GLAT/cos GLON; random (AUC 0.5).
- **Training diagnostic only:** 5-fold stratified CV AUC on the training release, with imputation fitted inside the folds.

## 7. Metrics and inference
- **Primary:** ROC-AUC of positives vs the rest of the pool, plus ΔAUC = AUC_ML − AUC_curv.
- **CI:** paired stratified bootstrap, 10,000 resamples with seed 42. Positives and unlabelled sources are resampled separately, and the same draw is used for all rankers.
- **Secondary:** recall in the top 5% and 10% of the pool, and the cumulative recovery curve (this is the note's single figure).
- **Exploratory (labelled as such; no multiplicity correction):**
  - |b|<10° vs ≥10°
  - MSP vs PSR positives
  - Discovery route (radio search of a LAT source vs gamma-ray blind search vs RF-targeted, e.g. TRAPUM), if it can be curated from the 3PC or FSSC lists

## 8. Known threats (to be reported, not hidden)
1. **Selection function of discoveries.** Curvature-guided searches favour the heuristic, and RF-targeted searches favour ML. Stratify where it can be curated.
2. **PU contamination.** The pool contains undiscovered pulsars, which deflates every AUC.
3. **Catalog-version differences** in features (3FGL vs 4FGL). The windows are analysed separately and not pooled.
4. **Small n in W2** (23). Report the CI and do not claim significance from W2 alone.
5. **Positional crosswalk** errors (ASSOC_FGL one-to-many: 7 cases). They are logged; affected positives are listed.

## 9. Deviations log
_(none yet)_
