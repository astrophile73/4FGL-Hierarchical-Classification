# Literature notes: novelty check for the G1 research note
_Checked 2026-11-27. Seven web searches, plus two papers read in depth (partly through model-extracted summaries, flagged below). This check is NOT exhaustive._

## Read in depth
### Bhat & Malyshev 2022, A&A 660, A87 (arXiv:2102.07642): the closest prior work
- **Temporal validation:** trained on 3FGL, then checked 3FGL-unassociated sources that are associated in 4FGL-DR2
  (340 sources: 258 AGN, 45 PSR, 37 other; Table 4, our arithmetic).
- **Cross-match accuracy 85–92% vs test accuracy 93–97%** (Table 2). PSR recall 0.51 vs 0.67, PSR precision 0.68 vs 0.78 (Table 4).
  They attribute the gap to parameter-reconstruction errors and differences between associated and unassociated sources.
- **No non-ML baseline** (no curvature-only comparison). **No ranking metric** (no AUC or top-k), only fixed-threshold
  precision/recall plus threshold sweeps (App. C). They do provide reliability diagrams (App. D).
- **Oversampling:** by repetition, applied to the training split (1000 random 70/30 splits, no k-fold). SMOTE vs repetition gave
  differences smaller than the training-randomness uncertainty (App. B). The leakage in our notebook is NOT a flaw in the literature.
- **Features:** Signif_Curve is the most important; GLAT is among the least important. This is consistent with our ablation (removing GLAT/GLON costs −0.014).
- **Caveat:** Sects. 5–6 were read only through a summary.

### TRAPUM UHF survey (arXiv:2602.20811, 2026)
- **Targets:** 79 unidentified 4FGL sources ranked by the Random Forest pulsar probability of Clark et al. (2023), then cut on |b|>10°,
  Dec < +20°, and 95% semi-major axis < 7'.
- **Discoveries:** 9 MSPs and 6 slow pulsars. 7 of the MSPs are confirmed gamma-ray counterparts (Sect. 3.5).
- **No yield broken down by selection method, and no comparison of ML vs curvature.**
- Sect. 5 (summary only, UNVERIFIED) mentions an expanded survey of 145 DR4 sources with 23 new pulsars so far.
- **Implication:** TRAPUM discoveries were *selected by an RF score*, so using them as test positives biases the test toward ML.
  DR4 associations from curvature-guided searches bias it toward curvature. **The selection function of the discoveries matters.**

## Seen via search only (abstract or snippet level; verify before citing)
- Malyshev 2023, RASTI 2, 735 (arXiv:2307.09584): covariate shift on multi-class Fermi classification, using importance weighting.
  Shift reduces precision/recall by 10–20% but changes the predicted probabilities little.
- Malyshev 2024 (arXiv:2412.04675): the dataset-shift framework, comparing covariate-shift and prior-shift models.
- Zhu et al. 2024, MNRAS 527, 1794 (arXiv:2311.03678): DR3 unassociated sources split by latitude.
  Balanced accuracy is 91% for a binary classifier at high latitude and 81% for a ternary one at low latitude (CV only).
- arXiv:2510.08654 (2025): RF/XGBoost pulsar candidates among 2257 DR4 unassociated sources. β_LP is the top feature.
- A&A 2026 (arXiv:2607.28723): TabularResCNN on 2563 DR4 unassociated sources, giving 1136 AGN and 202 PSR candidates (Zenodo 10.5281/zenodo.21701546).
- arXiv:2012.05251: deep learning on energy and time spectra, 3FGL → 4FGL-DR2 test, with 12 MSP candidates.
- Frail et al. (arXiv:1712.06609): an image-based radio search. It notes that classifiers would have ranked 2 known MSPs low.
- FAST discovery of PSR J0318+0253 (arXiv:2109.00715): targets chosen where 4 ML classifiers agreed.
- Third Fermi Pulsar Catalog (3PC; Smith et al. 2023, ApJ; arXiv:2307.11132): 294 pulsars, with 119 radio MSPs found in LAT sources.
  The FSSC page lists post-3PC discoveries.
- SMOTE-before-CV leakage is already documented outside astronomy: Blagus & Lusa 2015 (BMC Bioinformatics 16:363);
  Demircioğlu 2024 (Sci. Rep.), with up to +0.37 balanced-accuracy bias.

## Novelty verdict (provisional)
- **NOT novel:** validating against a later catalog release (Bhat & Malyshev 2022); covariate/prior-shift handling (Malyshev 2023/2024);
  DR4 pulsar candidate lists (two or more papers); SMOTE leakage as a phenomenon.
- **Apparent gap:** a head-to-head of an ML pulsar *ranking* against a curvature-only baseline, scored with ranking metrics
  (AUC, recall@k) on pulsars identified *later*, with the selection function of those discoveries made explicit.
- **Still to read before writing:** Clark et al. 2023 (the TRAPUM L-band RF, arXiv:2212.08528); arXiv:2510.08654; arXiv:2607.28723;
  Saz Parkinson et al. 2016 (arXiv:1602.00385); a systematic ADS search.

## Added 2026-11-27 (second pass)
### Clark et al. 2023, TRAPUM L-band (arXiv:2212.08528); Sects. 1–2.4 read verbatim, the rest via summary
- **Classifier:** RF trained on 4FGL (2020) with 5 features (PLEC_SigCurve, Variability2_Index, Signif_Avg, PLEC_Index, PLEC_Expfactor)
  and 3 classes. Validated on one random 33% hold-out: pulsar precision 82%, recall 71%. No AUC. Imbalance handling not stated.
- **Targets:** Dec < +20°, |b| ≥ 10°, 95% semi-major axis < 5', P(psr) ≥ 12%, leaving 79 sources. Result: 9 MSPs. Table 1 lists P(psr) for all 79.
- **No curvature-only comparison and no ranking/recall analysis.**
  Summary only: "five of the nine new pulsars had fairly low P(psr) < 25%". This hints that the ML ranking was not decisive.

### Pathania et al. 2025, Astropart. Phys. (arXiv:2510.08654); verbatim HTML
- **Model:** RF and XGBoost for pulsar vs AGN on DR4 (320 PSR / 3768 AGN). 8 features; β_LP is the top feature.
- **No non-ML baseline; no precision@k or recall@k.**
- **3FGL → DR3 shift test** (71 PSR + 1664 AGN new in DR3): balanced accuracy falls from ~90–99% in-distribution to ~49–74%.
  This test is defined by sources new to DR3, which differs from our pool-ranking design.
- **Prospective check:** 5 FAST post-DR4 pulsars, all with P > 0.7. No base rate is given, so this is not a precision estimate.
- **Apparent validity issues (OUR reading; verify before ever stating):** RFE feature selection on the full labelled set (Sect. 3.2),
  and SMOTE apparently applied before GridSearchCV. If confirmed, the leakage pattern also exists in the published literature, not only in
  our notebook. That would make the methods sentence more relevant, but must be handled carefully and verified.
- The candidate CSV is published as supplementary material.

### Verdict after the second pass
The gap still holds across all 4 papers read: **no curvature-only baseline and no ranking metrics on later-identified pulsars.**
