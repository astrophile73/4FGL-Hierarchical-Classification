"""Audit diagnostic: how much of the reported 98.84% macro CV score comes from leakage?

Read-only with respect to the pipeline. It uses 4fgl_macro_training_features.csv and
4fgl_inference_targets.csv, the Phase-1 outputs of main.ipynb.

Run from the repo root:
    .venv/Scripts/python audit/leakage_check.py
Outputs: audit/leakage_check_results.json (and a printed summary).
"""
import json, sys, time
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

SEED = 42
N_SPLITS, N_REPEATS = 5, 3
FEATURES = ['PL_Index', 'LP_SigCurv', 'Energy_Flux100', 'Variability_Index', 'Frac_Variability',
            'GLAT', 'GLON', 'HR_1_2', 'HR_2_3', 'HR_3_4', 'HR_4_5', 'HR_5_6', 'HR_6_7', 'HR_7_8']
SPATIAL = ['GLAT', 'GLON']


def clean(df):
    X = df[FEATURES].replace('--', np.nan).replace([np.inf, -np.inf], np.nan)
    return X.astype(float)


def rf(**kw):  # identical hyperparameters to main.ipynb macro model
    return RandomForestClassifier(n_estimators=200, max_depth=20, random_state=SEED, n_jobs=-1, **kw)


def score(y, yhat, classes):
    return {
        'balanced_accuracy': balanced_accuracy_score(y, yhat),
        'macro_f1': f1_score(y, yhat, average='macro'),
        'recall_per_class': dict(zip(classes, recall_score(y, yhat, average=None).round(4).tolist())),
    }


def cv_real(model, X, y, classes, n_splits=N_SPLITS, n_repeats=N_REPEATS):
    """Proper CV: all fitting (impute/scale/SMOTE/model) inside train folds; scored on REAL held-out sources."""
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    per_rep, ba, f1 = [], [], []
    for r in range(n_repeats):
        yhat = np.empty_like(y)
        splits = list(rskf.split(X, y))[r * n_splits:(r + 1) * n_splits]
        for tr, te in splits:
            m = clone(model).fit(X[tr], y[tr])
            yhat[te] = m.predict(X[te])
        s = score(y, yhat, classes)
        per_rep.append(s)
        ba.append(s['balanced_accuracy']); f1.append(s['macro_f1'])
    return {
        'balanced_accuracy_mean': float(np.mean(ba)), 'balanced_accuracy_std': float(np.std(ba)),
        'macro_f1_mean': float(np.mean(f1)), 'macro_f1_std': float(np.std(f1)),
        'recall_per_class_rep0': per_rep[0]['recall_per_class'],
    }


def main():
    t0 = time.time()
    df = pd.read_csv('4fgl_macro_training_features.csv')
    inf = pd.read_csv('4fgl_inference_targets.csv')
    X_df = clean(df)
    le = LabelEncoder()
    y = le.fit_transform(df['Macro_Class'])
    classes = le.classes_.tolist()
    X = X_df.values
    res = {'seed': SEED, 'cv': f'RepeatedStratifiedKFold({N_SPLITS}x{N_REPEATS})',
           'n_real': int(len(y)), 'class_counts': dict(zip(classes, np.bincount(y).tolist())),
           'versions': {m.__name__: m.__version__ for m in
                        [np, pd, __import__('sklearn'), __import__('imblearn')]},
           'python': sys.version.split()[0], 'macro': {}, 'micro': {}, 'shift': {}}

    # A. ORIGINAL protocol: impute+scale+SMOTE on ALL data, then CV on the balanced set.
    Xa = StandardScaler().fit_transform(SimpleImputer(strategy='median').fit_transform(X))
    Xb, yb = SMOTE(random_state=SEED).fit_resample(Xa, y)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    yhat = cross_val_predict(rf(), Xb, yb, cv=skf)
    res['macro']['A_original_smote_before_cv'] = score(yb, yhat, classes)
    res['macro']['A_original_smote_before_cv']['note'] = (
        f'{len(yb)} rows scored, {len(yb) - len(y)} of them synthetic')

    base = [('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]
    models = {
        'B_smote_inside_folds_rf': ImbPipeline(base + [('smote', SMOTE(random_state=SEED)), ('rf', rf())]),
        'C_rf_class_weight_no_smote': Pipeline(base + [('rf', rf(class_weight='balanced_subsample'))]),
        'D_logreg_balanced': Pipeline(base + [('lr', LogisticRegression(class_weight='balanced', max_iter=5000))]),
        'E_dummy_stratified': DummyClassifier(strategy='stratified', random_state=SEED),
        'E_dummy_majority': DummyClassifier(strategy='most_frequent'),
    }
    for name, m in models.items():
        res['macro'][name] = cv_real(m, X, y, classes)
        print(f'  done {name} ({time.time() - t0:.0f}s)', flush=True)

    # F. Spatial ablation (B without GLAT/GLON)
    keep = [i for i, f in enumerate(FEATURES) if f not in SPATIAL]
    res['macro']['F_smote_inside_no_glat_glon'] = cv_real(models['B_smote_inside_folds_rf'], X[:, keep], y, classes)

    # Micro classifiers: original reported resubstitution 1.00; here proper CV on real sources.
    for name, subs in {'bll_vs_fsrq': ['bll', 'fsrq'], 'msp_vs_psr': ['msp', 'psr']}.items():
        d = df[df['Sub_Class'].isin(subs)]
        Xm, ym = clean(d).values, LabelEncoder().fit_transform(d['Sub_Class'])
        pipe = ImbPipeline(base + [('smote', SMOTE(random_state=SEED)),
                                   ('rf', RandomForestClassifier(n_estimators=200, max_depth=15,
                                                                 random_state=SEED, n_jobs=-1))])
        res['micro'][name] = cv_real(pipe, Xm, ym, subs)
        res['micro'][name + '_no_glat_glon'] = cv_real(pipe, Xm[:, keep], ym, subs)

    # Covariate shift: adversarial validation (labelled training vs unass/unk inference set).
    Xi = clean(inf).values
    Xs = np.vstack([X, Xi]); ys = np.r_[np.zeros(len(X)), np.ones(len(Xi))].astype(int)
    adv = Pipeline([('imp', SimpleImputer(strategy='median')),
                    ('rf', RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1))])
    p = cross_val_predict(adv, Xs, ys, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                          method='predict_proba')[:, 1]
    res['shift']['adversarial_auc_train_vs_inference'] = float(roc_auc_score(ys, p))
    res['shift']['median_abs_glat_train'] = float(np.nanmedian(np.abs(X_df['GLAT'])))
    res['shift']['median_abs_glat_inference'] = float(np.nanmedian(np.abs(clean(inf)['GLAT'])))
    res['shift']['frac_abs_glat_lt10_train'] = float(np.mean(np.abs(X_df['GLAT']) < 10))
    res['shift']['frac_abs_glat_lt10_inference'] = float(np.mean(np.abs(clean(inf)['GLAT']) < 10))
    res['runtime_s'] = round(time.time() - t0, 1)

    with open('audit/leakage_check_results.json', 'w') as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
