"""Rankers (PROTOCOL.md §6). All hyperparameters are fixed; nothing is tuned on any holdout."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .catalogs import POSITION_FEATURES
from .config import SEED


def rf():
    return Pipeline([('imp', SimpleImputer(strategy='median')),
                     ('rf', RandomForestClassifier(n_estimators=500, max_features='sqrt', min_samples_leaf=2,
                                                   class_weight='balanced_subsample', random_state=SEED,
                                                   n_jobs=-1))])


def logreg():
    return Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler()),
                     ('lr', LogisticRegression(class_weight='balanced', max_iter=5000))])


def feature_sets(columns, curvature_feature):
    """Columns used by each fitted ranker."""
    return {
        'RF': list(columns),
        'RF_noPos': [c for c in columns if c not in POSITION_FEATURES],
        'LogReg_curv_var': [curvature_feature, 'ln_VarIdx'],
    }


FITTED = {'RF': rf, 'RF_noPos': rf, 'LogReg_curv_var': logreg}
PRIMARY_ML, PRIMARY_BASELINE = 'RF', 'Curvature'
