# housing_pipeline.py
"""
Shared ML pipeline components for the STUDENT PERFORMANCE
classification project.

Used by:
- 02_train_models_without_optuna
- 03_train_models_with_optuna
- FastAPI inference service
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier

# =============================================================================
# FEATURE GROUPS (MATCH DB + STREAMLIT)
# =============================================================================

NUMERICAL_FEATURES = [
    "age",
    "sleep_hours",
    "class_attendance",
    "study_hours",
]

CATEGORICAL_FEATURES = [
    "gender",
    "internet_access",
    "sleep_quality",
    "course",
    "study_method",
    "facility_rating",
    "exam_difficulty",
]

# =============================================================================
# PREPROCESSING
# =============================================================================

def build_preprocessing(use_pca: bool = False, pca_components: int = 10):
    """
    Build preprocessing pipeline for student classification.
    """

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    if use_pca:
        return Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("pca", PCA(n_components=pca_components, random_state=42)),
            ]
        )

    return preprocessor

# =============================================================================
# CLASSIFICATION MODELS
# =============================================================================

def make_estimator_for_name(name: str):
    """
    Factory for classification estimators.
    """

    if name == "logreg":
        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            n_jobs=-1,
        )

    elif name == "ridgeclf":
        return RidgeClassifier(class_weight="balanced")

    elif name == "histgb":
        return HistGradientBoostingClassifier(
            random_state=42,
            max_depth=6,
            learning_rate=0.1,
        )

    elif name == "xgboost":
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            n_jobs=-1,
        )

    else:
        raise ValueError(f"Unknown model name: {name}")
