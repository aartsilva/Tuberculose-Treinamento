"""
Funções de carregamento e preparação dos dados.

Este arquivo é o coração do trabalho:
- lê os CSVs;
- seleciona as variáveis;
- trata missings;
- aplica RobustScaler em variáveis numéricas;
- aplica OneHotEncoder nas categóricas;
- cria um ColumnTransformer da Scikit-Learn.
"""

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder

from config import FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET


def read_csv_safe(path):
    """Lê CSV tratando valores estranhos como ausentes."""
    missing_values = ["", " ", ".", "NA", "NaN", "nan", "NULL", "null"]
    return pd.read_csv(path, na_values=missing_values, low_memory=False)


def validate_columns(df, required_columns):
    """Confere se todas as colunas necessárias existem."""
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"As seguintes colunas não foram encontradas no CSV: {missing}")


def select_xy(df):
    """
    Recebe um dataframe bruto e devolve:
    X = preditores
    y = alvo ltfu
    """
    validate_columns(df, FEATURES + [TARGET])

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    # Garante alvo numérico 0/1.
    y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)

    # Variáveis numéricas como número.
    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Variáveis categóricas como texto.
    # Isso evita erro quando uma coluna tem mistura de número, texto e missing.
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("object")
        X[col] = X[col].where(~X[col].isna(), np.nan)

    return X, y


def make_onehot_encoder():
    """
    Cria OneHotEncoder compatível com versões novas e antigas do scikit-learn.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=20)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor():
    """
    Cria o ColumnTransformer pedido no roteiro.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_onehot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return preprocessor


def get_feature_names(preprocessor):
    """
    Recupera nomes das colunas após OneHotEncoder.
    Útil para explicabilidade.
    """
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        return [f"feature_{i}" for i in range(preprocessor.transform([[0] * len(FEATURES)]).shape[1])]
