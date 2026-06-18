"""
02_train_logistic.py

Treina regressão logística com ColumnTransformer + RandomizedSearchCV.

Como rodar rápido:
python src/02_train_logistic.py --sample 120000

Como rodar com todos os dados:
python src/02_train_logistic.py
"""

import argparse
import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

from config import TRAIN_PATH, TEST1_PATH, MODELS_DIR, REPORTS_DIR, RANDOM_STATE
from preprocessing import read_csv_safe, select_xy, make_preprocessor
from metrics_utils import evaluate_binary_model, save_json, print_metrics


def main(sample=None):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Lendo treino e teste1...")
    train = read_csv_safe(TRAIN_PATH)
    test1 = read_csv_safe(TEST1_PATH)

    if sample is not None and sample < len(train):
        print(f"Usando amostra de {sample} linhas do treino para acelerar.")
        train = train.sample(sample, random_state=RANDOM_STATE)

    X_train, y_train = select_xy(train)
    X_test1, y_test1 = select_xy(test1)

    preprocessor = make_preprocessor()

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", clf),
        ]
    )

    param_dist = {
        "model__C": np.logspace(-3, 2, 12),
        "model__penalty": ["l1", "l2"],
        "model__solver": ["liblinear"],
    }

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=8,
        scoring="f1",
        cv=3,
        verbose=2,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    print("Treinando regressão logística...")
    search.fit(X_train, y_train)

    print("Melhores hiperparâmetros:")
    print(search.best_params_)

    best_model = search.best_estimator_

    print("Avaliando no teste1...")
    y_prob = best_model.predict_proba(X_test1)[:, 1]
    metrics = evaluate_binary_model(y_test1, y_prob, threshold=0.5)
    metrics["best_params"] = search.best_params_
    metrics["model_name"] = "logistic_regression"

    print_metrics("REGRESSÃO LOGÍSTICA - TESTE1", metrics)

    model_path = MODELS_DIR / "logistic_regression.pkl"
    joblib.dump(best_model, model_path)
    print(f"Modelo salvo em: {model_path}")

    metrics_path = REPORTS_DIR / "metrics_logistic_test1.json"
    save_json(metrics, metrics_path)
    print(f"Métricas salvas em: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None, help="Quantidade de linhas para amostrar do treino.")
    args = parser.parse_args()
    main(sample=args.sample)
