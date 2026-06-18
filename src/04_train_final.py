"""
04_train_final.py

Escolhe o melhor modelo pelo F1-score no teste1.
Depois junta treino + teste1, retreina o melhor modelo e avalia no teste2.

Como rodar:
python src/04_train_final.py
"""

import json
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

from config import (
    TRAIN_PATH,
    TEST1_PATH,
    TEST2_PATH,
    MODELS_DIR,
    REPORTS_DIR,
    RANDOM_STATE,
)
from preprocessing import read_csv_safe, select_xy, make_preprocessor
from metrics_utils import evaluate_binary_model, save_json, print_metrics
from importlib.machinery import SourceFileLoader

# Carrega função build_model do script da rede neural, mesmo o arquivo começando com número.
nn_module = SourceFileLoader("train_nn", str(__import__("pathlib").Path(__file__).parent / "03_train_neural_network.py")).load_module()
build_model = nn_module.build_model


def load_metric(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def choose_best_model():
    logistic_path = REPORTS_DIR / "metrics_logistic_test1.json"
    nn_path = REPORTS_DIR / "metrics_neural_network_test1.json"

    if not logistic_path.exists() or not nn_path.exists():
        raise FileNotFoundError(
            "Rode antes os scripts 02_train_logistic.py e 03_train_neural_network.py."
        )

    log_metrics = load_metric(logistic_path)
    nn_metrics = load_metric(nn_path)

    if nn_metrics["f1"] > log_metrics["f1"]:
        return "neural_network", nn_metrics
    return "logistic_regression", log_metrics


def train_final_logistic(train_final, test2, old_metrics):
    X_train, y_train = select_xy(train_final)
    X_test2, y_test2 = select_xy(test2)

    best_params = old_metrics.get("best_params", {})

    C = best_params.get("model__C", 1.0)
    penalty = best_params.get("model__penalty", "l2")
    solver = best_params.get("model__solver", "liblinear")

    pipe = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            (
                "model",
                LogisticRegression(
                    C=C,
                    penalty=penalty,
                    solver=solver,
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    print("Retreinando regressão logística com treino + teste1...")
    pipe.fit(X_train, y_train)

    y_prob = pipe.predict_proba(X_test2)[:, 1]
    metrics = evaluate_binary_model(y_test2, y_prob)
    metrics["model_name"] = "final_logistic_regression"
    metrics["trained_on"] = "treino + teste1"
    metrics["tested_on"] = "teste2"

    joblib.dump(pipe, MODELS_DIR / "best_model_logistic.pkl")

    # Arquivo pequeno informando para a API qual modelo usar.
    save_json({"best_model_type": "logistic_regression"}, MODELS_DIR / "best_model_info.json")

    return metrics


def train_final_nn(train_final, test2):
    X_train_raw, y_train = select_xy(train_final)
    X_test2_raw, y_test2 = select_xy(test2)

    preprocessor = make_preprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test2 = preprocessor.transform(X_test2_raw)

    classes = np.array([0, 1])
    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    class_weight = {0: class_weights_array[0], 1: class_weights_array[1]}

    model = build_model(X_train.shape[1])

    early_stop = EarlyStopping(
        monitor="val_auc",
        patience=4,
        mode="max",
        restore_best_weights=True,
    )

    print("Retreinando rede neural com treino + teste1...")
    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=25,
        batch_size=512,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=1,
    )

    y_prob = model.predict(X_test2).ravel()
    metrics = evaluate_binary_model(y_test2, y_prob)
    metrics["model_name"] = "final_neural_network"
    metrics["trained_on"] = "treino + teste1"
    metrics["tested_on"] = "teste2"

    model.save(MODELS_DIR / "best_model_neural_network.keras")
    joblib.dump(preprocessor, MODELS_DIR / "best_model_nn_preprocessor.pkl")
    save_json({"best_model_type": "neural_network"}, MODELS_DIR / "best_model_info.json")

    return metrics


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    best_name, old_metrics = choose_best_model()

    print(f"Melhor modelo no teste1: {best_name}")
    print(f"F1 no teste1: {old_metrics['f1']:.4f}")

    treino = read_csv_safe(TRAIN_PATH)
    teste1 = read_csv_safe(TEST1_PATH)
    teste2 = read_csv_safe(TEST2_PATH)

    train_final = pd.concat([treino, teste1], ignore_index=True)

    if best_name == "logistic_regression":
        metrics = train_final_logistic(train_final, teste2, old_metrics)
    else:
        metrics = train_final_nn(train_final, teste2)

    print_metrics("MODELO FINAL - TESTE2", metrics)

    metrics_path = REPORTS_DIR / "metrics_final_test2.json"
    save_json(metrics, metrics_path)
    print(f"Métricas finais salvas em: {metrics_path}")


if __name__ == "__main__":
    main()
