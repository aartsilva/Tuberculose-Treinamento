"""
05_explainability.py

Gera importância das features usando Permutation Importance.

Como rodar:
python src/05_explainability.py
"""

import json
import joblib
import pandas as pd
import numpy as np

from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, make_scorer

from config import TEST2_PATH, MODELS_DIR, REPORTS_DIR, RANDOM_STATE
from preprocessing import read_csv_safe, select_xy, FEATURES


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    info_path = MODELS_DIR / "best_model_info.json"

    if not info_path.exists():
        raise FileNotFoundError("Rode primeiro: python src/04_train_final.py")

    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    best_type = info["best_model_type"]
    test2 = read_csv_safe(TEST2_PATH)

    X_test2, y_test2 = select_xy(test2)

    # Usa amostra para não demorar demais.
    if len(X_test2) > 1000:
        X_sample = X_test2.sample(1000, random_state=RANDOM_STATE)
        y_sample = y_test2.loc[X_sample.index]
    else:
        X_sample = X_test2
        y_sample = y_test2

    if best_type == "logistic_regression":
        model = joblib.load(MODELS_DIR / "best_model_logistic.pkl")

        print("Calculando permutation importance para regressão logística...")
        result = permutation_importance(
            model,
            X_sample,
            y_sample,
            scoring=make_scorer(f1_score),
            n_repeats=10,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        importance = pd.DataFrame(
            {
                "feature": FEATURES,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        ).sort_values("importance_mean", ascending=False)

    else:
        # Para rede neural, criamos uma classe wrapper para o sklearn conseguir avaliar.
        import tensorflow as tf

        keras_model = tf.keras.models.load_model(MODELS_DIR / "best_model_neural_network.keras")
        preprocessor = joblib.load(MODELS_DIR / "best_model_nn_preprocessor.pkl")

        class KerasWrapper:
            def predict(self, X):
                X_transformed = preprocessor.transform(X)
                prob = keras_model.predict(X_transformed, verbose=0).ravel()
                return (prob >= 0.5).astype(int)

        wrapper = KerasWrapper()

        print("Calculando permutation importance para rede neural...")
        result = permutation_importance(
            wrapper,
            X_sample,
            y_sample,
            scoring=make_scorer(f1_score),
            n_repeats=10,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )

        importance = pd.DataFrame(
            {
                "feature": FEATURES,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        ).sort_values("importance_mean", ascending=False)

    out_path = REPORTS_DIR / "permutation_importance.csv"
    importance.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"Importância das features salva em: {out_path}")
    print("\nTop 15 features:")
    print(importance.head(15))


if __name__ == "__main__":
    main()
