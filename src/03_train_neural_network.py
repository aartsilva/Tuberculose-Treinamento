"""
03_train_neural_network.py

Treina rede neural com Keras.

Como rodar rápido:
python src/03_train_neural_network.py --sample 120000 --epochs 10

Como rodar com todos os dados:
python src/03_train_neural_network.py --epochs 30
"""

import argparse
import joblib
import numpy as np

from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

from config import TRAIN_PATH, TEST1_PATH, MODELS_DIR, REPORTS_DIR, RANDOM_STATE
from preprocessing import read_csv_safe, select_xy, make_preprocessor
from metrics_utils import evaluate_binary_model, save_json, print_metrics


def build_model(input_dim):
    model = Sequential(
        [
            Dense(128, activation="relu", input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.30),
            Dense(64, activation="relu"),
            BatchNormalization(),
            Dropout(0.25),
            Dense(32, activation="relu"),
            Dropout(0.15),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    return model


def main(sample=None, epochs=20, batch_size=512):
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Lendo treino e teste1...")
    train = read_csv_safe(TRAIN_PATH)
    test1 = read_csv_safe(TEST1_PATH)

    if sample is not None and sample < len(train):
        print(f"Usando amostra de {sample} linhas do treino para acelerar.")
        train = train.sample(sample, random_state=RANDOM_STATE)

    X_train_raw, y_train = select_xy(train)
    X_test1_raw, y_test1 = select_xy(test1)

    print("Preparando dados com ColumnTransformer...")
    preprocessor = make_preprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test1 = preprocessor.transform(X_test1_raw)

    print(f"Formato após OneHotEncoder: {X_train.shape}")

    classes = np.array([0, 1])
    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    class_weight = {0: class_weights_array[0], 1: class_weights_array[1]}
    print(f"Pesos das classes: {class_weight}")

    model = build_model(X_train.shape[1])

    early_stop = EarlyStopping(
        monitor="val_auc",
        patience=4,
        mode="max",
        restore_best_weights=True,
    )

    print("Treinando rede neural...")
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=1,
    )

    print("Avaliando no teste1...")
    y_prob = model.predict(X_test1).ravel()
    metrics = evaluate_binary_model(y_test1, y_prob, threshold=0.5)
    metrics["model_name"] = "neural_network"
    metrics["epochs_requested"] = epochs

    print_metrics("REDE NEURAL - TESTE1", metrics)

    model_path = MODELS_DIR / "neural_network.keras"
    preprocessor_path = MODELS_DIR / "nn_preprocessor.pkl"

    model.save(model_path)
    joblib.dump(preprocessor, preprocessor_path)

    print(f"Modelo salvo em: {model_path}")
    print(f"Preprocessor salvo em: {preprocessor_path}")

    metrics_path = REPORTS_DIR / "metrics_neural_network_test1.json"
    save_json(metrics, metrics_path)
    print(f"Métricas salvas em: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None, help="Quantidade de linhas para amostrar do treino.")
    parser.add_argument("--epochs", type=int, default=20, help="Número máximo de épocas.")
    parser.add_argument("--batch-size", type=int, default=512, help="Tamanho do batch.")
    args = parser.parse_args()

    main(sample=args.sample, epochs=args.epochs, batch_size=args.batch_size)
