"""
API FastAPI para predição de abandono do tratamento.

Antes de rodar a API, execute:
python src/04_train_final.py

Depois rode:
uvicorn app.api:app --reload

Abra:
http://127.0.0.1:8000/docs
"""

import json
import sys
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Permite importar arquivos da pasta src.
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.append(str(SRC_DIR))

from config import MODELS_DIR, FEATURES  # noqa: E402


app = FastAPI(
    title="TBC Insight API",
    description="API para prever probabilidade de abandono do tratamento de tuberculose.",
    version="1.0.0",
)


class PacienteInput(BaseModel):
    idade_anos: float = Field(..., example=38)
    NU_ANO: int = Field(..., example=2025)
    NU_CONTATO: float = Field(0, example=0)

    SG_UF_NOT: Optional[str] = Field(None, example="33")
    CS_SEXO: Optional[str] = Field(None, example="M")
    CS_GESTANT: Optional[str] = Field(None, example="6")
    CS_RACA: Optional[str] = Field(None, example="4")
    CS_ESCOL_N: Optional[str] = Field(None, example="5")
    TRATAMENTO: Optional[str] = Field(None, example="1")
    INSTITUCIO: Optional[str] = Field(None, example="2")
    RAIOX_TORA: Optional[str] = Field(None, example="1")
    TESTE_TUBE: Optional[str] = Field(None, example="4")
    FORMA: Optional[str] = Field(None, example="1")
    AGRAVAIDS: Optional[str] = Field(None, example="2")
    AGRAVALCOO: Optional[str] = Field(None, example="2")
    AGRAVDIABE: Optional[str] = Field(None, example="2")
    AGRAVDOENC: Optional[str] = Field(None, example="2")
    AGRAVOUTRA: Optional[str] = Field(None, example="2")
    BACILOSC_E: Optional[str] = Field(None, example="3")
    BACILOS_E2: Optional[str] = Field(None, example="3")
    BACILOSC_O: Optional[str] = Field(None, example="3")
    CULTURA_ES: Optional[str] = Field(None, example="4")
    CULTURA_OU: Optional[str] = Field(None, example="4")
    HIV: Optional[str] = Field(None, example="4")
    HISTOPATOL: Optional[str] = Field(None, example="5")
    TRAT_SUPER: Optional[str] = Field(None, example="2")
    DOENCA_TRA: Optional[str] = Field(None, example="9")
    POP_LIBER: Optional[str] = Field(None, example="2")
    POP_RUA: Optional[str] = Field(None, example="2")
    POP_SAUDE: Optional[str] = Field(None, example="2")
    POP_IMIG: Optional[str] = Field(None, example="2")
    BENEF_GOV: Optional[str] = Field(None, example="2")
    AGRAVDROGA: Optional[str] = Field(None, example="2")
    AGRAVTABAC: Optional[str] = Field(None, example="2")
    TEST_MOLEC: Optional[str] = Field(None, example="5")
    TEST_SENSI: Optional[str] = Field(None, example="7")
    ANT_RETRO: Optional[str] = Field(None, example="2")


def load_model():
    info_path = MODELS_DIR / "best_model_info.json"

    if not info_path.exists():
        raise FileNotFoundError(
            "Modelo final não encontrado. Rode primeiro: python src/04_train_final.py"
        )

    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    model_type = info["best_model_type"]

    if model_type == "logistic_regression":
        model = joblib.load(MODELS_DIR / "best_model_logistic.pkl")
        return model_type, model, None

    import tensorflow as tf

    model = tf.keras.models.load_model(MODELS_DIR / "best_model_neural_network.keras")
    preprocessor = joblib.load(MODELS_DIR / "best_model_nn_preprocessor.pkl")
    return model_type, model, preprocessor


@app.get("/")
def home():
    return {
        "mensagem": "API TBC Insight funcionando.",
        "documentacao": "Acesse /docs para testar.",
    }


@app.post("/predict")
def predict(paciente: PacienteInput):
    model_type, model, preprocessor = load_model()

    data = paciente.model_dump()
    df = pd.DataFrame([data])

    # Garante que todas as features existam e estejam na ordem correta.
    for col in FEATURES:
        if col not in df.columns:
            df[col] = None

    df = df[FEATURES]

    if model_type == "logistic_regression":
        prob = float(model.predict_proba(df)[0, 1])
    else:
        X = preprocessor.transform(df)
        prob = float(model.predict(X, verbose=0).ravel()[0])

    if prob < 0.30:
        nivel = "baixo"
        recomendacao = "Acompanhamento de rotina."
    elif prob < 0.60:
        nivel = "moderado"
        recomendacao = "Considerar orientação reforçada e monitoramento mais próximo."
    else:
        nivel = "alto"
        recomendacao = "Priorizar busca ativa, acompanhamento intensivo e apoio ao paciente."

    return {
        "probabilidade_abandono": round(prob, 4),
        "probabilidade_percentual": round(prob * 100, 2),
        "nivel_risco": nivel,
        "recomendacao_clinica": recomendacao,
        "modelo_usado": model_type,
        "observacao": "Resultado é apoio à decisão, não substitui avaliação clínica.",
    }
