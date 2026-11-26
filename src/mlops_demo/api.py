"""API de inferência (FastAPI) servindo o modelo campeão.

Local:   uvicorn mlops_demo.api:app --reload
Docker:  docker run -p 8000:8000 mlops-demo
"""

import json
import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from mlops_demo import __version__, config

CAMINHO_MODELO = Path(os.environ.get("MODEL_PATH", config.MODELO_CAMPEAO))
CAMINHO_METRICAS = CAMINHO_MODELO.with_name("metrics.json")
N_ATRIBUTOS = len(config.FEATURES)
# nomes devolvidos na resposta: 1 = gamma (sinal), 0 = hádron (ruído de fundo)
NOME_POSITIVA = "gamma"
NOME_NEGATIVA = "hadron"

app = FastAPI(title="Triagem de eventos gamma (MAGIC, demo CI/CD/CT)", version=__version__)
_modelo = joblib.load(CAMINHO_MODELO)


class Evento(BaseModel):
    """Os atributos de um evento do telescópio, na mesma ordem do treino."""

    atributos: list[float] = Field(
        min_length=N_ATRIBUTOS,
        max_length=N_ATRIBUTOS,
        description=f"Exatamente {N_ATRIBUTOS} valores, nesta ordem: {', '.join(config.FEATURES)}",
    )


class Predicao(BaseModel):
    classe: str
    probabilidade_gamma: float
    versao_codigo: str


@app.get("/health")
def saude() -> dict:
    metricas = {}
    if CAMINHO_METRICAS.exists():
        metricas = json.loads(CAMINHO_METRICAS.read_text(encoding="utf-8"))
    return {"status": "ok", "versao_codigo": __version__, "metricas_do_campeao": metricas}


@app.post("/predict", response_model=Predicao)
def prever(evento: Evento) -> Predicao:
    entrada = pd.DataFrame([evento.atributos], columns=config.FEATURES)
    previsto = int(_modelo.predict(entrada)[0])
    # a coluna de gamma em predict_proba segue a ordem de classes_
    coluna = list(_modelo.classes_).index(config.CLASSE_POSITIVA)
    p_gamma = float(_modelo.predict_proba(entrada)[0][coluna])
    return Predicao(
        classe=NOME_POSITIVA if previsto == config.CLASSE_POSITIVA else NOME_NEGATIVA,
        probabilidade_gamma=round(p_gamma, 4),
        versao_codigo=__version__,
    )
