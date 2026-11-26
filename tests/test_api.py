"""Testes de contrato da API: formato de entrada e de saída do /predict."""

import pytest
from fastapi.testclient import TestClient

from mlops_demo import config
from mlops_demo.api import app

pytestmark = pytest.mark.api

cliente = TestClient(app)


def test_health_responde_ok():
    resposta = cliente.get("/health")
    assert resposta.status_code == 200
    assert resposta.json()["status"] == "ok"


def test_predict_devolve_o_formato_combinado(base_holdout):
    evento = base_holdout[config.FEATURES].iloc[0].tolist()
    resposta = cliente.post("/predict", json={"atributos": evento})
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert set(corpo) == {"classe", "probabilidade_gamma", "versao_codigo"}
    assert corpo["classe"] in {"gamma", "hadron"}
    assert 0.0 <= corpo["probabilidade_gamma"] <= 1.0


def test_predict_rejeita_quantidade_errada_de_atributos():
    resposta = cliente.post("/predict", json={"atributos": [1.0, 2.0]})
    assert resposta.status_code == 422


@pytest.mark.parametrize("n", [len(config.FEATURES) - 1, len(config.FEATURES) + 1])
def test_predict_rejeita_um_atributo_a_menos_ou_a_mais(base_holdout, n):
    """Sem o 422 a API preencheria/cortaria a lista e preveria com colunas trocadas."""
    linha = base_holdout[config.FEATURES].iloc[0].tolist()
    atributos = (linha + [0.0])[:n]
    resposta = cliente.post("/predict", json={"atributos": atributos})
    assert resposta.status_code == 422
    assert str(len(config.FEATURES)) in resposta.text


def test_predict_acerta_um_caso_gamma_e_um_hadron_claros(base_holdout, campeao):
    """Escolhe os casos mais claros (maior confiança do campeão) de cada classe."""
    X = base_holdout[config.FEATURES]
    coluna = list(campeao.classes_).index(config.CLASSE_POSITIVA)
    p_gamma = campeao.predict_proba(X)[:, coluna]
    i_gamma = int(p_gamma.argmax())
    i_hadron = int(p_gamma.argmin())
    for indice, esperado in ((i_gamma, "gamma"), (i_hadron, "hadron")):
        resposta = cliente.post("/predict", json={"atributos": X.iloc[indice].tolist()})
        assert resposta.json()["classe"] == esperado


def test_predict_devolve_probabilidade_da_coluna_positiva_gamma(base_holdout, campeao):
    X = base_holdout[config.FEATURES].iloc[[0]]
    coluna = list(campeao.classes_).index(config.CLASSE_POSITIVA)
    esperado = round(float(campeao.predict_proba(X)[0, coluna]), 4)
    resposta = cliente.post("/predict", json={"atributos": X.iloc[0].tolist()})
    assert resposta.status_code == 200
    assert resposta.json()["probabilidade_gamma"] == pytest.approx(esperado)
