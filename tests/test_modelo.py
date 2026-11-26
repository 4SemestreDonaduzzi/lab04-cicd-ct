"""Testes de modelo: limiares, baseline e comportamento (ML Test Score, Breck et al., 2017)."""

import json

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from mlops_demo import config
from mlops_demo.evaluate import avaliar, calcular_metricas
from mlops_demo.features import separar_xy
from mlops_demo.train import treinar

pytestmark = pytest.mark.modelo


def test_campeao_versionado_foi_certificado_pelo_portao():
    """O metrics.json do campeão registra o que o portão mediu ao aprová-lo e cumpre o contrato."""
    with open(config.METRICAS_CAMPEAO, encoding="utf-8") as arquivo:
        declaradas = json.load(arquivo)
    assert declaradas["recall_maligno"] >= config.RECALL_MINIMO, declaradas
    assert declaradas["precisao_maligna"] >= config.PRECISAO_MINIMA, declaradas


def test_campeao_versionado_nao_colapsou_na_base_de_referencia(campeao, base_holdout):
    """Sanidade: o artefato carrega, prevê e não perdeu o básico no holdout fixo.

    O contrato de 0,90 é medido pelo portão, no conjunto de avaliação da época da promoção.
    Depois que o modelo se adapta a dados novos, o holdout antigo pode cair um pouco.
    Por isso o piso aqui é menor.
    """
    metricas = avaliar(campeao, base_holdout)
    assert metricas["recall_maligno"] >= config.PISO_SANIDADE, metricas
    assert metricas["precisao_maligna"] >= config.PISO_SANIDADE, metricas


def test_treino_com_o_codigo_e_params_atuais_cumpre_o_contrato(modelo_atual, base_holdout):
    """Se alguém mexer no código ou no params.yaml, o CI treina de novo e confere os limiares."""
    metricas = avaliar(modelo_atual, base_holdout)
    assert metricas["recall_maligno"] >= config.RECALL_MINIMO, metricas
    assert metricas["precisao_maligna"] >= config.PRECISAO_MINIMA, metricas


def test_modelo_supera_o_baseline_ingenuo(modelo_atual, base_treino, base_holdout):
    X_tr, y_tr = separar_xy(base_treino)
    X_te, y_te = separar_xy(base_holdout)
    baseline = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    nota_baseline = calcular_metricas(y_te, baseline.predict(X_te))["acuracia"]
    nota_modelo = calcular_metricas(y_te, modelo_atual.predict(X_te))["acuracia"]
    assert nota_modelo > nota_baseline + 0.1


def test_treino_e_deterministico(base_treino, params, xy_holdout):
    X_te, _ = xy_holdout
    a = treinar(base_treino, params).predict_proba(X_te)
    b = treinar(base_treino, params).predict_proba(X_te)
    assert np.allclose(a, b)


def test_modelo_supera_baseline_em_um_lote_pequeno(base_treino, params):
    """No MAGIC, a regressão deve aprender além da classe majoritária em um lote pequeno."""
    pequeno = base_treino.sample(40, random_state=0)
    modelo = treinar(pequeno, params)
    X, y = separar_xy(pequeno)
    acuracia = (modelo.predict(X) == y).mean()
    baseline = y.value_counts(normalize=True).max()
    assert acuracia >= baseline + 0.05


def test_probabilidades_sao_validas(modelo_atual, xy_holdout):
    X_te, _ = xy_holdout
    probas = modelo_atual.predict_proba(X_te)
    assert np.all((probas >= 0) & (probas <= 1))
    assert np.allclose(probas.sum(axis=1), 1.0)


def test_modelo_atribui_probabilidade_gamma_maior_a_eventos_gamma(modelo_atual, xy_holdout):
    X_te, y_te = xy_holdout
    coluna = list(modelo_atual.classes_).index(config.CLASSE_POSITIVA)
    p_gamma = modelo_atual.predict_proba(X_te)[:, coluna]
    rotulos = y_te.to_numpy()
    media_gamma = p_gamma[rotulos == config.CLASSE_POSITIVA].mean()
    media_hadron = p_gamma[rotulos != config.CLASSE_POSITIVA].mean()
    assert media_gamma > media_hadron


def test_invariancia_a_ruido_minimo(modelo_atual, xy_holdout):
    """Teste de comportamento: ruído de 0,01% nos atributos quase nunca muda a previsão."""
    X_te, _ = xy_holdout
    rng = np.random.default_rng(0)
    ruidoso = X_te * (1 + rng.normal(0, 1e-4, X_te.shape))
    mudou = (modelo_atual.predict(X_te) != modelo_atual.predict(ruidoso)).mean()
    assert mudou <= 0.02


def test_direcional_fAlpha_maior_reduz_probabilidade_gamma(modelo_atual, xy_holdout):
    """Eventos gamma apontam à origem; ângulo fAlpha maior deve reduzir sua probabilidade."""
    X_te, _ = xy_holdout
    coluna = list(modelo_atual.classes_).index(config.CLASSE_POSITIVA)
    maior_alpha = X_te.copy()
    maior_alpha["fAlpha"] = np.minimum(maior_alpha["fAlpha"] + 5.0, 90.0)
    antes = modelo_atual.predict_proba(X_te)[:, coluna].mean()
    depois = modelo_atual.predict_proba(maior_alpha)[:, coluna].mean()
    assert depois < antes
