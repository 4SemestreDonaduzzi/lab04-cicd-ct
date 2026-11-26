"""Validação e simulação de drift para o contrato do MAGIC Gamma Telescope."""

import numpy as np
import pandas as pd
import pytest

from mlops_demo import config, monitor
from mlops_demo.data import aplicar_drift, gerar_lote, validar_dataframe

pytestmark = pytest.mark.dados


@pytest.fixture
def eventos():
    # Amostra determinística com momentos negativos e as dez medidas no formato da UCI.
    dados = pd.DataFrame(
        {
            "fLength": np.linspace(30.0, 40.0, 120),
            "fWidth": np.linspace(10.0, 15.0, 120),
            "fSize": np.linspace(2.4, 2.6, 120),
            "fConc": 0.4,
            "fConc1": 0.2,
            "fAsym": -3.0,
            "fM3Long": -2.0,
            "fM3Trans": -1.0,
            "fAlpha": 90.0,
            "fDist": 60.0,
        }
    )
    dados[config.ALVO] = np.arange(len(dados)) % 2
    return dados


@pytest.mark.parametrize(
    ("coluna", "impossivel"),
    [("fDist", -1), ("fConc", 1.01), ("fAlpha", 91), ("fWidth", 50), ("fConc1", 0.5)],
)
def test_aceita_momentos_negativos_e_rejeita_medida_impossivel(eventos, coluna, impossivel):
    assert validar_dataframe(eventos) == []
    eventos.loc[0, coluna] = impossivel
    assert any(coluna in erro for erro in validar_dataframe(eventos))


def test_drift_preserva_eventos_e_aplica_fator_na_escala_certa(eventos):
    original = eventos.copy()
    alterado = aplicar_drift(eventos, 0.75)
    esperado = original.copy()
    esperado[["fLength", "fWidth"]] *= 0.75
    esperado["fSize"] += np.log10(0.75)
    pd.testing.assert_frame_equal(alterado, esperado)
    pd.testing.assert_frame_equal(aplicar_drift(eventos, 1.0), original)
    pd.testing.assert_frame_equal(eventos, original)


@pytest.mark.parametrize("fator", [0.0, -1.0, np.inf, np.nan])
def test_simulacao_rejeita_fator_invalido(eventos, fator):
    with pytest.raises(ValueError, match="positivo e finito"):
        aplicar_drift(eventos, fator)


def test_uma_coluna_isolada_nao_dispara_retreino_e_duas_disparam(eventos):
    isolado = eventos.copy()
    isolado["fWidth"] *= 0.5
    dois_eixos = isolado.copy()
    dois_eixos["fLength"] *= 0.5
    casos = [(eventos, 0), (isolado, 1), (dois_eixos, 2), (aplicar_drift(eventos, 0.5), 3)]
    for lote, esperado in casos:
        tabela = monitor.detectar_drift(eventos, lote)
        assert int(tabela["drift"].sum()) == esperado
        assert monitor.deve_retreinar(esperado) is (esperado >= 2)


def test_monitor_recusa_lote_corrompido_antes_do_teste_ks(eventos, tmp_path, monkeypatch):
    base = tmp_path / "treino.csv"
    eventos.to_csv(base, index=False)
    monkeypatch.setattr(config, "BASE_TREINO", base)
    monkeypatch.setattr(config, "DADOS_LOTES", tmp_path / "lotes")
    config.DADOS_LOTES.mkdir()
    eventos.loc[0, "fConc"] = 1.5
    eventos.to_csv(config.DADOS_LOTES / "invalido.csv", index=False)
    with pytest.raises(ValueError, match=r"invalido\.csv.*fConc"):
        monitor.executar()


def test_monitor_decide_com_os_csvs_e_o_contrato_do_repositorio(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DADOS_LOTES", tmp_path / "lotes")
    monkeypatch.setattr(config, "RELATORIO_DRIFT", tmp_path / "drift.md")
    saida = tmp_path / "saida_actions"
    monkeypatch.setenv("GITHUB_OUTPUT", str(saida))

    for fator, esperado in [(1.0, False), (0.75, True)]:
        caminho = gerar_lote("amostra", n=120, fator_drift=fator, semente=1)
        assert validar_dataframe(pd.read_csv(caminho)) == []
        assert monitor.executar() is esperado
    assert saida.read_text().splitlines() == ["retreinar=false", "retreinar=true"]
