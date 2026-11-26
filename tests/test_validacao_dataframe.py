"""Regressões da validação dos CSVs antes de treinar ou comparar lotes."""

import pytest

from mlops_demo import config
from mlops_demo.data import validar_dataframe

pytestmark = pytest.mark.dados


@pytest.mark.parametrize(
    ("valor", "mensagem"),
    [("erro", "numérico"), (float("inf"), "infinitos"), (float("-inf"), "infinitos")],
)
def test_validador_identifica_atributo_invalido(base_treino, valor, mensagem):
    dados = base_treino.head(2).copy()
    coluna = config.FEATURES[0]
    dados[coluna] = valor

    assert any(coluna in erro and mensagem in erro for erro in validar_dataframe(dados))


def test_validador_rejeita_arquivo_so_com_cabecalho(base_treino):
    assert "arquivo sem linhas de dados" in validar_dataframe(base_treino.head(0))
