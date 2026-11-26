"""Configuração central do projeto: caminhos, nomes de colunas e limiares de qualidade."""

from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parents[2]

# dados
DADOS_BRUTOS = RAIZ / "data" / "raw"
DADOS_LOTES = DADOS_BRUTOS / "lotes"
DADOS_PROCESSADOS = RAIZ / "data" / "processed"
# arquivo original da UCI (sem cabeçalho), fonte da base e da reserva dos lotes
DATASET_ORIGINAL = RAIZ / "data" / "fonte" / "magic04.data"
BASE_TREINO = DADOS_BRUTOS / "base_treino.csv"
BASE_HOLDOUT = DADOS_BRUTOS / "base_holdout.csv"
TREINO_PROCESSADO = DADOS_PROCESSADOS / "treino.csv"
AVALIACAO_PROCESSADA = DADOS_PROCESSADOS / "avaliacao.csv"

# modelos e relatórios
MODELO_CAMPEAO = RAIZ / "models" / "champion" / "model.joblib"
METRICAS_CAMPEAO = RAIZ / "models" / "champion" / "metrics.json"
MODELO_CANDIDATO = RAIZ / "models" / "candidate" / "model.joblib"
METRICAS_CANDIDATO = RAIZ / "reports" / "candidate_metrics.json"
RELATORIO_GATE = RAIZ / "reports" / "gate.json"
RESUMO_GATE = RAIZ / "reports" / "gate.md"
RELATORIO_DRIFT = RAIZ / "reports" / "drift.md"
PARAMS = RAIZ / "params.yaml"

# esquema dos dados: MAGIC Gamma Telescope (UCI, id 159, CC BY 4.0), contrato definido na issue #1
FEATURES = [
    "fLength",  # eixo maior da elipse [mm]
    "fWidth",  # eixo menor da elipse [mm]
    "fSize",  # log10 da soma do conteúdo de todos os pixels
    "fConc",  # razão (soma dos 2 pixels mais altos) / fSize
    "fConc1",  # razão (pixel mais alto) / fSize
    "fAsym",  # distância do pixel mais alto ao centro, projetada no eixo maior [mm]
    "fM3Long",  # raiz cúbica do 3º momento ao longo do eixo maior [mm]
    "fM3Trans",  # raiz cúbica do 3º momento ao longo do eixo menor [mm]
    "fAlpha",  # ângulo do eixo maior com o vetor até a origem [graus]
    "fDist",  # distância da origem ao centro da elipse [mm]
]
ALVO = "classe"
# No CSV: 1 = gamma (g, sinal, classe POSITIVA) e 0 = hádron (h, ruído de fundo).
CLASSE_POSITIVA = 1
CLASSE_MALIGNA = CLASSE_POSITIVA  # apelido temporário: evaluate/api/testes ainda usam o nome antigo
# medidas de tamanho do chuveiro: usadas no drift simulado e no teste direcional
COLUNAS_TAMANHO = ["fLength", "fWidth", "fSize"]

# limiares de qualidade (o "contrato" do modelo, issue #1). Mudar aqui exige revisão em PR.
# Erro mais caro: falso positivo (ruído aceito como gamma), por isso a precisão é a principal.
PRECISAO_MINIMA = 0.80  # métrica principal: protege contra o falso positivo
RECALL_MINIMO = 0.70  # piso: o modelo não pode "fugir" de dizer gamma só para subir a precisão
TOLERANCIA_CAMPEAO = 0.0  # o candidato não pode ter PRECISÃO menor que o campeão (gate.py, #5)
PISO_SANIDADE = 0.60  # o campeão não pode colapsar no holdout fixo (abaixo dos dois limiares)
LIMITE_DRIFT_P = 0.01  # p-valor do teste KS
MIN_FEATURES_DRIFT = 3  # gatilho do CT: quantos atributos com drift disparam o retreino


def carregar_params() -> dict:
    """Lê o params.yaml (hiperparâmetros e semente)."""
    with open(PARAMS, encoding="utf-8") as arquivo:
        return yaml.safe_load(arquivo)
