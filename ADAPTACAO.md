# Adaptação do projeto: MAGIC Gamma Telescope

Equipe: Áquila, Eduardo, Letícia, Luiz, Rafael e William.

## 1. Problema e dados

O telescópio MAGIC registra milhares de chuveiros de partículas. O modelo separa os eventos que vêm de raios gamma, que são os que interessam para a física, do ruído de fundo causado por hádrons. A ideia é filtrar os eventos antes de os pesquisadores analisarem.

- **Fonte:** UCI Machine Learning Repository, dataset 159 (https://archive.ics.uci.edu/dataset/159).
- **Licença:** CC BY 4.0. São dados simulados (Monte Carlo), sem nenhum dado pessoal.
- **Tamanho:** 19.020 linhas, 10 atributos numéricos e nenhum valor faltando. Tiramos 115 linhas duplicadas antes de dividir os dados.
- **Classes:** `g` = gamma (12.332 linhas) e `h` = hádron (6.688 linhas). No CSV ficou `g` = 1 e `h` = 0.
- **No repositório:** treino com 3.000 linhas e holdout com 1.000, divididos de forma estratificada com semente fixa. O arquivo original fica em `data/fonte/`, para o CI não depender de internet.

Escolhemos esse dataset porque `fLength`, `fWidth` e `fSize` são medidas de tamanho e entraram direto no lugar das colunas de tamanho do projeto original, que são usadas no drift e no teste direcional.

## 2. Contrato de qualidade

- **Classe positiva:** gamma.
- **Erro mais caro:** o falso positivo, ou seja, aceitar ruído como se fosse gamma. Isso coloca um evento falso na análise. A própria UCI diz que classificar fundo como sinal é pior que o contrário. Um falso negativo só perde um evento no meio de milhares.
- **Precisão (gamma) ≥ 0,80.** É a métrica principal, porque protege contra o erro caro.
- **Recall (gamma) ≥ 0,70.** Serve de piso, para o modelo não parar de dizer gamma só para subir a precisão.
- **Contra o campeão:** o candidato não pode ter precisão menor que a do campeão, medida no mesmo conjunto. No projeto original essa comparação era por recall.
- **Gatilho de retreino:** pelo menos 2 dos 10 atributos com p < 0,01 no teste KS. Um atributo sozinho pode ser só variação da amostra.

Mantivemos a regressão logística com `class_weight: balanced`. Ela passou no contrato e não foi preciso mexer nos limiares.

## 3. O que mudou, arquivo por arquivo

**Dados e contrato (#3, PR #10)**
- `data/raw/base_treino.csv` e `base_holdout.csv`: trocados pelos dados do MAGIC.
- `data/fonte/`: o `magic04.data` e o `magic04.names` originais.
- `src/mlops_demo/data.py`: saiu o `load_breast_cancer`. Agora ele lê o arquivo da UCI, tira as duplicatas antes de dividir e recusa a base se alguma linha do holdout também estiver no treino.
- `src/mlops_demo/config.py`: as 10 features, `ALVO = "classe"`, a classe positiva, as colunas de tamanho e os limiares do contrato.

**Validação e drift (#4, PR #11)**
- `src/mlops_demo/data.py`: o `validar_dataframe` agora aceita valores negativos onde eles fazem sentido (`fAsym`, `fM3Long` e `fM3Trans`) e confere regras do domínio, como concentração entre 0 e 1 e ângulo entre 0 e 90.
- `src/mlops_demo/monitor.py`: o `aplicar_drift` multiplica `fLength` e `fWidth` pelo fator. No `fSize`, que já está em log10, ele soma `log10(fator)`.
- `tests/test_magic_drift.py` e `tests/test_validacao_dataframe.py`: testes novos para essas regras.

**Modelo, métricas e campeão (#5, PR #12)**
- `src/mlops_demo/evaluate.py`: mede precisão e recall da classe gamma.
- `src/mlops_demo/gate.py`: a terceira regra compara a precisão com a do campeão, e não mais o recall.
- `models/champion/`: o campeão inicial.

**API (#6, PR #13)**
- `src/mlops_demo/api.py`: recebe 10 atributos em vez de 30 e responde `gamma` ou `hadron`, com `probabilidade_gamma`.
- `demo/smoke_test.py` e `tests/test_api.py`: adaptados, com um teste novo que manda atributos de menos e de mais e espera o erro 422.

**Testes e cenários (#7, PR #14)**
- `tests/test_codigo.py`, `test_dados.py`, `test_modelo.py` e `test_api.py`: adaptados ao MAGIC, com um teste novo por camada. O teste direcional passou a usar o `fAlpha`: quanto maior o ângulo, menor deve ser a probabilidade de gamma.
- `demo/cenarios.py`: os cenários 1 a 5 refeitos. O cenário 5 gera recall alto com precisão abaixo do contrato.

**Treinamento contínuo (#8, PRs #15 e #16)**
- `data/raw/lotes/lote_drift.csv`: lote de 120 linhas gerado pelo `demo/gerar_lote.py` com drift de 0,75.
- `models/champion/`: novo campeão proposto pelo CT e aceito por nós.

Os PRs #10 a #13 entraram com o CI vermelho. Isso aconteceu porque os testes do projeto original, que esperavam 30 atributos e a classe "maligno", só foram adaptados na última etapa (#14). Depois dela a `main` ficou verde. Se fôssemos refazer, adaptaríamos os testes junto com cada etapa, para cada PR já entrar verde.

## 4. Evidências

| | O que mostra | Link |
|---|---|---|
| E1 | CI verde na `main` depois da adaptação | https://github.com/4SemestreDonaduzzi/lab04-cicd-ct/actions/runs/45747600570 |
| E2 | CI vermelho controlado: cenário 3 (modelo ruim), falha no step "Testes de modelo" | https://github.com/4SemestreDonaduzzi/lab04-cicd-ct/actions/runs/45934458519 (PR #18) |
| E3 | CT com drift e portão aprovando | https://github.com/4SemestreDonaduzzi/lab04-cicd-ct/actions/runs/45914351925 (PR do campeão: #16) |
| E4 | CT com o portão reprovando e o campeão sem mudar | https://github.com/4SemestreDonaduzzi/lab04-cicd-ct/actions/runs/45941109652 (PR #20) |
| E5 | CD verde com a tag `v1.0.0` e imagem no ghcr.io | https://github.com/4SemestreDonaduzzi/lab04-cicd-ct/actions/runs/45941975254 |

Página do pacote (pública): https://github.com/orgs/4SemestreDonaduzzi/packages/container/package/lab04-cicd-ct

Para testar: `docker run -p 8000:8000 ghcr.io/4semestredonaduzzi/lab04-cicd-ct:1.0.0` e depois `python demo/smoke_test.py http://localhost:8000`.

Na E3, o drift apareceu em 3 dos 10 atributos (`fWidth`, `fLength` e `fSize`), o modelo foi retreinado e o portão aprovou o candidato, com precisão de 0,842 contra 0,840 do campeão. Como a `PROMOCAO_AUTOMATICA` está `false`, o CT abriu um PR e uma pessoa fez o merge.

Na E4 usamos o cenário 5, com `class_weight` extremo para a classe gamma. O drift disparou de novo, mas o candidato chegou a recall de 0,999 com precisão de só 0,671, abaixo do mínimo de 0,80 e do campeão (0,842). O portão reprovou e o CT ficou vermelho. A prova de que o campeão não mudou é que o `models/champion/metrics.json` na `main` continua com precisão 0,842, nenhum PR de novo campeão foi aberto e não saiu nenhuma tag nova.

## 5. Limites e riscos

**O portão aprova pela diferença, mesmo quando ela é pequena demais.** Na E3 o candidato ganhou por 0,842 a 0,840. Em 1.060 amostras isso dá mais ou menos 2 acertos de diferença, o que está dentro do que pode variar só pela amostra. O portão não sabe disso e trocou o campeão mesmo assim. Uma margem mínima, ou um intervalo de confiança, deixaria essa decisão mais segura.

**O holdout não representa o mundo real.** A UCI avisa que a classe `h` está subestimada nesse dataset. Nos dados reais do telescópio o ruído é a maioria. Com mais hádrons, a mesma taxa de erro gera mais falsos positivos, então a precisão que medimos aqui é otimista.

**O portão só vê o que está no holdout.** Ele não percebe se aparece um tipo de evento que não existia na base, nem se o próprio drift muda a relação entre os atributos e a classe. O teste KS olha um atributo por vez e não pega mudanças que só aparecem na combinação deles.

**Em produção a gente monitoraria:**
- a proporção de eventos classificados como gamma ao longo do tempo;
- o drift dos atributos, que já fazemos, e também da probabilidade que o modelo devolve;
- a precisão real, quando os pesquisadores confirmarem uma amostra dos eventos;
- o tempo de resposta e os erros 422 da API, que mostram se os dados estão chegando no formato errado.

## 6. Contribuição de cada integrante e uso de IA

| Integrante | O que fez | Uso de IA (onde e para quê) |
|---|---|---|
| Áquila | Escolha do dataset e contrato (#1), criação do repo e primeiro CI (#2), organização das issues, correção do lote de drift (#15), merge do campeão (#16), imagem pública e este relatório | Claude Code para escrever as issues, revisar os PRs, corrigir o lote de drift e montar a primeira versão deste relatório. Revisei o texto e as mudanças |
| Luiz | Dados e contrato no `config.py` (#3, PR #10) e API com smoke test (#6, PR #13) | Claude Code para baixar e analisar o dataset (foi assim que apareceram as 115 duplicatas), escrever as mudanças em `data.py` e `config.py`, adaptar a API, o smoke test e os testes da API, e rodar os testes localmente |
| Rafael | Validação dos dados e drift (#4, PR #11) | OpenAI Codex para implementar a validação e o drift, escrever e rodar os testes e redigir o registro da issue |
| Eduardo | Métricas, portão e campeão inicial (#5, PR #12), testes e cenários (#7, PR #14) | IA para adaptar os testes e os cenários das quatro camadas ao MAGIC e conferir os pesos que geram o caso de precisão baixa |
| Letícia | Evidências E2, E4 e E5 (PRs #18 a #20 e a tag `v1.0.0`) | IA para estruturar e revisar a resposta da pergunta-chave e organizar o checklist de evidências |
| William | Primeiro lote de drift (PR #15) | Não informou |
