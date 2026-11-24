# Agrupamento temático da jurisprudência do TCU com representações aprendidas sem rótulos

Este estudo compara representações de texto para agrupar por tema os 16.045 enunciados do JurisTCU, sem uso de rótulos.
Os grupos do k-means são avaliados com NMI, ARI e silhueta; as áreas atribuídas pelo TCU entram apenas na avaliação.

| Notebook | Conteúdo | Selo |
|---|---|---|
| [`00_overview`](notebooks/00_overview.ipynb) | Estudo completo a partir dos artefatos salvos | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Igorbraziel/juristcu-clustering-tsdae/blob/main/notebooks/00_overview.ipynb) |
| [`01_data_corpus`](notebooks/01_data_corpus.ipynb) | Análise exploratória e corpus de frases sem rótulo | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Igorbraziel/juristcu-clustering-tsdae/blob/main/notebooks/01_data_corpus.ipynb) |
| [`02_embeddings`](notebooks/02_embeddings.ipynb) | Representações R1 a R4 e treinamento do TSDAE | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Igorbraziel/juristcu-clustering-tsdae/blob/main/notebooks/02_embeddings.ipynb) |
| [`03_clustering`](notebooks/03_clustering.ipynb) | k-means, silhueta por k, UMAP e HDBSCAN | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Igorbraziel/juristcu-clustering-tsdae/blob/main/notebooks/03_clustering.ipynb) |
| [`04_evaluation`](notebooks/04_evaluation.ipynb) | Métricas, testes de hipóteses e figuras | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Igorbraziel/juristcu-clustering-tsdae/blob/main/notebooks/04_evaluation.ipynb) |

O protocolo do estudo está em [docs/protocol.md](docs/protocol.md). Ele foi congelado em 16/11/2026, na tag `preregistro`, antes dos experimentos.
Desvios e decisões de implementação ficam em [docs/decisions.md](docs/decisions.md).

## 1. Pergunta de pesquisa

**Pergunta principal:** quais representações de texto agrupam melhor a jurisprudência do TCU por tema, sem o uso de rótulos?

Perguntas específicas:

- **P1.** Representações neurais agrupam melhor que a representação lexical clássica (TF-IDF com SVD)?
- **P2.** O treinamento não supervisionado no próprio acervo (TSDAE) melhora um modelo de linguagem em português?
- **P3 (exploratória).** Sem consultar as áreas, quantos grupos o coeficiente de silhueta sugere? Esse número se aproxima do número de áreas?

| ID | Hipótese | Previsão | Critério de confirmação |
|---|---|---|---|
| H1 | Representações neurais agrupam por tema melhor que a lexical clássica. | NMI de R3 e de R4 maior que a de R1. | IC de 95% de cada diferença, por bootstrap dos documentos, com limite inferior positivo. |
| H2 | O TSDAE no próprio acervo melhora o BERTimbau. | NMI de R3 maior que a de R2. | IC de 95% da diferença, por bootstrap dos documentos, com limite inferior positivo. |

- A métrica principal é a NMI entre os grupos do k-means e as áreas do TCU.
- ARI e silhueta são reportados como apoio à interpretação.
- Para R3, usa-se a média de 3 sementes de treinamento.

## 2. Motivação

- O TCU disponibiliza mais de 500 mil documentos produzidos desde 1992 (FERNANDES et al., 2026).
- Organizar o acervo por tema facilita a consulta a precedentes e revela assuntos recorrentes.
- Rotular documentos jurídicos é caro, e dados rotulados em português são escassos. Texto sem rótulo, por outro lado, é abundante.
- O artigo do JurisTCU avalia apenas busca. Não há análise de agrupamentos do acervo.

## 3. Metodologia

**Dados**

- Arquivo `doc.csv` do JurisTCU (`LeandroRibeiro/JurisTCU`, Hugging Face), com 16.045 documentos.
- ENUNCIADO: texto representado e agrupado.
- EXCERTO: fonte adicional de frases para o TSDAE.
- AREA e TEMA: usados apenas na avaliação.
- Corpus sem rótulo: 20 mil frases de ENUNCIADO e EXCERTO, com 5 a 80 palavras, sem duplicatas e com semente fixa.

**Representações**

| ID | Representação | Papel no estudo |
|---|---|---|
| R1 | TF-IDF com SVD truncado em 300 dimensões (LSA) | Referência clássica, sem rede neural |
| R2 | BERTimbau base, pooling pela média, sem ajuste | Limite inferior neural |
| R2-CLS (exploratória) | BERTimbau base, pooling pelo token CLS, sem ajuste | Controle de pooling para H2 |
| R3 | BERTimbau base + TSDAE no corpus de frases (pooling CLS) | Único treinamento do projeto, sem rótulos |
| R4 | multilingual-e5-small pronto (prefixo `query:`) | Modelo geral de referência |

- Todas as representações usam o ENUNCIADO e recebem normalização L2.
- R2-CLS isola o efeito do TSDAE, pois R3 usa CLS e R2 usa média. É um controle exploratório.

**TSDAE:** apaga 60% das palavras de cada frase. O BERTimbau aprende a reconstruir a frase a partir de um único vetor. Configuração: lote 8, taxa 3e-5, 1 época e 3 sementes.

**Agrupamento e avaliação**

- O ENUNCIADO é limpo por `data.clean_text` (marcação HTML e quebras de linha) antes de todas as representações.
- k-means com inicialização k-means++ e 10 reinícios; k igual ao número de áreas.
- Validação externa: NMI (principal) e ARI em relação a AREA.
- Análise secundária por TEMA: os temas com ao menos 20 documentos são filtrados antes do k-means. O k é igual ao número de temas mantidos (144).
- Validação interna: coeficiente de silhueta.
- P3: k de 5 a 50, com escolha do k de maior silhueta.
- Incerteza: IC de 95% por bootstrap pareado dos documentos (1.000 reamostragens). O bootstrap reamostra as atribuições já obtidas e não refaz o k-means.
- Visualização: projeção UMAP em 2 dimensões, colorida por área.
- Interpretação: termos característicos de cada grupo (TF-IDF por grupo).
- Extra opcional: HDBSCAN sobre a projeção UMAP.

## 4. Resultados

A preencher após os experimentos.

Tabela principal, em relação a AREA:

| Representação | NMI [IC 95%] | ARI | Silhueta |
|---|---|---|---|
| R1 | — | — | — |
| R2 | — | — | — |
| R2-CLS (exploratória) | — | — | — |
| R3 (média de 3 sementes) | — | — | — |
| R4 | — | — | — |

Plano de análise: [protocolo](docs/protocol.md#9-plano-de-análise).

## 5. Conclusões

A preencher.

Os critérios de interpretação foram definidos antes dos experimentos, no [protocolo](docs/protocol.md#10-interpretação-dos-cenários).

## Resumo final

Formato do passo 10 do [exemplo de projeto do Neuromatch Academy](https://deeplearning.neuromatch.io/projects/modelingsteps/Example_Deep_Learning_Project.html).

A preencher.

## Como executar

### No Colab

1. Abrir o notebook pelo selo da tabela no início deste arquivo.
2. Executar a célula de configuração. Ela clona o repositório, instala o pacote e monta o Google Drive (exceto no 04).
3. Seguir a ordem 01 → 02 → 03 → 04 → 00.
4. O notebook 02 exige GPU: Ambiente de execução → Alterar o tipo de ambiente de execução → GPU T4.

Observações:

- O 03 não requer GPU, mas a varredura de k leva várias horas em CPU.
- O grupo executou o 03 localmente. Ver [docs/decisions.md](docs/decisions.md).
- As tabelas do 03 ficam versionadas em `results/tables/`. O 04 lê apenas essas tabelas e não precisa do Drive.

Artefatos no Google Drive:

- Representações, modelos treinados e corpus de frases ficam no Drive, fora do git.
- Esses artefatos estão na pasta `juristcu`, compartilhada entre os integrantes.
- O código procura a pasta em `/content/drive/MyDrive/juristcu`.
- Quem não é o dono da pasta deve criar um atalho dela em "Meu Drive" ou definir a variável de ambiente `JURISTCU_ARTIFACTS` com o caminho da pasta.
- Localmente, o padrão é `./artifacts`. A variável `JURISTCU_ARTIFACTS` também substitui esse caminho.

### Localmente

```bash
uv sync                      # cria o ambiente (exige uv >= 0.13)
uv run pytest                # executa os testes
uv run pre-commit install    # instala os hooks do git
```

> **Aviso.** O `requirements.txt` é apenas referência das versões do ambiente local.
> Não usar no Colab.

Fluxo de contribuição: [CONTRIBUTING.md](CONTRIBUTING.md).

## Estrutura do repositório

```text
juristcu-clustering-tsdae/
├── README.md, CONTRIBUTING.md, CLAUDE.md, LICENSE
├── pyproject.toml, uv.lock    ambiente local (uv)
├── requirements.txt           referência de versões (não usar no Colab)
├── .github/                   CI e ruleset da main
├── docs/
│   ├── protocol.md            protocolo do estudo
│   └── decisions.md           decisões e desvios
├── src/juristcu/              pacote do estudo
├── notebooks/                 00_overview, 01_data_corpus, 02_embeddings, 03_clustering,
│                              04_evaluation
├── tests/                     testes sem GPU e sem download
└── results/
    ├── tables/                tabelas finais
    └── figures/               figuras finais
```

## Dados e licença

- Dados: [JurisTCU](https://huggingface.co/datasets/LeandroRibeiro/JurisTCU), sob licença CC BY 4.0.
- Os dados não são versionados neste repositório.
- Ao usar os dados, citar:
  FERNANDES, L. C.; RIBEIRO, L. S.; CASTRO, M. V. B.; PACHECO, L. A. S.; SANDES, E. F. O.
  JurisTCU: a Brazilian Portuguese information retrieval dataset with query relevance judgments.
  *Language Resources and Evaluation*, 2026.
  DOI: [10.1007/s10579-025-09881-w](https://doi.org/10.1007/s10579-025-09881-w).
- Código: licença MIT ([LICENSE](LICENSE)).

## Integrantes

- André Guilherme Alves do Carmo
- Gabriel Palhares Siqueira
- Igor Reis Braziel
- João Henrique Simielli
- Matheus Felipe de Borba Machado Vieira

Disciplina: Aprendizado de Máquina Não Supervisionado.
Instituição: Universidade Federal de Goiás (UFG).
