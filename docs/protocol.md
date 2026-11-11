# Protocolo do estudo

Agrupamento temático da jurisprudência do TCU com representações aprendidas sem rótulos.

| Item | Valor |
|---|---|
| Status | Rascunho. Será congelado na tag `preregistro`, antes de qualquer experimento. |
| Origem | Proposta v0.3, de 5 de novembro de 2026. |
| Disciplina | Aprendizado de Máquina Não Supervisionado (UFG) |
| Dados | JurisTCU, `LeandroRibeiro/JurisTCU`, licença CC BY 4.0 |
| Recursos | Google Colab com GPU T4 |
| Entregáveis | Apresentação oral e este repositório |

Regras do documento:

- Define o que será medido e como cada resultado será interpretado.
- Após o congelamento, não é editado.
- Desvios posteriores vão para [decisions.md](decisions.md), com data e motivo.

## 1. Objetivo

- Comparar representações de texto no agrupamento temático dos enunciados do TCU.
- Medir se o treinamento sem rótulos no próprio acervo (TSDAE) melhora um modelo em português.
- Usar as áreas do TCU apenas como referência de avaliação.

## 2. Perguntas

- **Principal.** Quais representações agrupam melhor a jurisprudência do TCU por tema, sem rótulos?
- **P1.** Representações neurais agrupam melhor que TF-IDF com SVD?
- **P2.** O TSDAE no próprio acervo melhora o BERTimbau?
- **P3 (exploratória).** Quantos grupos a silhueta sugere? Esse número se aproxima do número de áreas?

## 3. Hipóteses

Métrica principal: NMI entre os grupos do k-means e as áreas.

| ID | Hipótese | Previsão | Critério de confirmação |
|---|---|---|---|
| H1 | Representações neurais agrupam melhor que a lexical clássica. | NMI de R3 > R1 e NMI de R4 > R1. | IC de 95% de cada diferença com limite inferior positivo. |
| H2 | O TSDAE no próprio acervo melhora o BERTimbau. | NMI de R3 > R2. | IC de 95% da diferença com limite inferior positivo. |

- IC por bootstrap pareado dos documentos.
- R3 entra pela média das 3 sementes.
- ARI e silhueta apoiam a interpretação.

## 4. Dados

- Arquivo `doc.csv` do JurisTCU: 16.045 documentos, até junho de 2023.
- Carregar o arquivo diretamente. O visualizador do Hugging Face falha com este conjunto.

| Campo | Uso |
|---|---|
| ENUNCIADO | Texto representado e agrupado (cerca de 47 palavras). |
| EXCERTO | Fonte adicional de frases para o TSDAE (cerca de 660 palavras). |
| AREA | Referência principal de avaliação. |
| TEMA | Referência secundária de avaliação. |

A análise exploratória levanta o número de áreas e de temas e o equilíbrio entre eles.

## 5. Corpus de frases sem rótulo

- Fonte: ENUNCIADO e EXCERTO, sem rótulos nem metadados.
- Segmentação: NLTK para português, com abreviações jurídicas.
- Filtro: frases de 5 a 80 palavras.
- Limpeza: remoção de duplicatas.
- Amostra: 20 mil frases, semente fixa, a mesma para as 3 sementes do TSDAE.

## 6. Representações

Todas usam o ENUNCIADO e recebem normalização L2 antes do agrupamento.

| ID | Representação | Papel |
|---|---|---|
| R1 | TF-IDF com SVD truncado em 300 dimensões (LSA) | Referência clássica |
| R2 | BERTimbau base, pooling pela média, sem ajuste | Limite inferior neural |
| R2-CLS | BERTimbau base, pooling CLS, sem ajuste | Controle exploratório de H2 |
| R3 | BERTimbau base + TSDAE, pooling CLS | Único treinamento, sem rótulos |
| R4 | multilingual-e5-small, prefixo `query: ` | Modelo geral pronto |

- R2-CLS isola o efeito do TSDAE. R3 usa CLS; R2 usa média.
- A comparação R3 × R2-CLS é exploratória e não altera a decisão sobre H2.

## 7. Treinamento com TSDAE

| Item | Valor |
|---|---|
| Modelo inicial | BERTimbau base (cerca de 110 milhões de parâmetros) |
| Ruído | Deleção de 60% das palavras de cada frase |
| Arquitetura | Codificador e decodificador com pesos compartilhados |
| Pooling | Token CLS |
| Otimização | AdamW, taxa 3e-5, lote 8, agenda constante |
| Duração | 1 época sobre 20 mil frases (cerca de 2.500 passos) |
| Repetições | Sementes 0, 1 e 2 |
| Implementação | `DenoisingAutoEncoderLoss`, do sentence-transformers |

- Configuração recomendada pelos autores do método.
- Hiperparâmetros fixos. Apenas a perda de treino é monitorada.

## 8. Agrupamento e avaliação

| Etapa | Definição |
|---|---|
| Agrupamento principal | k-means++, 10 reinícios, k = número de áreas |
| Validação externa | NMI (principal) e ARI em relação a AREA |
| Validação interna | Coeficiente de silhueta |
| Análise por tema | Mantém temas com ao menos 20 documentos; novo k-means com k = número de temas mantidos |
| Escolha de k sem rótulos (P3) | k de 5 a 50; maior silhueta |
| Incerteza | Bootstrap dos documentos, 1.000 reamostragens, IC de 95% |
| Visualização | UMAP em 2 dimensões, colorido por área |
| Interpretação | Termos característicos de cada grupo (TF-IDF por grupo) |
| Extra opcional | HDBSCAN sobre a projeção UMAP |

- A análise por tema filtra os documentos antes do k-means.
- O bootstrap reamostra as atribuições já obtidas. Não refaz o agrupamento.

## 9. Plano de análise

- **Tabela principal:** cada representação com NMI (IC 95%), ARI e silhueta, por área e por tema.
- **Tabela de hipóteses:** H1 e H2 com diferença observada, IC e decisão.
- **Figura 1:** projeções UMAP lado a lado, coloridas por área.
- **Figura 2:** silhueta em função de k, por representação.
- **Análise qualitativa:** termos característicos de alguns grupos comparados às áreas.

## 10. Interpretação dos cenários

| Cenário | Interpretação |
|---|---|
| H1 confirmada | Representações neurais capturam a organização temática melhor que a contagem de palavras. |
| H1 não confirmada | No vocabulário jurídico, a sobreposição de termos já organiza o acervo por área. |
| H2 confirmada | O treinamento sem rótulos no próprio acervo agrega valor ao modelo em português. |
| H2 não confirmada | Os ganhos do TSDAE em outros domínios não se transferem para a jurisprudência do TCU. |
| R3 próximo de R4 | Um modelo adaptado só com o acervo se aproxima de um modelo geral treinado com muito mais dados. |

## 11. Controles de validade

- Áreas e temas não entram no treinamento nem na escolha de hiperparâmetros.
- O TSDAE usa frases dos documentos avaliados, nunca os rótulos.
- Todas as representações usam o mesmo texto, a mesma normalização e o mesmo k.
- Sementes fixas. Versões das bibliotecas registradas a cada execução.
- Resultados contrários às hipóteses são reportados como tal.

## 12. Escopo

Dentro:

- Representações da seção 6 e hipóteses H1 e H2.
- Corpus de frases e TSDAE com 3 sementes.
- k-means, validação externa e interna, silhueta por k, UMAP e análise qualitativa.

Fora:

- Busca semântica, consultas e julgamentos de relevância.
- Qualquer treinamento com rótulos.
- SimCSE, consultas sintéticas e modelos com mais de 1 bilhão de parâmetros.

Ordem de corte, em caso de atraso no cronograma:

1. HDBSCAN.
2. Análise por tema.
3. Mínimo: R1 a R4 com k-means, NMI, ARI e silhueta por área.

## 13. Limitações

- Área e tema são uma organização possível do acervo. NMI baixa não prova, sozinha, uma representação ruim.
- Fixar k pelo número de áreas usa informação dos rótulos. Por isso, a escolha de k pela silhueta é reportada junto.
- O multilingual-e5-small foi treinado com dados rotulados gerais. R4 não é puramente não supervisionado.
- O treinamento (cerca de 2.500 passos) é menor que o do artigo do TSDAE. Um resultado nulo em H2 tem alcance limitado.
- Um único domínio e apenas o campo ENUNCIADO.

## 14. Riscos

| Risco | Impacto | Mitigação |
|---|---|---|
| Queda da sessão do Colab | Perda de processamento | Representações e modelos salvos no Drive a cada etapa |
| TSDAE sem ganho | H2 não confirmada | Resultado válido; a análise qualitativa ajuda a explicá-lo |
| Áreas desbalanceadas | Métricas distorcidas | NMI e ARI juntos, com a distribuição das áreas |
| Frases ruidosas no corpus | Treinamento menos eficaz | Filtros e inspeção manual de uma amostra |
| Atraso no cronograma | Projeto incompleto | Ordem de corte da seção 12 |

## 15. Pendências para o congelamento

Mudanças em relação à proposta v0.3. O grupo aprova antes da tag `preregistro`.

- Inclusão de R2-CLS como controle exploratório.
- Protocolo da análise por tema: filtro antes do k-means e k próprio.
- Amostra de frases única para as 3 sementes do TSDAE.
- Agenda de aprendizado constante, sem decaimento de pesos.
- Limpeza de HTML e de quebras de linha também no ENUNCIADO, antes de todas as representações.

## 16. Integrantes

- André Guilherme Alves do Carmo
- Gabriel Palhares Siqueira
- Igor Reis Braziel
- João Henrique Simielli
- Matheus Felipe de Borba Machado Vieira

## 17. Referências

- CAMPELLO, R. J. G. B.; MOULAVI, D.; SANDER, J. Density-based clustering based on hierarchical density estimates. PAKDD, 2013.
- DEERWESTER, S. et al. Indexing by latent semantic analysis. Journal of the American Society for Information Science, v. 41, n. 6, p. 391–407, 1990.
- FERNANDES, L. C. et al. JurisTCU: a Brazilian Portuguese information retrieval dataset with query relevance judgments. Language Resources and Evaluation, 2026. DOI: 10.1007/s10579-025-09881-w.
- HUBERT, L.; ARABIE, P. Comparing partitions. Journal of Classification, v. 2, n. 1, p. 193–218, 1985.
- LLOYD, S. Least squares quantization in PCM. IEEE Transactions on Information Theory, v. 28, n. 2, p. 129–137, 1982.
- McINNES, L.; HEALY, J.; MELVILLE, J. UMAP: uniform manifold approximation and projection for dimension reduction. arXiv:1802.03426, 2018.
- NEUROMATCH ACADEMY. Example Deep Learning Project. Disponível em: https://deeplearning.neuromatch.io/projects/modelingsteps/Example_Deep_Learning_Project.html.
- REIMERS, N.; GUREVYCH, I. Sentence-BERT: sentence embeddings using Siamese BERT-networks. EMNLP-IJCNLP, 2019.
- ROUSSEEUW, P. J. Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. Journal of Computational and Applied Mathematics, v. 20, p. 53–65, 1987.
- SOUZA, F.; NOGUEIRA, R.; LOTUFO, R. BERTimbau: pretrained BERT models for Brazilian Portuguese. BRACIS, 2020.
- STREHL, A.; GHOSH, J. Cluster ensembles: a knowledge reuse framework for combining multiple partitions. JMLR, v. 3, p. 583–617, 2002.
- VINCENT, P. et al. Extracting and composing robust features with denoising autoencoders. ICML, 2008.
- WANG, K.; REIMERS, N.; GUREVYCH, I. TSDAE: using Transformer-based sequential denoising auto-encoder for unsupervised sentence embedding learning. Findings of EMNLP, 2021.
- WANG, L. et al. Multilingual E5 text embeddings: a technical report. arXiv:2402.05672, 2024.
