# Decisões e desvios

Decisões de implementação e desvios em relação ao [protocolo](protocol.md). Uma linha por decisão.

| Data | Decisão | Motivo |
|---|---|---|
| 2026-11-09 | Ambiente local com uv (`pyproject.toml` e `uv.lock`). No Colab, instalação com `uv pip install --system --no-sources`. | O lock garante reprodutibilidade local. No Colab, `--system` instala no Python do ambiente, que já traz torch com CUDA. `--no-sources` ignora o índice de torch CPU. |
| 2026-11-09 | `requirements.txt` serve apenas como referência de versões. É gerado pelo `uv export` sem o torch. | As versões fixas do lock local quebrariam o ambiente do Colab. As versões reais de cada execução ficam em `environment_NN.json`, no diretório de artefatos. |
| 2026-11-09 | Python 3.13. | É a versão do Colab em novembro de 2026. |
| 2026-11-09 | Nomes de arquivos e de notebooks em inglês. | Padroniza com os identificadores do código. A proposta v0.3 usava nomes em português. |
| 2026-11-09 | Código no pacote `src/juristcu`. Notebooks apenas chamam funções. | Permite testes automatizados e reduz conflitos em notebooks. |
| 2026-11-09 | Apenas `00_overview.ipynb` mantém saídas. Os demais notebooks são salvos limpos. | Diffs menores e repositório leve. O `00_overview` apresenta o estudo ao leitor. |
| 2026-11-09 | TSDAE com `SentenceTransformerTrainer`. A deleção de palavras é implementada no projeto. | O exemplo oficial do sentence-transformers apaga 40% das palavras, não 60%. O protocolo exige 60%. |
| 2026-11-09 | Ruído do TSDAE: deleção de 60% das palavras separadas por espaço, não de tokens do tokenizador. | Segue o artigo do TSDAE, que apaga palavras. Dispensa recursos do NLTK. A proposta v0.3 citava "tokens". |
| 2026-11-09 | TSDAE com agenda constante e sem decaimento de pesos. | A proposta v0.3 não fixava esses itens. Os valores seguem o exemplo de referência do TSDAE. |
| 2026-11-09 | TSDAE com fp16 e `max_seq_length` de 128. | O fp16 acelera o treino na T4, que não suporta bf16. O limite de 128 tokens reduz o uso de memória. |
| 2026-11-09 | Segmentação com o recurso `punkt_tab` do NLTK e abreviações jurídicas no Punkt (ex.: "art.", "inc."). | O NLTK 3.9 substituiu o recurso `punkt` por `punkt_tab`. Sem as abreviações, "art. 25" quebraria a frase. |
| 2026-11-09 | Avisos de pesos ausentes ao carregar o decodificador do TSDAE são esperados. | O BERTimbau não traz as camadas de atenção cruzada do decodificador. Essas camadas são inicializadas e treinadas pelo TSDAE. |
| 2026-11-09 | Proposta v0.3 convertida em protocolo (`docs/protocol.md`). | Documento curto e em tópicos, fonte única do estudo. |
| 2026-11-09 | Proteção da `main` por ruleset, sem exceção para o dono. | Garante PR e CI verde em toda mudança. Rulesets valem em repositório público no plano gratuito. |
| 2026-11-11 | `doc.csv` lido só com as colunas usadas, todas como texto. O carregamento confere as colunas e as 16.045 linhas. | Evita que a inferência de tipos do `datasets` varie entre blocos do arquivo. Um separador errado gera erro claro. |
| 2026-11-11 | Limpeza antes da segmentação (`data.clean_text`): remove etiquetas e entidades HTML e troca quebras de linha, reais ou escritas como texto, por espaço. | O artigo do JurisTCU mostra `<b>` no ENUNCIADO e `\\n` no EXCERTO. Sem a limpeza, essas marcas entram nas frases do TSDAE. |
| 2026-11-11 | Limpeza revista após a inspeção manual: etiquetas de bloco viram espaço; as de formatação são removidas sem espaço; `[...]` é removido; o espaço antes de pontuação é retirado. | 99,6% dos EXCERTOS têm HTML. Trocar toda etiqueta por espaço gerava "bens , por" e "Plenário ) .". |
| 2026-11-11 | Corpus segmentado por parágrafo (`data.split_paragraphs`): etiquetas de bloco e números de parágrafo colados (ex.: "8.Este", "20.Quando") marcam fronteiras; o número é descartado. Números de item no início do parágrafo (ex.: "2. No âmbito", "9.3.9. cadastre") também são removidos. | O Punkt não separava "técnicos: 20.Quando" e juntava duas frases. As duas primeiras execuções mostraram os padrões. |
| 2026-11-11 | Abreviações jurídicas do Punkt ficam em `config.LEGAL_ABBREVIATIONS`. | Constantes ficam apenas em `config.py`. A lista pode crescer após a inspeção manual do corpus. |
| 2026-11-11 | Corpus: frases extraídas documento a documento (ENUNCIADO, depois EXCERTO); amostra devolvida na ordem sorteada; SHA-256 do `sentences.json` registrado em `corpus_funnel.csv`. | A ordem fixa torna a amostra reprodutível. O hash permite conferir que todos treinam com o mesmo corpus. |
| 2026-11-12 | Perda do TSDAE registrada a cada 50 passos (`TsdaeConfig.logging_steps`) e gravada em `loss_history.json`, na pasta de cada modelo. | O protocolo monitora apenas a perda de treino. O arquivo junto do modelo preserva a curva se a sessão cair e alimenta a tabela e a figura da perda. |
| 2026-11-15 | Fora do Colab, o corpus só é reproduzido com o NLTK 3.9.1 ou 3.9.2 (SHA-256 igual ao da Parte 1). | Com o NLTK 3.10.4 do lock local, a segmentação gera 1 frase a mais em cada etapa e muda toda a amostra. |
| 2026-11-16 | Modelos do TSDAE treinados em um Mac com Apple M5 (MPS, fp32) descartados. O treino final usa o Colab com GPU T4 e fp16, como no protocolo, antes da tag `preregistro`. | No Mac, nas 3 sementes, a perda caiu até cerca de 3,2 e subiu até cerca de 6,4 no fim, com picos de gradiente. No Colab, as 3 sementes terminaram estáveis, com perda final de 2,65, 2,66 e 2,65. O TSDAE não usa rótulos e segue os hiperparâmetros do protocolo, então treinar antes da tag não afeta a avaliação. |
| 2026-11-16 | Aprovado pelo grupo: R2-CLS (BERTimbau sem ajuste, pooling CLS) como controle exploratório de H2. | R3 usa CLS e R2 usa média. A diferença R3 − R2 mistura TSDAE e pooling. R3 − R2-CLS isola o TSDAE. |
| 2026-11-16 | Aprovado pelo grupo: na análise por tema, filtro dos temas com ao menos 20 documentos antes do k-means; k igual ao número de temas mantidos (144); mesmo bootstrap. | A proposta v0.3 não definia k nem o momento do filtro. Filtrar antes evita avaliar documentos de temas excluídos. |
| 2026-11-16 | Aprovado pelo grupo: amostra de 20 mil frases única para as 3 sementes do TSDAE. | A variação entre sementes passa a medir apenas o treinamento. |
| 2026-11-16 | Aprovado pelo grupo: agenda de aprendizado constante, sem decaimento de pesos. | Segue o exemplo de referência do TSDAE. A proposta v0.3 não fixava esses itens. |
| 2026-11-16 | Aprovado pelo grupo: `data.clean_text` aplicado também ao ENUNCIADO, antes de R1 a R4. | Remove a marcação HTML de 3,6% dos enunciados. Vale para todas as representações, então não favorece nenhuma. |
| 2026-11-16 | Protocolo congelado na tag `preregistro`. Mudanças posteriores entram apenas neste arquivo. | Fixa hipóteses, métricas e critérios antes de qualquer avaliação com rótulos. |
| 2026-11-23 | R3 usa os modelos do TSDAE treinados no Colab em 16/11/2026, salvos no Drive. O notebook `02_embeddings` só treina uma semente se o modelo dela faltar. | Evita sobrescrever os modelos finais e gastar cerca de 30 minutos de GPU a cada execução. O treino não muda: mesmas sementes, corpus e hiperparâmetros. |
