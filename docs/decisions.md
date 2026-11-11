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
| 2026-11-11 | Abreviações jurídicas do Punkt ficam em `config.LEGAL_ABBREVIATIONS`. | Constantes ficam apenas em `config.py`. A lista pode crescer após a inspeção manual do corpus. |
| 2026-11-11 | Corpus: frases extraídas documento a documento (ENUNCIADO, depois EXCERTO); amostra devolvida na ordem sorteada; SHA-256 do `sentences.json` registrado em `corpus_funnel.csv`. | A ordem fixa torna a amostra reprodutível. O hash permite conferir que todos treinam com o mesmo corpus. |

## Pendências para o congelamento

O grupo aprova estes itens antes da tag `preregistro`. Lista também no [protocolo](protocol.md#15-pendências-para-o-congelamento).

| Proposta | Motivo |
|---|---|
| Incluir R2-CLS (BERTimbau sem ajuste, pooling CLS) como controle exploratório de H2. | R3 usa CLS e R2 usa média. A diferença R3 − R2 mistura TSDAE e pooling. R3 − R2-CLS isola o TSDAE. |
| Análise por tema: filtrar temas com ao menos 20 documentos antes do k-means; k igual ao número de temas mantidos; mesmo bootstrap. | A proposta v0.3 não definia k nem o momento do filtro. Filtrar antes evita avaliar documentos de temas excluídos. |
| Amostra de 20 mil frases única para as 3 sementes do TSDAE. | A variação entre sementes passa a medir apenas o treinamento. |
| Agenda de aprendizado constante, sem decaimento de pesos. | Segue o exemplo de referência do TSDAE. A proposta v0.3 não fixava esses itens. |
| Aplicar `data.clean_text` também ao ENUNCIADO, antes de R1 a R4. | Remove `<b>` e quebras escritas como texto do texto agrupado. Vale para todas as representações, então não favorece nenhuma. |
