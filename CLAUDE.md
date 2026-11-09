# CLAUDE.md

Guia para agentes e colaboradores deste repositório.

## Projeto

- Agrupamento temático dos enunciados do JurisTCU, sem rótulos.
- Compara representações (R1 a R4) com k-means e mede NMI, ARI e silhueta.
- Protocolo do estudo: [docs/protocol.md](docs/protocol.md).

## Idioma

- Código em inglês: variáveis, funções, classes, módulos e arquivos.
- Em português formal: comentários, docstrings, mensagens, notebooks (Markdown) e documentação.

## Estilo de escrita

- Frases curtas e diretas. Uma ideia por frase.
- Listas em vez de parágrafos longos.
- Voz ativa e tempo presente.
- Sem gírias, sem primeira pessoa e sem adjetivos vagos.
- Comentários e docstrings não citam documentos nem números de seção. Usar só o assunto: `# TSDAE`, não `# TSDAE (seção 4.4)`.

Exemplo:

- Bom: `"""Remove frases duplicadas e preserva a ordem."""`
- Ruim: `"""Essa função basicamente vai pegar a lista e tirar as frases que se repetem."""`

## Glossário

| Termo | Uso |
|---|---|
| representação | Vetores de um método (R1 a R4). Não usar "embedding" no texto. |
| agrupamento | Resultado do k-means ou do HDBSCAN. |
| grupo | Um elemento do agrupamento. Não usar "cluster" no texto. |
| área, tema | Campos AREA e TEMA do JurisTCU. |
| enunciado | Campo ENUNCIADO; é o texto representado. |
| semente | Valor que fixa a aleatoriedade. |

## Docstrings

- Estilo Google, em português.
- Primeira linha no presente do indicativo: "Carrega os documentos."
- Seções em inglês, com estes nomes: `Args:`, `Returns:`, `Raises:`, `Attributes:` e `Note:`.
- O ruff verifica o formato dessas seções (convenção Google).

## Código

- Type hints em todas as funções públicas.
- Funções pequenas e, quando possível, puras.
- Semente sempre como parâmetro explícito.
- Constantes apenas em `src/juristcu/config.py`.
- Sem `print` em `src/`. Funções devolvem valores; notebooks exibem.
- Lógica fica em `src/`. Notebooks só chamam funções e mostram resultados.

## Validade experimental

- AREA e TEMA são carregados em `data.py` e usados só na avaliação e na análise descritiva.
- Rótulos nunca entram no treinamento nem na escolha de hiperparâmetros.
- Não editar `docs/protocol.md` após a tag `preregistro`.
- Todo desvio do protocolo vai para `docs/decisions.md`, com data e motivo.

## Notebooks e Colab

- Apenas `00_overview.ipynb` mantém saídas. Os demais são salvos limpos.
- No Colab, instalar com `uv pip install --system --no-sources`.
- Nunca usar `uv sync` nem `requirements.txt` no Colab.
- Representações e modelos ficam no Google Drive, nunca no git.
- Nenhum token em notebook. Usar Colab Secrets.

## Comandos

```bash
uv sync                              # cria o ambiente local
uv run pytest                        # executa os testes
uv run pre-commit run --all-files    # lint, formatação e notebooks
```

## Fluxo de trabalho

- A `main` é protegida. Toda mudança entra por pull request.
- O CI (`lint-and-test`) precisa passar antes do merge.
- Detalhes em [CONTRIBUTING.md](CONTRIBUTING.md).
