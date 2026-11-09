# Como contribuir

Guia de trabalho no repositório, localmente e no Colab.
Regras de idioma, estilo e código: [CLAUDE.md](CLAUDE.md).

## Configuração local

1. Instalar o [uv](https://docs.astral.sh/uv/getting-started/installation/), versão 0.13 ou superior.
2. Criar o ambiente e instalar os hooks:

```bash
uv sync                              # cria o ambiente local
uv run pre-commit install            # ativa os hooks antes de cada commit
uv run pytest                        # executa os testes
uv run pre-commit run --all-files    # lint, formatação e notebooks
```

## Regra da `main`

- A `main` é protegida. Ninguém faz push direto, nem o dono do repositório.
- Toda mudança entra por pull request (PR).
- O CI `lint-and-test` precisa passar antes do merge.
- As conversas do PR precisam estar resolvidas.
- O merge é sempre por squash.
- Qualquer integrante pode fazer o merge, inclusive do próprio PR.
- Não há aprovação obrigatória. A revisão de um colega é recomendada.

## Fluxo com git

1. Atualizar a `main`: `git switch main && git pull`.
2. Criar a branch no formato `nome/tarefa`: `git switch -c nome/filtro-frases`.
3. Fazer commits no padrão [Conventional Commits](https://www.conventionalcommits.org/pt-br/), com descrição em português.
4. Enviar a branch: `git push -u origin nome/filtro-frases`.
5. Abrir o PR no GitHub. O título do PR segue o mesmo padrão. Ele se torna a mensagem do commit de squash.
6. Após o merge, apagar a branch e começar a próxima a partir da `main` atualizada.

Tipos de commit mais usados:

| Tipo | Uso | Exemplo |
|---|---|---|
| `feat` | Nova funcionalidade | `feat: implementa filtro de frases` |
| `fix` | Correção de erro | `fix: corrige semente do bootstrap` |
| `docs` | Documentação | `docs: descreve o corpus no README` |
| `test` | Testes | `test: cobre frases com abreviações` |
| `refactor` | Mudança sem efeito no comportamento | `refactor: separa a normalização L2` |
| `chore` | Configuração e dependências | `chore: atualiza o uv.lock` |

## Fluxo no Colab

Pré-requisitos:

- Cada integrante autoriza o app do Colab na própria conta do GitHub. O pedido aparece no primeiro salvamento.
- Usar Chrome ou Firefox.

Passo a passo:

1. Criar a branch pela web: aba **Code** → seletor de branch → digitar o nome (`nome/tarefa`) → **Create branch from main**.
2. Abrir o notebook pelo selo "Open in Colab" do [README](README.md).
3. Para testar código da branch, trocar `BRANCH = "main"` pelo nome da branch na célula de configuração.
4. Executar a célula de configuração e editar o notebook.
5. Nos notebooks 01 a 03, marcar **Omitir a saída das células de código ao salvar este notebook** (Editar → Configurações do notebook). O `00_overview` é a exceção: ele mantém as saídas.
6. Salvar: Arquivo → **Salvar uma cópia no GitHub**.
   - Repositório: `Igorbraziel/juristcu-clustering-tsdae`.
   - Branch: a branch de trabalho. Nunca a `main`.
   - Caminho: `notebooks/NN_nome.ipynb`, igual ao arquivo original.
7. Se o notebook gerou tabelas ou figuras, baixá-las pelo painel **Arquivos** do Colab (`results/tables/` e `results/figures/`) e enviá-las pela web na mesma branch (**Add file** → **Upload files**). O disco do Colab é apagado ao fim da sessão.
8. Abrir o PR pela web, a partir da branch de trabalho.

Observações:

- Após mudanças em `src/` na branch, reiniciar a sessão (Ambiente de execução → Reiniciar sessão) e executar de novo a célula de configuração.
- O Colab não executa os hooks locais. O CI acusa notebooks com saídas e arquivos fora do padrão.
- Plano B: baixar o `.ipynb` (Arquivo → Fazer o download → Baixar o .ipynb) e enviá-lo pela web na branch certa (**Add file** → **Upload files**).

## Regras para notebooks

- Um notebook por PR.
- Nunca abrir dois PRs no mesmo `.ipynb`.
- Lógica nova vai para `src/juristcu/`, com teste em `tests/`. O notebook só chama funções e mostra resultados.
- Em caso de conflito em notebook, refazer a branch a partir da `main` atualizada. Não resolver conflitos no JSON do `.ipynb`.

## Implementação de funções

- As funções do esqueleto levantam `NotImplementedError`.
- Os testes correspondentes estão marcados com `xfail`.
- Ao implementar uma função, remover o marcador `xfail` do teste correspondente.
- Com `xfail_strict = true`, um teste que passa com o marcador aparece como XPASS e falha o CI.

## Dependências

- Alterar dependências apenas com `uv add` (ou `uv add --dev`). Não editar o `uv.lock` à mão.
- O hook do pre-commit atualiza o `uv.lock` e o `requirements.txt`.
- Usar somente limites inferiores (`>=`), sempre abaixo das versões instaladas no Colab.
- O `requirements.txt` é apenas referência de versões do ambiente local. Não usar no Colab.

## Pré-registro

- O [protocolo](docs/protocol.md) define o estudo. Após a tag `preregistro`, não é editado.
- Todo desvio do protocolo vai para [docs/decisions.md](docs/decisions.md), com data e motivo.
- A tag `preregistro` é criada após a reunião que congelar as hipóteses:

```bash
git switch main && git pull
git tag preregistro
git push origin preregistro
```

## Segurança

- O repositório é público.
- Nenhum token, senha ou chave em notebook ou em código.
- No Colab, usar Colab Secrets (ícone de chave na barra lateral).
