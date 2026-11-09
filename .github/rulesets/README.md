# Proteção da `main`

O arquivo `protect-main.json` define o ruleset da branch padrão.

- **O que a regra faz.**
  - Exige pull request para toda mudança na `main`.
  - Aceita merge sem aprovação; o próprio autor pode fazer o merge.
  - Exige conversas resolvidas e permite apenas squash merge.
  - Exige o status check `lint-and-test` (job do CI) aprovado.
  - Bloqueia push direto, force push e exclusão da `main`.
  - Não há exceções: a regra vale também para o dono do repositório.
- **Pré-requisito.** O CI precisa ter sido executado ao menos uma vez com sucesso (aba Actions).
- **Opções de merge.** Em Settings → General → Pull Requests:
  - Marcar apenas "Allow squash merging".
  - Marcar "Automatically delete head branches".
- **Importação do ruleset.**
  1. Abrir Settings → Rules → Rulesets.
  2. Clicar em New ruleset → Import a ruleset.
  3. Selecionar `.github/rulesets/protect-main.json`.
  4. Conferir: Enforcement status = Active; Bypass list vazia; status check `lint-and-test`.
  5. Clicar em Create.
- **Teste.** Um `git push origin main` deve ser rejeitado. Um PR com o CI aprovado deve permitir o merge.
- **Aviso: visibilidade.** No plano gratuito, rulesets só valem em repositórios públicos. Se o repositório se tornar privado, a proteção deixa de valer.
- **Aviso: nome do job.** Se o nome do job em `.github/workflows/ci.yml` mudar, atualizar o status check no ruleset. Caso contrário, nenhum PR consegue fazer merge.
