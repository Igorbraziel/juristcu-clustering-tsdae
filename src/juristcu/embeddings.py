"""Representações R1 a R4 dos enunciados.

- R1: TF-IDF com SVD truncado (LSA).
- R2 e R2-CLS: BERTimbau sem ajuste, com pooling pela média e pelo token CLS.
- R3: BERTimbau após o TSDAE, com o pooling CLS salvo no modelo.
- R4: multilingual-e5-small com o prefixo "query: ".
- Todas as representações passam por l2_normalize antes do agrupamento.
- Esqueleto: as funções levantam NotImplementedError até serem implementadas.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from juristcu.config import ENCODE_BATCH_SIZE, ENCODE_MAX_SEQ_LENGTH, LSA_DIM, SEED

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


def encode_lsa(
    texts: Sequence[str],
    n_components: int = LSA_DIM,
    seed: int = SEED,
) -> np.ndarray:
    """Calcula a representação R1: TF-IDF seguido de SVD truncado.

    Usa TfidfVectorizer e TruncatedSVD(n_components, random_state=seed), do scikit-learn.

    Args:
        texts: Textos a representar (ENUNCIADO de cada documento).
        n_components: Número de dimensões do SVD.
        seed: Semente do SVD.

    Returns:
        Matriz (n_textos, n_components), sem normalização.

    Note:
        - O TruncatedSVD usa o algoritmo "randomized" por padrão. Sem random_state, o resultado
          muda a cada execução.
    """
    raise NotImplementedError


def load_sentence_model(
    name_or_path: str | Path,
    pooling: str | None = None,
    max_seq_length: int = ENCODE_MAX_SEQ_LENGTH,
) -> SentenceTransformer:
    """Carrega um modelo de frases do sentence-transformers.

    - pooling=None mantém o pooling salvo no modelo: CLS em R3 e média em R4.
    - pooling="mean" ou "cls" monta Transformer + Pooling a partir do modelo base. Assim se
      obtêm R2 (média) e R2-CLS a partir do BERTimbau.

    Args:
        name_or_path: Identificador no Hugging Face ou diretório local.
        pooling: Estratégia de pooling: None, "mean" ou "cls".
        max_seq_length: Comprimento máximo em tokens na codificação.

    Returns:
        Modelo pronto para encode_texts.

    Raises:
        ValueError: Se pooling não for None, "mean" nem "cls".

    Note:
        - Com pooling=None, um modelo sem configuração do sentence-transformers (como o
          BERTimbau) recebe pooling pela média de forma implícita. Para R2, passar "mean".
        - Ajustar model.max_seq_length também quando pooling=None.
    """
    raise NotImplementedError


def encode_texts(
    model: SentenceTransformer,
    texts: Sequence[str],
    prefix: str | None = None,
    batch_size: int = ENCODE_BATCH_SIZE,
) -> np.ndarray:
    """Codifica textos com um modelo de frases.

    O prefixo é repassado como model.encode(..., prompt=prefix).

    Args:
        model: Modelo devolvido por load_sentence_model.
        texts: Textos a codificar (ENUNCIADO de cada documento).
        prefix: Prefixo de cada texto. None não acrescenta prefixo.
        batch_size: Número de textos por lote.

    Returns:
        Matriz (n_textos, dimensão do modelo), sem normalização.

    Note:
        - R4 exige prefix=E5_PREFIX. Sem o prefixo, o E5 perde qualidade.
        - Manter normalize_embeddings=False. A normalização fica em l2_normalize, igual para
          todas as representações.
    """
    raise NotImplementedError


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Divide cada linha pela sua norma L2.

    Linhas nulas permanecem nulas, sem divisão por zero.

    Args:
        x: Matriz com uma linha por documento.

    Returns:
        Matriz do mesmo formato, com linhas de norma 1.
    """
    raise NotImplementedError
