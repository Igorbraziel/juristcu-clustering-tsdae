"""Construção do corpus de frases sem rótulo para o TSDAE.

- O corpus usa apenas ENUNCIADO e EXCERTO. Áreas e temas nunca entram aqui.
- Esqueleto: as funções levantam NotImplementedError até serem implementadas.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from juristcu.config import MAX_WORDS, MIN_WORDS, N_SENTENCES, SEED


def ensure_nltk_resources() -> None:
    """Baixa o modelo Punkt do NLTK, se ainda não estiver instalado.

    Note:
        - O recurso correto é "punkt_tab", não "punkt".
        - O PunktTokenizer do NLTK 3.9 ou superior lê apenas punkt_tab.
        - O NLTK 3.10 ignora diretórios de dados com escrita para grupo ou outros.
          Criar o destino com permissão 0o700 antes do download.
    """
    raise NotImplementedError


def split_sentences(text: str) -> list[str]:
    """Divide um texto em frases com o Punkt para português.

    O tokenizador recebe estas abreviações jurídicas: art, arts, inc, incs, al, n, nº, p, pp,
    min, rel, ac, proc, fl, fls, sr, sra, dr, dra.

    Args:
        text: Texto de entrada.

    Returns:
        Frases na ordem do texto, sem espaços nas extremidades.

    Raises:
        LookupError: Se o recurso punkt_tab não estiver instalado.

    Note:
        - As abreviações entram em PunktTokenizer("portuguese")._params.abbrev_types. É API
          interna do NLTK e pode mudar entre versões.
        - A lista de abreviações deve ficar em config.py, que ainda não tem essa constante.
        - Criar o tokenizador uma única vez e reutilizá-lo (por exemplo, com functools.cache).
    """
    raise NotImplementedError


def filter_by_length(
    sentences: Iterable[str],
    min_words: int = MIN_WORDS,
    max_words: int = MAX_WORDS,
) -> list[str]:
    """Mantém as frases com número de palavras entre os limites, inclusive.

    As palavras são contadas com str.split().

    Args:
        sentences: Frases de entrada.
        min_words: Número mínimo de palavras.
        max_words: Número máximo de palavras.

    Returns:
        Frases aceitas, na ordem original.
    """
    raise NotImplementedError


def deduplicate(sentences: Iterable[str]) -> list[str]:
    """Remove frases duplicadas e preserva a ordem da primeira ocorrência.

    A comparação é exata.

    Args:
        sentences: Frases de entrada.

    Returns:
        Frases sem repetição.
    """
    raise NotImplementedError


def build_corpus(
    df: pd.DataFrame,
    n_sentences: int = N_SENTENCES,
    seed: int = SEED,
) -> list[str]:
    """Monta a amostra fixa de frases para o treinamento do TSDAE.

    Etapas:

    1. split_sentences sobre ENUNCIADO e EXCERTO. Textos ausentes são ignorados.
    2. filter_by_length.
    3. deduplicate.
    4. Amostragem sem reposição de n_sentences frases, com a semente.

    Args:
        df: Documentos com as colunas ENUNCIADO e EXCERTO. As demais colunas são ignoradas.
        n_sentences: Tamanho da amostra.
        seed: Semente da amostragem.

    Returns:
        Lista com n_sentences frases distintas.

    Raises:
        ValueError: Se houver menos frases válidas que n_sentences.

    Note:
        - Ler apenas TEXT_COLUMN e EXCERPT_COLUMN. AREA e TEMA nunca entram no corpus.
        - A mesma amostra serve às 3 sementes do TSDAE. Aqui a semente é SEED, não TSDAE_SEEDS.
    """
    raise NotImplementedError
