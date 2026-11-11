"""Construção do corpus de frases sem rótulo para o TSDAE.

O corpus usa apenas ENUNCIADO e EXCERTO. Áreas e temas nunca entram aqui.
"""

from __future__ import annotations

import functools
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from juristcu.config import (
    EXCERPT_COLUMN,
    LEGAL_ABBREVIATIONS,
    MAX_WORDS,
    MIN_WORDS,
    N_SENTENCES,
    NLTK_LANGUAGE,
    NLTK_RESOURCE,
    SEED,
    TEXT_COLUMN,
)
from juristcu.data import clean_text

if TYPE_CHECKING:
    from nltk.tokenize import PunktTokenizer


def ensure_nltk_resources() -> None:
    """Baixa o modelo Punkt do NLTK, se ainda não estiver instalado.

    Raises:
        LookupError: Se o recurso continuar ausente após o download.

    Note:
        - O recurso correto é "punkt_tab", não "punkt".
        - O PunktTokenizer do NLTK 3.9 ou superior lê apenas punkt_tab.
        - O NLTK 3.10 ignora diretórios de dados com escrita para grupo ou outros.
          Por isso, o destino é criado com permissão 0o700.
    """
    import nltk

    if _has_punkt():
        return
    target = Path(nltk.downloader.Downloader().default_download_dir())
    target.mkdir(mode=0o700, parents=True, exist_ok=True)
    if str(target) not in nltk.data.path:
        nltk.data.path.append(str(target))
    nltk.download(NLTK_RESOURCE, download_dir=str(target), quiet=True, raise_on_error=True)
    if not _has_punkt():
        raise LookupError(f"Recurso {NLTK_RESOURCE} do NLTK não encontrado após o download.")


def split_sentences(text: str) -> list[str]:
    """Divide um texto em frases com o Punkt para português.

    O tokenizador recebe as abreviações jurídicas de LEGAL_ABBREVIATIONS (ex.: "art.", "inc.").

    Args:
        text: Texto de entrada.

    Returns:
        Frases na ordem do texto, sem espaços nas extremidades.

    Raises:
        LookupError: Se o recurso punkt_tab não estiver instalado.

    Note:
        - As abreviações entram em PunktTokenizer("portuguese")._params.abbrev_types. É API
          interna do NLTK e pode mudar entre versões.
        - O tokenizador é criado uma única vez e reutilizado.
    """
    sentences = _tokenizer().tokenize(text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


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
    return [s for s in sentences if min_words <= len(s.split()) <= max_words]


def deduplicate(sentences: Iterable[str]) -> list[str]:
    """Remove frases duplicadas e preserva a ordem da primeira ocorrência.

    A comparação é exata.

    Args:
        sentences: Frases de entrada.

    Returns:
        Frases sem repetição.
    """
    return list(dict.fromkeys(sentences))


def extract_sentences(df: pd.DataFrame) -> list[str]:
    """Extrai todas as frases de ENUNCIADO e EXCERTO, na ordem dos documentos.

    Cada documento contribui primeiro com as frases do ENUNCIADO e depois com as do EXCERTO.
    Os textos passam por clean_text antes da segmentação. Textos ausentes são ignorados.

    Args:
        df: Documentos com as colunas ENUNCIADO e EXCERTO. As demais colunas são ignoradas.

    Returns:
        Frases sem filtro de tamanho e sem remoção de duplicatas.
    """
    sentences: list[str] = []
    for statement, excerpt in zip(df[TEXT_COLUMN], df[EXCERPT_COLUMN], strict=True):
        for text in (statement, excerpt):
            cleaned = clean_text(text)
            if cleaned:
                sentences.extend(split_sentences(cleaned))
    return sentences


def corpus_funnel(df: pd.DataFrame) -> pd.DataFrame:
    """Conta as frases em cada etapa da construção do corpus.

    Args:
        df: Documentos com as colunas ENUNCIADO e EXCERTO.

    Returns:
        Tabela com as colunas step e sentences, nas etapas "segmentadas",
        "filtro de tamanho" e "sem duplicatas".
    """
    sentences = extract_sentences(df)
    filtered = filter_by_length(sentences)
    unique = deduplicate(filtered)
    return pd.DataFrame(
        {
            "step": ["segmentadas", "filtro de tamanho", "sem duplicatas"],
            "sentences": [len(sentences), len(filtered), len(unique)],
        }
    )


def build_corpus(
    df: pd.DataFrame,
    n_sentences: int = N_SENTENCES,
    seed: int = SEED,
) -> list[str]:
    """Monta a amostra fixa de frases para o treinamento do TSDAE.

    Etapas:

    1. extract_sentences sobre ENUNCIADO e EXCERTO, com limpeza e split_sentences.
    2. filter_by_length.
    3. deduplicate.
    4. Amostragem sem reposição de n_sentences frases, com a semente.

    Args:
        df: Documentos com as colunas ENUNCIADO e EXCERTO. As demais colunas são ignoradas.
        n_sentences: Tamanho da amostra.
        seed: Semente da amostragem.

    Returns:
        Lista com n_sentences frases distintas, na ordem sorteada.

    Raises:
        ValueError: Se houver menos frases válidas que n_sentences.

    Note:
        - Apenas TEXT_COLUMN e EXCERPT_COLUMN são lidos. AREA e TEMA nunca entram no corpus.
        - A mesma amostra serve às 3 sementes do TSDAE. Aqui a semente é SEED, não TSDAE_SEEDS.
    """
    candidates = deduplicate(filter_by_length(extract_sentences(df)))
    if len(candidates) < n_sentences:
        raise ValueError(
            f"Há {len(candidates)} frases válidas, menos que as {n_sentences} pedidas."
        )
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(candidates), size=n_sentences, replace=False)
    return [candidates[i] for i in indices]


def _has_punkt() -> bool:
    """Indica se o recurso punkt_tab para português está instalado."""
    import nltk

    try:
        nltk.data.find(f"tokenizers/{NLTK_RESOURCE}/{NLTK_LANGUAGE}")
    except LookupError:
        return False
    return True


@functools.cache
def _tokenizer() -> PunktTokenizer:
    """Cria o Punkt para português com as abreviações jurídicas."""
    from nltk.tokenize import PunktTokenizer

    tokenizer = PunktTokenizer(NLTK_LANGUAGE)
    tokenizer._params.abbrev_types.update(LEGAL_ABBREVIATIONS)
    return tokenizer
