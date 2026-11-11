"""Carregamento do JurisTCU, limpeza de texto e distribuição dos rótulos."""

from __future__ import annotations

import html
import math
import re
from collections.abc import Sequence

import pandas as pd

from juristcu.config import (
    AREA_COLUMN,
    DATA_FILE,
    DATASET_ID,
    EXCERPT_COLUMN,
    HTML_BLOCK_TAGS,
    MIN_TOPIC_SIZE,
    N_DOCUMENTS,
    TEXT_COLUMN,
    TOPIC_COLUMN,
)

_BLOCK_TAG = re.compile(rf"</?(?:{'|'.join(HTML_BLOCK_TAGS)})\b[^>]*>", re.IGNORECASE)
_HTML_TAG = re.compile(r"<[^>]+>")
_LINE_BREAK = re.compile(r"\\[nrt]|[\r\n\t]")
_OMISSION = re.compile(r"\[\s*(?:\.\s*){3}\]|\[\s*…\s*\]")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([,.;:!?)\]])")
_SPACE_AFTER_BRACKET = re.compile(r"([(\[])\s+")
_WHITESPACE = re.compile(r"\s+")
_PARAGRAPH_BREAK = re.compile(rf"{_BLOCK_TAG.pattern}|{_LINE_BREAK.pattern}", re.IGNORECASE)
_PARAGRAPH_NUMBER = re.compile(r"(?<![\w.,/-])\d{1,3}\.(?=[A-ZÀ-ÖØ-Þ])")


def load_documents(
    columns: Sequence[str] = (TEXT_COLUMN, EXCERPT_COLUMN, AREA_COLUMN, TOPIC_COLUMN),
    dataset_id: str = DATASET_ID,
    data_file: str = DATA_FILE,
    expected_rows: int | None = N_DOCUMENTS,
) -> pd.DataFrame:
    """Carrega os documentos do JurisTCU.

    Usa load_dataset(dataset_id, data_files=data_file, split="train"), do pacote datasets.

    Args:
        columns: Colunas mantidas no resultado.
        dataset_id: Identificador do conjunto no Hugging Face ou diretório local.
        data_file: Arquivo CSV dentro do conjunto.
        expected_rows: Número esperado de documentos. None desativa a conferência.

    Returns:
        Tabela com uma linha por documento e as colunas pedidas, todas como texto.

    Raises:
        ValueError: Se faltar alguma coluna pedida ou se o número de linhas divergir do esperado.

    Note:
        - O arquivo doc.csv é lido diretamente. O visualizador do Hugging Face falha neste conjunto.
        - Todas as colunas são lidas como texto. Assim, a inferência de tipos do datasets não
          varia entre blocos do arquivo.
        - Valores ausentes ficam como None.
    """
    # Importação tardia: o datasets é pesado e só é necessário aqui.
    from datasets import Features, Value, load_dataset

    columns = list(columns)
    dataset = load_dataset(
        dataset_id,
        data_files=data_file,
        split="train",
        usecols=columns,
        features=Features({column: Value("string") for column in columns}),
    )
    df = dataset.to_pandas()

    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"Colunas ausentes em {data_file}: {missing}. Conferir o separador do CSV."
        )
    if expected_rows is not None and len(df) != expected_rows:
        raise ValueError(f"{data_file} tem {len(df)} linhas; esperadas {expected_rows}.")
    return df[columns]


def clean_text(text: str | None) -> str:
    """Remove marcação HTML e normaliza espaços de um texto.

    Etapas:

    1. Troca etiquetas de bloco (ex.: "<p>", "<br>") por espaço.
    2. Remove as demais etiquetas (ex.: "<b>", "<span>") sem inserir espaço.
    3. Converte entidades HTML (ex.: "&nbsp;") em caracteres.
    4. Troca marcas de omissão ("[...]") e quebras de linha, inclusive as escritas como
       texto, por espaço.
    5. Reduz espaços repetidos e remove o espaço antes de pontuação e depois de parênteses.

    Args:
        text: Texto de entrada. None e NaN viram texto vazio.

    Returns:
        Texto limpo.
    """
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    text = _BLOCK_TAG.sub(" ", str(text))
    text = _HTML_TAG.sub("", text)
    text = html.unescape(text)
    text = _OMISSION.sub(" ", text)
    text = _LINE_BREAK.sub(" ", text)
    text = _WHITESPACE.sub(" ", text)
    text = _SPACE_BEFORE_PUNCTUATION.sub(r"\1", text)
    text = _SPACE_AFTER_BRACKET.sub(r"\1", text)
    return text.strip()


def split_paragraphs(text: str | None) -> list[str]:
    """Divide um texto em parágrafos limpos.

    Fronteiras de parágrafo:

    - Etiquetas HTML de bloco (ex.: "<p>", "<br>") e quebras de linha.
    - Números de parágrafo colados à palavra seguinte (ex.: "8.Este", "20.Quando").
      O número é descartado.

    Args:
        text: Texto de entrada. None e NaN viram lista vazia.

    Returns:
        Parágrafos não vazios, já processados por clean_text, na ordem do texto.

    Note:
        Um número de parágrafo é reconhecido quando tem até 3 dígitos, não vem depois de letra,
        dígito, ponto, vírgula, barra ou hífen e é seguido de ponto e letra maiúscula. Assim,
        "Lei 8.666" e "Acórdão 1.131/2009" não são afetados.
    """
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return []
    blocks = _PARAGRAPH_BREAK.split(str(text))
    paragraphs = []
    for block in blocks:
        for part in _PARAGRAPH_NUMBER.split(clean_text(block)):
            part = part.strip()
            if part:
                paragraphs.append(part)
    return paragraphs


def tag_frequency(texts: pd.Series) -> pd.DataFrame:
    """Conta as etiquetas HTML de abertura em uma coluna de texto.

    Args:
        texts: Textos de entrada. Valores ausentes são ignorados.

    Returns:
        Tabela com as colunas tag, count e block, em ordem decrescente de contagem.
        A coluna block indica se a etiqueta separa parágrafos.
    """
    tags = texts.dropna().str.findall(r"<([A-Za-z][A-Za-z0-9]*)").explode().dropna().str.lower()
    table = tags.value_counts().rename("count").rename_axis("tag").reset_index()
    table["block"] = table["tag"].isin(HTML_BLOCK_TAGS)
    return table


def label_distribution(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Conta os documentos de cada categoria de um campo.

    Args:
        df: Documentos carregados por load_documents.
        column: Campo analisado, em geral AREA_COLUMN ou TOPIC_COLUMN.

    Returns:
        Tabela com as colunas column, count e proportion, em ordem decrescente de contagem.
        Empates seguem a ordem alfabética. Valores ausentes formam uma categoria própria.
    """
    counts = df[column].value_counts(dropna=False).rename("count").rename_axis(column)
    table = counts.reset_index()
    table["proportion"] = table["count"] / len(df)
    table = table.sort_values(
        ["count", column], ascending=[False, True], kind="stable", na_position="last"
    )
    return table.reset_index(drop=True)


def summarize_distribution(
    distribution: pd.DataFrame,
    min_size: int = MIN_TOPIC_SIZE,
) -> dict[str, float]:
    """Resume o equilíbrio de uma distribuição de rótulos.

    Args:
        distribution: Tabela devolvida por label_distribution.
        min_size: Tamanho mínimo de categoria usado na contagem de categorias mantidas.

    Returns:
        Dicionário com as chaves:
        - "categories": número de categorias.
        - "documents": número de documentos.
        - "largest_share": proporção da maior categoria.
        - "smallest_count": tamanho da menor categoria.
        - "imbalance_ratio": maior categoria dividida pela menor.
        - "normalized_entropy": entropia dividida pelo seu máximo (1 indica equilíbrio total).
        - "categories_min_size": categorias com ao menos min_size documentos.
        - "share_min_size": proporção de documentos nessas categorias.
    """
    counts = distribution["count"].to_numpy(dtype=float)
    total = counts.sum()
    proportions = counts / total
    n_categories = len(counts)
    entropy = -sum(p * math.log(p) for p in proportions if p > 0)
    max_entropy = math.log(n_categories) if n_categories > 1 else 1.0
    kept = counts >= min_size
    return {
        "categories": n_categories,
        "documents": int(total),
        "largest_share": float(proportions.max()),
        "smallest_count": int(counts.min()),
        "imbalance_ratio": float(counts.max() / counts.min()),
        "normalized_entropy": float(entropy / max_entropy),
        "categories_min_size": int(kept.sum()),
        "share_min_size": float(counts[kept].sum() / total),
    }
