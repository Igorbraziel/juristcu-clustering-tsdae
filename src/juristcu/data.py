"""Carregamento do JurisTCU e distribuição dos rótulos.

Esqueleto: as funções levantam NotImplementedError até serem implementadas.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from juristcu.config import AREA_COLUMN, EXCERPT_COLUMN, TEXT_COLUMN, TOPIC_COLUMN


def load_documents(
    columns: Sequence[str] = (TEXT_COLUMN, EXCERPT_COLUMN, AREA_COLUMN, TOPIC_COLUMN),
) -> pd.DataFrame:
    """Carrega os documentos do JurisTCU.

    Usa load_dataset(DATASET_ID, data_files=DATA_FILE, split="train"), do pacote datasets.

    Args:
        columns: Colunas mantidas no resultado.

    Returns:
        Tabela com uma linha por documento e as colunas pedidas.

    Note:
        - Ler o arquivo doc.csv diretamente. O visualizador do Hugging Face falha neste conjunto.
        - Conferir que o resultado tem N_DOCUMENTS (16.045) linhas.
        - Importar datasets dentro da função, para manter a importação do módulo rápida.
    """
    raise NotImplementedError


def label_distribution(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Conta os documentos de cada categoria de um campo.

    Args:
        df: Documentos carregados por load_documents.
        column: Campo analisado, em geral AREA_COLUMN ou TOPIC_COLUMN.

    Returns:
        Tabela com as colunas column, count e proportion, em ordem decrescente de contagem.
    """
    raise NotImplementedError
