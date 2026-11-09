"""Interpretação dos grupos e figuras do relatório.

- Termos característicos de cada grupo (TF-IDF por grupo).
- Figura 1: projeções UMAP coloridas por área.
- Figura 2: silhueta em função de k.
- Esqueleto: as funções levantam NotImplementedError até serem implementadas.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def top_terms_per_cluster(
    texts: Sequence[str],
    labels: np.ndarray,
    n_terms: int = 10,
) -> dict[int, list[str]]:
    """Lista os termos mais característicos de cada grupo.

    Os textos de cada grupo formam um único documento. O TF-IDF é calculado sobre esses
    documentos, e os termos de maior peso descrevem o grupo.

    Args:
        texts: Texto de cada documento.
        labels: Grupo de cada documento.
        n_terms: Número de termos por grupo.

    Returns:
        Dicionário do grupo para os seus termos, em ordem decrescente de peso.

    Note:
        - O scikit-learn só oferece palavras vazias em inglês (stop_words="english").
        - Para português, passar uma lista própria, definida em config.py.
    """
    raise NotImplementedError


def plot_umap_grid(projections: Mapping[str, np.ndarray], colors: Sequence[str]) -> Figure:
    """Desenha as projeções UMAP lado a lado (Figura 1).

    Args:
        projections: Projeção 2D de cada representação, indexada pelo nome (por exemplo, "R1").
        colors: Área de cada documento, na ordem das linhas das projeções. Define a cor dos
            pontos.

    Returns:
        Figura com um painel por representação.
    """
    raise NotImplementedError


def plot_silhouette_curves(sweeps: Mapping[str, pd.DataFrame]) -> Figure:
    """Desenha a silhueta em função de k para cada representação (Figura 2).

    Args:
        sweeps: Resultado de clustering.silhouette_sweep, indexado pelo nome da representação.

    Returns:
        Figura com uma curva por representação.
    """
    raise NotImplementedError
