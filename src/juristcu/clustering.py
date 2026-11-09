"""Agrupamento das representações.

- k-means com inicialização k-means++ e 10 reinícios (algoritmo principal).
- Varredura de k pela silhueta, sem rótulos (P3).
- Projeção UMAP em 2 dimensões e HDBSCAN como extra opcional.
- Esqueleto: as funções levantam NotImplementedError até serem implementadas.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from juristcu.config import K_RANGE, KMEANS_N_INIT, SEED, UMAP_N_COMPONENTS


def run_kmeans(
    x: np.ndarray,
    k: int,
    seed: int = SEED,
    n_init: int = KMEANS_N_INIT,
) -> np.ndarray:
    """Agrupa as linhas de uma representação com k-means.

    Usa sklearn.cluster.KMeans(n_clusters=k, init="k-means++", n_init=n_init,
    random_state=seed).

    Args:
        x: Representação normalizada, uma linha por documento.
        k: Número de grupos.
        seed: Semente da inicialização.
        n_init: Número de reinícios.

    Returns:
        Grupo de cada linha, com valores de 0 a k - 1.

    Note:
        - Passar n_init explicitamente. O padrão "auto" do scikit-learn faz um único reinício
          com k-means++.
    """
    raise NotImplementedError


def silhouette_sweep(
    x: np.ndarray,
    k_values: Iterable[int] = K_RANGE,
    seed: int = SEED,
) -> pd.DataFrame:
    """Calcula a silhueta do k-means para cada valor de k (P3).

    Cada k usa run_kmeans e evaluation.internal_score.

    Args:
        x: Representação normalizada, uma linha por documento.
        k_values: Valores de k avaliados.
        seed: Semente do k-means.

    Returns:
        Tabela com as colunas k e silhouette, uma linha por valor de k.

    Note:
        - Não usa rótulos. O k sugerido é o de maior silhueta.
    """
    raise NotImplementedError


def project_umap(
    x: np.ndarray,
    seed: int = SEED,
    n_components: int = UMAP_N_COMPONENTS,
) -> np.ndarray:
    """Projeta uma representação em poucas dimensões com UMAP.

    Args:
        x: Representação normalizada, uma linha por documento.
        seed: Semente do UMAP (random_state).
        n_components: Número de dimensões da projeção.

    Returns:
        Matriz (n_documentos, n_components).

    Note:
        - random_state torna o resultado reprodutível, mas força n_jobs=1. O aviso do
          umap-learn é esperado.
        - A projeção serve à visualização e ao HDBSCAN. NMI, ARI e silhueta usam a
          representação original.
    """
    raise NotImplementedError


def run_hdbscan(x_2d: np.ndarray, min_cluster_size: int) -> np.ndarray:
    """Agrupa a projeção UMAP com HDBSCAN (extra opcional).

    Usa sklearn.cluster.HDBSCAN, que estima o número de grupos.

    Args:
        x_2d: Projeção devolvida por project_umap.
        min_cluster_size: Número mínimo de documentos em um grupo.

    Returns:
        Grupo de cada linha. O valor -1 indica ruído.

    Note:
        - Os pontos de ruído (-1) exigem tratamento explícito antes do cálculo da NMI.
        - Passar copy explicitamente. O scikit-learn 1.9 emite FutureWarning quando ele é
          omitido.
    """
    raise NotImplementedError
