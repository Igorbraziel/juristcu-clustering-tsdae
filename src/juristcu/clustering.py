"""Agrupamento das representações.

- k-means com inicialização k-means++ e 10 reinícios (algoritmo principal).
- Varredura de k pela silhueta, sem rótulos.
- Projeção UMAP em 2 dimensões e HDBSCAN como extra opcional.
- Tabelas das atribuições, das varreduras e do k de maior silhueta, prontas para CSV.
- Retomada da varredura a partir da tabela gravada.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans

from juristcu import evaluation
from juristcu.config import (
    HDBSCAN_MIN_CLUSTER_SIZE,
    K_RANGE,
    KMEANS_N_INIT,
    SEED,
    UMAP_MIN_DIST,
    UMAP_N_COMPONENTS,
    UMAP_N_NEIGHBORS,
)


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
        Grupo de cada linha, com valores inteiros de 0 a k - 1.

    Note:
        - Passar n_init explicitamente. O padrão "auto" do scikit-learn faz um único reinício
          com k-means++.
    """
    model = KMeans(n_clusters=k, init="k-means++", n_init=n_init, random_state=seed)
    return model.fit_predict(x)


def silhouette_sweep(
    x: np.ndarray,
    k_values: Iterable[int] = K_RANGE,
    seed: int = SEED,
    sample_size: int | None = None,
) -> pd.DataFrame:
    """Calcula a silhueta do k-means para cada valor de k.

    Cada k usa run_kmeans e evaluation.internal_score, com a mesma semente.

    Args:
        x: Representação normalizada, uma linha por documento.
        k_values: Valores de k avaliados, na ordem da tabela.
        seed: Semente do k-means e da subamostra da silhueta.
        sample_size: Número de documentos sorteados para a silhueta. None usa todos.

    Returns:
        Tabela com as colunas k (int) e silhouette (float), uma linha por valor de k.

    Note:
        - Varredura de k pela silhueta, sem rótulos.
        - sample_size=None usa todos os documentos. A subamostra serve aos testes.
        - O k sugerido é o de maior silhueta. Ver best_k_table.
    """
    rows = []
    for k in k_values:
        labels = run_kmeans(x, k, seed=seed)
        score = evaluation.internal_score(x, labels, seed=seed, sample_size=sample_size)
        rows.append({"k": int(k), "silhouette": score})
    table = pd.DataFrame(rows, columns=["k", "silhouette"])
    return table.astype({"k": "int64", "silhouette": "float64"})


def project_umap(
    x: np.ndarray,
    seed: int = SEED,
    n_components: int = UMAP_N_COMPONENTS,
    n_neighbors: int = UMAP_N_NEIGHBORS,
    min_dist: float = UMAP_MIN_DIST,
) -> np.ndarray:
    """Projeta uma representação em poucas dimensões com UMAP.

    Usa umap.UMAP com a distância euclidiana, random_state=seed e n_jobs=1.

    Args:
        x: Representação normalizada, uma linha por documento.
        seed: Semente do UMAP (random_state).
        n_components: Número de dimensões da projeção.
        n_neighbors: Número de vizinhos do grafo do UMAP.
        min_dist: Distância mínima entre os pontos da projeção.

    Returns:
        Matriz float32 (n_documentos, n_components).

    Note:
        - n_jobs=1 é explícito. random_state já força uma única thread, e o valor explícito
          evita o aviso do umap-learn.
        - A projeção serve à visualização e ao HDBSCAN. NMI, ARI e silhueta usam a
          representação original.
    """
    # Importação tardia: o umap compila funções com numba e tem carregamento pesado.
    import umap

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=seed,
        n_jobs=1,
    )
    return np.asarray(reducer.fit_transform(x), dtype=np.float32)


def run_hdbscan(
    x_2d: np.ndarray,
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
) -> np.ndarray:
    """Agrupa a projeção UMAP com HDBSCAN (extra opcional).

    Usa sklearn.cluster.HDBSCAN(min_cluster_size=min_cluster_size, copy=True), que estima o
    número de grupos.

    Args:
        x_2d: Projeção devolvida por project_umap.
        min_cluster_size: Número mínimo de documentos em um grupo.

    Returns:
        Grupo de cada linha. O valor -1 indica ruído.

    Note:
        - Os pontos de ruído (-1) exigem tratamento explícito antes do cálculo da NMI.
        - copy=True garante que x_2d não é alterada. O valor explícito evita o FutureWarning
          do scikit-learn 1.9, cujo padrão passa de False para True no 1.10.
    """
    model = HDBSCAN(min_cluster_size=min_cluster_size, copy=True)
    return model.fit_predict(x_2d)


def assignments_table(preds: Mapping[str, np.ndarray], documents: np.ndarray) -> pd.DataFrame:
    """Monta a tabela das atribuições de grupo de cada documento.

    Args:
        preds: Grupo de cada documento, indexado pelo nome da representação.
        documents: Índice de cada documento, na ordem das linhas de load_documents.

    Returns:
        Tabela com a coluna document e uma coluna int64 por representação, na ordem de
        preds.

    Raises:
        ValueError: Se algum agrupamento tiver tamanho diferente de documents ou rótulos que
            não sejam inteiros.

    Note:
        - A tabela serve para gravar as atribuições em CSV para a avaliação.
    """
    index = np.asarray(documents, dtype=np.int64)
    columns = {"document": index}
    for name, labels in preds.items():
        values = np.asarray(labels)
        if len(values) != len(index):
            raise ValueError(
                f"O agrupamento {name} tem {len(values)} linhas; esperado {len(index)}."
            )
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"O agrupamento {name} não tem rótulos inteiros: {values.dtype}.")
        columns[name] = values.astype(np.int64)
    return pd.DataFrame(columns)


def best_k_table(sweeps: Mapping[str, pd.DataFrame], n_areas: int) -> pd.DataFrame:
    """Resume a varredura de k de cada representação.

    Args:
        sweeps: Resultado de silhouette_sweep, indexado pelo nome da representação.
        n_areas: Número de áreas, usado apenas para comparação.

    Returns:
        Tabela com uma linha por representação, na ordem de sweeps, e as colunas:
        - "representation": nome da representação.
        - "best_k": k de maior silhueta. Em empate, o menor k.
        - "best_silhouette": silhueta no best_k.
        - "silhouette_at_n_areas": silhueta em k = n_areas, ou NaN fora da varredura.
        - "n_areas": número de áreas.

    Raises:
        ValueError: Se alguma varredura estiver vazia.

    Note:
        - Não usa rótulos. O número de áreas não interfere na escolha de best_k.
    """
    rows = []
    for name, sweep in sweeps.items():
        if sweep.empty:
            raise ValueError(f"A varredura de {name} está vazia.")
        best_silhouette = sweep["silhouette"].max()
        best_k = sweep.loc[sweep["silhouette"] == best_silhouette, "k"].min()
        at_n_areas = sweep.loc[sweep["k"] == n_areas, "silhouette"]
        rows.append(
            {
                "representation": name,
                "best_k": int(best_k),
                "best_silhouette": float(best_silhouette),
                "silhouette_at_n_areas": float(at_n_areas.iloc[0]) if len(at_n_areas) else np.nan,
                "n_areas": int(n_areas),
            }
        )
    columns = ["representation", "best_k", "best_silhouette", "silhouette_at_n_areas", "n_areas"]
    return pd.DataFrame(rows, columns=columns)


def sweep_table(sweeps: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Reúne as varreduras de k em uma única tabela, em formato longo.

    Args:
        sweeps: Resultado de silhouette_sweep, indexado pelo nome da representação.

    Returns:
        Tabela com as colunas representation, k e silhouette, ordenada pela ordem de sweeps
        e, em cada representação, por k.

    Note:
        - A tabela serve para gravar a silhueta por k em CSV e para retomar a varredura.
    """
    columns = ["representation", "k", "silhouette"]
    frames = [
        sweep.sort_values("k", kind="stable").assign(representation=name)[columns]
        for name, sweep in sweeps.items()
    ]
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def completed_sweeps(
    table: pd.DataFrame,
    names: Iterable[str],
    k_values: Iterable[int] = K_RANGE,
) -> dict[str, pd.DataFrame]:
    """Recupera as varreduras completas de uma tabela gravada por sweep_table.

    Args:
        table: Tabela com as colunas representation, k e silhouette.
        names: Representações procuradas, na ordem do resultado.
        k_values: Valores de k que uma varredura completa contém.

    Returns:
        Varredura de cada representação completa, no formato de silhouette_sweep e ordenada
        por k. Representações ausentes ou incompletas ficam de fora.

    Note:
        - Permite retomar a varredura sem refazer as representações já concluídas.
    """
    expected = {int(k) for k in k_values}
    sweeps = {}
    for name in names:
        rows = table.loc[table["representation"] == name, ["k", "silhouette"]]
        if set(rows["k"].astype(int)) == expected and len(rows) == len(expected):
            sweep = rows.sort_values("k", kind="stable").reset_index(drop=True)
            sweeps[name] = sweep.astype({"k": "int64", "silhouette": "float64"})
    return sweeps
