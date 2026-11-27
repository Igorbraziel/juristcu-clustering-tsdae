"""Interpretação dos grupos e figuras do relatório.

- Termos característicos de cada grupo (TF-IDF por grupo).
- Figura 1: projeções UMAP coloridas por área.
- Figura 2: silhueta em função de k.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from juristcu.config import PT_STOPWORDS, TERM_TOKEN_PATTERN, UMAP_GRID_COLUMNS

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
        Dicionário do grupo para os seus termos, em ordem decrescente de peso. Os grupos
        aparecem em ordem crescente.

    Raises:
        ValueError: Se texts e labels tiverem tamanhos diferentes.

    Note:
        - Usa as palavras vazias de config.PT_STOPWORDS e só palavras com letras
          (config.TERM_TOKEN_PATTERN).
        - Empates de peso seguem a ordem alfabética, para um resultado reprodutível.
        - Um grupo com menos de n_terms termos devolve apenas os termos com peso positivo.
        - Passar os textos já limpos com data.clean_text, para que etiquetas HTML não virem termos.
    """
    labels = np.asarray(labels)
    if len(texts) != len(labels):
        raise ValueError(f"texts tem {len(texts)} itens e labels tem {len(labels)}.")

    grouped = pd.Series(list(texts), dtype=object).fillna("").groupby(labels).agg(" ".join)
    vectorizer = TfidfVectorizer(stop_words=list(PT_STOPWORDS), token_pattern=TERM_TOKEN_PATTERN)
    weights = vectorizer.fit_transform(grouped.to_list()).toarray()
    vocabulary = vectorizer.get_feature_names_out()

    terms: dict[int, list[str]] = {}
    for group, row in zip(grouped.index, weights, strict=True):
        order = np.argsort(-row, kind="stable")[:n_terms]
        terms[int(group)] = [str(vocabulary[i]) for i in order if row[i] > 0]
    return terms


def plot_umap_grid(projections: Mapping[str, np.ndarray], colors: Sequence[str]) -> Figure:
    """Desenha as projeções UMAP lado a lado (Figura 1).

    Args:
        projections: Projeção 2D de cada representação, indexada pelo nome (por exemplo, "R1").
        colors: Área de cada documento, na ordem das linhas das projeções. Define a cor dos
            pontos.

    Returns:
        Figura com um painel por representação.

    Raises:
        ValueError: Se projections estiver vazio ou se alguma projeção não tiver uma linha por
            documento e 2 colunas.

    Note:
        - A mesma área tem a mesma cor em todos os painéis.
        - As áreas maiores são desenhadas primeiro, para não cobrir as menores.
        - Os eixos do UMAP não têm unidade, por isso ficam sem marcações.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if not projections:
        raise ValueError("projections está vazio. Passar ao menos uma projeção.")
    categories = pd.Series(list(colors), dtype=object).fillna("Sem área").astype(str)
    for name, x_2d in projections.items():
        if x_2d.shape != (len(categories), 2):
            raise ValueError(
                f"A projeção {name} tem formato {x_2d.shape}; esperado ({len(categories)}, 2)."
            )

    palette = _category_palette(sorted(categories.unique()))
    # Ordem de desenho: da área maior para a menor.
    drawing_order = categories.value_counts().index.to_list()

    n_panels = len(projections)
    n_cols = min(UMAP_GRID_COLUMNS, n_panels)
    n_rows = math.ceil(n_panels / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows), squeeze=False, layout="constrained"
    )
    for ax, (name, x_2d) in zip(axes.flat, projections.items(), strict=False):
        for category in drawing_order:
            mask = (categories == category).to_numpy()
            ax.scatter(
                x_2d[mask, 0],
                x_2d[mask, 1],
                s=2,
                alpha=0.6,
                color=palette[category],
                linewidths=0,
                rasterized=True,
            )
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.flat[n_panels:]:
        ax.set_visible(False)

    handles = [
        Line2D([], [], marker="o", linestyle="", color=palette[category], label=category)
        for category in sorted(palette)
    ]
    fig.legend(
        handles=handles,
        title="Área",
        loc="outside lower center",
        ncol=min(5, len(handles)),
        frameon=False,
    )
    return fig


def plot_silhouette_curves(
    sweeps: Mapping[str, pd.DataFrame],
    reference_k: int | None = None,
) -> Figure:
    """Desenha a silhueta em função de k para cada representação (Figura 2).

    Args:
        sweeps: Resultado de clustering.silhouette_sweep, indexado pelo nome da representação.
        reference_k: Valor de k marcado com uma linha vertical, como o número de áreas. None
            não desenha a linha.

    Returns:
        Figura com uma curva por representação.

    Raises:
        ValueError: Se sweeps estiver vazio ou se alguma tabela não tiver as colunas k e
            silhouette.

    Note:
        - O ponto destacado em cada curva é o k de maior silhueta (P3).
    """
    import matplotlib.pyplot as plt

    if not sweeps:
        raise ValueError("sweeps está vazio. Passar ao menos uma varredura.")
    for name, sweep in sweeps.items():
        missing = {"k", "silhouette"} - set(sweep.columns)
        if missing:
            raise ValueError(f"A varredura {name} não tem as colunas {sorted(missing)}.")

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for name, sweep in sweeps.items():
        ordered = sweep.sort_values("k")
        (line,) = ax.plot(ordered["k"], ordered["silhouette"], label=name)
        best = ordered.loc[ordered["silhouette"].idxmax()]
        ax.scatter(best["k"], best["silhouette"], color=line.get_color(), zorder=3)
    if reference_k is not None:
        ax.axvline(
            reference_k,
            color="gray",
            linestyle="--",
            linewidth=1,
            label=f"Número de áreas (k = {reference_k})",
        )
    ax.set_xlabel("k (número de grupos)")
    ax.set_ylabel("Silhueta")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    return fig


def _category_palette(categories: Sequence[str]) -> dict[str, tuple[float, ...]]:
    """Associa cada categoria a uma cor de uma paleta qualitativa do matplotlib."""
    import matplotlib as mpl

    cmap = mpl.colormaps["tab10" if len(categories) <= 10 else "tab20"]
    return {category: cmap(i % cmap.N) for i, category in enumerate(categories)}
