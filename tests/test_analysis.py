"""Especificação executável de analysis.py."""

from collections.abc import Iterator

import matplotlib
import numpy as np
import pandas as pd
import pytest

from juristcu import analysis
from juristcu.config import PT_STOPWORDS

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TEXTS = [
    "A licitação exige pregão eletrônico em 2021.",
    "O pregão da licitação foi anulado pelo órgão.",
    "A aposentadoria do servidor depende do tempo de serviço.",
    "O servidor pediu a revisão da aposentadoria.",
]
LABELS = np.array([0, 0, 1, 1])


@pytest.fixture(autouse=True)
def close_figures() -> Iterator[None]:
    yield
    plt.close("all")


def test_top_terms_describe_each_group() -> None:
    terms = analysis.top_terms_per_cluster(TEXTS, LABELS, n_terms=2)
    assert list(terms) == [0, 1]
    assert set(terms[0]) == {"licitação", "pregão"}
    assert set(terms[1]) == {"aposentadoria", "servidor"}


def test_top_terms_skip_stopwords_and_numbers() -> None:
    terms = analysis.top_terms_per_cluster(TEXTS, LABELS, n_terms=50)
    all_terms = {term for group_terms in terms.values() for term in group_terms}
    assert not all_terms & set(PT_STOPWORDS)
    assert "2021" not in all_terms


def test_top_terms_respect_n_terms() -> None:
    terms = analysis.top_terms_per_cluster(TEXTS, LABELS, n_terms=3)
    assert all(len(group_terms) == 3 for group_terms in terms.values())


def test_top_terms_accept_noise_label() -> None:
    labels = np.array([-1, -1, 4, 4])
    terms = analysis.top_terms_per_cluster(TEXTS, labels, n_terms=2)
    assert list(terms) == [-1, 4]


def test_top_terms_reject_size_mismatch() -> None:
    with pytest.raises(ValueError, match="labels"):
        analysis.top_terms_per_cluster(TEXTS, LABELS[:-1])


def test_stopwords_are_lowercase_and_unique() -> None:
    assert all(word == word.lower() for word in PT_STOPWORDS)
    assert len(PT_STOPWORDS) == len(set(PT_STOPWORDS))


def test_umap_grid_has_one_visible_panel_per_representation() -> None:
    rng = np.random.default_rng(0)
    projections = {name: rng.normal(size=(6, 2)) for name in ["R1", "R2", "R3", "R4", "R5"]}
    colors = ["Licitação", "Pessoal", "Licitação", "Contrato", "Pessoal", "Licitação"]
    fig = analysis.plot_umap_grid(projections, colors)
    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert [ax.get_title() for ax in visible] == list(projections)
    labels = [text.get_text() for text in fig.legends[0].get_texts()]
    assert labels == ["Contrato", "Licitação", "Pessoal"]


def test_umap_grid_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="R1"):
        analysis.plot_umap_grid({"R1": np.zeros((3, 2))}, ["a", "b"])


def test_umap_grid_rejects_empty_projections() -> None:
    with pytest.raises(ValueError, match="vazio"):
        analysis.plot_umap_grid({}, [])


def _sweep(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"k": range(5, 5 + len(values)), "silhouette": values})


def test_silhouette_curves_draw_one_line_per_representation() -> None:
    sweeps = {"R1": _sweep([0.1, 0.3, 0.2]), "R2": _sweep([0.2, 0.1, 0.05])}
    fig = analysis.plot_silhouette_curves(sweeps)
    ax = fig.axes[0]
    assert [line.get_label() for line in ax.get_lines()] == ["R1", "R2"]
    # Um ponto destacado por curva, no k de maior silhueta.
    best = [tuple(collection.get_offsets()[0]) for collection in ax.collections]
    assert best == [(6.0, 0.3), (5.0, 0.2)]


def test_silhouette_curves_draw_reference_k() -> None:
    fig = analysis.plot_silhouette_curves({"R1": _sweep([0.1, 0.3])}, reference_k=10)
    labels = [line.get_label() for line in fig.axes[0].get_lines()]
    assert labels == ["R1", "Número de áreas (k = 10)"]


def test_silhouette_curves_reject_missing_columns() -> None:
    with pytest.raises(ValueError, match="silhouette"):
        analysis.plot_silhouette_curves({"R1": pd.DataFrame({"k": [5, 6]})})
