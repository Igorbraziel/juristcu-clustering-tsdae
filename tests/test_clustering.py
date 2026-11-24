"""Especificação executável de clustering.py."""

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, silhouette_score

from juristcu import clustering, config

SWEEP_COLUMNS = ["k", "silhouette"]
BEST_K_COLUMNS = ["representation", "best_k", "best_silhouette", "silhouette_at_n_areas", "n_areas"]


@pytest.fixture(scope="module")
def blobs() -> tuple[np.ndarray, np.ndarray]:
    # Quatro grupos bem separados, com semente fixa.
    x, labels = make_blobs(n_samples=120, centers=4, cluster_std=0.5, random_state=0)
    return x, labels


def sweep(k_values: list[int], scores: list[float]) -> pd.DataFrame:
    # Varredura sintética, no formato de silhouette_sweep.
    return pd.DataFrame({"k": k_values, "silhouette": scores})


# k-means


def test_run_kmeans_recovers_true_groups(blobs: tuple[np.ndarray, np.ndarray]) -> None:
    x, labels = blobs
    preds = clustering.run_kmeans(x, k=4, seed=0)
    assert adjusted_rand_score(labels, preds) == pytest.approx(1.0)


def test_run_kmeans_labels_are_integers_in_range(blobs: tuple[np.ndarray, np.ndarray]) -> None:
    x, _ = blobs
    preds = clustering.run_kmeans(x, k=4, seed=0)
    assert preds.shape == (len(x),)
    assert np.issubdtype(preds.dtype, np.integer)
    assert set(preds.tolist()) == set(range(4))


def test_run_kmeans_is_deterministic_with_seed(blobs: tuple[np.ndarray, np.ndarray]) -> None:
    x, _ = blobs
    first = clustering.run_kmeans(x, k=5, seed=3)
    second = clustering.run_kmeans(x, k=5, seed=3)
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("n_init", [1, 10])
def test_run_kmeans_accepts_n_init(blobs: tuple[np.ndarray, np.ndarray], n_init: int) -> None:
    x, _ = blobs
    preds = clustering.run_kmeans(x, k=4, seed=0, n_init=n_init)
    assert preds.shape == (len(x),)
    assert set(preds.tolist()) <= set(range(4))


def test_run_kmeans_passes_n_init(
    blobs: tuple[np.ndarray, np.ndarray], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Registra os argumentos do KMeans sem alterar o comportamento.
    calls = []
    original = clustering.KMeans

    def spy(**kwargs: object) -> object:
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(clustering, "KMeans", spy)
    x, _ = blobs
    clustering.run_kmeans(x, k=4, seed=7, n_init=3)
    clustering.run_kmeans(x, k=4, seed=7)
    assert calls == [
        {"n_clusters": 4, "init": "k-means++", "n_init": 3, "random_state": 7},
        {"n_clusters": 4, "init": "k-means++", "n_init": config.KMEANS_N_INIT, "random_state": 7},
    ]


# Varredura de k


def test_silhouette_sweep_has_one_row_per_k(blobs: tuple[np.ndarray, np.ndarray]) -> None:
    x, _ = blobs
    k_values = [6, 2, 4]
    table = clustering.silhouette_sweep(x, k_values, seed=0)
    assert list(table.columns) == SWEEP_COLUMNS
    assert table["k"].tolist() == k_values
    assert pd.api.types.is_integer_dtype(table["k"])
    assert pd.api.types.is_float_dtype(table["silhouette"])


def test_silhouette_sweep_peaks_at_true_k(blobs: tuple[np.ndarray, np.ndarray]) -> None:
    x, _ = blobs
    table = clustering.silhouette_sweep(x, range(2, 7), seed=0)
    assert table["k"].tolist() == [2, 3, 4, 5, 6]
    assert table.loc[table["silhouette"].idxmax(), "k"] == 4


def test_silhouette_sweep_uses_all_documents_by_default(
    blobs: tuple[np.ndarray, np.ndarray],
) -> None:
    # Sem sample_size, a silhueta usa todos os documentos e o k-means recebe a semente.
    x, _ = blobs
    table = clustering.silhouette_sweep(x, [3, 4], seed=2)
    for k, score in zip(table["k"], table["silhouette"], strict=True):
        labels = KMeans(
            n_clusters=k, init="k-means++", n_init=config.KMEANS_N_INIT, random_state=2
        ).fit_predict(x)
        assert score == pytest.approx(silhouette_score(x, labels, metric="euclidean"))


def test_silhouette_sweep_sample_is_reproducible(blobs: tuple[np.ndarray, np.ndarray]) -> None:
    # A mesma semente sorteia a mesma subamostra.
    x, _ = blobs
    first = clustering.silhouette_sweep(x, range(2, 5), seed=1, sample_size=60)
    second = clustering.silhouette_sweep(x, range(2, 5), seed=1, sample_size=60)
    pd.testing.assert_frame_equal(first, second)


# UMAP


@pytest.fixture(scope="module")
def small_blobs() -> np.ndarray:
    x, _ = make_blobs(n_samples=60, centers=3, n_features=8, cluster_std=0.5, random_state=0)
    return x


def test_project_umap_returns_float32_projection(small_blobs: np.ndarray) -> None:
    projection = clustering.project_umap(small_blobs, seed=0, n_neighbors=5)
    assert projection.shape == (len(small_blobs), 2)
    assert projection.dtype == np.float32


def test_project_umap_uses_config_defaults(
    small_blobs: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Registra os argumentos do UMAP sem alterar o comportamento.
    import umap

    calls = []
    original = umap.UMAP

    def spy(**kwargs: object) -> object:
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(umap, "UMAP", spy)
    clustering.project_umap(small_blobs, seed=4)
    assert calls == [
        {
            "n_components": config.UMAP_N_COMPONENTS,
            "n_neighbors": config.UMAP_N_NEIGHBORS,
            "min_dist": config.UMAP_MIN_DIST,
            "metric": "euclidean",
            "random_state": 4,
            "n_jobs": 1,
        }
    ]


def test_project_umap_is_deterministic_with_seed(small_blobs: np.ndarray) -> None:
    first = clustering.project_umap(small_blobs, seed=0, n_neighbors=5)
    second = clustering.project_umap(small_blobs, seed=0, n_neighbors=5)
    np.testing.assert_array_equal(first, second)


# HDBSCAN


@pytest.fixture(scope="module")
def dense_blobs_with_outlier() -> np.ndarray:
    # Três grupos densos e um ponto muito distante, na última linha.
    x, _ = make_blobs(
        n_samples=90, centers=[[0, 0], [10, 10], [-10, 10]], cluster_std=0.3, random_state=0
    )
    return np.vstack([x, [[100.0, -100.0]]])


def test_run_hdbscan_marks_isolated_point_as_noise(dense_blobs_with_outlier: np.ndarray) -> None:
    preds = clustering.run_hdbscan(dense_blobs_with_outlier, min_cluster_size=10)
    assert preds.shape == (len(dense_blobs_with_outlier),)
    assert preds[-1] == -1
    assert len(set(preds.tolist()) - {-1}) >= 3


def test_run_hdbscan_emits_no_future_warning(dense_blobs_with_outlier: np.ndarray) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        clustering.run_hdbscan(dense_blobs_with_outlier, min_cluster_size=10)


def test_run_hdbscan_passes_parameters(
    dense_blobs_with_outlier: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Registra os argumentos do HDBSCAN sem alterar o comportamento.
    calls = []
    original = clustering.HDBSCAN

    def spy(**kwargs: object) -> object:
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(clustering, "HDBSCAN", spy)
    clustering.run_hdbscan(dense_blobs_with_outlier)
    clustering.run_hdbscan(dense_blobs_with_outlier, min_cluster_size=10)
    assert calls == [
        {"min_cluster_size": config.HDBSCAN_MIN_CLUSTER_SIZE, "copy": True},
        {"min_cluster_size": 10, "copy": True},
    ]


# Tabela das atribuições


def test_assignments_table_has_document_then_representations() -> None:
    documents = np.array([0, 1, 2])
    preds = {"R4": np.array([1, 0, 1], dtype=np.int32), "R1": np.array([2, 2, 0])}
    table = clustering.assignments_table(preds, documents)
    assert list(table.columns) == ["document", "R4", "R1"]
    assert table["document"].tolist() == [0, 1, 2]
    assert table["R4"].tolist() == [1, 0, 1]
    assert table["R1"].tolist() == [2, 2, 0]
    assert (table.dtypes == np.int64).all()


def test_assignments_table_keeps_noise_label() -> None:
    table = clustering.assignments_table({"R1": np.array([-1, 0])}, np.array([0, 1]))
    assert table["R1"].tolist() == [-1, 0]


def test_assignments_table_rejects_non_integer_labels() -> None:
    # Rótulos fracionários ou NaN indicariam uma tabela corrompida.
    with pytest.raises(ValueError):
        clustering.assignments_table({"R1": np.array([0.0, np.nan, 2.0])}, np.arange(3))


def test_assignments_table_rejects_length_mismatch() -> None:
    preds = {"R1": np.array([0, 1, 0]), "R2": np.array([0, 1])}
    with pytest.raises(ValueError):
        clustering.assignments_table(preds, np.array([0, 1, 2]))


# Tabela do melhor k


def test_best_k_table_has_one_row_per_representation() -> None:
    sweeps = {
        "R2": sweep([5, 6, 7], [0.1, 0.3, 0.2]),
        "R1": sweep([5, 6, 7], [0.4, 0.2, 0.1]),
    }
    table = clustering.best_k_table(sweeps, n_areas=6)
    assert list(table.columns) == BEST_K_COLUMNS
    assert table["representation"].tolist() == ["R2", "R1"]
    assert table["best_k"].tolist() == [6, 5]
    assert table["best_silhouette"].tolist() == pytest.approx([0.3, 0.4])
    assert table["silhouette_at_n_areas"].tolist() == pytest.approx([0.3, 0.2])
    assert table["n_areas"].tolist() == [6, 6]


def test_best_k_table_has_integer_k() -> None:
    table = clustering.best_k_table({"R1": sweep([5, 10], [0.1, 0.3])}, n_areas=10)
    assert table["best_k"].dtype == np.int64
    assert table["n_areas"].dtype == np.int64


def test_best_k_table_rejects_empty_sweep() -> None:
    with pytest.raises(ValueError):
        clustering.best_k_table({"R1": sweep([], [])}, n_areas=10)


def test_best_k_table_breaks_ties_with_smallest_k() -> None:
    # A varredura fora de ordem confirma que o empate não depende da posição da linha.
    sweeps = {"R1": sweep([9, 5, 7], [0.4, 0.2, 0.4])}
    table = clustering.best_k_table(sweeps, n_areas=5)
    assert table.loc[0, "best_k"] == 7
    assert table.loc[0, "best_silhouette"] == pytest.approx(0.4)


def test_best_k_table_gives_nan_outside_sweep() -> None:
    sweeps = {"R1": sweep([5, 6], [0.1, 0.2])}
    table = clustering.best_k_table(sweeps, n_areas=12)
    assert np.isnan(table.loc[0, "silhouette_at_n_areas"])
    assert table.loc[0, "n_areas"] == 12


# Tabela das varreduras


def test_sweep_table_is_long_format() -> None:
    sweeps = {
        "R3-seed0": sweep([6, 5], [0.3, 0.1]),
        "R1": sweep([5, 6, 7], [0.4, 0.2, 0.1]),
    }
    table = clustering.sweep_table(sweeps)
    assert list(table.columns) == ["representation", "k", "silhouette"]
    assert len(table) == 5
    assert table["representation"].tolist() == ["R3-seed0"] * 2 + ["R1"] * 3
    assert table["k"].tolist() == [5, 6, 5, 6, 7]
    assert table["silhouette"].tolist() == pytest.approx([0.1, 0.3, 0.4, 0.2, 0.1])


# Retomada da varredura


def test_completed_sweeps_keeps_only_complete_representations() -> None:
    # R1 completa, R2 incompleta e R3 ausente; a ordem segue names.
    saved = pd.DataFrame(
        {
            "representation": ["R2", "R1", "R1", "R1"],
            "k": [5, 7, 5, 6],
            "silhouette": [0.2, 0.3, 0.1, 0.4],
        }
    )
    sweeps = clustering.completed_sweeps(saved, ["R1", "R2", "R3"], range(5, 8))
    assert list(sweeps) == ["R1"]
    assert list(sweeps["R1"].columns) == SWEEP_COLUMNS
    assert sweeps["R1"]["k"].tolist() == [5, 6, 7]
    assert sweeps["R1"]["silhouette"].tolist() == [0.1, 0.4, 0.3]


def test_completed_sweeps_accepts_empty_table() -> None:
    empty = clustering.sweep_table({})
    assert clustering.completed_sweeps(empty, ["R1"], range(5, 8)) == {}


def test_completed_sweeps_round_trips_sweep_table(blobs: tuple[np.ndarray, np.ndarray]) -> None:
    x, _ = blobs
    original = {"R1": clustering.silhouette_sweep(x, range(2, 5), seed=0)}
    restored = clustering.completed_sweeps(clustering.sweep_table(original), ["R1"], range(2, 5))
    pd.testing.assert_frame_equal(restored["R1"], original["R1"])
