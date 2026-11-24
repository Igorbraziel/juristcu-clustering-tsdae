"""Especificação executável de evaluation.py."""

import dataclasses

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import normalized_mutual_info_score, silhouette_score

from juristcu import evaluation
from juristcu.evaluation import ConfidenceInterval

not_implemented = pytest.mark.xfail(raises=NotImplementedError, reason="A implementar.")


@pytest.fixture
def partitions() -> tuple[np.ndarray, np.ndarray]:
    # Rótulos e um agrupamento ruidoso, com semente fixa.
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 4, size=200)
    y_pred = np.where(rng.random(200) < 0.7, y_true, rng.integers(0, 4, size=200))
    return y_true, y_pred


def test_confidence_interval_is_frozen() -> None:
    interval = ConfidenceInterval(estimate=0.5, low=0.4, high=0.6)
    with pytest.raises(dataclasses.FrozenInstanceError):
        interval.low = 0.0  # type: ignore[misc]


def test_external_scores_are_one_for_identical_partitions() -> None:
    labels = np.array([0, 0, 1, 1, 2, 2])
    scores = evaluation.external_scores(labels, labels)
    assert set(scores) == {"nmi", "ari"}
    assert scores["nmi"] == pytest.approx(1.0)
    assert scores["ari"] == pytest.approx(1.0)


def test_external_scores_ignore_label_permutation() -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([2, 2, 0, 0, 1, 1])
    scores = evaluation.external_scores(y_true, y_pred)
    assert scores["nmi"] == pytest.approx(1.0)
    assert scores["ari"] == pytest.approx(1.0)


def test_external_scores_use_arithmetic_nmi() -> None:
    # Partições com entropias diferentes: a média aritmética difere da geométrica.
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    scores = evaluation.external_scores(y_true, y_pred)
    arithmetic = normalized_mutual_info_score(y_true, y_pred, average_method="arithmetic")
    geometric = normalized_mutual_info_score(y_true, y_pred, average_method="geometric")
    assert scores["nmi"] == pytest.approx(arithmetic)
    assert scores["nmi"] != pytest.approx(geometric)


def test_external_scores_return_python_floats(
    partitions: tuple[np.ndarray, np.ndarray],
) -> None:
    y_true, y_pred = partitions
    scores = evaluation.external_scores(y_true, y_pred)
    assert all(type(value) is float for value in scores.values())


@pytest.fixture
def blobs() -> tuple[np.ndarray, np.ndarray]:
    # Três grupos bem separados, com semente fixa.
    x, labels = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=0)
    return x, labels


def test_internal_score_matches_euclidean_silhouette(
    blobs: tuple[np.ndarray, np.ndarray],
) -> None:
    x, labels = blobs
    score = evaluation.internal_score(x, labels)
    assert isinstance(score, float)
    assert score > 0.5
    assert score == pytest.approx(silhouette_score(x, labels, metric="euclidean"))


def test_internal_score_sample_is_reproducible(blobs: tuple[np.ndarray, np.ndarray]) -> None:
    # A mesma semente sorteia a mesma subamostra.
    x, labels = blobs
    first = evaluation.internal_score(x, labels, seed=1, sample_size=100)
    second = evaluation.internal_score(x, labels, seed=1, sample_size=100)
    assert first == second


def test_filter_min_size_keeps_labels_at_threshold() -> None:
    y_true = np.array(["a"] * 3 + ["b"] * 2 + ["c"])
    mask = evaluation.filter_min_size(y_true, min_size=2)
    assert mask.dtype == bool
    assert mask.tolist() == [True] * 5 + [False]


def test_filter_min_size_never_keeps_missing_labels() -> None:
    # None e NaN aparecem duas vezes cada, mas não são mantidos.
    y_true = np.array(["a", "a", None, None, np.nan, np.nan, "b"], dtype=object)
    mask = evaluation.filter_min_size(y_true, min_size=2)
    assert mask.shape == y_true.shape
    assert mask.tolist() == [True, True, False, False, False, False, False]


def test_filter_min_size_accepts_numeric_labels() -> None:
    y_true = np.array([1.0, 1.0, 2.0, np.nan, np.nan, 3.0, 3.0, 3.0])
    mask = evaluation.filter_min_size(y_true, min_size=2)
    assert mask.shape == y_true.shape
    assert mask.tolist() == [True, True, False, False, False, True, True, True]


@not_implemented
def test_bootstrap_nmi_ci_is_reproducible(partitions: tuple[np.ndarray, np.ndarray]) -> None:
    y_true, y_pred = partitions
    first = evaluation.bootstrap_nmi_ci(y_true, [y_pred], n_resamples=200, seed=3)
    second = evaluation.bootstrap_nmi_ci(y_true, [y_pred], n_resamples=200, seed=3)
    assert first == second


@not_implemented
def test_bootstrap_nmi_ci_contains_estimate(partitions: tuple[np.ndarray, np.ndarray]) -> None:
    y_true, y_pred = partitions
    interval = evaluation.bootstrap_nmi_ci(y_true, [y_pred], n_resamples=200, seed=3)
    assert interval.low <= interval.estimate <= interval.high


@not_implemented
def test_bootstrap_difference_of_identical_partitions_is_zero(
    partitions: tuple[np.ndarray, np.ndarray],
) -> None:
    y_true, y_pred = partitions
    interval = evaluation.bootstrap_nmi_difference(
        y_true, [y_pred], [y_pred], n_resamples=200, seed=3
    )
    assert interval.estimate == pytest.approx(0.0)
    assert interval.low == pytest.approx(0.0)
    assert interval.high == pytest.approx(0.0)


@not_implemented
def test_bootstrap_difference_averages_partitions(
    partitions: tuple[np.ndarray, np.ndarray],
) -> None:
    # Diferença = média da NMI de a - média da NMI de b.
    y_true, y_pred = partitions
    preds_a = [y_true, y_pred]
    preds_b = [y_pred]
    interval = evaluation.bootstrap_nmi_difference(
        y_true, preds_a, preds_b, n_resamples=200, seed=3
    )
    nmi_pred = evaluation.external_scores(y_true, y_pred)["nmi"]
    expected = (1.0 + nmi_pred) / 2 - nmi_pred
    assert interval.estimate == pytest.approx(expected)
    assert interval.low <= interval.estimate <= interval.high


@not_implemented
def test_bootstrap_difference_is_reproducible(
    partitions: tuple[np.ndarray, np.ndarray],
) -> None:
    y_true, y_pred = partitions
    args = (y_true, [y_true], [y_pred])
    first = evaluation.bootstrap_nmi_difference(*args, n_resamples=200, seed=5)
    second = evaluation.bootstrap_nmi_difference(*args, n_resamples=200, seed=5)
    assert first == second
