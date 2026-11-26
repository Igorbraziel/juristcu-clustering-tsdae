"""Especificação executável de evaluation.py."""

import dataclasses

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import normalized_mutual_info_score, silhouette_score

from juristcu import evaluation
from juristcu.evaluation import ConfidenceInterval


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


def test_external_scores_use_adjusted_rand() -> None:
    # Partições independentes: o ARI é negativo, ao contrário do índice de Rand sem ajuste.
    scores = evaluation.external_scores(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))
    assert scores["ari"] == pytest.approx(-0.5)


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
    expected = silhouette_score(x, labels, metric="euclidean", sample_size=100, random_state=1)
    assert first == pytest.approx(expected)
    assert first != pytest.approx(evaluation.internal_score(x, labels))


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


@pytest.mark.parametrize("min_size", [0, -1])
def test_filter_min_size_drops_missing_labels_without_threshold(min_size: int) -> None:
    y_true = np.array(["a", None, np.nan], dtype=object)
    mask = evaluation.filter_min_size(y_true, min_size=min_size)
    assert mask.tolist() == [True, False, False]


def test_filter_min_size_accepts_numeric_labels() -> None:
    y_true = np.array([1.0, 1.0, 2.0, np.nan, np.nan, 3.0, 3.0, 3.0])
    mask = evaluation.filter_min_size(y_true, min_size=2)
    assert mask.shape == y_true.shape
    assert mask.tolist() == [True, True, False, False, False, True, True, True]


def test_noise_aware_scores_ignore_noise() -> None:
    # O ruído (-1) fica fora da NMI e do ARI e entra na fração de ruído.
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([5, 5, 7, 7, -1, -1])
    scores = evaluation.noise_aware_scores(y_true, y_pred)
    assert set(scores) == {"nmi", "ari", "n_groups", "noise_share"}
    assert scores["nmi"] == pytest.approx(1.0)
    assert scores["ari"] == pytest.approx(1.0)
    assert scores["n_groups"] == 2
    assert scores["noise_share"] == pytest.approx(2 / 6)


def test_noise_aware_scores_with_only_noise() -> None:
    scores = evaluation.noise_aware_scores(np.array([0, 1, 1]), np.array([-1, -1, -1]))
    assert scores["n_groups"] == 0
    assert scores["noise_share"] == pytest.approx(1.0)
    assert np.isnan(scores["nmi"])
    assert np.isnan(scores["ari"])


def test_bootstrap_nmi_ci_is_reproducible(partitions: tuple[np.ndarray, np.ndarray]) -> None:
    y_true, y_pred = partitions
    first = evaluation.bootstrap_nmi_ci(y_true, [y_pred], n_resamples=200, seed=3)
    second = evaluation.bootstrap_nmi_ci(y_true, [y_pred], n_resamples=200, seed=3)
    assert first == second


def test_bootstrap_nmi_ci_contains_estimate(partitions: tuple[np.ndarray, np.ndarray]) -> None:
    y_true, y_pred = partitions
    interval = evaluation.bootstrap_nmi_ci(y_true, [y_pred], n_resamples=200, seed=3)
    assert interval.low <= interval.estimate <= interval.high


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


def test_bootstrap_difference_is_reproducible(
    partitions: tuple[np.ndarray, np.ndarray],
) -> None:
    y_true, y_pred = partitions
    args = (y_true, [y_true], [y_pred])
    first = evaluation.bootstrap_nmi_difference(*args, n_resamples=200, seed=5)
    second = evaluation.bootstrap_nmi_difference(*args, n_resamples=200, seed=5)
    assert first == second


def test_bootstrap_nmi_ci_estimate_is_mean_nmi(
    partitions: tuple[np.ndarray, np.ndarray],
) -> None:
    y_true, y_pred = partitions
    interval = evaluation.bootstrap_nmi_ci(y_true, [y_true, y_pred], n_resamples=50, seed=3)
    nmi_pred = evaluation.external_scores(y_true, y_pred)["nmi"]
    assert interval.estimate == pytest.approx((1.0 + nmi_pred) / 2)


def test_bootstrap_nmi_ci_does_not_depend_on_label_names(
    partitions: tuple[np.ndarray, np.ndarray],
) -> None:
    y_true, y_pred = partitions
    names = np.array(["Licitação", "Pessoal", "Contrato", "Convênio"])
    numeric = evaluation.bootstrap_nmi_ci(y_true, [y_pred], n_resamples=100, seed=3)
    text = evaluation.bootstrap_nmi_ci(names[y_true], [y_pred], n_resamples=100, seed=3)
    # A ordem dos rótulos muda só o arredondamento das somas.
    assert dataclasses.astuple(numeric) == pytest.approx(dataclasses.astuple(text))


def test_bootstrap_nmi_ci_accepts_missing_labels() -> None:
    y_true = np.array(["a", "a", None, None, "b", "b"], dtype=object)
    interval = evaluation.bootstrap_nmi_ci(y_true, [np.array([0, 0, 1, 1, 2, 2])], n_resamples=20)
    assert interval.estimate == pytest.approx(1.0)


def test_bootstrap_nmi_ci_rejects_empty_preds(partitions: tuple[np.ndarray, np.ndarray]) -> None:
    y_true, _ = partitions
    with pytest.raises(ValueError, match="vazio"):
        evaluation.bootstrap_nmi_ci(y_true, [])


def test_bootstrap_nmi_ci_rejects_size_mismatch(
    partitions: tuple[np.ndarray, np.ndarray],
) -> None:
    y_true, y_pred = partitions
    with pytest.raises(ValueError, match="documentos"):
        evaluation.bootstrap_nmi_ci(y_true, [y_pred[:-1]])


def test_bootstrap_uses_same_nmi_as_external_scores(
    partitions: tuple[np.ndarray, np.ndarray],
) -> None:
    # A estimativa pontual do bootstrap coincide com a NMI de external_scores.
    y_true, y_pred = partitions
    interval = evaluation.bootstrap_nmi_ci(y_true, [y_pred], n_resamples=10, seed=0)
    assert interval.estimate == pytest.approx(evaluation.external_scores(y_true, y_pred)["nmi"])


def test_bootstrap_difference_has_positive_bound_for_better_partition(
    partitions: tuple[np.ndarray, np.ndarray],
) -> None:
    # Os rótulos verdadeiros superam um agrupamento ruidoso em toda reamostragem.
    y_true, y_pred = partitions
    interval = evaluation.bootstrap_nmi_difference(
        y_true, [y_true], [y_pred], n_resamples=200, seed=5
    )
    assert interval.low > 0


@pytest.mark.parametrize("empty_side", ["a", "b"])
def test_bootstrap_difference_rejects_empty_side(
    partitions: tuple[np.ndarray, np.ndarray], empty_side: str
) -> None:
    y_true, y_pred = partitions
    preds_a, preds_b = ([], [y_pred]) if empty_side == "a" else ([y_pred], [])
    with pytest.raises(ValueError, match=f"preds_{empty_side}"):
        evaluation.bootstrap_nmi_difference(y_true, preds_a, preds_b)
