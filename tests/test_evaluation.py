"""Especificação executável de evaluation.py."""

import dataclasses

import numpy as np
import pytest

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


@not_implemented
def test_external_scores_are_one_for_identical_partitions() -> None:
    labels = np.array([0, 0, 1, 1, 2, 2])
    scores = evaluation.external_scores(labels, labels)
    assert set(scores) == {"nmi", "ari"}
    assert scores["nmi"] == pytest.approx(1.0)
    assert scores["ari"] == pytest.approx(1.0)


@not_implemented
def test_external_scores_ignore_label_permutation() -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([2, 2, 0, 0, 1, 1])
    scores = evaluation.external_scores(y_true, y_pred)
    assert scores["nmi"] == pytest.approx(1.0)
    assert scores["ari"] == pytest.approx(1.0)


@not_implemented
def test_filter_min_size_keeps_labels_at_threshold() -> None:
    y_true = np.array(["a"] * 3 + ["b"] * 2 + ["c"])
    mask = evaluation.filter_min_size(y_true, min_size=2)
    assert mask.dtype == bool
    assert mask.tolist() == [True] * 5 + [False]


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
