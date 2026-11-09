"""Confere as constantes de config.py com o protocolo do estudo."""

import dataclasses

import pytest

from juristcu import config
from juristcu.config import TsdaeConfig


def test_corpus_constants_match_proposal() -> None:
    # Frases de 5 a 80 palavras; amostra de 20 mil frases.
    assert config.MIN_WORDS == 5
    assert config.MAX_WORDS == 80
    assert config.N_SENTENCES == 20_000


def test_representation_constants_match_proposal() -> None:
    # LSA em 300 dimensões; prefixo do E5.
    assert config.LSA_DIM == 300
    assert config.E5_PREFIX == "query: "


def test_tsdae_defaults_match_proposal() -> None:
    # Deleção de 60%, lote 8, taxa 3e-5, 1 época, pooling CLS.
    tsdae = TsdaeConfig()
    assert tsdae.deletion_ratio == 0.6
    assert tsdae.batch_size == 8
    assert tsdae.learning_rate == 3e-5
    assert tsdae.num_epochs == 1
    assert tsdae.pooling == "cls"


def test_tsdae_config_is_frozen() -> None:
    tsdae = TsdaeConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        tsdae.batch_size = 16  # type: ignore[misc]


def test_tsdae_seeds_are_three_distinct_values() -> None:
    assert len(config.TSDAE_SEEDS) == 3
    assert len(set(config.TSDAE_SEEDS)) == 3


def test_clustering_constants_match_proposal() -> None:
    # k de 5 a 50, inclusive; k-means com 10 reinícios.
    assert list(config.K_RANGE) == list(range(5, 51))
    assert config.KMEANS_N_INIT == 10


def test_evaluation_constants_match_proposal() -> None:
    # Temas com ao menos 20 documentos; 1.000 reamostragens; IC de 95%.
    assert config.MIN_TOPIC_SIZE == 20
    assert config.N_BOOTSTRAP == 1_000
    assert config.CI_LEVEL == 0.95
