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


def test_nmi_uses_arithmetic_average() -> None:
    # Média aritmética das entropias, padrão do scikit-learn.
    assert config.NMI_AVERAGE_METHOD == "arithmetic"


def test_umap_and_hdbscan_defaults() -> None:
    # UMAP em 2 componentes, 15 vizinhos e distância mínima 0,1; HDBSCAN com 20 documentos.
    assert config.UMAP_N_COMPONENTS == 2
    assert config.UMAP_N_NEIGHBORS == 15
    assert config.UMAP_MIN_DIST == 0.1
    assert config.HDBSCAN_MIN_CLUSTER_SIZE == 20


def test_representation_names() -> None:
    # Sete nomes únicos; R3 aparece uma vez por semente do TSDAE.
    names = config.REPRESENTATION_NAMES
    r3_names = tuple(f"R3-seed{seed}" for seed in config.TSDAE_SEEDS)
    assert names == ("R1", "R2", "R2-CLS", *r3_names, "R4")
    assert len(set(names)) == 7
