"""Validação externa e interna dos agrupamentos, com bootstrap.

- Rótulos (AREA e TEMA) entram apenas na avaliação, nunca no treinamento.
- NMI calculada com a normalização de config.NMI_AVERAGE_METHOD em todas as funções.
- Esqueleto: as funções de bootstrap levantam NotImplementedError até serem implementadas.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from juristcu.config import CI_LEVEL, MIN_TOPIC_SIZE, N_BOOTSTRAP, NMI_AVERAGE_METHOD, SEED


@dataclass(frozen=True)
class ConfidenceInterval:
    """Estimativa pontual com intervalo de confiança.

    Attributes:
        estimate: Valor calculado com todos os documentos.
        low: Limite inferior do intervalo.
        high: Limite superior do intervalo.
    """

    estimate: float
    low: float
    high: float


def external_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compara um agrupamento com os rótulos do TCU.

    Usa normalized_mutual_info_score e adjusted_rand_score, do scikit-learn.

    Args:
        y_true: Área ou tema de cada documento.
        y_pred: Grupo de cada documento.

    Returns:
        Dicionário com as chaves "nmi" e "ari", com valores float.

    Note:
        - A NMI usa average_method=NMI_AVERAGE_METHOD. O bootstrap deve usar a mesma definição.
    """
    nmi = normalized_mutual_info_score(y_true, y_pred, average_method=NMI_AVERAGE_METHOD)
    ari = adjusted_rand_score(y_true, y_pred)
    return {"nmi": float(nmi), "ari": float(ari)}


def internal_score(
    x: np.ndarray,
    labels: np.ndarray,
    seed: int = SEED,
    sample_size: int | None = None,
) -> float:
    """Calcula o coeficiente de silhueta de um agrupamento.

    Args:
        x: Representação normalizada, uma linha por documento.
        labels: Grupo de cada documento.
        seed: Semente da subamostra, usada apenas quando sample_size não é None.
        sample_size: Número de documentos sorteados. None usa todos.

    Returns:
        Silhueta média, entre -1 e 1.

    Raises:
        ValueError: Se houver um único grupo ou tantos grupos quanto documentos.

    Note:
        - Usa a distância euclidiana, a mesma do k-means. Com vetores de norma 1, ela ordena os
          pares como a distância do cosseno.
        - O custo cresce com o quadrado do número de documentos. sample_size reduz o tempo,
          mas muda o valor da silhueta.
    """
    score = silhouette_score(
        x, labels, metric="euclidean", sample_size=sample_size, random_state=seed
    )
    return float(score)


def filter_min_size(y_true: np.ndarray, min_size: int = MIN_TOPIC_SIZE) -> np.ndarray:
    """Marca os documentos cujo rótulo tem ao menos min_size documentos.

    Args:
        y_true: Tema de cada documento.
        min_size: Número mínimo de documentos por rótulo, inclusive.

    Returns:
        Máscara booleana com o mesmo tamanho de y_true.

    Note:
        - Na análise por TEMA, aplicar a máscara antes do k-means.
        - O k-means por TEMA usa k igual ao número de temas mantidos.
        - Rótulos ausentes (None ou NaN) nunca são mantidos.
    """
    labels = pd.Series(np.asarray(y_true, dtype=object))
    counts = labels.map(labels.value_counts(dropna=True))
    return (counts.ge(min_size) & labels.notna()).to_numpy(dtype=bool)


def noise_aware_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compara com os rótulos um agrupamento que marca ruído com -1.

    Args:
        y_true: Área ou tema de cada documento.
        y_pred: Grupo de cada documento. O valor -1 indica ruído.

    Returns:
        Dicionário com as chaves:
        - "nmi" e "ari": calculadas como em external_scores, só nos documentos fora do ruído.
          Valem NaN quando todos os documentos são ruído.
        - "n_groups": número de grupos, sem contar o ruído.
        - "noise_share": fração dos documentos marcados como ruído.

    Note:
        - Serve ao HDBSCAN, que não fixa o número de grupos nem cobre todos os documentos.
        - O resultado não se compara diretamente ao do k-means.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    kept = y_pred != -1
    n_groups = int(np.unique(y_pred[kept]).size)
    noise_share = float(np.mean(~kept)) if len(y_pred) else 0.0
    if not kept.any():
        scores = {"nmi": float("nan"), "ari": float("nan")}
    else:
        scores = external_scores(y_true[kept], y_pred[kept])
    return {**scores, "n_groups": n_groups, "noise_share": noise_share}


def bootstrap_nmi_ci(
    y_true: np.ndarray,
    preds: Sequence[np.ndarray],
    n_resamples: int = N_BOOTSTRAP,
    seed: int = SEED,
    level: float = CI_LEVEL,
) -> ConfidenceInterval:
    """Estima o intervalo de confiança da NMI média por bootstrap dos documentos.

    - Cada reamostragem sorteia documentos com reposição, com np.random.default_rng(seed).
    - Em cada reamostragem, calcula a média da NMI dos agrupamentos.
    - Um único agrupamento usa uma lista de um elemento. As 3 sementes de R3 usam três.
    - O intervalo é percentil: quantis (1 - level) / 2 e (1 + level) / 2.

    Args:
        y_true: Área ou tema de cada documento.
        preds: Agrupamentos da mesma representação, um por semente.
        n_resamples: Número de reamostragens.
        seed: Semente do bootstrap.
        level: Nível de confiança.

    Returns:
        NMI média com todos os documentos e limites do intervalo.

    Raises:
        ValueError: Se preds estiver vazio.

    Note:
        - O bootstrap reamostra as atribuições já obtidas. O k-means não é refeito.
        - NMI calculada como em external_scores.
    """
    raise NotImplementedError


def bootstrap_nmi_difference(
    y_true: np.ndarray,
    preds_a: Sequence[np.ndarray],
    preds_b: Sequence[np.ndarray],
    n_resamples: int = N_BOOTSTRAP,
    seed: int = SEED,
    level: float = CI_LEVEL,
) -> ConfidenceInterval:
    """Estima o intervalo de confiança da diferença de NMI entre duas representações.

    - Diferença = média da NMI dos agrupamentos de a - média da NMI dos agrupamentos de b.
    - Cada lado aceita vários agrupamentos, como as 3 sementes de R3.
    - Reamostragem e intervalo seguem bootstrap_nmi_ci.

    Args:
        y_true: Área ou tema de cada documento.
        preds_a: Partições da primeira representação.
        preds_b: Partições da segunda representação.
        n_resamples: Número de reamostragens.
        seed: Semente do bootstrap.
        level: Nível de confiança.

    Returns:
        Diferença com todos os documentos e limites do intervalo.

    Raises:
        ValueError: Se preds_a ou preds_b estiver vazio.

    Note:
        - Bootstrap pareado: os mesmos índices sorteados valem para todos os agrupamentos.
        - H1 e H2 se confirmam quando o limite inferior é positivo.
        - NMI calculada como em external_scores.
    """
    raise NotImplementedError
