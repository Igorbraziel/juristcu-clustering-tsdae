"""Especificação executável de corpus.py."""

import re

import nltk
import pandas as pd
import pytest

from juristcu import corpus
from juristcu.config import AREA_COLUMN, EXCERPT_COLUMN, TEXT_COLUMN, TOPIC_COLUMN


@pytest.fixture
def punkt_portuguese() -> None:
    # Os testes não fazem download. Sem o recurso local, o teste é pulado.
    try:
        nltk.data.find("tokenizers/punkt_tab/portuguese")
    except LookupError:
        pytest.skip("Recurso punkt_tab do NLTK não instalado.")


@pytest.fixture
def simple_splitter(monkeypatch: pytest.MonkeyPatch) -> None:
    # Substitui o Punkt por uma divisão simples; os testes de build_corpus não usam o NLTK.
    def split(text: str) -> list[str]:
        return [part.strip() for part in re.split(r"(?<=\.)\s+", text) if part.strip()]

    monkeypatch.setattr(corpus, "split_sentences", split)


@pytest.fixture
def documents() -> pd.DataFrame:
    # DataFrame sintético, com colunas de rótulo que o corpus deve ignorar.
    statements = [
        f"O órgão deve publicar o edital número {i} com antecedência mínima. "
        f"A comissão analisa a proposta {i} conforme o regulamento vigente."
        for i in range(10)
    ]
    excerpts = [
        f"O relator examinou o processo {i} e acompanhou a unidade técnica. "
        f"A decisão {i} determina a correção das falhas apontadas na auditoria."
        for i in range(10)
    ]
    return pd.DataFrame(
        {
            TEXT_COLUMN: statements,
            EXCERPT_COLUMN: excerpts,
            AREA_COLUMN: ["rotulo_area_a"] * 5 + ["rotulo_area_b"] * 5,
            TOPIC_COLUMN: ["rotulo_tema_a"] * 5 + ["rotulo_tema_b"] * 5,
        }
    )


def _words(n: int) -> str:
    return " ".join(["palavra"] * n)


def test_filter_by_length_is_inclusive_at_both_limits() -> None:
    sentences = [_words(4), _words(5), _words(80), _words(81)]
    assert corpus.filter_by_length(sentences, min_words=5, max_words=80) == [
        _words(5),
        _words(80),
    ]


def test_filter_by_length_uses_config_defaults() -> None:
    sentences = [_words(4), _words(5), _words(80), _words(81)]
    assert corpus.filter_by_length(sentences) == [_words(5), _words(80)]


def test_deduplicate_preserves_first_occurrence_order() -> None:
    sentences = ["c", "a", "c", "b", "a"]
    assert corpus.deduplicate(sentences) == ["c", "a", "b"]


@pytest.mark.usefixtures("punkt_portuguese")
def test_split_sentences_keeps_legal_abbreviations() -> None:
    text = (
        "A contratação direta exige justificativa, nos termos do art. 25 da lei de licitações. "
        "O gestor deve comprovar a inviabilidade de competição."
    )
    sentences = corpus.split_sentences(text)
    assert len(sentences) == 2
    assert "art. 25" in sentences[0]


@pytest.mark.usefixtures("simple_splitter")
def test_build_corpus_is_deterministic_with_seed(documents: pd.DataFrame) -> None:
    first = corpus.build_corpus(documents, n_sentences=10, seed=0)
    second = corpus.build_corpus(documents, n_sentences=10, seed=0)
    assert first == second
    assert len(first) == 10
    assert len(set(first)) == 10


@pytest.mark.usefixtures("simple_splitter")
def test_build_corpus_ignores_label_columns(documents: pd.DataFrame) -> None:
    # Rótulos diferentes não podem alterar o corpus.
    shuffled = documents.assign(
        **{AREA_COLUMN: documents[AREA_COLUMN][::-1].to_numpy(), TOPIC_COLUMN: "rotulo_outro"}
    )
    without_labels = documents.drop(columns=[AREA_COLUMN, TOPIC_COLUMN])
    reference = corpus.build_corpus(documents, n_sentences=10, seed=0)
    assert corpus.build_corpus(shuffled, n_sentences=10, seed=0) == reference
    assert corpus.build_corpus(without_labels, n_sentences=10, seed=0) == reference
    assert not any("rotulo_" in sentence for sentence in reference)


@pytest.mark.usefixtures("simple_splitter")
def test_build_corpus_raises_when_sample_exceeds_valid_sentences(
    documents: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError):
        corpus.build_corpus(documents, n_sentences=1_000, seed=0)


@pytest.mark.usefixtures("punkt_portuguese")
def test_split_sentences_keeps_other_legal_abbreviations() -> None:
    text = (
        "O pagamento observa o inc. II do art. 5º da norma, conforme o voto do Min. relator. "
        "A unidade técnica concorda."
    )
    sentences = corpus.split_sentences(text)
    assert len(sentences) == 2
    assert "inc. II do art. 5º" in sentences[0]


@pytest.mark.usefixtures("simple_splitter")
def test_extract_sentences_cleans_markup_and_skips_missing() -> None:
    df = pd.DataFrame(
        {
            TEXT_COLUMN: ["<b>SÚMULA TCU 1:</b> O gestor deve justificar a escolha.", None],
            EXCERPT_COLUMN: ["Primeira frase do excerto.\\nSegunda frase.", "Outra frase."],
        }
    )
    assert corpus.extract_sentences(df) == [
        "SÚMULA TCU 1: O gestor deve justificar a escolha.",
        "Primeira frase do excerto.",
        "Segunda frase.",
        "Outra frase.",
    ]


@pytest.mark.usefixtures("simple_splitter")
def test_corpus_funnel_counts_each_step(documents: pd.DataFrame) -> None:
    duplicated = pd.concat([documents, documents], ignore_index=True)
    funnel = corpus.corpus_funnel(duplicated)
    assert funnel["step"].to_list() == ["segmentadas", "filtro de tamanho", "sem duplicatas"]
    assert funnel["sentences"].to_list() == [80, 80, 40]
