"""Testes de data.py, sem download."""

from pathlib import Path

import pandas as pd
import pytest

from juristcu import data
from juristcu.config import AREA_COLUMN, EXCERPT_COLUMN, TEXT_COLUMN, TOPIC_COLUMN


@pytest.fixture
def local_dataset(tmp_path: Path) -> Path:
    # CSV local com colunas extras, texto em várias linhas e valor ausente.
    df = pd.DataFrame(
        {
            "KEY": ["1", "2", "3"],
            TEXT_COLUMN: ["<b>Enunciado</b> um.", "Enunciado dois.", "Enunciado três."],
            EXCERPT_COLUMN: ["Linha um.\nLinha dois.", None, "Excerto três."],
            AREA_COLUMN: ["Licitação", "Pessoal", "Licitação"],
            TOPIC_COLUMN: ["Edital", "Aposentadoria", "Edital"],
            "NUMACORDAO": [354.0, 12.0, 7.0],
        }
    )
    df.to_csv(tmp_path / "doc.csv", index=False)
    return tmp_path


def test_load_documents_reads_local_csv(local_dataset: Path) -> None:
    df = data.load_documents(dataset_id=str(local_dataset), expected_rows=3)
    assert df.columns.to_list() == [TEXT_COLUMN, EXCERPT_COLUMN, AREA_COLUMN, TOPIC_COLUMN]
    assert len(df) == 3
    assert df.loc[0, EXCERPT_COLUMN] == "Linha um.\nLinha dois."
    assert df[EXCERPT_COLUMN].isna().sum() == 1


def test_load_documents_checks_row_count(local_dataset: Path) -> None:
    with pytest.raises(ValueError, match="linhas"):
        data.load_documents(dataset_id=str(local_dataset), expected_rows=4)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("<b>SÚMULA TCU 289:</b> A exigência", "SÚMULA TCU 289: A exigência"),
        ("Primeira linha.\\nSegunda linha.", "Primeira linha. Segunda linha."),
        ("Texto\r\ncom\tquebras.", "Texto com quebras."),
        ("Valor&nbsp;de&nbsp;R$ 10", "Valor de R$ 10"),
        ("  espaços   repetidos  ", "espaços repetidos"),
        (None, ""),
        (float("nan"), ""),
    ],
)
def test_clean_text(raw: object, expected: str) -> None:
    assert data.clean_text(raw) == expected  # type: ignore[arg-type]


def test_label_distribution_counts_and_sorts() -> None:
    df = pd.DataFrame({AREA_COLUMN: ["b", "a", "b", "c", "a", "b", None]})
    table = data.label_distribution(df, AREA_COLUMN)
    assert table.columns.to_list() == [AREA_COLUMN, "count", "proportion"]
    assert table["count"].to_list() == [3, 2, 1, 1]
    assert table[AREA_COLUMN].to_list()[:3] == ["b", "a", "c"]
    assert table[AREA_COLUMN].isna().iloc[-1]
    assert table["proportion"].sum() == pytest.approx(1.0)


def test_summarize_distribution_on_balanced_labels() -> None:
    df = pd.DataFrame({TOPIC_COLUMN: ["a"] * 20 + ["b"] * 20 + ["c"] * 5})
    summary = data.summarize_distribution(data.label_distribution(df, TOPIC_COLUMN), min_size=20)
    assert summary["categories"] == 3
    assert summary["documents"] == 45
    assert summary["smallest_count"] == 5
    assert summary["imbalance_ratio"] == pytest.approx(4.0)
    assert summary["categories_min_size"] == 2
    assert summary["share_min_size"] == pytest.approx(40 / 45)
    assert 0 < summary["normalized_entropy"] < 1


def test_clean_text_removes_inline_tags_without_spaces() -> None:
    raw = "o <i>caput</i>, conforme o <span>Acórdão 1.131/2009</span> ( Plenário ) ."
    assert data.clean_text(raw) == "o caput, conforme o Acórdão 1.131/2009 (Plenário)."


def test_clean_text_separates_block_tags_and_omissions() -> None:
    raw = "<p>[...] Primeira frase.</p><p>Segunda frase.</p>"
    assert data.clean_text(raw) == "Primeira frase. Segunda frase."


def test_split_paragraphs_uses_block_tags() -> None:
    raw = "<p>Voto:</p><p>O relator <b>examina</b> o caso.</p><br>Outro parágrafo."
    assert data.split_paragraphs(raw) == ["Voto:", "O relator examina o caso.", "Outro parágrafo."]


def test_split_paragraphs_removes_glued_paragraph_numbers() -> None:
    raw = "[...] 8.Este Tribunal decide. A proposta não tem estudos técnicos: 20.Quando se disputa."
    assert data.split_paragraphs(raw) == [
        "Este Tribunal decide. A proposta não tem estudos técnicos:",
        "Quando se disputa.",
    ]


@pytest.mark.parametrize(
    "text",
    [
        "Conforme a Lei 8.666/1993, o órgão decide.",
        "Ver o Acórdão 1.131/2009 - Plenário.",
        "Acórdão: 9.2. determinar à unidade a correção.",
        "Valor de R$ 1.500.000,00 pago a maior.",
    ],
)
def test_split_paragraphs_keeps_numbers_that_are_not_paragraph_labels(text: str) -> None:
    assert data.split_paragraphs(text) == [text]


def test_split_paragraphs_handles_missing_text() -> None:
    assert data.split_paragraphs(None) == []
    assert data.split_paragraphs(float("nan")) == []


def test_tag_frequency_counts_opening_tags() -> None:
    texts = pd.Series(["<p>a</p><p><b>b</b></p>", None, "<BR>c"])
    table = data.tag_frequency(texts)
    assert table.set_index("tag")["count"].to_dict() == {"p": 2, "b": 1, "br": 1}
    assert table.set_index("tag")["block"].to_dict() == {"p": True, "b": False, "br": True}
