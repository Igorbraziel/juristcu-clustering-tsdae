"""Especificação executável das funções de embeddings.py.

Os testes dos modelos de frases usam um BERT minúsculo, criado localmente, sem download.
"""

from pathlib import Path

import numpy as np
import pytest

from juristcu import embeddings

TEXTS = [
    "o tribunal determinou a anulação do certame",
    "a licitação exige publicidade do edital",
    "o contrato administrativo admite alteração unilateral",
    "a dispensa de licitação exige justificativa do preço",
    "o gestor responde pelo dano ao erário",
    "a tomada de contas especial apura o débito",
    "o pregão eletrônico é a modalidade preferencial",
    "a aposentadoria depende do registro pelo tribunal",
    "o sobrepreço no contrato gera débito ao responsável",
    "a pensão civil exige a comprovação da dependência",
    "o edital não pode restringir a competitividade",
    "a multa decorre da omissão no dever de prestar contas",
]

VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "query", ":"] + sorted(
    {word for text in TEXTS for word in text.split()}
)


@pytest.fixture(scope="module")
def tiny_bert(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Cria um BERT minúsculo, com pesos aleatórios, em um diretório temporário."""
    from transformers import BertConfig, BertModel, BertTokenizerFast, set_seed

    directory = tmp_path_factory.mktemp("tiny-bert")
    vocab_file = directory / "vocab.txt"
    vocab_file.write_text("\n".join(VOCAB), encoding="utf-8")
    set_seed(0)
    config = BertConfig(
        vocab_size=len(VOCAB),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
    )
    BertModel(config).save_pretrained(directory)
    BertTokenizerFast(vocab_file=str(vocab_file)).save_pretrained(directory)
    return directory


# R1: LSA


def test_encode_lsa_returns_requested_dimension() -> None:
    x = embeddings.encode_lsa(TEXTS, n_components=5, seed=0)
    assert x.shape == (len(TEXTS), 5)


def test_encode_lsa_is_deterministic_with_seed() -> None:
    first = embeddings.encode_lsa(TEXTS, n_components=5, seed=7)
    second = embeddings.encode_lsa(TEXTS, n_components=5, seed=7)
    np.testing.assert_array_equal(first, second)


# Normalização L2


def test_l2_normalize_gives_unit_rows() -> None:
    x = np.random.default_rng(0).normal(size=(20, 8))
    norms = np.linalg.norm(embeddings.l2_normalize(x), axis=1)
    np.testing.assert_allclose(norms, 1.0)


def test_l2_normalize_keeps_zero_rows() -> None:
    x = np.array([[3.0, 4.0], [0.0, 0.0]])
    normalized = embeddings.l2_normalize(x)
    np.testing.assert_allclose(normalized, [[0.6, 0.8], [0.0, 0.0]])
    assert np.isfinite(normalized).all()


def test_l2_normalize_does_not_change_input() -> None:
    x = np.array([[3.0, 4.0]])
    embeddings.l2_normalize(x)
    np.testing.assert_array_equal(x, [[3.0, 4.0]])


# Modelos de frases


def test_load_sentence_model_rejects_invalid_pooling(tiny_bert: Path) -> None:
    with pytest.raises(ValueError):
        embeddings.load_sentence_model(tiny_bert, pooling="max")


@pytest.mark.parametrize("pooling", ["mean", "cls"])
def test_load_sentence_model_sets_pooling(tiny_bert: Path, pooling: str) -> None:
    model = embeddings.load_sentence_model(tiny_bert, pooling=pooling, max_seq_length=32)
    assert model[1].pooling_mode == pooling
    assert model.max_seq_length == 32


def test_cls_pooling_uses_first_token(tiny_bert: Path) -> None:
    model = embeddings.load_sentence_model(tiny_bert, pooling="cls")
    x = embeddings.encode_texts(model, TEXTS[:2])
    tokens = model.encode(TEXTS[:2], output_value="token_embeddings")
    np.testing.assert_allclose(x, np.stack([t[0].numpy() for t in tokens]), atol=1e-6)


def test_load_saved_model_keeps_saved_pooling(tiny_bert: Path, tmp_path: Path) -> None:
    # Simula R3: modelo salvo com pooling CLS, carregado com pooling=None.
    embeddings.load_sentence_model(tiny_bert, pooling="cls").save(str(tmp_path))
    model = embeddings.load_sentence_model(tmp_path, max_seq_length=20)
    assert model[1].pooling_mode == "cls"
    assert model.max_seq_length == 20


def test_encode_texts_returns_one_row_per_text(tiny_bert: Path) -> None:
    model = embeddings.load_sentence_model(tiny_bert, pooling="mean")
    x = embeddings.encode_texts(model, TEXTS, batch_size=4)
    assert isinstance(x, np.ndarray)
    assert x.shape == (len(TEXTS), 16)


def test_encode_texts_applies_prefix(tiny_bert: Path) -> None:
    model = embeddings.load_sentence_model(tiny_bert, pooling="mean")
    with_prefix = embeddings.encode_texts(model, TEXTS[:3], prefix="query: ")
    concatenated = embeddings.encode_texts(model, [f"query: {t}" for t in TEXTS[:3]])
    without_prefix = embeddings.encode_texts(model, TEXTS[:3])
    np.testing.assert_allclose(with_prefix, concatenated, atol=1e-6)
    assert not np.allclose(with_prefix, without_prefix)
