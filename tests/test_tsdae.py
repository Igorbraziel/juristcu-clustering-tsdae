"""Especificação executável das funções puras de tsdae.py.

O treinamento (train_tsdae) exige GPU e download e não é testado aqui.
"""

import random

import pytest

from juristcu import tsdae

LONG_TEXT = " ".join(f"w{i}" for i in range(2_000))


def test_delete_words_with_zero_ratio_keeps_all_words() -> None:
    text = "o tribunal determinou a anulação do certame"
    assert tsdae.delete_words(text, ratio=0.0, rng=random.Random(0)) == text


def test_delete_words_with_full_ratio_keeps_one_word() -> None:
    text = "o tribunal determinou a anulação do certame"
    noisy = tsdae.delete_words(text, ratio=1.0, rng=random.Random(0))
    assert len(noisy.split()) == 1
    assert noisy in text.split()


def test_delete_words_keeps_about_forty_percent() -> None:
    # Com ratio = 0,6, cerca de 40% das palavras permanecem.
    noisy = tsdae.delete_words(LONG_TEXT, ratio=0.6, rng=random.Random(0))
    kept = len(noisy.split()) / len(LONG_TEXT.split())
    assert kept == pytest.approx(0.4, abs=0.05)


def test_delete_words_preserves_word_order() -> None:
    noisy = tsdae.delete_words(LONG_TEXT, ratio=0.6, rng=random.Random(0))
    indices = [int(word[1:]) for word in noisy.split()]
    assert indices == sorted(indices)


def test_delete_words_is_deterministic_with_seed() -> None:
    first = tsdae.delete_words(LONG_TEXT, ratio=0.6, rng=random.Random(7))
    second = tsdae.delete_words(LONG_TEXT, ratio=0.6, rng=random.Random(7))
    assert first == second


@pytest.mark.parametrize("ratio", [-0.1, 1.5])
def test_delete_words_rejects_invalid_ratio(ratio: float) -> None:
    with pytest.raises(ValueError):
        tsdae.delete_words("texto qualquer", ratio=ratio, rng=random.Random(0))


@pytest.mark.parametrize("text", ["", "   "])
def test_delete_words_returns_text_without_words_unchanged(text: str) -> None:
    assert tsdae.delete_words(text, ratio=0.6, rng=random.Random(0)) == text


def test_make_denoising_pairs_has_noisy_column_first() -> None:
    sentences = ["primeira frase de teste", "segunda frase de teste", "terceira frase"]
    pairs = tsdae.make_denoising_pairs(sentences, ratio=0.6, seed=0)
    # A primeira coluna é a entrada com ruído; a ordem trocada treina ao contrário.
    assert tuple(pairs) == ("noisy", "text")
    assert len(pairs["noisy"]) == len(pairs["text"]) == len(sentences)
    assert pairs["text"] == sentences


def test_make_denoising_pairs_is_deterministic_with_seed() -> None:
    sentences = [LONG_TEXT, LONG_TEXT]
    first = tsdae.make_denoising_pairs(sentences, ratio=0.6, seed=1)
    second = tsdae.make_denoising_pairs(sentences, ratio=0.6, seed=1)
    assert first == second


def test_make_denoising_pairs_changes_with_seed() -> None:
    first = tsdae.make_denoising_pairs([LONG_TEXT], ratio=0.6, seed=0)
    second = tsdae.make_denoising_pairs([LONG_TEXT], ratio=0.6, seed=1)
    assert first["noisy"] != second["noisy"]
