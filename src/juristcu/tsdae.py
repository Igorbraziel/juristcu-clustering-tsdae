"""Treinamento sem rótulos do BERTimbau com TSDAE.

- O ruído é próprio do projeto: deleção de palavras com semente explícita.
- O treinamento usa o SentenceTransformerTrainer do sentence-transformers.
- O histórico da perda é gravado ao lado de cada modelo treinado.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from juristcu.config import TSDAE_LOSS_FILE, TsdaeConfig


def delete_words(text: str, ratio: float, rng: random.Random) -> str:
    """Apaga palavras de uma frase ao acaso.

    - Cada palavra é mantida com probabilidade 1 - ratio, de forma independente.
    - As palavras mantidas preservam a ordem e são unidas por um espaço.
    - Se nenhuma palavra sobrar, uma palavra sorteada com rng é devolvida.
    - Texto sem palavras é devolvido sem alteração.

    Args:
        text: Frase original.
        ratio: Probabilidade de apagar cada palavra, entre 0 e 1.
        rng: Gerador aleatório. A função não usa o estado global do módulo random.

    Returns:
        Frase com ruído.

    Raises:
        ValueError: Se ratio estiver fora do intervalo [0, 1].

    Note:
        - Tokenizar com str.split(), sem recurso do NLTK.
        - Não copiar o exemplo oficial do sentence-transformers. Ele usa random() < ratio para
          manter a palavra e, com ratio = 0,6, apaga apenas 40%.
    """
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"ratio deve estar entre 0 e 1; recebido {ratio}.")
    words = text.split()
    if not words:
        return text
    # random() fica em [0, 1): a palavra permanece com probabilidade 1 - ratio.
    kept = [word for word in words if rng.random() >= ratio]
    if not kept:
        kept = [rng.choice(words)]
    return " ".join(kept)


def make_denoising_pairs(
    sentences: Sequence[str],
    ratio: float,
    seed: int,
) -> dict[str, list[str]]:
    """Gera os pares de frase com ruído e frase original.

    O ruído é materializado uma única vez: um random.Random(seed) percorre as frases em ordem,
    com delete_words.

    Args:
        sentences: Frases originais.
        ratio: Probabilidade de apagar cada palavra.
        seed: Semente do ruído.

    Returns:
        Dicionário com as chaves "noisy" e "text", nessa ordem, e listas do mesmo tamanho.

    Note:
        - A ordem das chaves define a ordem das colunas do conjunto de dados.
        - A primeira coluna é a entrada com ruído.
    """
    rng = random.Random(seed)
    noisy = [delete_words(sentence, ratio, rng) for sentence in sentences]
    return {"noisy": noisy, "text": list(sentences)}


def train_tsdae(
    sentences: Sequence[str],
    seed: int,
    output_dir: Path,
    # TsdaeConfig é congelada. Compartilhar a instância padrão é seguro.
    config: TsdaeConfig = TsdaeConfig(),  # noqa: B008
) -> Path:
    """Treina o BERTimbau com TSDAE e salva apenas o codificador.

    Etapas:

    1. transformers.set_seed(seed).
    2. Codificador com módulos de sentence_transformers.sentence_transformer.modules:
       Transformer(config.base_model, max_seq_length=config.max_seq_length) e
       Pooling(embedding_dimension, pooling_mode=config.pooling).
    3. Dataset: datasets.Dataset.from_dict(make_denoising_pairs(sentences,
       config.deletion_ratio, seed)).
    4. Loss: DenoisingAutoEncoderLoss(model, decoder_name_or_path=None,
       tie_encoder_decoder=True), de sentence_transformers.sentence_transformer.losses.
    5. SentenceTransformerTrainingArguments com seed, data_seed, learning_rate, weight_decay,
       lr_scheduler_type, warmup_steps, per_device_train_batch_size, num_train_epochs, fp16 e
       logging_steps tirados de config; save_strategy="no"; report_to="none".
    6. SentenceTransformerTrainer(model=..., args=..., train_dataset=..., loss=...).train().
    7. model.save(str(output_dir)).
    8. Histórico da perda (trainer.state.log_history) em output_dir/TSDAE_LOSS_FILE.

    Args:
        sentences: Corpus de frases devolvido por corpus.build_corpus.
        seed: Semente do treinamento e do ruído (um valor de TSDAE_SEEDS).
        output_dir: Diretório onde o codificador é salvo.
        config: Hiperparâmetros do TSDAE.

    Returns:
        O próprio output_dir, para uso em embeddings.load_sentence_model.

    Note:
        - Chamar set_seed antes de criar o modelo e a função de perda.
        - A atenção cruzada do decodificador é inicializada ao acaso.
        - Colunas na ordem ("noisy", "text"). A ordem trocada treina ao contrário, sem erro.
        - O caminho antigo sentence_transformers.losses emite DeprecationWarning. Avisos de
          pesos ausentes (atenção cruzada) ao carregar o decodificador são esperados.
        - fp16 exige GPU com CUDA. Fora dela, passar dataclasses.replace(config, fp16=False).
    """
    # Importação tardia: torch, transformers e sentence-transformers são pesados.
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.sentence_transformer.losses import DenoisingAutoEncoderLoss
    from sentence_transformers.sentence_transformer.modules import Pooling, Transformer
    from transformers import set_seed

    output_dir = Path(output_dir)
    set_seed(seed)
    transformer = Transformer(config.base_model, max_seq_length=config.max_seq_length)
    pooling = Pooling(transformer.get_embedding_dimension(), pooling_mode=config.pooling)
    model = SentenceTransformer(modules=[transformer, pooling])

    dataset = Dataset.from_dict(make_denoising_pairs(sentences, config.deletion_ratio, seed))
    loss = DenoisingAutoEncoderLoss(model, decoder_name_or_path=None, tie_encoder_decoder=True)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        seed=seed,
        data_seed=seed,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_steps=config.warmup_steps,
        per_device_train_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        fp16=config.fp16,
        logging_steps=config.logging_steps,
        save_strategy="no",
        report_to="none",
    )
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=dataset, loss=loss)
    trainer.train()

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(output_dir))
    history = json.dumps(trainer.state.log_history, indent=2)
    (output_dir / TSDAE_LOSS_FILE).write_text(history, encoding="utf-8")
    return output_dir


def load_loss_history(model_dir: Path) -> list[dict[str, Any]]:
    """Carrega o histórico da perda gravado por train_tsdae.

    Args:
        model_dir: Diretório do modelo treinado.

    Returns:
        Registros do treinamento, na ordem em que foram gravados.

    Raises:
        FileNotFoundError: Se o diretório não tiver o arquivo TSDAE_LOSS_FILE.
    """
    path = Path(model_dir) / TSDAE_LOSS_FILE
    return json.loads(path.read_text(encoding="utf-8"))


def loss_table(log_history: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Monta a curva da perda de treino.

    Args:
        log_history: Registros devolvidos por load_loss_history.

    Returns:
        Tabela com as colunas step, epoch e loss, uma linha por registro da perda.

    Raises:
        ValueError: Se nenhum registro tiver a perda.
    """
    rows = [
        {"step": entry["step"], "epoch": entry["epoch"], "loss": entry["loss"]}
        for entry in log_history
        if "loss" in entry
    ]
    if not rows:
        raise ValueError("O histórico não tem registros da perda.")
    return pd.DataFrame(rows, columns=["step", "epoch", "loss"])


def training_summary(log_history: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Resume um treinamento a partir do histórico da perda.

    Args:
        log_history: Registros devolvidos por load_loss_history.

    Returns:
        Dicionário com as chaves:
        - "steps": número de passos do treinamento.
        - "final_loss": último registro da perda.
        - "train_loss": perda média de todo o treinamento.
        - "train_runtime": tempo de treinamento, em segundos.

    Raises:
        ValueError: Se faltar o registro final do treinamento ou o registro da perda.
    """
    final = next((entry for entry in log_history if "train_loss" in entry), None)
    if final is None:
        raise ValueError("O histórico não tem o registro final do treinamento.")
    curve = loss_table(log_history)
    return {
        "steps": float(final["step"]),
        "final_loss": float(curve["loss"].iloc[-1]),
        "train_loss": float(final["train_loss"]),
        "train_runtime": float(final["train_runtime"]),
    }
