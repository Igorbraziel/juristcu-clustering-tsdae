"""Constantes e hiperparâmetros do estudo."""

from dataclasses import dataclass

# Dados
DATASET_ID = "LeandroRibeiro/JurisTCU"
DATA_FILE = "doc.csv"
TEXT_COLUMN = "ENUNCIADO"
EXCERPT_COLUMN = "EXCERTO"
AREA_COLUMN = "AREA"
TOPIC_COLUMN = "TEMA"
N_DOCUMENTS = 16_045

# Corpus de frases sem rótulo
MIN_WORDS = 5
MAX_WORDS = 80
N_SENTENCES = 20_000
SEED = 42

# Representações
LSA_DIM = 300
BERTIMBAU_ID = "neuralmind/bert-base-portuguese-cased"
E5_ID = "intfloat/multilingual-e5-small"
E5_PREFIX = "query: "
ENCODE_MAX_SEQ_LENGTH = 256
ENCODE_BATCH_SIZE = 64


@dataclass(frozen=True)
class TsdaeConfig:
    """Configuração do treinamento com TSDAE.

    Attributes:
        base_model: Modelo inicial do Hugging Face.
        deletion_ratio: Fração de palavras apagadas em cada frase.
        batch_size: Número de frases por lote.
        learning_rate: Taxa de aprendizado do AdamW.
        num_epochs: Número de épocas sobre o corpus.
        pooling: Estratégia de pooling do codificador.
        weight_decay: Decaimento de pesos do AdamW.
        lr_scheduler_type: Agenda da taxa de aprendizado.
        warmup_steps: Passos de aquecimento.
        max_seq_length: Comprimento máximo em tokens.
        fp16: Uso de precisão mista (a T4 não suporta bf16).
    """

    base_model: str = BERTIMBAU_ID
    deletion_ratio: float = 0.6
    batch_size: int = 8
    learning_rate: float = 3e-5
    num_epochs: int = 1
    pooling: str = "cls"
    weight_decay: float = 0.0
    lr_scheduler_type: str = "constant"
    warmup_steps: int = 0
    max_seq_length: int = 128
    fp16: bool = True


TSDAE_SEEDS = (0, 1, 2)

# Agrupamento e avaliação
KMEANS_N_INIT = 10
K_RANGE = range(5, 51)
MIN_TOPIC_SIZE = 20
N_BOOTSTRAP = 1_000
CI_LEVEL = 0.95
UMAP_N_COMPONENTS = 2
