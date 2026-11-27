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
# Etiquetas HTML que separam parágrafos; as demais são de formatação dentro da linha.
HTML_BLOCK_TAGS = (
    "p",
    "br",
    "div",
    "li",
    "ul",
    "ol",
    "tr",
    "td",
    "th",
    "table",
    "blockquote",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
)
NLTK_RESOURCE = "punkt_tab"
NLTK_LANGUAGE = "portuguese"
# Abreviações jurídicas acrescentadas ao Punkt, em minúsculas e sem o ponto final.
LEGAL_ABBREVIATIONS = (
    "art",
    "arts",
    "inc",
    "incs",
    "al",
    "n",
    "nº",
    "p",
    "pp",
    "min",
    "rel",
    "ac",
    "proc",
    "fl",
    "fls",
    "sr",
    "sra",
    "dr",
    "dra",
)

# Representações
LSA_DIM = 300
BERTIMBAU_ID = "neuralmind/bert-base-portuguese-cased"
E5_ID = "intfloat/multilingual-e5-small"
E5_PREFIX = "query: "
ENCODE_MAX_SEQ_LENGTH = 256
ENCODE_BATCH_SIZE = 64
# Nomes das representações salvas, na ordem das tabelas. R3 aparece uma vez por semente.
REPRESENTATION_NAMES = ("R1", "R2", "R2-CLS", "R3-seed0", "R3-seed1", "R3-seed2", "R4")


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
        logging_steps: Intervalo, em passos, entre registros da perda de treino.
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
    logging_steps: int = 50


TSDAE_SEEDS = (0, 1, 2)
# Histórico da perda gravado ao lado de cada modelo treinado.
TSDAE_LOSS_FILE = "loss_history.json"

# Agrupamento e avaliação
KMEANS_N_INIT = 10
K_RANGE = range(5, 51)
MIN_TOPIC_SIZE = 20
N_BOOTSTRAP = 1_000
CI_LEVEL = 0.95
# Normalização da NMI: média aritmética das entropias, padrão do scikit-learn.
NMI_AVERAGE_METHOD = "arithmetic"
UMAP_N_COMPONENTS = 2
# Padrões do umap-learn, fixados para reprodutibilidade.
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
# Número de painéis por linha na figura das projeções UMAP.
UMAP_GRID_COLUMNS = 4
HDBSCAN_MIN_CLUSTER_SIZE = 20

# Termos característicos de cada grupo
# Palavras com ao menos 2 letras; números e códigos ficam de fora.
TERM_TOKEN_PATTERN = r"(?u)\b[^\W\d_]{2,}\b"
# Palavras vazias do português: lista do NLTK, com "tém" corrigido para "têm".
# Fixada antes de ver os resultados; não inclui termos do domínio jurídico.
PT_STOPWORDS = (
    "a",
    "à",
    "ao",
    "aos",
    "aquela",
    "aquelas",
    "aquele",
    "aqueles",
    "aquilo",
    "as",
    "às",
    "até",
    "com",
    "como",
    "da",
    "das",
    "de",
    "dela",
    "delas",
    "dele",
    "deles",
    "depois",
    "do",
    "dos",
    "e",
    "é",
    "ela",
    "elas",
    "ele",
    "eles",
    "em",
    "entre",
    "era",
    "eram",
    "éramos",
    "essa",
    "essas",
    "esse",
    "esses",
    "esta",
    "está",
    "estamos",
    "estão",
    "estas",
    "estava",
    "estavam",
    "estávamos",
    "este",
    "esteja",
    "estejam",
    "estejamos",
    "estes",
    "esteve",
    "estive",
    "estivemos",
    "estiver",
    "estivera",
    "estiveram",
    "estivéramos",
    "estiverem",
    "estivermos",
    "estivesse",
    "estivessem",
    "estivéssemos",
    "estou",
    "eu",
    "foi",
    "fomos",
    "for",
    "fora",
    "foram",
    "fôramos",
    "forem",
    "formos",
    "fosse",
    "fossem",
    "fôssemos",
    "fui",
    "há",
    "haja",
    "hajam",
    "hajamos",
    "hão",
    "havemos",
    "hei",
    "houve",
    "houvemos",
    "houver",
    "houvera",
    "houverá",
    "houveram",
    "houvéramos",
    "houverão",
    "houverei",
    "houverem",
    "houveremos",
    "houveria",
    "houveriam",
    "houveríamos",
    "houvermos",
    "houvesse",
    "houvessem",
    "houvéssemos",
    "isso",
    "isto",
    "já",
    "lhe",
    "lhes",
    "mais",
    "mas",
    "me",
    "mesmo",
    "meu",
    "meus",
    "minha",
    "minhas",
    "muito",
    "na",
    "não",
    "nas",
    "nem",
    "no",
    "nos",
    "nós",
    "nossa",
    "nossas",
    "nosso",
    "nossos",
    "num",
    "numa",
    "o",
    "os",
    "ou",
    "para",
    "pela",
    "pelas",
    "pelo",
    "pelos",
    "por",
    "qual",
    "quando",
    "que",
    "quem",
    "são",
    "se",
    "seja",
    "sejam",
    "sejamos",
    "sem",
    "será",
    "serão",
    "serei",
    "seremos",
    "seria",
    "seriam",
    "seríamos",
    "seu",
    "seus",
    "só",
    "somos",
    "sou",
    "sua",
    "suas",
    "também",
    "te",
    "tem",
    "têm",
    "temos",
    "tenha",
    "tenham",
    "tenhamos",
    "tenho",
    "terá",
    "terão",
    "terei",
    "teremos",
    "teria",
    "teriam",
    "teríamos",
    "teu",
    "teus",
    "teve",
    "tinha",
    "tinham",
    "tínhamos",
    "tive",
    "tivemos",
    "tiver",
    "tivera",
    "tiveram",
    "tivéramos",
    "tiverem",
    "tivermos",
    "tivesse",
    "tivessem",
    "tivéssemos",
    "tu",
    "tua",
    "tuas",
    "um",
    "uma",
    "você",
    "vocês",
    "vos",
)
