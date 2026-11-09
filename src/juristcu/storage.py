"""Persistência de artefatos e registro do ambiente.

- Representações e modelos ficam fora do git: no Google Drive (Colab) ou em ./artifacts.
- Tabelas e figuras ficam em results/, dentro do repositório.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure


def artifacts_dir() -> Path:
    """Devolve o diretório de artefatos e o cria, se necessário.

    A escolha segue esta ordem:

    1. Variável de ambiente JURISTCU_ARTIFACTS.
    2. /content/drive/MyDrive/juristcu, no Colab.
    3. ./artifacts, relativo ao diretório de trabalho.

    Returns:
        Caminho absoluto do diretório.

    Raises:
        FileNotFoundError: Se, no Colab, o Google Drive não estiver montado.

    Note:
        - No Colab, montar o Drive antes da chamada.
        - Sem essa verificação, o diretório seria criado no disco temporário da sessão.
    """
    configured = os.environ.get("JURISTCU_ARTIFACTS")
    if configured:
        path = Path(configured).expanduser()
    elif _in_colab():
        path = _drive_artifacts_dir()
    else:
        path = Path("artifacts")
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_embeddings(name: str, x: np.ndarray) -> Path:
    """Salva uma representação em formato .npy.

    O arquivo fica em artifacts_dir()/embeddings/<name>.npy.

    Args:
        name: Nome do arquivo, sem extensão (por exemplo, "r3_seed0").
        x: Matriz com uma linha por documento.

    Returns:
        Caminho do arquivo gravado.
    """
    path = _embeddings_path(name)
    np.save(path, x)
    return path


def load_embeddings(name: str) -> np.ndarray:
    """Carrega uma representação salva por save_embeddings.

    Args:
        name: Nome do arquivo, sem extensão.

    Returns:
        Matriz gravada, com o tipo original.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
    """
    return np.load(_embeddings_path(name))


def save_table(df: pd.DataFrame, name: str) -> Path:
    """Salva uma tabela em CSV na pasta results/tables do repositório.

    O índice do DataFrame não é gravado. Usar reset_index() antes, se ele tiver significado.

    Args:
        df: Tabela a salvar.
        name: Nome do arquivo, sem extensão.

    Returns:
        Caminho do arquivo gravado.

    Raises:
        FileNotFoundError: Se a raiz do repositório não for encontrada.

    Note:
        - A raiz é procurada primeiro a partir do diretório de trabalho.
        - No Colab, o pacote é instalado fora do clone. A célula de configuração muda o
          diretório de trabalho para a raiz do clone.
    """
    path = _results_dir("tables") / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def save_figure(fig: Figure, name: str, dpi: int = 150) -> Path:
    """Salva uma figura em PNG na pasta results/figures do repositório.

    Args:
        fig: Figura do matplotlib.
        name: Nome do arquivo, sem extensão.
        dpi: Resolução em pontos por polegada.

    Returns:
        Caminho do arquivo gravado.

    Raises:
        FileNotFoundError: Se a raiz do repositório não for encontrada.

    Note:
        - O pre-commit rejeita arquivos acima de 1 MB. Reduzir dpi se a figura passar disso.
        - A raiz do repositório é localizada como em save_table.
    """
    path = _results_dir("figures") / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def log_environment() -> dict[str, str]:
    """Reúne as versões das bibliotecas e os dados da GPU.

    Returns:
        Dicionário com as chaves:
        - "python": versão do Python.
        - Uma chave por biblioteca do estudo, como "torch" e "nltk".
        - "cuda": "True" ou "False".
        - "gpu": nome do dispositivo ou "nenhuma".
    """
    packages = (
        "torch",
        "transformers",
        "sentence-transformers",
        "scikit-learn",
        "numpy",
        "umap-learn",
        "nltk",
    )
    environment = {"python": platform.python_version()}
    for package in packages:
        environment[package] = _package_version(package)
    environment.update(_gpu_info())
    return environment


def save_environment(name: str) -> dict[str, str]:
    """Registra o ambiente de execução em JSON no diretório de artefatos.

    O arquivo fica em artifacts_dir()/environment_<name>.json.

    Args:
        name: Identificação da execução (por exemplo, "02").

    Returns:
        Dicionário gravado, igual ao de log_environment().
    """
    environment = log_environment()
    path = artifacts_dir() / f"environment_{name}.json"
    path.write_text(json.dumps(environment, indent=2), encoding="utf-8")
    return environment


def _in_colab() -> bool:
    """Indica se o código é executado no Google Colab."""
    return "google.colab" in sys.modules


def _drive_artifacts_dir() -> Path:
    """Devolve o diretório de artefatos no Google Drive.

    Raises:
        FileNotFoundError: Se o Drive não estiver montado.
    """
    drive_root = Path("/content/drive/MyDrive")
    if not drive_root.is_dir():
        raise FileNotFoundError(
            "Google Drive não montado. Execute drive.mount('/content/drive') antes."
        )
    return drive_root / "juristcu"


def _embeddings_path(name: str) -> Path:
    """Monta o caminho do arquivo .npy de uma representação."""
    directory = artifacts_dir() / "embeddings"
    directory.mkdir(exist_ok=True)
    return directory / f"{name}.npy"


def _repo_root() -> Path:
    """Localiza a raiz do repositório, identificada pela pasta src/juristcu.

    A busca sobe a partir do diretório de trabalho e, depois, a partir deste arquivo.

    Raises:
        FileNotFoundError: Se nenhum diretório contiver src/juristcu.
    """
    for start in (Path.cwd().resolve(), Path(__file__).resolve().parent):
        for candidate in (start, *start.parents):
            if (candidate / "src" / "juristcu").is_dir():
                return candidate
    raise FileNotFoundError(
        "Raiz do repositório não encontrada. Execute a partir do clone do repositório."
    )


def _results_dir(kind: str) -> Path:
    """Devolve results/<kind> na raiz do repositório e cria a pasta, se necessário."""
    path = _repo_root() / "results" / kind
    path.mkdir(parents=True, exist_ok=True)
    return path


def _package_version(package: str) -> str:
    """Devolve a versão instalada de um pacote ou "não instalado"."""
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "não instalado"


def _gpu_info() -> dict[str, str]:
    """Informa a disponibilidade de CUDA e o nome da GPU."""
    # Importação tardia: o torch é pesado e só é necessário aqui.
    import torch

    available = torch.cuda.is_available()
    gpu = torch.cuda.get_device_name(0) if available else "nenhuma"
    return {"cuda": str(available), "gpu": gpu}
