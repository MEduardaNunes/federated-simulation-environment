import argparse
import math
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import seaborn as sns
from datetime import datetime
import shutil
from pathlib import Path


MOVING_AVERAGE_WINDOW = 10


def _find_seed_metrics(input_dir):
    requested_dir = Path(input_dir).expanduser()
    project_root = Path(__file__).resolve().parents[2]
    candidates = (
        [requested_dir]
        if requested_dir.is_absolute()
        else [Path.cwd() / requested_dir, project_root / requested_dir]
    )

    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_dir():
            csv_paths = sorted(candidate.rglob("round_metrics*.csv"))
            if csv_paths:
                return candidate, csv_paths

    searched_dirs = ", ".join(str(path.resolve()) for path in candidates)
    raise FileNotFoundError(
        "Nenhum CSV round_metrics*.csv encontrado. "
        f"Diretórios pesquisados: {searched_dirs}"
    )


def _plot_smoothed_metric(ax, df, metric, moving_average, color):
    ax.plot(
        df["server_round"],
        df[metric],
        color=color,
        linewidth=1,
        alpha=0.25,
    )
    ax.plot(
        df["server_round"],
        df[moving_average],
        color=color,
        linewidth=2.5,
        label=f"Média móvel ({MOVING_AVERAGE_WINDOW} rodadas)",
    )
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))


def aggregate_metrics_and_plots(input_dir, output_dir, dpi=300):
    """
    Agrega métricas por rodada de vários CSVs de seeds e gera CSV e gráficos.

    Os arquivos de entrada devem se chamar ``round_metrics*.csv`` e conter
    ``server_round``, ``accuracy``, ``loss`` e ``total_bytes_comunication``.
    """
    input_dir, csv_paths = _find_seed_metrics(input_dir)
    output_dir = Path(output_dir)

    required_columns = (
        "server_round",
        "accuracy",
        "loss",
        "total_bytes_comunication",
    )
    frames = []
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path)
        frame.columns = frame.columns.str.strip()
        missing_columns = set(required_columns).difference(frame.columns)
        if missing_columns:
            raise ValueError(
                f"{csv_path} não contém as colunas obrigatórias: "
                f"{', '.join(sorted(missing_columns))}"
            )

        frame = frame[list(required_columns)].rename(
            columns={
                "server_round": "round",
                "total_bytes_comunication": "bytes",
            }
        )
        for column in ("round", "accuracy", "loss", "bytes"):
            frame[column] = pd.to_numeric(frame[column], errors="raise")
            if frame[column].isna().any() or not frame[column].map(
                math.isfinite
            ).all():
                raise ValueError(f"{csv_path} contém valores inválidos em {column}")

        if (frame["round"] % 1 != 0).any():
            raise ValueError(f"{csv_path} contém números de rodada não inteiros")
        if frame["round"].duplicated().any():
            raise ValueError(f"{csv_path} contém rodadas duplicadas")
        frame["round"] = frame["round"].astype(int)
        frames.append(frame)

    expected_rounds = set(frames[0]["round"])
    for csv_path, frame in zip(csv_paths[1:], frames[1:]):
        if set(frame["round"]) != expected_rounds:
            raise ValueError(
                f"{csv_path} não possui o mesmo conjunto de rodadas dos demais CSVs"
            )

    all_metrics = pd.concat(frames, ignore_index=True)
    aggregated = (
        all_metrics.groupby("round", as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            loss_mean=("loss", "mean"),
            loss_std=("loss", "std"),
            bytes_mean=("bytes", "mean"),
            bytes_std=("bytes", "std"),
        )
        .sort_values("round")
        .reset_index(drop=True)
    )
    aggregated = aggregated[
        [
            "round",
            "accuracy_mean",
            "accuracy_std",
            "loss_mean",
            "loss_std",
            "bytes_mean",
            "bytes_std",
        ]
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(output_dir / "aggregated_metrics.csv", index=False)

    sns.set_theme(style="whitegrid", palette="colorblind")
    plots = (
        (
            "accuracy_curve.png",
            "Acurácia global por rodada",
            "Acurácia",
            "accuracy_mean",
            "accuracy_std",
            sns.color_palette("colorblind")[1],
        ),
        (
            "loss_curve.png",
            "Loss global por rodada",
            "Loss",
            "loss_mean",
            "loss_std",
            sns.color_palette("colorblind")[0],
        ),
        (
            "communication_bytes.png",
            "Bytes de comunicação acumulados por rodada",
            "Bytes acumulados",
            "bytes_mean",
            "bytes_std",
            sns.color_palette("colorblind")[2],
        ),
    )
    for filename, title, ylabel, mean_column, std_column, color in plots:
        fig, ax = plt.subplots(figsize=(10, 6))
        rounds = aggregated["round"]
        means = aggregated[mean_column]
        stds = aggregated[std_column].fillna(0)
        ax.plot(rounds, means, color=color, linewidth=2.5, label="Média")
        ax.fill_between(
            rounds,
            means - stds,
            means + stds,
            color=color,
            alpha=0.2,
            label="Média ± desvio padrão",
        )
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Rodada")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
        ax.legend()
        sns.despine()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    print(f"Métricas agregadas de {len(csv_paths)} arquivos salvas em {output_dir}")
    return aggregated


def save_config_metrics_and_plots(
    csv_path="round_metrics.csv", 
    output_dir_base="results",
    output_dir=None,
):
    """
    Cria uma pasta com a data atual, salva as métricas obtidas durante o treinamento,
    o arquivo de configuração, e os plots finais (acurácia, loss e bytes de comunicação).

    Parâmetros:
        - csv_path: caminho para o arquivo csv com as métricas do treinamento.
        - output_dir_base: caminho para a pasta onde serão salvos os resultados;
        - output_dir: diretório de saída exato, opcional.
    """
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = os.path.join(output_dir_base, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Copia o arquivo de métricas .csv do treinamento
    if os.path.exists(csv_path):
        metrics_dir = os.path.join(output_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_copy_path = os.path.join(
            metrics_dir, os.path.basename(csv_path)
        )
        if os.path.abspath(csv_path) != os.path.abspath(metrics_copy_path):
            shutil.copy2(csv_path, metrics_copy_path)

    # Copia o arquivo de configurações do treinamento
    toml_path = "pyproject.toml"
    if os.path.exists(toml_path):
        shutil.copy2(toml_path, os.path.join(output_dir, "pyproject.toml"))

    # Leitura dos dados
    expected_columns = [
        "server_round",
        "loss",
        "accuracy",
        "total_bytes_comunication",
    ]
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    if not set(expected_columns).issubset(df.columns):
        header_is_data = len(df.columns) == len(expected_columns)
        if header_is_data:
            try:
                for column in df.columns:
                    float(column)
            except ValueError:
                header_is_data = False

        if header_is_data:
            df = pd.read_csv(csv_path, header=None, names=expected_columns)
        else:
            missing_columns = sorted(set(expected_columns).difference(df.columns))
            raise ValueError(
                f"{csv_path} não contém as colunas obrigatórias: "
                f"{', '.join(missing_columns)}. "
                f"Colunas encontradas: {', '.join(df.columns.astype(str))}"
            )

    if df.empty:
        raise ValueError(f"{csv_path} não contém métricas para gerar os gráficos.")

    df = df.sort_values("server_round").reset_index(drop=True)
    window = min(MOVING_AVERAGE_WINDOW, len(df))
    df["loss_moving_average"] = df["loss"].rolling(
        window=window, min_periods=1, center=True
    ).mean()
    df["accuracy_moving_average"] = df["accuracy"].rolling(
        window=window, min_periods=1, center=True
    ).mean()

    # Ajuste de estilo e fonte
    sns.set_theme(style="whitegrid", palette="colorblind")

    # Converter Bytes para Megabytes (MB) para facilitar a leitura visual
    if "total_bytes_comunication" in df.columns:
        df["total_mb_comunication"] = df["total_bytes_comunication"] / (
            1024 * 1024
        )

    # -------------------------------------------------------------
    # Gráfico 1: Evolução do Loss
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_smoothed_metric(
        ax, df, "loss", "loss_moving_average", sns.color_palette("colorblind")[0]
    )
    ax.legend()
    ax.set_title(
        "Evolução da Perda (Loss) por Rodada",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Rodada do Servidor (Server Round)")
    ax.set_ylabel("Perda Global (Loss) — média móvel")

    sns.despine()
    plt.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "loss_per_round.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # -------------------------------------------------------------
    # Gráfico 2: Evolução da Acurácia
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_smoothed_metric(
        ax,
        df,
        "accuracy",
        "accuracy_moving_average",
        sns.color_palette("colorblind")[1],
    )
    ax.legend()
    max_acc = df["accuracy_moving_average"].max()
    max_round = df.loc[df["accuracy_moving_average"].idxmax(), "server_round"]

    ax.axhline(
        y=max_acc, 
        color="#D55E00", 
        linestyle="--", 
        linewidth=1.5, 
        alpha=0.85
    )

    ax.text(
        x=df["server_round"].min(), 
        y=max_acc + (max_acc * 0.02), # Pequeno deslocamento vertical acima da linha
        s=f"Máx. média móvel: {max_acc * 100:.2f}% (Rodada {int(max_round)})",
        color="#D55E00", 
        fontweight="bold", 
        fontsize=10
    )

    ax.set_title(
        "Evolução da Acurácia Global por Rodada",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Rodada do Servidor (Server Round)")
    ax.set_ylabel("Acurácia global — média móvel")
    ax.set_ylim(0, max(df["accuracy_moving_average"]) * 1.25)
    sns.despine()
    plt.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "accuracy_per_round.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # -------------------------------------------------------------
    # Gráfico 3: Tráfego de Comunicação (MB)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        data=df,
        x="server_round",
        y="total_mb_comunication",
        linewidth=2.5,
        ax=ax
    )
    ax.set_title(
        "Comunicação Acumulada por Rodada (MB)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Rodada do Servidor (Server Round)")
    ax.set_ylabel("Total de Dados Transmitidos (MB)")

    sns.despine()
    plt.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "communication_per_round.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # -------------------------------------------------------------
    # Gráfico 4: Dashboard Combinado (3 Subplots)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Resumo do Treinamento Federado (Métricas por Rodada)",
        fontsize=16,
        fontweight="bold",
        y=1.03,
    )

    # Subplot 1: Loss
    _plot_smoothed_metric(
        axes[0],
        df,
        "loss",
        "loss_moving_average",
        sns.color_palette("colorblind")[0],
    )
    axes[0].legend(fontsize=8)
    axes[0].set_title("Perda Global (Loss)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Rodada")
    axes[0].set_ylabel("Loss — média móvel")

    # Subplot 2: Accuracy
    _plot_smoothed_metric(
        axes[1],
        df,
        "accuracy",
        "accuracy_moving_average",
        sns.color_palette("colorblind")[1],
    )
    axes[1].legend(fontsize=8)
    axes[1].set_title(
        "Acurácia Global", fontsize=12, fontweight="bold"
    )
    max_acc = df["accuracy_moving_average"].max()
    max_round = df.loc[df["accuracy_moving_average"].idxmax(), "server_round"]

    # Adiciona a linha horizontal no nível máximo
    axes[1].axhline(
        y=max_acc, 
        color="#D55E00", 
        linestyle="--", 
        linewidth=1.5, 
        alpha=0.85
    )

    # Adiciona a anotação acima da linha
    axes[1].text(
        x=df["server_round"].min(), 
        y=max_acc + (max_acc * 0.02),
        s=f"Máx. média móvel: {max_acc * 100:.2f}% (R{int(max_round)})",
        color="#D55E00", 
        fontweight="bold", 
        fontsize=9
    )
    axes[1].set_xlabel("Rodada")
    axes[1].set_ylabel("Acurácia — média móvel")

    # Subplot 3: Communication
    sns.lineplot(
        data=df,
        x="server_round",
        y="total_mb_comunication",
        linewidth=2.5,
        ax=axes[2],
    )
    axes[2].set_title(
        "Tráfego de Comunicação Acumulado", fontsize=12, fontweight="bold"
    )
    axes[2].set_xlabel("Rodada")
    axes[2].set_ylabel("MBs Acumulados")

    sns.despine(fig=fig)
    plt.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "metrics_dashboard.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    print("Métricas, configurações e plots salvos com sucesso.")