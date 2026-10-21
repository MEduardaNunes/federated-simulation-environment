import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime
import shutil


def save_config_metrics_and_plots(
    csv_path="round_metrics.csv", 
    output_dir_base="plots"
):
    """
    Cria uma pasta com a data atual, salva as métricas obtidas durante o treinamento,
    o arquivo de configuração, e os plots finais (acurácia, loss e bytes de comunicação).

    Parâmetros:
        - csv_path: caminho para o arquivo csv com as métricas do treinamento.
        - output_dir_base: caminho para a pasta onde serão salvos os resultados;
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Cria o caminho subpasta/horario (ex: plots/2026-10-18_16-14-56)
    output_dir = os.path.join(output_dir_base, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Copia o arquivo de métricas .csv do treinamento
    if os.path.exists(csv_path):
        metrics_copy_path = os.path.join(output_dir, os.path.basename(csv_path))
        shutil.copy2(csv_path, metrics_copy_path)

    # Copia o arquivo de configurações do treinamento
    toml_path = "pyproject.toml"
    if os.path.exists(toml_path):
        shutil.copy2(toml_path, os.path.join(output_dir, "pyproject.toml"))

    # Leitura dos dados
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

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
    sns.lineplot(
        data=df,
        x="server_round",
        y="loss",
        linewidth=2.5,
        ax=ax,
    )
    ax.set_title(
        "Evolução da Perda (Loss) por Rodada",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Rodada do Servidor (Server Round)")
    ax.set_ylabel("Perda Global (Loss)")
    ax.set_xticks(df["server_round"])

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
    sns.lineplot(
        data=df,
        x="server_round",
        y="accuracy",
        linewidth=2.5,
        ax=ax,
    )
    ax.set_title(
        "Evolução da Acurácia Global por Rodada",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Rodada do Servidor (Server Round)")
    ax.set_ylabel("Acurácia Top-1")
    ax.set_xticks(df["server_round"])
    ax.set_ylim(0, max(df["accuracy"]) * 1.25)
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
    sns.lineplot(
        data=df,
        x="server_round",
        y="loss",
        linewidth=2.5,
        ax=axes[0],
    )
    axes[0].set_title("Perda Global (Loss)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Rodada")
    axes[0].set_ylabel("Loss")
    axes[0].set_xticks(df["server_round"])

    # Subplot 2: Accuracy
    sns.lineplot(
        data=df,
        x="server_round",
        y="accuracy",
        linewidth=2.5,
        ax=axes[1],
    )
    axes[1].set_title(
        "Acurácia Global Top-1", fontsize=12, fontweight="bold"
    )
    axes[1].set_xlabel("Rodada")
    axes[1].set_ylabel("Acurácia")
    axes[1].set_xticks(df["server_round"])

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