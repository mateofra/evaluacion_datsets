import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MST_COLORS = [
    "#F6EDE4",
    "#F3E7DB",
    "#F7EACE",
    "#EACEAF",
    "#D8B28F",
    "#BC9273",
    "#996C51",
    "#6F4731",
    "#422E23",
    "#2B1E16",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Genera ejemplos de graficos de comparacion MST FreiHAND vs HaGRID."
    )
    parser.add_argument(
        "--input-csv",
        default="csv/comparativa_mst_enriquecida_freihand_vs_hagrid.csv",
        help="CSV con columnas count_* y pct_* por nivel MST.",
    )
    parser.add_argument(
        "--output-dir",
        default="graficos/ejemplos_mst",
        help="Directorio de salida para los ejemplos de graficos.",
    )
    return parser.parse_args()


def detect_sources(df):
    count_cols = [c for c in df.columns if c.startswith("count_")]
    pct_cols = [c for c in df.columns if c.startswith("pct_")]

    if len(count_cols) != 2 or len(pct_cols) != 2:
        raise RuntimeError("Se esperaban exactamente dos columnas count_* y dos columnas pct_*.")

    src_a = count_cols[0].replace("count_", "")
    src_b = count_cols[1].replace("count_", "")
    return src_a, src_b, count_cols, pct_cols


def save_grouped_counts(df, count_cols, labels, output_path):
    x = np.arange(len(df))
    w = 0.38

    plt.figure(figsize=(12, 6))
    b1 = plt.bar(x - w / 2, df[count_cols[0]], width=w, label=labels[0], edgecolor="black")
    b2 = plt.bar(x + w / 2, df[count_cols[1]], width=w, label=labels[1], edgecolor="black")

    plt.title("Ejemplo 1: Barras agrupadas de conteo MST")
    plt.xlabel("Nivel MST")
    plt.ylabel("Numero de imagenes")
    plt.xticks(x, df["mst"].astype(int).tolist())
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend()

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                plt.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{int(h)}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_percentage_lines(df, pct_cols, labels, output_path):
    x = df["mst"].astype(int).to_numpy()

    plt.figure(figsize=(12, 6))
    plt.plot(x, df[pct_cols[0]], marker="o", linewidth=2.2, label=f"% {labels[0]}")
    plt.plot(x, df[pct_cols[1]], marker="o", linewidth=2.2, label=f"% {labels[1]}")

    plt.title("Ejemplo 2: Curvas de porcentaje por nivel MST")
    plt.xlabel("Nivel MST")
    plt.ylabel("Porcentaje")
    plt.xticks(x)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_stacked_percentage(df, pct_cols, labels, output_path):
    x = np.arange(len(df))

    plt.figure(figsize=(12, 6))
    plt.bar(x, df[pct_cols[0]], label=labels[0], edgecolor="black")
    plt.bar(x, df[pct_cols[1]], bottom=df[pct_cols[0]], label=labels[1], edgecolor="black")

    plt.title("Ejemplo 3: Barras apiladas de porcentaje (composicion)")
    plt.xlabel("Nivel MST")
    plt.ylabel("Porcentaje combinado")
    plt.xticks(x, df["mst"].astype(int).tolist())
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_radar_chart(df, pct_cols, labels, output_path):
    categories = [str(int(v)) for v in df["mst"].tolist()]
    vals_a = df[pct_cols[0]].to_list()
    vals_b = df[pct_cols[1]].to_list()

    # Cerrar poligono.
    vals_a += vals_a[:1]
    vals_b += vals_b[:1]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, polar=True)

    ax.plot(angles, vals_a, linewidth=2, label=labels[0])
    ax.fill(angles, vals_a, alpha=0.15)
    ax.plot(angles, vals_b, linewidth=2, label=labels[1])
    ax.fill(angles, vals_b, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title("Ejemplo 4: Radar de distribucion porcentual MST", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_single_dataset_category_colors(df, count_col, dataset_label, output_path):
    x = np.arange(len(df))
    y = df[count_col].to_numpy(dtype=float)

    plt.figure(figsize=(12, 6))
    bars = plt.bar(x, y, color=MST_COLORS, edgecolor="black")

    plt.title(f"Distribucion MST por categoria - {dataset_label}")
    plt.xlabel("Nivel MST")
    plt.ylabel("Numero de imagenes")
    plt.xticks(x, df["mst"].astype(int).tolist())
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    for bar in bars:
        h = bar.get_height()
        if h > 0:
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                h + 1,
                f"{int(h)}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main():
    args = parse_args()
    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"No existe el CSV: {args.input_csv}")

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    src_a, src_b, count_cols, pct_cols = detect_sources(df)

    save_single_dataset_category_colors(
        df=df,
        count_col=count_cols[0],
        dataset_label=src_a,
        output_path=os.path.join(args.output_dir, f"barras_mst_{src_a}.png"),
    )
    save_single_dataset_category_colors(
        df=df,
        count_col=count_cols[1],
        dataset_label=src_b,
        output_path=os.path.join(args.output_dir, f"barras_mst_{src_b}.png"),
    )

    print(f"Graficas por dataset generadas en: {args.output_dir}")


if __name__ == "__main__":
    main()
