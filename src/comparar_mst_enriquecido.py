import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Comparativa MST enriquecida entre dos fuentes con muestras balanceadas."
    )
    parser.add_argument(
        "--input-unified",
        default="csv/dataset_unificado_freihand500_hagrid_subset100.csv",
        help="CSV unificado con columna source y mst_score.",
    )
    parser.add_argument("--source-a", default="freihand", help="Fuente A.")
    parser.add_argument("--source-b", default="hagrid", help="Fuente B.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla de muestreo.")
    parser.add_argument(
        "--target-balanced-samples",
        type=int,
        default=None,
        help="Cantidad objetivo por fuente. Si supera lo disponible, remuestrea con reemplazo.",
    )
    parser.add_argument(
        "--output-plot",
        default="graficos/comparativa_mst_enriquecida_freihand_vs_hagrid.png",
        help="Ruta del grafico de salida.",
    )
    parser.add_argument(
        "--output-csv",
        default="csv/comparativa_mst_enriquecida_freihand_vs_hagrid.csv",
        help="CSV de salida con conteos y porcentajes por MST.",
    )
    return parser.parse_args()


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def get_balanced_subsets(df, source_a, source_b, seed, target_n=None):
    df_a = df[df["source"].astype(str).str.lower() == source_a.lower()].copy()
    df_b = df[df["source"].astype(str).str.lower() == source_b.lower()].copy()

    if df_a.empty or df_b.empty:
        raise RuntimeError(
            f"No hay datos para comparar: {source_a}={len(df_a)}, {source_b}={len(df_b)}"
        )

    base_n = min(len(df_a), len(df_b))
    n = target_n if target_n is not None else base_n

    if n <= 0:
        raise RuntimeError("No hay muestras suficientes para balancear")

    replace_a = n > len(df_a)
    replace_b = n > len(df_b)

    df_a = df_a.sample(n=n, replace=replace_a, random_state=seed)
    df_b = df_b.sample(n=n, replace=replace_b, random_state=seed)
    return df_a, df_b, n, base_n, replace_a, replace_b


def mst_distribution(df):
    mst = pd.to_numeric(df["mst_score"], errors="coerce").dropna().astype(int)
    mst = mst[(mst >= 1) & (mst <= 10)]
    counts = mst.value_counts().reindex(range(1, 11), fill_value=0).sort_index()
    total = int(counts.sum())
    pct = (counts / total * 100.0) if total > 0 else counts.astype(float)

    mean_val = float(mst.mean()) if len(mst) > 0 else np.nan
    median_val = float(mst.median()) if len(mst) > 0 else np.nan
    light_pct = float((mst.isin([1, 2, 3, 4]).mean() * 100.0)) if len(mst) > 0 else np.nan

    return counts, pct, total, mean_val, median_val, light_pct


def export_table(output_csv, counts_a, pct_a, counts_b, pct_b, source_a, source_b, balanced_n):
    rows = []
    for mst in range(1, 11):
        rows.append(
            {
                "mst": mst,
                f"count_{source_a}": int(counts_a.loc[mst]),
                f"pct_{source_a}": round(float(pct_a.loc[mst]), 4),
                f"count_{source_b}": int(counts_b.loc[mst]),
                f"pct_{source_b}": round(float(pct_b.loc[mst]), 4),
                "balanced_samples_per_source": int(balanced_n),
            }
        )
    pd.DataFrame(rows).to_csv(output_csv, index=False)


def plot_enriched(
    output_plot,
    counts_a,
    pct_a,
    counts_b,
    pct_b,
    source_a,
    source_b,
    balanced_n,
    stats_a,
    stats_b,
):
    mst_levels = np.arange(1, 11)
    x = np.arange(len(mst_levels))
    w = 0.38

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

    b1 = ax1.bar(x - w / 2, counts_a.values, width=w, label=f"{source_a} (n={balanced_n})")
    b2 = ax1.bar(x + w / 2, counts_b.values, width=w, label=f"{source_b} (n={balanced_n})")

    for sep in [3.5, 6.5]:
        ax1.axvline(sep, color="gray", linestyle="--", linewidth=1)
        ax2.axvline(sep, color="gray", linestyle="--", linewidth=1)

    ax1.set_title("Comparativa MST enriquecida: FreiHAND vs HaGRID (muestras balanceadas)")
    ax1.set_ylabel("Numero de imagenes")
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.legend(loc="upper right")

    # Etiquetas de conteo sobre barras.
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 1,
                    f"{int(h)}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax2.plot(x, pct_a.values, marker="o", linewidth=2, label=f"% {source_a}")
    ax2.plot(x, pct_b.values, marker="o", linewidth=2, label=f"% {source_b}")
    ax2.set_ylabel("Porcentaje por nivel MST")
    ax2.set_xlabel("Escala MST")
    ax2.set_xticks(x)
    ax2.set_xticklabels(mst_levels)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    ax2.legend(loc="upper right")

    text = (
        f"Resumen {source_a}: media={stats_a[0]:.2f}, mediana={stats_a[1]:.2f}, MST1-4={stats_a[2]:.2f}%\n"
        f"Resumen {source_b}: media={stats_b[0]:.2f}, mediana={stats_b[1]:.2f}, MST1-4={stats_b[2]:.2f}%\n"
        "Bloques: claro (1-4), medio (5-7), oscuro (8-10)"
    )
    fig.text(
        0.015,
        0.01,
        text,
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "gray", "alpha": 0.85},
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output_plot, dpi=180)
    plt.close()


def main():
    args = parse_args()

    if not os.path.exists(args.input_unified):
        raise FileNotFoundError(f"No existe el CSV de entrada: {args.input_unified}")

    df = pd.read_csv(args.input_unified)
    if df.empty:
        raise RuntimeError("El CSV unificado esta vacio")
    if "source" not in df.columns or "mst_score" not in df.columns:
        raise RuntimeError("Faltan columnas requeridas: source y mst_score")

    ensure_parent(args.output_plot)
    ensure_parent(args.output_csv)

    df_a, df_b, balanced_n, base_n, replace_a, replace_b = get_balanced_subsets(
        df,
        args.source_a,
        args.source_b,
        args.seed,
        target_n=args.target_balanced_samples,
    )

    counts_a, pct_a, _, mean_a, median_a, light_a = mst_distribution(df_a)
    counts_b, pct_b, _, mean_b, median_b, light_b = mst_distribution(df_b)

    export_table(
        output_csv=args.output_csv,
        counts_a=counts_a,
        pct_a=pct_a,
        counts_b=counts_b,
        pct_b=pct_b,
        source_a=args.source_a,
        source_b=args.source_b,
        balanced_n=balanced_n,
    )

    plot_enriched(
        output_plot=args.output_plot,
        counts_a=counts_a,
        pct_a=pct_a,
        counts_b=counts_b,
        pct_b=pct_b,
        source_a=args.source_a,
        source_b=args.source_b,
        balanced_n=balanced_n,
        stats_a=(mean_a, median_a, light_a),
        stats_b=(mean_b, median_b, light_b),
    )

    print(f"Grafico enriquecido guardado en: {args.output_plot}")
    print(f"Tabla enriquecida guardada en: {args.output_csv}")
    print(f"Balance maximo sin reemplazo: {base_n}")
    print(f"Muestras balanceadas por fuente: {balanced_n}")
    print(
        "Remuestreo con reemplazo: "
        f"{args.source_a}={'si' if replace_a else 'no'}, "
        f"{args.source_b}={'si' if replace_b else 'no'}"
    )


if __name__ == "__main__":
    main()
