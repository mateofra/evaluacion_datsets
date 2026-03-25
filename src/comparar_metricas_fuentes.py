import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from auditoria_extendida import (
    compute_accessory_occlusion_proxies,
    compute_demographic_metrics,
    compute_image_condition_metrics,
    compute_laterality_metrics,
    sample_existing_images,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compara FreiHAND vs HaGRID con muestras balanceadas y grafico unico de metricas."
    )
    parser.add_argument(
        "--input-unified",
        default="csv/dataset_unificado_freihand500_hagrid_subset100.csv",
        help="CSV unificado de entrada con columna source.",
    )
    parser.add_argument(
        "--source-a",
        default="freihand",
        help="Nombre de la primera fuente en columna source.",
    )
    parser.add_argument(
        "--source-b",
        default="hagrid",
        help="Nombre de la segunda fuente en columna source.",
    )
    parser.add_argument(
        "--max-image-samples-per-source",
        type=int,
        default=800,
        help="Maximo de imagenes por fuente para metricas de iluminacion/fondo.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para muestreo reproducible.",
    )
    parser.add_argument(
        "--output-csv",
        default="csv/comparativa_metricas_freihand_vs_hagrid.csv",
        help="CSV de salida con comparacion de metricas.",
    )
    parser.add_argument(
        "--output-plot",
        default="graficos/comparativa_metricas_freihand_vs_hagrid.png",
        help="Grafico de salida comparando todas las metricas numericas.",
    )
    return parser.parse_args()


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def source_subset(df, source_name):
    return df[df["source"].astype(str).str.lower() == source_name.lower()].copy()


def compute_metrics(df_source, max_image_samples, seed):
    metrics = {}

    demographic_metrics, _, _ = compute_demographic_metrics(df_source)
    metrics.update(demographic_metrics)
    metrics.update(compute_accessory_occlusion_proxies(df_source))

    laterality_metrics, _ = compute_laterality_metrics(df_source)
    metrics.update(laterality_metrics)

    image_paths = sample_existing_images(
        df=df_source,
        max_samples=max_image_samples,
        seed=seed,
    )
    image_metrics, _ = compute_image_condition_metrics(image_paths)
    metrics.update(image_metrics)

    return metrics


def to_float_or_nan(value):
    numeric = pd.to_numeric([value], errors="coerce")
    return float(numeric[0]) if not pd.isna(numeric[0]) else np.nan


def to_numeric_union_metrics(metrics_a, metrics_b):
    all_metrics = sorted(set(metrics_a.keys()) | set(metrics_b.keys()))
    rows = []

    for key in all_metrics:
        a_num = to_float_or_nan(metrics_a.get(key, np.nan))
        b_num = to_float_or_nan(metrics_b.get(key, np.nan))
        if np.isnan(a_num) and np.isnan(b_num):
            continue

        delta = np.nan
        if not np.isnan(a_num) and not np.isnan(b_num):
            delta = float(b_num - a_num)

        delta_pct = np.nan
        if not np.isnan(delta) and a_num != 0.0:
            delta_pct = float((delta / a_num) * 100.0)

        rows.append(
            {
                "metric": key,
                "source_a_value": a_num,
                "source_b_value": b_num,
                "delta_b_minus_a": delta,
                "delta_pct_vs_a": delta_pct,
            }
        )

    return pd.DataFrame(rows)


def plot_all_metrics_heatmap(df_cmp, source_a, source_b, output_plot):
    if df_cmp.empty:
        raise RuntimeError("No hay metricas numericas compartidas para graficar.")

    values = df_cmp[["source_a_value", "source_b_value"]].to_numpy(dtype=float)
    valid_mask = ~np.isnan(values)

    mins = np.nanmin(values, axis=1, keepdims=True)
    maxs = np.nanmax(values, axis=1, keepdims=True)
    denom = maxs - mins

    norm = np.full(values.shape, np.nan, dtype=float)
    for i in range(values.shape[0]):
        row_vals = values[i]
        row_valid = valid_mask[i]
        if not row_valid.any():
            continue

        rmin = mins[i, 0]
        rmax = maxs[i, 0]
        if np.isnan(rmin) or np.isnan(rmax):
            continue
        if rmax == rmin:
            norm[i, row_valid] = 0.5
        else:
            norm[i, row_valid] = (row_vals[row_valid] - rmin) / (rmax - rmin)

    n_metrics = len(df_cmp)
    fig_h = max(8, int(n_metrics * 0.45))

    plt.figure(figsize=(11, fig_h))
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad(color="#D9D9D9")
    plt.imshow(norm, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    plt.colorbar(label="Valor relativo por metrica")

    plt.xticks([0, 1], [source_a, source_b])
    plt.yticks(np.arange(n_metrics), df_cmp["metric"].tolist())
    plt.title("Comparacion FreiHAND vs HaGRID (muestras balanceadas)")

    for i in range(n_metrics):
        for j in range(2):
            raw_val = values[i, j]
            text = "N/A" if np.isnan(raw_val) else f"{raw_val:.4g}"
            plt.text(j, i, text, ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_plot, dpi=180)
    plt.close()


def main():
    args = parse_args()

    if args.max_image_samples_per_source <= 0:
        raise ValueError("--max-image-samples-per-source debe ser mayor que 0")

    if not os.path.exists(args.input_unified):
        raise FileNotFoundError(f"No existe el CSV de entrada: {args.input_unified}")

    ensure_parent_dir(args.output_csv)
    ensure_parent_dir(args.output_plot)

    df = pd.read_csv(args.input_unified)
    if df.empty:
        raise RuntimeError("El CSV unificado esta vacio")
    if "source" not in df.columns:
        raise RuntimeError("El CSV unificado no tiene columna source")

    df_a = source_subset(df, args.source_a)
    df_b = source_subset(df, args.source_b)

    if df_a.empty or df_b.empty:
        raise RuntimeError(
            f"No hay filas suficientes para comparar. {args.source_a}={len(df_a)}, {args.source_b}={len(df_b)}"
        )

    n = min(len(df_a), len(df_b))
    df_a_bal = df_a.sample(n=n, random_state=args.seed)
    df_b_bal = df_b.sample(n=n, random_state=args.seed)

    metrics_a = compute_metrics(
        df_source=df_a_bal,
        max_image_samples=min(args.max_image_samples_per_source, n),
        seed=args.seed,
    )
    metrics_b = compute_metrics(
        df_source=df_b_bal,
        max_image_samples=min(args.max_image_samples_per_source, n),
        seed=args.seed,
    )

    cmp_df = to_numeric_union_metrics(metrics_a, metrics_b)
    if cmp_df.empty:
        raise RuntimeError("No se pudieron calcular metricas numericas para comparar")

    cmp_df.insert(1, "source_a", args.source_a)
    cmp_df.insert(2, "source_b", args.source_b)
    cmp_df.insert(3, "balanced_samples_per_source", n)

    cmp_df.to_csv(args.output_csv, index=False)
    plot_all_metrics_heatmap(cmp_df, args.source_a, args.source_b, args.output_plot)

    print(f"Comparacion guardada en CSV: {args.output_csv}")
    print(f"Grafico guardado en: {args.output_plot}")
    print(f"Muestras balanceadas por fuente: {n}")


if __name__ == "__main__":
    main()
