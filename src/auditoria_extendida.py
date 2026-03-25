import argparse
import os
import re
from collections import Counter

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Auditoria extendida de sesgos para datasets de manos"
    )
    parser.add_argument(
        "--input-unified",
        default="csv/dataset_unificado_freihand_hagrid.csv",
        help="CSV unificado de entrada.",
    )
    parser.add_argument(
        "--output-metrics",
        default="csv/auditoria_extendida_metricas.csv",
        help="CSV de salida con metricas resumidas.",
    )
    parser.add_argument(
        "--output-report",
        default="reportes/reporte_auditoria_extendida.md",
        help="Reporte markdown de salida.",
    )
    parser.add_argument(
        "--output-plots-dir",
        default="graficos/auditoria_extendida",
        help="Directorio para guardar graficos de la auditoria.",
    )
    parser.add_argument(
        "--output-image-features",
        default="csv/auditoria_extendida_features_imagen.csv",
        help="CSV opcional con features por imagen muestreada.",
    )
    parser.add_argument(
        "--max-image-samples",
        type=int,
        default=3000,
        help="Maximo de imagenes a muestrear para metricas de iluminacion/fondo.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para muestreo reproducible.",
    )
    return parser.parse_args()


def ensure_output_dirs(*paths):
    for path in paths:
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)


def to_metric_rows(metric_dict):
    rows = []
    for key, value in metric_dict.items():
        if isinstance(value, (np.floating, float)):
            value = float(value)
        rows.append({"metric": key, "value": value})
    return rows


def normalize_text(v):
    if pd.isna(v):
        return None
    s = str(v).strip().lower()
    return s if s else None


def infer_age_bucket(value):
    text = normalize_text(value)
    if text is None:
        return "unknown"

    if "child" in text or "kid" in text or "ni" in text:
        return "child"
    if "teen" in text or "adolesc" in text:
        return "teen"
    if "elder" in text or "senior" in text or "old" in text or "anc" in text:
        return "senior"
    if "adult" in text:
        return "adult"

    numbers = re.findall(r"\d+", text)
    if numbers:
        age = int(numbers[0])
        if age < 13:
            return "child"
        if age < 18:
            return "teen"
        if age < 60:
            return "adult"
        return "senior"

    return "unknown"


def summarize_counter(prefix, counter):
    total = sum(counter.values())
    out = {f"{prefix}.total": total}
    for key, count in counter.items():
        out[f"{prefix}.{key}.count"] = int(count)
        out[f"{prefix}.{key}.pct"] = round((count / total) * 100, 2) if total else 0.0
    return out


def has_informative_categories(counter):
    if not counter:
        return False
    for key, count in counter.items():
        if key != "unknown" and count > 0:
            return True
    return False


def remove_metric_prefixes(metrics, prefixes):
    for key in list(metrics.keys()):
        if any(key.startswith(prefix) for prefix in prefixes):
            metrics.pop(key, None)


def compute_source_metrics(df):
    metrics = {}
    counter = Counter()

    source_series = df.get("source")
    if source_series is None:
        metrics["source.total"] = 0
        metrics["source.mode"] = "desconocido"
        return metrics, counter

    for value in source_series:
        src = normalize_text(value) or "unknown"
        counter[src] += 1

    metrics.update(summarize_counter("source", counter))

    has_freihand = counter.get("freihand", 0) > 0
    has_hagrid = counter.get("hagrid", 0) > 0
    if has_freihand and has_hagrid:
        mode = "ambos"
    elif has_freihand:
        mode = "solo_freihand"
    elif has_hagrid:
        mode = "solo_hagrid"
    else:
        mode = "desconocido"
    metrics["source.mode"] = mode

    return metrics, counter


def compute_demographic_metrics(df):
    metrics = {}

    age_series = df.get("age")
    age_counter = Counter()
    if age_series is not None:
        for value in age_series:
            age_counter[infer_age_bucket(value)] += 1
    metrics.update(summarize_counter("age", age_counter))

    gender_series = df.get("gender")
    gender_counter = Counter()
    if gender_series is not None:
        for value in gender_series:
            g = normalize_text(value) or "unknown"
            gender_counter[g] += 1
    metrics.update(summarize_counter("gender", gender_counter))

    if "hand_scale" in df.columns:
        hs = pd.to_numeric(df["hand_scale"], errors="coerce").dropna()
        metrics["hand_scale.count"] = int(hs.shape[0])
        if not hs.empty:
            metrics["hand_scale.mean"] = round(float(hs.mean()), 5)
            metrics["hand_scale.std"] = round(float(hs.std()), 5)
            metrics["hand_scale.min"] = round(float(hs.min()), 5)
            metrics["hand_scale.max"] = round(float(hs.max()), 5)

    return metrics, age_counter, gender_counter


def compute_accessory_occlusion_proxies(df):
    metrics = {}

    if "num_bboxes" in df.columns:
        num_bboxes = pd.to_numeric(df["num_bboxes"], errors="coerce")
        valid = num_bboxes.dropna()
        metrics["occlusion.samples_with_bbox"] = int(valid.shape[0])
        if not valid.empty:
            multi = (valid > 1).sum()
            metrics["occlusion.multi_bbox.count"] = int(multi)
            metrics["occlusion.multi_bbox.pct"] = round(float(multi / len(valid) * 100), 2)

    if "num_hands_detected" in df.columns:
        nh = pd.to_numeric(df["num_hands_detected"], errors="coerce")
        valid = nh.dropna()
        metrics["hands_detected.samples"] = int(valid.shape[0])
        if not valid.empty:
            two_plus = (valid >= 2).sum()
            metrics["hands_detected.two_or_more.count"] = int(two_plus)
            metrics["hands_detected.two_or_more.pct"] = round(float(two_plus / len(valid) * 100), 2)

    return metrics


def compute_laterality_metrics(df):
    metrics = {}
    counter = Counter()

    if "leading_hand" in df.columns:
        for value in df["leading_hand"]:
            text = normalize_text(value)
            if text in {"left", "right"}:
                counter[text] += 1
            elif text is None:
                counter["unknown"] += 1
            else:
                counter[text] += 1

    metrics.update(summarize_counter("laterality", counter))

    left = counter.get("left", 0)
    right = counter.get("right", 0)
    if left > 0 and right > 0:
        metrics["laterality.balance_ratio_left_right"] = round(float(left / right), 4)

    return metrics, counter


def sample_existing_images(df, max_samples, seed):
    if "image_path" not in df.columns:
        return []

    paths = [p for p in df["image_path"].dropna().astype(str).tolist() if os.path.exists(p)]
    if not paths:
        return []

    if len(paths) <= max_samples:
        return paths

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(paths), size=max_samples, replace=False)
    return [paths[i] for i in idx]


def compute_image_condition_metrics(image_paths):
    metrics = {
        "image_metrics.sampled_images": int(len(image_paths)),
    }
    if not image_paths:
        return metrics, pd.DataFrame()

    mean_brightness = []
    contrast_std = []
    shadow_ratio = []
    highlight_ratio = []
    cool_warm_delta = []
    edge_density = []
    lap_var = []
    sample_paths = []

    for path in image_paths:
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        sample_paths.append(path)

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        mean_brightness.append(float(hsv[:, :, 2].mean()))
        contrast_std.append(float(gray.std()))
        shadow_ratio.append(float((gray < 40).mean()))
        highlight_ratio.append(float((gray > 220).mean()))

        b_mean = float(bgr[:, :, 0].mean())
        r_mean = float(bgr[:, :, 2].mean())
        cool_warm_delta.append(b_mean - r_mean)

        edges = cv2.Canny(gray, 100, 200)
        edge_density.append(float((edges > 0).mean()))
        lap_var.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

    if not mean_brightness:
        return metrics, pd.DataFrame()

    brightness_np = np.asarray(mean_brightness)
    contrast_np = np.asarray(contrast_std)
    shadow_np = np.asarray(shadow_ratio)
    highlight_np = np.asarray(highlight_ratio)
    coolwarm_np = np.asarray(cool_warm_delta)
    edge_np = np.asarray(edge_density)
    lap_np = np.asarray(lap_var)

    metrics.update(
        {
            "lighting.mean_brightness": round(float(brightness_np.mean()), 3),
            "lighting.low_brightness_pct": round(float((brightness_np < 70).mean() * 100), 2),
            "lighting.strong_shadows_pct": round(float((shadow_np > 0.25).mean() * 100), 2),
            "lighting.highlights_pct": round(float((highlight_np > 0.15).mean() * 100), 2),
            "lighting.low_contrast_pct": round(float((contrast_np < 35).mean() * 100), 2),
            "lighting.cool_temperature_pct": round(float((coolwarm_np > 8).mean() * 100), 2),
            "lighting.warm_temperature_pct": round(float((coolwarm_np < -8).mean() * 100), 2),
            "lighting.backlight_proxy_pct": round(
                float(((brightness_np < 80) & (contrast_np < 30)).mean() * 100), 2
            ),
            "background.edge_density_mean": round(float(edge_np.mean()), 4),
            "background.edge_density_high_pct": round(float((edge_np > 0.12).mean() * 100), 2),
            "background.laplacian_var_mean": round(float(lap_np.mean()), 2),
            "background.laplacian_var_high_pct": round(float((lap_np > 250).mean() * 100), 2),
        }
    )

    features_df = pd.DataFrame(
        {
            "image_path": sample_paths,
            "brightness": mean_brightness,
            "contrast_std": contrast_std,
            "shadow_ratio": shadow_ratio,
            "highlight_ratio": highlight_ratio,
            "cool_warm_delta": cool_warm_delta,
            "edge_density": edge_density,
            "laplacian_var": lap_var,
        }
    )

    return metrics, features_df


def save_count_plot(counter, title, x_label, y_label, output_path):
    if not counter:
        return
    keys = list(counter.keys())
    values = [counter[k] for k in keys]

    plt.figure(figsize=(9, 5))
    plt.bar(keys, values, edgecolor="black")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_hist_plot(series, title, x_label, output_path, bins=30):
    if series.empty:
        return

    plt.figure(figsize=(9, 5))
    plt.hist(series, bins=bins, edgecolor="black")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Conteo")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def remove_file_if_exists(path):
    if os.path.exists(path):
        os.remove(path)


def save_plots(
    age_counter,
    gender_counter,
    laterality_counter,
    features_df,
    output_dir,
    include_age,
    include_gender,
    include_laterality,
):
    os.makedirs(output_dir, exist_ok=True)

    age_plot = os.path.join(output_dir, "edad_buckets.png")
    gender_plot = os.path.join(output_dir, "genero_distribucion.png")
    laterality_plot = os.path.join(output_dir, "lateralidad_distribucion.png")

    if include_age:
        save_count_plot(
            counter=age_counter,
            title="Distribucion por Edad (bucket)",
            x_label="Bucket",
            y_label="Muestras",
            output_path=age_plot,
        )
    else:
        remove_file_if_exists(age_plot)

    if include_gender:
        save_count_plot(
            counter=gender_counter,
            title="Distribucion por Genero",
            x_label="Genero",
            y_label="Muestras",
            output_path=gender_plot,
        )
    else:
        remove_file_if_exists(gender_plot)

    if include_laterality:
        save_count_plot(
            counter=laterality_counter,
            title="Distribucion de Lateralidad",
            x_label="Lateralidad",
            y_label="Muestras",
            output_path=laterality_plot,
        )
    else:
        remove_file_if_exists(laterality_plot)

    if features_df.empty:
        return

    save_hist_plot(
        series=features_df["brightness"],
        title="Iluminacion: Brillo medio por imagen",
        x_label="Brillo (canal V HSV)",
        output_path=os.path.join(output_dir, "iluminacion_brillo_hist.png"),
    )
    save_hist_plot(
        series=features_df["contrast_std"],
        title="Iluminacion: Contraste por imagen",
        x_label="Desviacion estandar (gris)",
        output_path=os.path.join(output_dir, "iluminacion_contraste_hist.png"),
    )
    save_hist_plot(
        series=features_df["edge_density"],
        title="Fondo: Densidad de bordes",
        x_label="Edge density",
        output_path=os.path.join(output_dir, "fondo_edge_density_hist.png"),
    )


def build_report(
    metrics,
    image_samples,
    plots_dir,
    include_age,
    include_gender,
    include_laterality,
    source_counter,
):
    def m(key, default="N/A"):
        return metrics.get(key, default)

    lines = [
        "# Reporte de Auditoria Extendida",
        "",
        "Este reporte cubre 5 dimensiones adicionales de sesgo y robustez.",
        "",
        "## Datasets usados en esta auditoria",
        f"- Modo de uso: {m('source.mode', 'desconocido')}.",
        f"- Total de muestras auditadas: {m('source.total', 0)}.",
        "",
        "## 2) Accesorios y Oclusiones Artificiales (Proxy)",
        "- No hay etiquetas directas de joyas/unas/mangas en el dataset unificado actual.",
        f"- Proxy de oclusion por multiples cajas (num_bboxes > 1): {m('occlusion.multi_bbox.count', 0)} muestras ({m('occlusion.multi_bbox.pct', 0)}%).",
        f"- Escenas con 2 o mas manos detectadas: {m('hands_detected.two_or_more.count', 0)} ({m('hands_detected.two_or_more.pct', 0)}%).",
        "",
        "## 3) Condiciones de Iluminacion",
        f"- Imagenes analizadas: {image_samples}.",
        f"- Brillo medio: {m('lighting.mean_brightness', 'N/A')}.",
        f"- Baja iluminacion: {m('lighting.low_brightness_pct', 'N/A')}%.",
        f"- Sombras fuertes: {m('lighting.strong_shadows_pct', 'N/A')}%.",
        f"- Posible contraluz (proxy): {m('lighting.backlight_proxy_pct', 'N/A')}%.",
        f"- Temperatura de color: fria={m('lighting.cool_temperature_pct', 'N/A')}%, calida={m('lighting.warm_temperature_pct', 'N/A')}%.",
        "",
        "## 4) Complejidad de Fondo (Proxy)",
        f"- Densidad de bordes media: {m('background.edge_density_mean', 'N/A')}.",
        f"- Escenas con fondo complejo (edge density alta): {m('background.edge_density_high_pct', 'N/A')}%.",
        f"- Varianza de Laplaciano media: {m('background.laplacian_var_mean', 'N/A')}.",
        "",
        "## Notas",
        "- Varias metricas son proxys y no reemplazan etiquetado manual especializado.",
        "- Para accesorios (joyas, unas, mangas) se recomienda una ronda de etiquetado adicional.",
        f"- Graficos exportados en: {plots_dir}",
    ]

    if source_counter:
        source_items = sorted(source_counter.items(), key=lambda x: x[0])
        source_lines = [
            f"- {src}: {count} muestras ({m(f'source.{src}.pct', 0)}%)."
            for src, count in source_items
        ]
        idx = lines.index("## 2) Accesorios y Oclusiones Artificiales (Proxy)")
        lines[idx:idx] = source_lines + [""]

    if include_age or include_gender:
        idx = lines.index("## 2) Accesorios y Oclusiones Artificiales (Proxy)")
        section = ["## 1) Diversidad Etaria y Morfologica"]
        if include_age:
            section.append(
                f"- Distribucion por edad: child={m('age.child.count', 0)}, teen={m('age.teen.count', 0)}, adult={m('age.adult.count', 0)}, senior={m('age.senior.count', 0)}, unknown={m('age.unknown.count', 0)}."
            )
        if include_gender:
            section.append(
                f"- Distribucion de genero: {m('gender.total', 0)} muestras con metadata; unknown={m('gender.unknown.count', 0)}."
            )
        section.append(
            f"- Hand scale (proxy morfologico): count={m('hand_scale.count', 0)}, mean={m('hand_scale.mean', 'N/A')}, min={m('hand_scale.min', 'N/A')}, max={m('hand_scale.max', 'N/A')}."
        )
        section.append("")
        lines[idx:idx] = section

    if include_laterality:
        idx = lines.index("## Notas")
        section = [
            "## 5) Sesgo de Lateralidad",
            f"- Left: {m('laterality.left.count', 0)}, Right: {m('laterality.right.count', 0)}, Unknown: {m('laterality.unknown.count', 0)}.",
            f"- Ratio left/right: {m('laterality.balance_ratio_left_right', 'N/A')}.",
            "",
        ]
        lines[idx:idx] = section

    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    if args.max_image_samples <= 0:
        raise ValueError("--max-image-samples debe ser mayor que 0")

    ensure_output_dirs(args.output_metrics, args.output_report, args.output_image_features)
    os.makedirs(args.output_plots_dir, exist_ok=True)

    if not os.path.exists(args.input_unified):
        raise FileNotFoundError(f"No existe CSV unificado: {args.input_unified}")

    df = pd.read_csv(args.input_unified)
    if df.empty:
        raise RuntimeError("El CSV unificado esta vacio")

    metrics = {}
    source_metrics, source_counter = compute_source_metrics(df)
    metrics.update(source_metrics)
    demographic_metrics, age_counter, gender_counter = compute_demographic_metrics(df)
    metrics.update(demographic_metrics)
    metrics.update(compute_accessory_occlusion_proxies(df))
    laterality_metrics, laterality_counter = compute_laterality_metrics(df)
    metrics.update(laterality_metrics)

    include_age = has_informative_categories(age_counter)
    include_gender = has_informative_categories(gender_counter)
    include_laterality = has_informative_categories(laterality_counter)

    if not include_age:
        remove_metric_prefixes(metrics, ["age."])
    if not include_gender:
        remove_metric_prefixes(metrics, ["gender."])
    if not include_laterality:
        remove_metric_prefixes(metrics, ["laterality."])

    image_paths = sample_existing_images(
        df=df,
        max_samples=args.max_image_samples,
        seed=args.seed,
    )
    image_metrics, features_df = compute_image_condition_metrics(image_paths)
    metrics.update(image_metrics)

    metrics_df = pd.DataFrame(to_metric_rows(metrics))
    metrics_df.to_csv(args.output_metrics, index=False)
    features_df.to_csv(args.output_image_features, index=False)

    save_plots(
        age_counter=age_counter,
        gender_counter=gender_counter,
        laterality_counter=laterality_counter,
        features_df=features_df,
        output_dir=args.output_plots_dir,
        include_age=include_age,
        include_gender=include_gender,
        include_laterality=include_laterality,
    )

    report = build_report(
        metrics,
        image_samples=image_metrics.get("image_metrics.sampled_images", 0),
        plots_dir=args.output_plots_dir,
        include_age=include_age,
        include_gender=include_gender,
        include_laterality=include_laterality,
        source_counter=source_counter,
    )
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Metricas guardadas en: {args.output_metrics}")
    print(f"Features por imagen guardadas en: {args.output_image_features}")
    print(f"Graficos guardados en: {args.output_plots_dir}")
    print(f"Reporte guardado en: {args.output_report}")


if __name__ == "__main__":
    main()
