import argparse
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- CONFIGURACION ---
DATASET_ROOT = "datasets/FreiHAND_pub_v2"
RGB_DIR = os.path.join(DATASET_ROOT, "training", "rgb")
PATH_XYZ = os.path.join(DATASET_ROOT, "training_xyz.json")
PATH_K = os.path.join(DATASET_ROOT, "training_K.json")
PATH_SCALE = os.path.join(DATASET_ROOT, "training_scale.json")

MAX_GREEN_BG = 32560
NUM_SAMPLES = 5000
PATCH_RADIUS = 5
SEED = 42

OUTPUT_CSV = "csv/auditoria_freihand_muestra.csv"
OUTPUT_PLOT = "graficos/reporte_diversidad_freihand.png"
OUTPUT_REPORT = "reportes/reporte_freihand.md"

MONK_SCALE = {
    1: (246, 237, 228),
    2: (243, 231, 219),
    3: (247, 234, 208),
    4: (234, 206, 175),
    5: (216, 178, 143),
    6: (188, 146, 115),
    7: (153, 108, 81),
    8: (111, 71, 49),
    9: (66, 46, 35),
    10: (43, 30, 22),
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def project_3d_point_to_2d(point_3d, camera_matrix):
    point_3d = np.asarray(point_3d, dtype=np.float64)
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    projected = camera_matrix @ point_3d
    if projected[2] <= 0:
        return None
    x = projected[0] / projected[2]
    y = projected[1] / projected[2]
    return int(round(x)), int(round(y))


def extract_mean_patch_rgb(image_rgb, cx, cy, radius):
    height, width, _ = image_rgb.shape
    x0 = max(0, cx - radius)
    x1 = min(width, cx + radius)
    y0 = max(0, cy - radius)
    y1 = min(height, cy + radius)
    if x0 >= x1 or y0 >= y1:
        return None
    roi = image_rgb[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    return roi.mean(axis=(0, 1))


def get_closest_mst(rgb_color):
    rgb_color = np.asarray(rgb_color, dtype=np.float64)
    best_level = None
    best_dist = float("inf")
    for level, mst_rgb in MONK_SCALE.items():
        dist = np.linalg.norm(rgb_color - np.asarray(mst_rgb, dtype=np.float64))
        if dist < best_dist:
            best_dist = dist
            best_level = level
    return best_level


def build_markdown_report(df, scale_values, green_limit, sample_size):
    if df.empty:
        return (
            "# Reporte FreiHAND\n\n"
            "No se pudieron extraer muestras validas para el analisis de tonos.\n"
        )

    light_count = int(df["mst_score"].isin([1, 2, 3, 4]).sum())
    light_ratio = (light_count / len(df)) * 100

    lines = [
        "# Reporte FreiHAND",
        "",
        "## Configuracion de auditoria",
        f"- Universo analizado: primeras {green_limit} imagenes de entrenamiento (fondo verde).",
        f"- Muestra aleatoria: {sample_size} imagenes.",
        "- Punto usado para piel: landmark 9 (centro de palma).",
        "",
        "## Hallazgo de sesgo de tono",
        (
            f"- Proporcion MST 1-4 en la muestra: {light_ratio:.2f}% "
            f"({light_count}/{len(df)})."
        ),
        "- Interpretacion: el dataset FreiHAND presenta concentracion en tonos claros.",
        "- Accion recomendada: complementar con un dataset mas diverso en tono de piel (por ejemplo HAGRID o 11K Hands).",
        "",
        "## Relevancia para escenarios de video",
        (
            "- FreiHAND incluye hand scale por muestra, util para robustez en video cuando la mano "
            "se acerca o se aleja de la camara."
        ),
    ]

    if scale_values is not None and len(scale_values) > 0:
        scales_np = np.asarray(scale_values, dtype=np.float64)
        lines.extend(
            [
                f"- Rango hand scale observado: {scales_np.min():.4f} a {scales_np.max():.4f}.",
                f"- Media hand scale: {scales_np.mean():.4f}.",
                (
                    "- Esto permite calibrar mejor el tamano aparente de la mano y reducir perdidas "
                    "de tracking por cambios de distancia en webcam."
                ),
            ]
        )

    lines.append("")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Auditoria de tonos de piel en FreiHAND con landmark 9"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=NUM_SAMPLES,
        help="Cantidad de imagenes a muestrear dentro de las primeras 32560.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Semilla para muestreo aleatorio reproducible.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples debe ser mayor que 0")

    for path in [OUTPUT_CSV, OUTPUT_PLOT, OUTPUT_REPORT]:
        output_dir = os.path.dirname(path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    for required_path in [PATH_XYZ, PATH_K, RGB_DIR]:
        if not os.path.exists(required_path):
            raise FileNotFoundError(f"No existe la ruta requerida: {required_path}")

    xyz_all = load_json(PATH_XYZ)
    k_all = load_json(PATH_K)
    scale_all = load_json(PATH_SCALE) if os.path.exists(PATH_SCALE) else None

    total_by_meta = min(len(xyz_all), len(k_all))
    green_limit = min(MAX_GREEN_BG, total_by_meta)
    sample_size = min(args.num_samples, green_limit)

    rng = np.random.default_rng(args.seed)
    sampled_indices = rng.choice(green_limit, size=sample_size, replace=False)

    print(
        f"Iniciando auditoria FreiHAND: {sample_size} muestras aleatorias "
        f"de las primeras {green_limit} imagenes."
    )

    resultados = []
    for i, idx in enumerate(sampled_indices, start=1):
        img_name = f"{int(idx):08d}.jpg"
        img_path = os.path.join(RGB_DIR, img_name)
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pt_2d = project_3d_point_to_2d(xyz_all[int(idx)][9], k_all[int(idx)])
        if pt_2d is None:
            continue

        cx, cy = pt_2d
        mean_rgb = extract_mean_patch_rgb(image_rgb, cx, cy, PATCH_RADIUS)
        if mean_rgb is None:
            continue

        mst_level = get_closest_mst(mean_rgb)
        if mst_level is None:
            continue
        row = {
            "idx": int(idx),
            "archivo": img_name,
            "mst_score": int(mst_level),
            "r": float(mean_rgb[0]),
            "g": float(mean_rgb[1]),
            "b": float(mean_rgb[2]),
        }
        if scale_all is not None:
            row["hand_scale"] = float(scale_all[int(idx)])
        resultados.append(row)

        if i % 50 == 0:
            print(f"Procesadas {i} / {sample_size} muestras...")

    df = pd.DataFrame(resultados)
    if df.empty:
        print("No se obtuvieron muestras validas.")
        return

    df["mst_score"] = pd.Categorical(df["mst_score"], categories=list(range(1, 11)))
    conteo_tonos = df["mst_score"].value_counts().sort_index()

    plt.figure(figsize=(12, 6))
    colores_hex = [
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
    conteo_tonos.plot(kind="bar", color=colores_hex, edgecolor="black")
    plt.title("Distribucion de Tonos de Piel (Escala Monk) - FreiHAND")
    plt.xlabel("Escala MST")
    plt.ylabel("Numero de imagenes")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT)

    df.to_csv(OUTPUT_CSV, index=False)
    report_text = build_markdown_report(df, scale_all, green_limit, sample_size)
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as file:
        file.write(report_text)

    light_ratio = (df["mst_score"].isin([1, 2, 3, 4]).sum() / len(df)) * 100
    print(f"Auditoria completada con {len(df)} muestras validas.")
    print(f"Porcentaje MST 1-4: {light_ratio:.2f}%")
    print(f"CSV: {OUTPUT_CSV}")
    print(f"Grafico: {OUTPUT_PLOT}")
    print(f"Reporte: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()