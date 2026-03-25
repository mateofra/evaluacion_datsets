import argparse
import json
import os
from collections import Counter

import cv2
import numpy as np
import pandas as pd

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

# Rangos aproximados sugeridos para analisis de diversidad.
RACE_TO_MST_RANGE = {
    "white": (1, 3),
    "black": (7, 10),
    "african": (7, 10),
    "indian": (4, 6),
    "latino": (4, 6),
    "hispanic": (4, 6),
    "asian": (2, 5),
    "middle eastern": (4, 7),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Integra HaGRID y FreiHAND en un formato unificado para auditoria de diversidad"
    )
    parser.add_argument(
        "--hagrid-annotations",
        default="datasets/hagrid_annotations/train",
        help="Carpeta con JSON de anotaciones de HaGRID por gesto.",
    )
    parser.add_argument(
        "--hagrid-images",
        default="datasets/hagrid_dataset",
        help="Carpeta raiz de imagenes de HaGRID (subcarpetas por gesto).",
    )
    parser.add_argument(
        "--gestures",
        nargs="+",
        default=["palm", "fist", "like", "ok"],
        help="Gestos a integrar.",
    )
    parser.add_argument(
        "--max-per-gesture",
        type=int,
        default=1000,
        help="Maximo de imagenes por gesto para no procesar todo el dataset.",
    )
    parser.add_argument(
        "--patch-radius",
        type=int,
        default=5,
        help="Radio del parche de color alrededor del landmark 9.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para muestreo aleatorio reproducible.",
    )
    parser.add_argument(
        "--freihand-csv",
        default="csv/auditoria_freihand_muestra.csv",
        help="CSV generado por el analisis de FreiHAND.",
    )
    parser.add_argument(
        "--freihand-max-samples",
        type=int,
        default=500,
        help="Limite de muestras de FreiHAND para comparativas.",
    )
    parser.add_argument(
        "--output-unified",
        default="csv/dataset_unificado_freihand_hagrid.csv",
        help="Archivo CSV de salida con el formato unificado.",
    )
    parser.add_argument(
        "--output-diversity",
        default="csv/reporte_diversidad_hagrid.csv",
        help="CSV con conteos de raza en HaGRID.",
    )
    return parser.parse_args()


def race_to_mst_range(race_label):
    if not race_label:
        return None
    race_key = race_label.strip().lower()
    for key, mst_range in RACE_TO_MST_RANGE.items():
        if key in race_key:
            return mst_range
    return None


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


def resolve_image_path(images_root, gesture, image_id):
    base = os.path.join(images_root, gesture, image_id)
    for extension in [".jpg", ".jpeg", ".png"]:
        candidate = base + extension
        if os.path.exists(candidate):
            return candidate
    return None


def extract_patch_color_from_landmark9(image_bgr, hand_landmarks, patch_radius):
    if image_bgr is None or not hand_landmarks:
        return None

    first_hand = hand_landmarks[0]
    if len(first_hand) <= 9:
        return None

    point9 = first_hand[9]
    if len(point9) < 2:
        return None

    h, w, _ = image_bgr.shape
    x = int(round(float(point9[0]) * w))
    y = int(round(float(point9[1]) * h))

    x0 = max(0, x - patch_radius)
    x1 = min(w, x + patch_radius)
    y0 = max(0, y - patch_radius)
    y1 = min(h, y + patch_radius)
    if x0 >= x1 or y0 >= y1:
        return None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    roi = image_rgb[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    return roi.mean(axis=(0, 1))


def load_hagrid_records(args):
    rows = []
    race_counter = Counter()
    rng = np.random.default_rng(args.seed)

    for gesture in args.gestures:
        json_path = os.path.join(args.hagrid_annotations, f"{gesture}.json")
        if not os.path.exists(json_path):
            print(f"Aviso: no existe anotacion para gesto '{gesture}': {json_path}")
            continue

        with open(json_path, "r", encoding="utf-8") as file:
            annotations = json.load(file)

        image_ids = list(annotations.keys())
        if not image_ids:
            continue

        available_ids = []
        for image_id in image_ids:
            if resolve_image_path(args.hagrid_images, gesture, image_id) is not None:
                available_ids.append(image_id)

        if not available_ids:
            print(
                f"Aviso: no hay imagenes locales para gesto '{gesture}' en "
                f"{os.path.join(args.hagrid_images, gesture)}"
            )
            continue

        sample_size = min(args.max_per_gesture, len(available_ids))
        sampled_ids = rng.choice(available_ids, size=sample_size, replace=False)

        for image_id in sampled_ids:
            info = annotations.get(image_id, {})
            image_path = resolve_image_path(args.hagrid_images, gesture, image_id)
            image_bgr = cv2.imread(image_path) if image_path else None

            hand_landmarks = info.get("hand_landmarks")
            mean_rgb = extract_patch_color_from_landmark9(
                image_bgr=image_bgr,
                hand_landmarks=hand_landmarks,
                patch_radius=args.patch_radius,
            )
            mst_color = None
            if mean_rgb is not None:
                mst_level = get_closest_mst(mean_rgb)
                if mst_level is not None:
                    mst_color = int(mst_level)

            meta = info.get("meta", {})
            race = (meta.get("race") or [None])[0]
            gender = (meta.get("gender") or [None])[0]
            age = (meta.get("age") or [None])[0]
            race_counter[str(race)] += 1 if race is not None else 0

            mst_range = race_to_mst_range(race)
            mst_from_race = None
            mst_range_label = None
            if mst_range is not None:
                mst_from_race = int(round((mst_range[0] + mst_range[1]) / 2))
                mst_range_label = f"{mst_range[0]}-{mst_range[1]}"

            row = {
                "source": "hagrid",
                "sample_id": str(image_id),
                "image_path": image_path,
                "gesture": gesture,
                "mst_score": mst_color,
                "mst_proxy_from_race": mst_from_race,
                "mst_proxy_range": mst_range_label,
                "race": race,
                "gender": gender,
                "age": age,
                "hand_scale": np.nan,
                "landmark_source": "mediapipe_2d",
            }
            rows.append(row)

    diversity_rows = []
    total_race = sum(race_counter.values())
    for race, count in race_counter.items():
        if race in ["None", "nan"]:
            continue
        pct = (count / total_race) * 100 if total_race > 0 else 0.0
        diversity_rows.append({"race": race, "count": count, "percentage": round(pct, 2)})

    diversity_df = pd.DataFrame(diversity_rows)
    if not diversity_df.empty:
        diversity_df = diversity_df.sort_values(by="count", ascending=False)
    return pd.DataFrame(rows), diversity_df


def load_freihand_records(path_csv, max_samples=None, seed=42):
    if not os.path.exists(path_csv):
        print(f"Aviso: no existe CSV de FreiHAND: {path_csv}")
        return pd.DataFrame()

    df = pd.read_csv(path_csv)
    if df.empty:
        return pd.DataFrame()

    if "idx" not in df.columns or "archivo" not in df.columns:
        raise ValueError("El CSV de FreiHAND no tiene columnas esperadas: idx, archivo")

    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("--freihand-max-samples debe ser mayor que 0")
        if len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=seed)

    out = pd.DataFrame(
        {
            "source": "freihand",
            "sample_id": df["idx"].astype(str),
            "image_path": "datasets/FreiHAND_pub_v2/training/rgb/" + df["archivo"].astype(str),
            "gesture": None,
            "mst_score": df.get("mst_score"),
            "mst_proxy_from_race": np.nan,
            "mst_proxy_range": None,
            "race": None,
            "gender": None,
            "age": None,
            "hand_scale": df.get("hand_scale"),
            "landmark_source": "freihand_3d_projected",
        }
    )
    return out


def main():
    args = parse_args()

    hagrid_df, diversity_df = load_hagrid_records(args)
    freihand_df = load_freihand_records(
        args.freihand_csv,
        max_samples=args.freihand_max_samples,
        seed=args.seed,
    )

    frames = [df for df in [hagrid_df, freihand_df] if not df.empty]
    if not frames:
        raise RuntimeError("No se cargaron datos de HaGRID ni de FreiHAND.")

    unified_df = pd.concat(frames, ignore_index=True)
    unified_df.to_csv(args.output_unified, index=False)
    print(f"Dataset unificado guardado en: {args.output_unified}")
    print(f"Filas totales: {len(unified_df)}")

    if not diversity_df.empty:
        diversity_df.to_csv(args.output_diversity, index=False)
        print(f"Reporte de diversidad HaGRID guardado en: {args.output_diversity}")
        print(diversity_df.to_string(index=False))
    else:
        print("No se pudo generar reporte de diversidad de HaGRID (sin metadata de raza).")


if __name__ == "__main__":
    main()
