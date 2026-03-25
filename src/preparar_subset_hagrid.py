import argparse
import json
import os
import shutil
from collections import Counter

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Genera un subset estratificado de HaGRID por gesto y lateralidad"
    )
    parser.add_argument(
        "--annotations-dir",
        default="datasets/ann_train_val",
        help="Directorio con JSON de anotaciones de HaGRID.",
    )
    parser.add_argument(
        "--output-annotations-dir",
        default="datasets/hagrid_annotations/train_subset",
        help="Directorio de salida para JSON subset.",
    )
    parser.add_argument(
        "--output-manifest",
        default="csv/hagrid_subset_manifest.csv",
        help="CSV de salida con ids seleccionados.",
    )
    parser.add_argument(
        "--gestures",
        nargs="+",
        default=["palm", "fist", "like", "ok"],
        help="Gestos a incluir en el subset.",
    )
    parser.add_argument(
        "--max-per-gesture",
        type=int,
        default=5000,
        help="Maximo de ejemplos por gesto.",
    )
    parser.add_argument(
        "--target-left-ratio",
        type=float,
        default=0.5,
        help="Fraccion objetivo para mano izquierda dentro del subset.",
    )
    parser.add_argument(
        "--images-root",
        default="datasets/hagrid_dataset",
        help="Raiz de imagenes de HaGRID por gesto.",
    )
    parser.add_argument(
        "--output-images-root",
        default="datasets/hagrid_dataset_subset",
        help="Destino para copiar imagenes del subset (si --copy-images).",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Si se habilita, copia imagenes locales al subset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para muestreo reproducible.",
    )
    return parser.parse_args()


def normalize_hand(v):
    if v is None:
        return "unknown"
    text = str(v).strip().lower()
    if text in {"left", "right"}:
        return text
    return "unknown"


def resolve_image_path(images_root, gesture, image_id):
    base = os.path.join(images_root, gesture, image_id)
    for ext in [".jpg", ".jpeg", ".png"]:
        candidate = base + ext
        if os.path.exists(candidate):
            return candidate
    return None


def stratified_sample(records, max_samples, target_left_ratio, rng):
    left = []
    right = []
    unknown = []

    for image_id, info in records:
        hand = normalize_hand(info.get("leading_hand"))
        if hand == "left":
            left.append((image_id, info))
        elif hand == "right":
            right.append((image_id, info))
        else:
            unknown.append((image_id, info))

    rng.shuffle(left)
    rng.shuffle(right)
    rng.shuffle(unknown)

    selected = []
    target_left = int(round(max_samples * target_left_ratio))

    take_left = min(target_left, len(left))
    selected.extend(left[:take_left])

    remaining = max_samples - len(selected)
    take_right = min(remaining, len(right))
    selected.extend(right[:take_right])

    remaining = max_samples - len(selected)
    if remaining > 0:
        selected.extend(unknown[:remaining])

    if len(selected) < max_samples:
        used_ids = {img_id for img_id, _ in selected}
        fallback = left[take_left:] + right[take_right:] + unknown[remaining:]
        rng.shuffle(fallback)
        for img_id, info in fallback:
            if img_id in used_ids:
                continue
            selected.append((img_id, info))
            if len(selected) >= max_samples:
                break

    return selected[:max_samples]


def main():
    args = parse_args()
    if args.max_per_gesture <= 0:
        raise ValueError("--max-per-gesture debe ser mayor que 0")
    if not (0.0 <= args.target_left_ratio <= 1.0):
        raise ValueError("--target-left-ratio debe estar entre 0 y 1")

    os.makedirs(args.output_annotations_dir, exist_ok=True)
    manifest_dir = os.path.dirname(args.output_manifest)
    if manifest_dir:
        os.makedirs(manifest_dir, exist_ok=True)
    if args.copy_images:
        os.makedirs(args.output_images_root, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    manifest_rows = []
    summary = []

    for gesture in args.gestures:
        json_path = os.path.join(args.annotations_dir, f"{gesture}.json")
        if not os.path.exists(json_path):
            print(f"Aviso: no existe anotacion para gesto '{gesture}': {json_path}")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)

        records = list(annotations.items())
        if not records:
            print(f"Aviso: sin registros en {json_path}")
            continue

        sample_size = min(args.max_per_gesture, len(records))
        selected = stratified_sample(
            records=records,
            max_samples=sample_size,
            target_left_ratio=args.target_left_ratio,
            rng=rng,
        )

        selected_dict = {image_id: info for image_id, info in selected}
        out_json = os.path.join(args.output_annotations_dir, f"{gesture}.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(selected_dict, f, ensure_ascii=False)

        counter = Counter()
        copied = 0

        for image_id, info in selected:
            hand = normalize_hand(info.get("leading_hand"))
            counter[hand] += 1
            image_path = resolve_image_path(args.images_root, gesture, image_id)
            has_local_image = image_path is not None

            if args.copy_images and has_local_image:
                dst_dir = os.path.join(args.output_images_root, gesture)
                os.makedirs(dst_dir, exist_ok=True)
                dst_file = os.path.join(dst_dir, os.path.basename(image_path))
                shutil.copy2(image_path, dst_file)
                copied += 1

            manifest_rows.append(
                {
                    "gesture": gesture,
                    "image_id": image_id,
                    "leading_hand": hand,
                    "has_local_image": has_local_image,
                    "image_path": image_path,
                }
            )

        summary.append(
            {
                "gesture": gesture,
                "selected": len(selected),
                "left": counter.get("left", 0),
                "right": counter.get("right", 0),
                "unknown": counter.get("unknown", 0),
                "copied_images": copied,
            }
        )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(args.output_manifest, index=False)

    if summary:
        summary_df = pd.DataFrame(summary)
        print(summary_df.to_string(index=False))

    print(f"Subset annotations en: {args.output_annotations_dir}")
    print(f"Manifest en: {args.output_manifest}")


if __name__ == "__main__":
    main()
