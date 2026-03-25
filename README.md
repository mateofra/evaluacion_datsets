# evaluacion_datsets

Proyecto para auditar diversidad de tono de piel en FreiHAND y unificarlo con HaGRID en un CSV común.

## Requisitos

- Python 3.13+
- uv (gestor de entornos y paquetes)
- Dependencias del proyecto (definidas en [pyproject.toml](pyproject.toml))

Instalación rápida con uv:

```powershell
uv sync
```

## Crear tu propio directorio `datasets/`

Los scripts esperan que los datos estén dentro de [datasets/](datasets/) en la raíz del proyecto.

En Windows (PowerShell):

```powershell
mkdir datasets\FreiHAND_pub_v2\training\rgb
mkdir datasets\hagrid_annotations\train
mkdir datasets\hagrid_dataset
```

`mkdir` en Windows crea directorios intermedios, por eso no hace falta un equivalente de `-p`.

En Linux/macOS:

```bash
mkdir -p datasets/FreiHAND_pub_v2/training/rgb
mkdir -p datasets/hagrid_annotations/train
mkdir -p datasets/hagrid_dataset
```

Estructura final esperada (mínima):

```text
datasets/
	FreiHAND_pub_v2/
		training_xyz.json
		training_K.json
		training_scale.json
		training/
			rgb/
	hagrid_annotations/
		train/
			palm.json
			fist.json
			like.json
			ok.json
	hagrid_dataset/
		palm/
		fist/
		like/
		ok/
```

## Descargar datasets

Si ahora tienes más almacenamiento, puedes trabajar con descarga grande (incluso completa) y luego procesar más muestras.

### Opción rápida para volumen grande (Kaggle CLI)

1. Instala Kaggle CLI:

```powershell
uv add kaggle
```

2. Configura tus credenciales de Kaggle en C:\Users\<tu_usuario>\.kaggle\kaggle.json.
3. Descarga HaGRID completo (zip):

```powershell
uv run kaggle datasets download -d kapitanov/hagrid -p datasets --unzip
```

4. Reubica el contenido para que coincida con las rutas del proyecto:
	- JSON de anotaciones en [datasets/hagrid_annotations/train/](datasets/hagrid_annotations/train/)
	- Imágenes por gesto en [datasets/hagrid_dataset/](datasets/hagrid_dataset/)

### 1) FreiHAND

1. Ir a la página oficial: https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html
2. Descargar el dataset de entrenamiento (`FreiHAND_pub_v2`).
3. Descomprimir y copiar el contenido dentro de [datasets/FreiHAND_pub_v2/](datasets/FreiHAND_pub_v2/).

Verifica que existan estos archivos:

- [datasets/FreiHAND_pub_v2/training_xyz.json](datasets/FreiHAND_pub_v2/training_xyz.json)
- [datasets/FreiHAND_pub_v2/training_K.json](datasets/FreiHAND_pub_v2/training_K.json)
- [datasets/FreiHAND_pub_v2/training_scale.json](datasets/FreiHAND_pub_v2/training_scale.json)
- Imágenes en [datasets/FreiHAND_pub_v2/training/rgb](datasets/FreiHAND_pub_v2/training/rgb)

### 2) HaGRID

1. Ir a la fuente oficial (Kaggle): https://www.kaggle.com/datasets/kapitanov/hagrid
2. Descargar:
	 - Imágenes del dataset
	 - Anotaciones JSON por gesto
3. Copiar anotaciones en [datasets/hagrid_annotations/train/](datasets/hagrid_annotations/train/).
4. Copiar imágenes por gesto en [datasets/hagrid_dataset/](datasets/hagrid_dataset/), por ejemplo:
	 - [datasets/hagrid_dataset/palm/](datasets/hagrid_dataset/palm/)
	 - [datasets/hagrid_dataset/fist/](datasets/hagrid_dataset/fist/)
	 - [datasets/hagrid_dataset/like/](datasets/hagrid_dataset/like/)
	 - [datasets/hagrid_dataset/ok/](datasets/hagrid_dataset/ok/)

## Flujo recomendado

1. Ejecutar auditoría de FreiHAND para generar [csv/auditoria_freihand_muestra.csv](csv/auditoria_freihand_muestra.csv).

```powershell
uv run python src/freihands.py --num-samples 5000
```

Para una muestra más grande:

```powershell
uv run python src/freihands.py --num-samples 20000
```

2. Unificar FreiHAND + HaGRID y generar CSV final.

```powershell
uv run python src/unificar_hagrid_freihand.py \
	--hagrid-annotations datasets/hagrid_annotations/train \
	--hagrid-images datasets/hagrid_dataset \
	--gestures palm fist like ok \
	--max-per-gesture 1000 \
	--freihand-csv csv/auditoria_freihand_muestra.csv
```

Para una integración más grande:

```powershell
uv run python src/unificar_hagrid_freihand.py \
	--hagrid-annotations datasets/hagrid_annotations/train \
	--hagrid-images datasets/hagrid_dataset \
	--gestures palm fist like ok \
	--max-per-gesture 5000 \
	--freihand-csv csv/auditoria_freihand_muestra.csv \
	--freihand-max-samples 5000 \
	--output-unified csv/dataset_unificado_freihand5000_hagrid5000.csv \
	--output-diversity csv/reporte_diversidad_hagrid5000.csv
```

## Auditoria extendida (5 dimensiones)

Se incorporo una auditoria adicional con estas dimensiones:

1. Diversidad etaria y morfologica.
2. Accesorios y oclusiones artificiales (proxys).
3. Condiciones de iluminacion y contraluz (proxys).
4. Complejidad de fondo (proxys).
5. Sesgo de lateralidad (left vs right).

Ejecutar sobre un CSV unificado:

```powershell
uv run python src/auditoria_extendida.py \
	--input-unified csv/dataset_unificado_freihand5000_hagrid5000.csv \
	--output-metrics csv/auditoria_extendida_metricas_5000.csv \
	--output-image-features csv/auditoria_extendida_features_imagen_5000.csv \
	--output-plots-dir graficos/auditoria_extendida_5000 \
	--output-report reportes/reporte_auditoria_extendida_5000.md \
	--max-image-samples 3000
```

Salidas nuevas de la auditoria extendida:

- CSV de features por imagen muestreada.
- Graficos por dimension (edad, genero, lateralidad, brillo, contraste y complejidad de fondo).

## Subset estratificado de HaGRID (gesto + lateralidad)

Si quieres preparar un subset balanceado antes de unificar:

```powershell
uv run python src/preparar_subset_hagrid.py \
	--annotations-dir datasets/ann_train_val \
	--output-annotations-dir datasets/hagrid_annotations/train_subset \
	--output-manifest csv/hagrid_subset_manifest.csv \
	--gestures palm fist like ok \
	--max-per-gesture 5000 \
	--target-left-ratio 0.5
```

Opcional: copiar imagenes locales disponibles del subset.

```powershell
uv run python src/preparar_subset_hagrid.py \
	--annotations-dir datasets/ann_train_val \
	--output-annotations-dir datasets/hagrid_annotations/train_subset \
	--output-manifest csv/hagrid_subset_manifest.csv \
	--gestures palm fist like ok \
	--max-per-gesture 5000 \
	--target-left-ratio 0.5 \
	--copy-images \
	--images-root datasets/hagrid_dataset \
	--output-images-root datasets/hagrid_dataset_subset
```

Salida principal:

- [csv/dataset_unificado_freihand_hagrid.csv](csv/dataset_unificado_freihand_hagrid.csv)
- [csv/reporte_diversidad_hagrid.csv](csv/reporte_diversidad_hagrid.csv)

## Problemas comunes

- Error `No existe la ruta requerida`: revisa que FreiHAND esté exactamente en [datasets/FreiHAND_pub_v2/](datasets/FreiHAND_pub_v2/).
- Aviso `no existe anotacion para gesto`: falta el JSON de ese gesto en [datasets/hagrid_annotations/train/](datasets/hagrid_annotations/train/).
- Aviso `no hay imagenes locales para gesto`: falta la carpeta o imágenes del gesto en [datasets/hagrid_dataset/](datasets/hagrid_dataset/).
