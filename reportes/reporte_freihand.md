# Reporte FreiHAND

## Configuracion de auditoria
- Universo analizado: primeras 32560 imagenes de entrenamiento (fondo verde).
- Muestra aleatoria: 5000 imagenes.
- Punto usado para piel: landmark 9 (centro de palma).

## Hallazgo de sesgo de tono
- Proporcion MST 1-4 en la muestra: 5.50% (275/5000).
- Interpretacion: el dataset FreiHAND presenta concentracion en tonos claros.
- Accion recomendada: complementar con un dataset mas diverso en tono de piel (por ejemplo HAGRID o 11K Hands).

## Relevancia para escenarios de video
- FreiHAND incluye hand scale por muestra, util para robustez en video cuando la mano se acerca o se aleja de la camara.
- Rango hand scale observado: 0.0159 a 0.0364.
- Media hand scale: 0.0285.
- Esto permite calibrar mejor el tamano aparente de la mano y reducir perdidas de tracking por cambios de distancia en webcam.
