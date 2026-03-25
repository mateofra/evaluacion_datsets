# Consideraciones para abordar los cambios con los datasets actuales

## Contexto actual del proyecto

Con los datos disponibles hoy en el repositorio:

- FreiHAND domina el volumen utilizable de imagen.
- HaGRID aporta menos ejemplos efectivos en la comparativa actual.
- d
Esto justifica aplicar medidas activas de balance para evitar sesgo del modelo hacia el bloque medio.

## Acciones Proactivas (Fase 1)

### 1) Oversampling de extremos

Propuesta:

"Durante el entrenamiento, clonare las imagenes de los niveles MST 1, 2, 3 y 10 para que el modelo las vea con mas frecuencia y compense su baja cantidad".

Como implementarlo con lo actual:

- Construir un sampler ponderado por clase MST.
- Asignar mayor peso de muestreo a 1, 2, 3 y 10.
- Mantener validacion/test sin oversampling para no inflar metricas.

Sugerencia tecnica:

- Objetivo minimo: que cada extremo alcance una frecuencia efectiva cercana a niveles medios en entrenamiento.
- Registrar en el experimento el factor de repeticion por cada MST extremo.

### 2) Aumento de datos cromatico (Color Jittering)

Propuesta:

"Dado que el MST 10 es escaso, aplicare transformaciones de brillo y contraste a imagenes de MST 8 y 9 para simular artificialmente el nivel 10".

Como implementarlo con lo actual:

- Aplicar augmentations solo en entrenamiento.
- Partir de muestras MST 8 y 9 y generar variantes con:
  - menor brillo
  - menor saturacion
  - ajuste de contraste y gamma
- Limitar la intensidad para no degradar textura de mano ni introducir artefactos irreales.

Sugerencia tecnica:

- Guardar trazabilidad de augmentations (parametros por imagen).
- Revisar visualmente una muestra de control para validar realismo.

### 3) Hibridacion balanceada FreiHAND + HaGRID

Propuesta:

"No usare todo FreiHAND y todo HaGRID, sino que creare un set de entrenamiento balanceado donde cada bloque (Claro, Medio, Oscuro) tenga un peso similar, evitando que el bloque Medio (5-7) aplaste el aprendizaje de los demas".

Definicion de bloques:

- Claro: MST 1-4
- Medio: MST 5-7
- Oscuro: MST 8-10

Como implementarlo con lo actual:

- Crear un dataset de entrenamiento por cuotas de bloque, no por volumen bruto.
- Balancear por bloque y, dentro de cada bloque, mezclar fuentes (FreiHAND/HaGRID) en proporcion controlada.
- Si un bloque queda corto, compensar con oversampling + augmentations antes de aumentar bloque medio.

Sugerencia tecnica:

- Fijar una meta de proporcion por bloque (por ejemplo, 33/33/33 o una variante justificada).
- Reportar en cada corrida la composicion final por bloque y por fuente.

## Como poner esto en el informe

Puedes incluir una seccion llamada:

## Plan de mitigacion de sesgo (Fase 1)

Texto sugerido para pegar casi literal:

"A partir del analisis de distribucion MST, se identifico una concentracion de muestras en el bloque medio (5-7) y escasez en extremos (1-3 y 10). Para reducir sesgo de aprendizaje, se implementaran tres acciones: (1) oversampling de extremos MST 1, 2, 3 y 10 en entrenamiento; (2) aumento cromatico controlado sobre MST 8-9 para reforzar el extremo oscuro y aproximar casos MST 10; y (3) construccion de un set hibrido balanceado por bloques Claro (1-4), Medio (5-7) y Oscuro (8-10), evitando que el volumen mayoritario del bloque medio domine la optimizacion. Estas medidas se aplicaran solo en entrenamiento y se evaluaran en validacion/test sin rebalance artificial, para medir impacto real en generalizacion y equidad entre tonos".

## Evidencia minima recomendada en el informe

Incluye estas tablas/figuras para cerrar con buena calidad tecnica:

- Distribucion MST antes de balance (conteo y porcentaje).
- Distribucion MST despues de balance (objetivo de entrenamiento).
- Matriz de acciones por bloque (oversampling, jitter, mezcla de fuentes).
- Comparativa de metricas del modelo por bloque MST (no solo metricas globales).

## Criterio de exito de la Fase 1

Se considera exitosa la fase si:

- Disminuye la brecha de desempeno entre bloques Claro/Medio/Oscuro.
- No cae significativamente la metrica global.
- Se mantiene coherencia visual de las imagenes aumentadas.
- La composicion del entrenamiento queda documentada y reproducible.
