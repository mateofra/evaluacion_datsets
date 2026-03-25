# Reporte de Auditoria Extendida

Este reporte cubre 5 dimensiones adicionales de sesgo y robustez.

## Datasets usados en esta auditoria
- Modo de uso: solo_freihand.
- Total de muestras auditadas: 20000.

- freihand: 20000 muestras (100.0%).

## 2) Accesorios y Oclusiones Artificiales (Proxy)
- No hay etiquetas directas de joyas/unas/mangas en el dataset unificado actual.
- Proxy de oclusion por multiples cajas (num_bboxes > 1): 0 muestras (0%).
- Escenas con 2 o mas manos detectadas: 0 (0%).

## 3) Condiciones de Iluminacion
- Imagenes analizadas: 3000.
- Brillo medio: 118.405.
- Baja iluminacion: 1.1%.
- Sombras fuertes: 7.27%.
- Posible contraluz (proxy): 0.07%.
- Temperatura de color: fria=0.13%, calida=94.7%.

## 4) Complejidad de Fondo (Proxy)
- Densidad de bordes media: 0.0328.
- Escenas con fondo complejo (edge density alta): 0.0%.
- Varianza de Laplaciano media: 309.95.

## Notas
- Varias metricas son proxys y no reemplazan etiquetado manual especializado.
- Para accesorios (joyas, unas, mangas) se recomienda una ronda de etiquetado adicional.
- Graficos exportados en: graficos/auditoria_extendida_20000
