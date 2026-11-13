"""Questions that force web research plus a judgment, and one search that should miss.

Each demo question starts from a fresh state.
"""

DEMO_QUESTIONS: list[tuple[str, str]] = [
    (
        "sentiment",
        "Busca cobertura reciente de los últimos resultados trimestrales de NVIDIA y puntúa el sentimiento general de esas fuentes.",
    ),
    (
        "percentage difference",
        "Busca los precios actuales de las acciones de NVIDIA (NVDA) y AMD, y calcula la diferencia porcentual del precio de NVIDIA respecto al de AMD.",
    ),
    (
        "missing sources",
        "¿Cuál fue el EBITDA exacto no publicado de la empresa ficticia Nimbus Lunar Mining en el último trimestre? Si no hay fuentes, no inventes el número y explica qué falta.",
    ),
]
