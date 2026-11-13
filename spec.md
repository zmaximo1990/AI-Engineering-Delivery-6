# Pre-entrega 6: Orquestador multi-agente especializado

## ¿Qué debes construir?

Debes entregar un prototipo funcional de un Orquestador Multi-Agente de Análisis e Investigación. El sistema debe ser capaz de procesar una solicitud de usuario que requiera al menos dos dominios de especialización distintos y una fase de síntesis final.

## Requerimientos Técnicos:
Topología Jerárquica: Un nodo Supervisor que actúa como router inteligente y controlador de flujo.
Mínimo de 2 Agentes Especialistas:
Agente de Búsqueda/Investigación: Con herramientas para consultar fuentes externas (puedes usar Tavily o una búsqueda simulada sobre tu Vector DB de pre-entregas anteriores).
Agente de Análisis/Cómputo: Especializado en procesar los datos obtenidos (análisis de sentimiento, cálculos matemáticos o validación de esquemas).
Estado Compartido Estructurado: Un esquema de State en LangGraph que permita rastrear qué agente ha contribuido con qué información, evitando la pérdida de contexto en la comunicación asíncrona.
Flujo de Supervisión: El supervisor debe decidir si la tarea está completa o si algún especialista debe refinar su output antes de dar la respuesta final.
Pasos Sugeridos
Define tu Estado: Diseña una clase TypedDict que herede de MessagesState. Piensa si necesitas campos extra como next_agent o task_completed.
Crea los Agentes Especialistas: Define funciones que utilicen create_react_agent de LangGraph o prompts específicos para cada rol. Asegúrate de que cada uno tenga herramientas acotadas.
Implementa el Supervisor: Este es el cerebro. Su prompt debe ser claro: "Dada la conversación actual, ¿quién debe intervenir ahora o es momento de finalizar?". Debe mapear sus respuestas a los nombres de los nodos del grafo.
Construye el Grafo: Une los puntos. Usa add_node para cada agente y para el supervisor. Define las Conditional Edges que conecten al supervisor con los especialistas.
Prueba la Interacción: Lanza una consulta que obligue al supervisor a enviar la tarea al Investigador, recibir el dato, enviarlo al Analista y finalmente cerrar la conversación.
Errores Comunes a Evitar
El "Supervisor Infinito": No definir una condición de parada clara, haciendo que el supervisor y los agentes entren en un bucle de correcciones eternas. Tip: Implementa un contador de pasos o un criterio de "Suficiencia" estricto.
Contaminación de Contexto: Pasar todo el historial a todos los agentes en cada turno. Para sistemas escalables, a veces es mejor que el especialista solo reciba la instrucción específica y el contexto necesario, no toda la metadata del sistema.

## Entregable

Instrucciones de la Práctica
Estructura del Repositorio:

Crea un archivo state.py para definir el esquema de datos compartido.
Crea un directorio agents/ con los especialistas (ej. research_agent.py, analyst_agent.py).
Define el grafo principal en main.py o graph.py.
Implementación del Grafo:

Utiliza StateGraph de LangGraph.
Define al menos un nodo de "Supervisor" que decida dinámicamente el flujo usando Literal en el retorno para las aristas condicionales.
Herramientas (Tools):

Los agentes deben tener al menos una herramienta funcional (puedes usar TavilySearchResults o una función propia que consulte una base de datos).
Validación:

Implementa un nodo de Validation o asegura que el Supervisor tenga una rúbrica personalizada en su prompt para validar los resultados de los especialistas antes de dar el END.
Documentación:

Tu README debe incluir un diagrama del grafo (puedes generarlo con graph.get_graph().draw_mermaid_png()).
Explica brevemente por qué elegiste esa topología y cómo manejas los posibles conflictos entre agentes.