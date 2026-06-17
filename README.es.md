<!--
  README del repo de GitHub para Interview Research Agent — versión en español
  Repo: github.com/JorgePulgar/interview-preparation-agent
  Coloca este archivo como README.es.md en la raíz del repo.
-->

<p align="right"><sub><a href="./README.md">English</a> · <b>Español</b></sub></p>

<h1 align="center">Interview Research Agent</h1>

<p align="center">
  <b>Un agente con LangGraph que investiga una empresa antes de una entrevista y redacta un briefing en Markdown.</b><br>
  Human-in-the-loop en cada puerta · dos niveles de búsqueda web en paralelo · inspección de ejecución que demuestra que el grafo corre como se diseñó.<br>
  <sub>Python · LangGraph · Azure OpenAI (Azure AI Foundry) · Tavily</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/LangGraph-0.2+-1C3C3C" alt="LangGraph">
  <img src="https://img.shields.io/badge/Azure_OpenAI-Foundry-0078D4?logo=microsoftazure&logoColor=white" alt="Azure OpenAI">
  <img src="https://img.shields.io/badge/Tavily-web_search-FF6154" alt="Tavily">
</p>

---

> 🇬🇧 *Also available in English: [README.md](README.md)*

## TL;DR

- **Qué hace**: le das el nombre de una empresa e investiga la empresa + su stack tecnológico real, redacta 3 preguntas concretas para la entrevista, monta un briefing en Markdown y lo guarda en `briefings/`.
- **Human-in-the-loop**: el grafo se pausa (`interrupt()`) dos veces — una en las preguntas, otra en el briefing. Apruebas con `ok`, pides un retoque de redacción, o pides que **vuelva a buscar en la web**.
- **Dos niveles de paralelismo**: `research` y `tech_stack` corren como sub-grafos en paralelo; dentro de cada uno, 3 búsquedas web salen a la vez con el patrón map-reduce de `Send`. 6 trabajadores en vuelo.
- **Generar y revisar son nodos separados** — una decisión deliberada para que reanudar tras una pausa no vuelva a llamar al LLM y cambie en silencio lo que ya aprobaste.
- **Guarda contra alucinaciones en el stack**: el prompt le dice al modelo que indique qué secciones están vacías en vez de inventarse un stack que no encontró.
- **`inspect_run.py`** ejecuta el grafo con `subgraphs=True` para que puedas comprobar que los sub-grafos y el fan-out en paralelo ocurrieron de verdad — una ejecución normal se ve idéntica a una secuencial, así que "salió un briefing" no prueba nada.

## Resumen del proyecto

Es un agente construido con **LangGraph**. Un agente, aquí, no es magia: es un grafo de pasos (nodos) unidos por flechas (edges), con un estado compartido (un diccionario) que viaja entre ellos. Algunos nodos llaman a un LLM (Azure OpenAI) o a un buscador web (Tavily); otros simplemente **pausan** para que una persona revise el resultado y diga `ok` o pida cambios.

Lo interesante no es el briefing — es el cableado. El grafo compone tres **sub-grafos** compilados como si fueran nodos normales, corre dos de ellos en paralelo, despliega cada uno en búsquedas web paralelas con map-reduce y mantiene a la persona al mando en dos puertas de interrupción. El diseño lidia con un filo afilado de LangGraph: cuando un grafo se reanuda tras `interrupt()`, el nodo pausado se re-ejecuta desde el principio — así que cualquier llamada al LLM colocada antes de la pausa regeneraría contenido en cada reanudación. La solución es estructural, no un parche: generación y revisión viven en nodos distintos.

---

## Tabla de contenidos

- [Qué hace](#qué-hace)
- [Características clave](#características-clave)
- [Arquitectura](#arquitectura)
- [Decisiones de diseño clave](#decisiones-de-diseño-clave)
- [Comprobar que los sub-grafos y el map-reduce corren de verdad](#comprobar-que-los-sub-grafos-y-el-map-reduce-corren-de-verdad)
- [Stack tecnológico](#stack-tecnológico)
- [Requisitos](#requisitos)
- [Instalación local](#instalación-local)
- [Configuración (variables de entorno)](#configuración-variables-de-entorno)
- [Uso](#uso)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Limitaciones conocidas](#limitaciones-conocidas)

---

## Qué hace

A partir del nombre de una empresa:

1. **Investiga en paralelo** qué hace la empresa + sus noticias recientes (sub-grafo `research`) y su **stack tecnológico** real (sub-grafo `tech_stack`).
2. **Genera 3 preguntas concretas** para la entrevista — nada genérico — y se **pausa** para que las revises.
3. **Monta el briefing final** en Markdown y se **pausa** otra vez para tu visto bueno.
4. Al aprobar, **guarda el `.md`** en `briefings/briefing-<empresa>-<timestamp>.md`.

En cada pausa puedes:

- escribir `ok` para aprobar;
- pedir un **retoque de redacción** (p.ej. *"cambia la pregunta 2 por una de IA"*) — solo cambia eso, el resto se mantiene idéntico, palabra por palabra;
- pedir **datos nuevos** (p.ej. *"vuelve a buscar sus frameworks"*) — el agente **busca de verdad** otra vez y reconstruye con la info actualizada.

La diferencia entre "retoque de redacción" y "datos nuevos" la decide un clasificador LLM, así que *"acorta la sección de noticias"* nunca dispara una búsqueda, mientras que *"vuelve a revisar su stack"* sí.

[↑ Volver arriba](#tabla-de-contenidos)

---

## Características clave

- **Human-in-the-loop en dos puertas**: `interrupt()` en las preguntas y en el briefing. La ejecución se detiene, te muestra el artefacto y espera `ok` / edición / re-búsqueda.
- **Paralelismo en dos niveles**: (A) `research` ‖ `tech_stack` como hermanos desde `START`; (B) dentro de cada uno, 3 trabajadores `search_one` a la vez. `briefing` corre más tarde, él solo, sin paralelismo interno.
- **Búsqueda map-reduce con `Send`**: una función de fan-out devuelve un `Send` por query; los trabajadores concatenan en un canal `search_results` con un reducer `Annotated[list[str], operator.add]`; un único nodo `synthesize` resume cuando todos terminan.
- **Feedback enrutado por intención**: un clasificador LLM ordena cada rechazo en `edit`, `research` o `tech_stack`, y el grafo enruta en consecuencia — reescritura barata vs. re-búsqueda real.
- **Guarda de frases genéricas en las preguntas**: una lista negra (`día a día`, `cultura`, `valores`, …) se comprueba tras generar; si aparece una frase prohibida, el nodo reintenta en silencio hasta 3 veces antes de mostrarte nada.
- **Extracción del stack a prueba de alucinaciones**: el prompt instruye al modelo a indicar qué secciones están vacías en vez de inventar tecnologías que no encontró.

[↑ Volver arriba](#tabla-de-contenidos)

---

## Arquitectura

![Grafo del agente](graph.png)

Hay un **grafo padre** (el agente principal). Tres de sus nodos no son funciones sueltas — son grafos más pequeños compilados (sub-grafos) que se hacen cargo de una fase entera:

| Componente | Tipo | Rol |
|---|---|---|
| `research` | sub-grafo (map-reduce) | Busca "qué hacen" + noticias recientes, en 3 queries paralelas, y luego resume. |
| `tech_stack` | sub-grafo (map-reduce) | Infiere el stack real a partir del blog de ingeniería + ofertas de empleo + StackShare, en 3 queries paralelas. |
| `generate_questions_node` | nodo (LLM) | Genera/edita las 3 preguntas. Sin `interrupt()` aquí, a propósito. |
| `review_questions_node` | nodo (interrupt) | Pausa para la revisión humana de las preguntas; clasifica el feedback. |
| `briefing` | sub-grafo (genera + revisa) | Redacta el briefing y se pausa para el visto bueno; el bucle de edición es interno. |
| `write_file_node` | nodo | Escribe el briefing aprobado en `briefings/`. |

**Flujo de ejecución.** `START` hace fan-out a `research` y `tech_stack` en paralelo. Ambos terminan, luego el grafo genera las preguntas y se pausa en `review_questions_node`. Con `ok` entra al sub-grafo `briefing`, que redacta y se pausa otra vez. Con `ok` ahí, `write_file_node` guarda el `.md` y el grafo llega a `END`. En cualquiera de las pausas, una petición de "datos nuevos" enruta de vuelta a los sub-grafos de búsqueda y luego retorna a la fase que lo pidió (lo registra `search_return`).

En una frase: *`research` y `tech_stack` investigan en paralelo (cada uno dispara sus búsquedas en paralelo); cuando apruebas las preguntas, el sub-grafo `briefing` redacta el documento por su cuenta y se pausa para tu visto bueno.*

[↑ Volver arriba](#tabla-de-contenidos)

---

## Decisiones de diseño clave

1. **Generar y revisar están en nodos separados.** En LangGraph, al reanudar tras `interrupt()`, el nodo pausado se re-ejecuta desde el principio. Si la llamada al LLM viviera antes de la pausa, regeneraría contenido en cada reanudación y lo que aprobaste no sería lo que se guarda. Por eso un nodo **genera** (llama al LLM) y otro **revisa** (solo pausa y lee el estado). Al reanudar, no se regenera nada.

2. **Feedback enrutado por intención.** Un pequeño clasificador (`_classify_feedback`, una llamada al LLM) decide si tu petición es `edit` (reescribir con los datos que ya hay — barato, sin buscar), `research` o `tech_stack` (volver a buscar). Los nodos de búsqueda guardan un `search_return` para saber a qué fase regresar después — las preguntas o el briefing.

3. **Los sub-grafos se comunican por nombre de clave del estado.** `research`, `tech_stack` y `briefing` son grafos compilados usados como nodos dentro del padre. Las claves que existen en **ambos** (padre e hijo) cruzan al entrar/salir; las que solo existen en el hijo (como `search_results`) quedan internas. Se compilan **sin checkpointer** para heredar el del padre — por eso el `interrupt()` que vive dentro del sub-grafo `briefing` burbujea hasta el `invoke()` de arriba igual que el de las preguntas.

4. **Map-reduce con `Send` para la búsqueda en paralelo.** Cada sub-grafo de búsqueda no lanza sus queries en fila:
   - **MAP** — una función de fan-out devuelve una lista de `Send(...)`, uno por query; LangGraph crea N copias del trabajador `search_one` y las ejecuta a la vez.
   - **REDUCE** — cada trabajador escribe en `search_results`, un canal con reducer `Annotated[list[str], operator.add]` que **concatena** en vez de pisar. Al hacer fan-in, un único nodo `synthesize` resume con el LLM.
   - Cada sub-grafo declara un `output_schema` (solo su clave de resultado) para que, al correr `research` y `tech_stack` en el mismo superstep, no choquen escribiendo claves compartidas (`company`, …) y disparen el `InvalidUpdateError` de LangGraph.

[↑ Volver arriba](#tabla-de-contenidos)

---

## Comprobar que los sub-grafos y el map-reduce corren de verdad

Una ejecución normal produce la misma salida que una versión secuencial, así que "salió un briefing" no prueba nada. `inspect_run.py` ejecuta el grafo con `stream(..., subgraphs=True)` e imprime cada nodo con su namespace:

```powershell
python inspect_run.py "Vercel"
```

Qué buscar en la salida:

- `search_one` **×3** bajo `('research:...')` y **×3** bajo `('tech_stack:...')` → el fan-out con `Send` ocurrió (6 trabajadores). Si además sus eventos aparecen **intercalados** entre los dos namespaces, es prueba de que corrieron en paralelo.
- `synthesize` **×1** por cada namespace → el fan-in del reduce.
- **Namespaces no vacíos** (`('briefing:...')`, etc.) frente a `ns=()` de los nodos del padre → los sub-grafos corrieron de verdad.

[↑ Volver arriba](#tabla-de-contenidos)

---

## Stack tecnológico

**Lenguaje y runtime**
- Python 3.10+

**Orquestación**
- LangGraph (`StateGraph`, sub-grafos, map-reduce con `Send`, `interrupt()` / `Command`, checkpointer `MemorySaver`)

**Modelo y búsqueda**
- Azure OpenAI / Azure AI Foundry vía `langchain-openai` (`AzureChatOpenAI`, p.ej. `gpt-4o`, `temperature=0.3`)
- Búsqueda web Tavily vía `langchain-tavily` (`TavilySearch`, `max_results=3`)

**Config**
- `python-dotenv` para cargar el `.env`

[↑ Volver arriba](#tabla-de-contenidos)

---

## Requisitos

- Python 3.10+
- Un recurso de **Azure OpenAI / Foundry** con un *deployment* de un modelo de chat (p.ej. `gpt-4o`)
- Una **API key de Tavily** (https://tavily.com)

---

## Instalación local

```powershell
# 1. Clonar el repo
git clone https://github.com/JorgePulgar/interview-preparation-agent.git
cd interview-preparation-agent

# 2. Crear y activar un entorno virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate          # macOS / Linux

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## Configuración (variables de entorno)

Copia la plantilla y rellena tus claves reales:

```powershell
copy .env.example .env               # Windows
# cp .env.example .env                # macOS / Linux
```

Edita `.env`:

```dotenv
# --- Azure OpenAI (Azure AI Foundry) ---
AZURE_OPENAI_ENDPOINT=https://<tu-recurso>.openai.azure.com/   # o .services.ai.azure.com
AZURE_OPENAI_API_KEY=...
OPENAI_API_VERSION=2024-10-21        # versión de la API, NO la del modelo
AZURE_OPENAI_DEPLOYMENT=gpt-4o       # el NOMBRE DEL DEPLOYMENT, no del modelo
```

```dotenv
# --- Tavily (búsqueda web) ---
TAVILY_API_KEY=tvly-...
```

> ⚠️ Dos errores típicos:
> - `OPENAI_API_VERSION` es la versión de la **API** (`2024-10-21`), no la del modelo (`2024-11-20`). Usa la que muestre tu *deployment*.
> - `AZURE_OPENAI_DEPLOYMENT` es el **nombre que le diste al deployment** en Foundry, que puede no coincidir con el nombre del modelo.
>
> El archivo `.env` está en `.gitignore`: **nunca subas tus claves al repo.**

---

## Uso

```powershell
python interview_agent.py "Stripe"
```

Si no pasas nombre, usa `Stripe` por defecto. El agente se pausa por consola; responde `ok` o escribe qué cambiar. Al terminar verás la ruta del `.md` generado en `briefings/`.

---

## Estructura del proyecto

```
.
├── interview_agent.py   # el agente (grafo padre + sub-grafos + map-reduce) — comentado en español
├── inspect_run.py       # ejecuta con stream(subgraphs=True) para ver sub-grafos y map-reduce
├── requirements.txt     # dependencias
├── .env.example         # plantilla de variables de entorno
├── graph.drawio         # diagrama del grafo (editable en draw.io)
├── graph.png            # imagen del diagrama (exportada de graph.drawio)
└── briefings/           # salida: los .md generados (ignorada por git)
```

---

## Limitaciones conocidas

- **El clasificador de feedback usa el LLM**, así que es fiable pero no perfecto. Si una petición ambigua se interpreta mal, reformúlala (p.ej. *"vuelve a buscar..."* para forzar una re-búsqueda).
- **`MemorySaver` guarda el estado en memoria** — se pierde al cerrar el programa. Para algo persistente se usaría un checkpointer con base de datos.
- **Los comentarios del código están en español.** El fuente está muy comentado como artefacto didáctico para un equipo hispanohablante; el README es bilingüe pero los comentarios en línea no.
- **Aún no hay suite de tests automáticos.** La corrección del cableado del grafo se verifica a mano con `inspect_run.py` en vez de con `pytest`.

[↑ Volver arriba](#tabla-de-contenidos)
