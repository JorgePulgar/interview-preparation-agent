"""
Interview Research Agent — LangGraph + Human-in-the-Loop
=========================================================
Investiga una empresa antes de una entrevista y genera un briefing estructurado.
Al aprobar el briefing final, lo guarda en un archivo Markdown.

------------------------------------------------------------------------------
¿QUÉ ES ESTO? (lectura rápida para el equipo)
------------------------------------------------------------------------------
Es un "agente" construido con LangGraph. Un agente, aquí, no es magia: es un
GRAFO de pasos (nodos) conectados por flechas (edges). El estado (un diccionario
compartido) va pasando de un nodo a otro. Algunos nodos llaman a un modelo de
lenguaje (LLM) o a un buscador web (Tavily); otros simplemente PAUSAN para que
una persona revise el resultado y diga "ok" o pida cambios.

Flujo general (las flechas son los 'edges' del grafo):

  START
    ├─→ [research] ──────┐   (estos dos SUB-GRAFOS corren EN PARALELO)
    └─→ [tech_stack] ────┤
                         ▼
            generate_questions_node ──→ review_questions_node
                  ↑                         │  interrupt(): 'ok' / feedback
                  │   feedback de edición   │
                  └─────────────────────────┤
                  ┌──── feedback de datos ───┤   (re-buscar research/tech_stack
                  │                          │    y volver aquí)
                [research] / [tech_stack]  ◄─┘
                         │ ok
                         ▼
                    [briefing]  (SUB-GRAFO: genera + revisa con interrupt)
                  ↑      │
                  │      │ ok ─────────────→ write_file_node ──→ END
                  └──────┤
                  ┌──── feedback de datos ───┤   (re-buscar y volver al briefing)
                [research] / [tech_stack]  ◄─┘

------------------------------------------------------------------------------
CUATRO IDEAS CLAVE DEL DISEÑO (importante entenderlas)
------------------------------------------------------------------------------
  1) Generación y revisión SEPARADAS en nodos distintos.
     En LangGraph, cuando el grafo se reanuda tras un interrupt() (la pausa),
     el nodo que estaba pausado SE RE-EJECUTA DESDE EL PRINCIPIO. Si la llamada
     al LLM estuviera ANTES del interrupt(), se volvería a generar contenido
     nuevo en cada reanudación, y lo que aprobaste no sería lo que se guarda.
     Por eso: un nodo GENERA (llama al LLM) y otro nodo REVISA (solo pausa y
     lee el estado). Así, al reanudar, no se regenera nada.

  2) Enrutado por INTENCIÓN del feedback.
     Cuando la persona rechaza algo, un pequeño clasificador (otra llamada al
     LLM) decide qué tipo de petición es:
       - "edit"       → reescribir con los datos que ya tenemos (barato, sin buscar),
       - "research"   → volver a BUSCAR info general/noticias y regenerar,
       - "tech_stack" → volver a BUSCAR el stack/frameworks y regenerar.
     Los nodos de búsqueda guardan en el estado un 'search_return' para saber a
     qué fase regresar después de buscar (a las preguntas o al briefing).

  3) SUB-GRAFOS (subgraphs).
     Un sub-grafo es un grafo completo que se USA COMO SI FUERA UN NODO dentro
     de otro grafo. Aquí 'research', 'tech_stack' y 'briefing' son sub-grafos
     compilados. El grafo padre los invoca como nodos normales; por dentro, cada
     uno tiene sus propios nodos y flechas. Ventaja: encapsula una fase entera
     (sus pasos internos no ensucian el grafo principal) y se puede razonar/
     probar por separado.
     · Comunicación padre↔hijo: por NOMBRE DE CLAVE. Las claves del estado que
       existen en AMBOS (p.ej. 'company', 'research', 'briefing') se pasan al
       entrar y se devuelven al salir. Las claves que SOLO existen en el hijo
       (p.ej. 'search_results') son internas y no se ven desde fuera.
     · Checkpointer e interrupt(): el sub-grafo se compila SIN checkpointer; hereda
       el del padre. Por eso un interrupt() dentro del sub-grafo 'briefing' burbujea
       hasta el invoke() de arriba y se puede reanudar con Command(resume=...).

  4) MAP-REDUCE con Send (búsqueda en paralelo dentro de cada sub-grafo).
     Los sub-grafos de búsqueda no lanzan sus queries una tras otra. Usan el
     patrón map-reduce de LangGraph:
       - MAP   : una función de "fan-out" devuelve una lista de Send(...), uno por
                 query. LangGraph crea N copias del nodo trabajador 'search_one'
                 y las ejecuta EN PARALELO, cada una con su query.
       - REDUCE: cada trabajador escribe su resultado en un canal de tipo lista con
                 un REDUCER (Annotated[list, operator.add]) que CONCATENA en vez de
                 sobrescribir. Cuando todos terminan (fan-in), un único nodo
                 'synthesize' lee la lista completa y la resume con el LLM.

------------------------------------------------------------------------------
REQUISITOS
------------------------------------------------------------------------------
  pip install -r requirements.txt

Variables de entorno necesarias (Azure OpenAI / Foundry + Tavily), normalmente
en un archivo .env:
  AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, OPENAI_API_VERSION,
  AZURE_OPENAI_DEPLOYMENT, TAVILY_API_KEY
"""

# --- Librerías estándar de Python ---------------------------------------------
import os         # acceso a variables de entorno y rutas de archivos
import re         # expresiones regulares (para limpiar el nombre del archivo)
import operator   # operator.add: lo usamos como REDUCER para concatenar listas
import datetime   # fecha/hora (para el nombre del archivo y la cabecera del .md)

# 'TypedDict' nos deja describir la forma del estado (qué claves tiene y de qué
# tipo). 'Optional[str]' significa "puede ser un str o None". 'Annotated' nos deja
# "etiquetar" un tipo con metadatos; LangGraph usa esa etiqueta para saber qué
# REDUCER aplicar cuando varios nodos escriben en la misma clave a la vez.
from typing import TypedDict, Optional, Annotated

# python-dotenv: carga el archivo .env y vuelca sus valores en las variables de
# entorno del proceso (os.environ).
from dotenv import load_dotenv

# IMPORTANTE: llamamos a load_dotenv() ANTES de importar/crear los clientes de
# Azure y Tavily, porque esos clientes leen las claves de las variables de
# entorno en el momento en que se crean.
load_dotenv()

# --- Librerías de LangChain / LangGraph ---------------------------------------
# AzureChatOpenAI: cliente del modelo de chat alojado en Azure OpenAI / Foundry.
from langchain_openai import AzureChatOpenAI
# TavilySearch: herramienta de búsqueda web (devuelve resultados de internet).
from langchain_tavily import TavilySearch
# StateGraph: el constructor del grafo. START y END son los nodos especiales de
# inicio y fin.
from langgraph.graph import StateGraph, START, END
# MemorySaver: guarda el estado del grafo en memoria entre pausas (necesario para
# que interrupt() pueda pausar y luego reanudar en el mismo punto).
from langgraph.checkpoint.memory import MemorySaver
# interrupt: pausa el grafo y devuelve el control a quien lo invocó.
# Command: objeto que usamos para REANUDAR el grafo pasándole la respuesta humana.
# Send: el "sobre" del map-reduce. Send("nodo", payload) le dice a LangGraph
#       "crea una copia del nodo 'nodo' y ejecútala con este payload como estado".
from langgraph.types import interrupt, Command, Send


# ---------------------------------------------------------------------------
# 1. Estado del grafo
# ---------------------------------------------------------------------------
# El "estado" es un diccionario que viaja por todo el grafo. Cada nodo recibe el
# estado actual y devuelve un diccionario con las claves que quiere ACTUALIZAR
# (no hace falta devolver el estado entero, solo lo que cambia).
#
# Definimos su forma con TypedDict para que quede claro qué guarda cada campo.

class InterviewState(TypedDict):
    company: str                        # nombre de la empresa (lo da el usuario al arrancar)
    research: str                       # texto resumen que produce el sub-grafo 'research'
    tech_stack: str                     # texto del stack que produce el sub-grafo 'tech_stack'

    smart_questions: Optional[str]      # las 3 preguntas (candidatas o ya aprobadas)
    previous_questions: Optional[str]   # preguntas que se le mostraron al humano; base para editar
    questions_feedback: Optional[str]   # el cambio concreto que pidió el humano sobre las preguntas

    briefing: Optional[str]             # el briefing en Markdown (candidato o aprobado)
    previous_briefing: Optional[str]    # briefing que se mostró al humano; base para editar
    briefing_feedback: Optional[str]    # el cambio concreto que pidió el humano sobre el briefing

    # --- Campos de control del enrutado dinámico ---
    route: Optional[str]                # decisión tomada en la revisión: a qué nodo ir después
    search_return: Optional[str]        # tras (re)buscar, a qué fase volver (preguntas o briefing)
    search_feedback: Optional[str]      # texto para afinar la query cuando re-buscamos

    briefing_path: Optional[str]        # ruta del archivo .md generado al final


# ---------------------------------------------------------------------------
# 2. Dependencias (clientes y constantes que usan todos los nodos)
# ---------------------------------------------------------------------------

# Cliente del LLM en Azure OpenAI (Azure AI Foundry).
# AzureChatOpenAI lee automáticamente estas variables de entorno:
#   AZURE_OPENAI_ENDPOINT   -> el endpoint del recurso (la URL)
#   AZURE_OPENAI_API_KEY    -> la clave del recurso
#   OPENAI_API_VERSION      -> la versión de la API (ej. 2024-10-21; OJO: NO es
#                              la versión del modelo como "2024-11-20")
# Y nosotros le pasamos explícitamente:
#   azure_deployment -> el NOMBRE DEL DEPLOYMENT que le diste a gpt-4o en Foundry
#                       (puede ser distinto de "gpt-4o").
#   temperature=0.3  -> baja "creatividad" -> respuestas más estables/repetibles.
llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    api_version=os.environ.get("OPENAI_API_VERSION", "2024-10-21"),
    temperature=0.3,
)

# Cliente de búsqueda web. max_results=3 -> trae como mucho 3 resultados por query.
search = TavilySearch(max_results=3)

# Lista negra de frases "de relleno" que NO queremos ver en las preguntas.
# Si el modelo las usa, lo detectamos y regeneramos (ver generate_questions_node).
GENERIC_PHRASES = [
    "día a día",
    "cultura de la empresa",
    "valores de la empresa",
    "oportunidades de crecimiento",
    "cómo es el equipo",
    "qué os diferencia",
    "cuál es vuestra visión",
]

# Reglas que reutilizamos en VARIOS prompts de preguntas. Tenerlas en una sola
# constante evita repetir el mismo texto y que se desincronicen.
QUESTION_RULES = (
    '- Cada pregunta debe referenciar algo CONCRETO (un producto, una tecnología\n'
    '  específica, una noticia reciente, una decisión técnica).\n'
    '- Prohibido usar frases genéricas como "día a día", "cultura", "valores",\n'
    '  "oportunidades de crecimiento", "qué os diferencia".\n'
    '- Las preguntas deben demostrar que investigaste, no que googleaste el nombre.'
)


# --- Funciones auxiliares (helpers). El "_" inicial indica "uso interno" -------

def _tavily_contents(query: str) -> list[str]:
    """Ejecuta una búsqueda en Tavily y devuelve solo el texto de cada resultado.

    Detalle importante de la migración de librería:
    TavilySearch (paquete 'langchain-tavily') devuelve un DICCIONARIO con la
    clave 'results'. El antiguo TavilySearchResults devolvía directamente una
    LISTA. Por eso normalizamos aquí: si viene dict, sacamos response['results'];
    si no, asumimos que ya es la lista.
    """
    response = search.invoke(query)
    results = response.get("results", []) if isinstance(response, dict) else response
    # De cada resultado nos quedamos solo con el campo 'content' (el texto).
    return [r.get("content", "") for r in results]


def _has_generic(text: str) -> bool:
    """Devuelve True si el texto contiene alguna de las frases genéricas prohibidas."""
    low = text.lower()  # pasamos a minúsculas para comparar sin importar mayúsculas
    # any(...) -> True si AL MENOS UNA frase de la lista aparece en el texto.
    return any(phrase in low for phrase in GENERIC_PHRASES)


def _strip_code_fence(text: str) -> str:
    """Quita un envoltorio de bloque de código (```markdown ... ```) si el modelo
    lo añade. A veces el LLM "envuelve" el Markdown en ```; eso haría que el .md
    se viera como un bloque de código en vez de como Markdown real."""
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()[1:]                      # quita la primera línea (```markdown)
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]                          # quita la última línea (```)
        t = "\n".join(lines).strip()
    return t


def _classify_feedback(feedback: str) -> str:
    """Clasifica el feedback del humano en una de tres categorías usando el LLM.

    Devuelve uno de:
      'research'   -> hay que volver a BUSCAR info general/noticias de la empresa.
      'tech_stack' -> hay que volver a BUSCAR el stack/tecnologías/frameworks.
      'edit'       -> solo reescribir/reformatear lo que ya hay (sin buscar nada).

    Esto es lo que permite que "vuelve a buscar sus frameworks" dispare una
    búsqueda real, mientras que "acorta la sección de noticias" no.
    """
    # Le pedimos al modelo que responda con UNA sola palabra.
    out = llm.invoke(
        "Clasifica la petición del usuario en UNA sola palabra.\n"
        '- "research": pide BUSCAR de nuevo información general, noticias o qué hace la empresa.\n'
        '- "tech_stack": pide BUSCAR de nuevo el stack, tecnologías, frameworks o herramientas.\n'
        '- "edit": solo pide reescribir, reformular, acortar, cambiar tono o formato de lo ya existente.\n\n'
        f'Petición: "{feedback}"\n\n'
        "Responde solo con: research, tech_stack o edit."
    ).content.lower()

    # Interpretamos la respuesta de forma tolerante (por si el modelo añade algo
    # más que la palabra). El orden importa: comprobamos tech_stack primero.
    if "tech" in out or "stack" in out:
        return "tech_stack"
    if "research" in out:
        return "research"
    return "edit"  # por defecto, tratamos el feedback como una simple edición


# ---------------------------------------------------------------------------
# 3. Sub-grafos de búsqueda (research y tech_stack) con MAP-REDUCE
# ---------------------------------------------------------------------------
# Los dos sub-grafos comparten la MISMA estructura interna:
#
#     START ──(fan-out: un Send por query)──► search_one ─┐
#                                             search_one ─┤─► synthesize ─► END
#                                             search_one ─┘
#                                            (en PARALELO)   (fan-in)
#
# El nodo trabajador 'search_one' es COMÚN a ambos sub-grafos (misma función).
# Lo que cambia es: (a) qué queries genera el fan-out, y (b) qué hace el
# 'synthesize' final (resumir noticias vs. extraer el stack).

# --- Estados internos de los sub-grafos ---
# 'search_results' es la clave clave del map-reduce: Annotated[list[str], operator.add]
# significa "es una lista de str y, cuando varios nodos escriban en ella a la vez,
# CONCATÉNALAS (no las sobrescribas)". Sin este reducer, los Send paralelos darían
# error por escribir todos en la misma clave.
# 'search_results' SOLO existe aquí (no en InterviewState) -> es interno al sub-grafo.

class ResearchState(TypedDict):
    company: str                                  # entra desde el padre
    search_feedback: Optional[str]                # entra desde el padre (afina la query)
    search_results: Annotated[list[str], operator.add]   # acumulador interno (map-reduce)
    research: str                                 # sale hacia el padre


class TechStackState(TypedDict):
    company: str
    search_feedback: Optional[str]
    search_results: Annotated[list[str], operator.add]
    tech_stack: str                               # sale hacia el padre


# Esquemas de SALIDA de los sub-grafos de búsqueda. CLAVE para el paralelismo:
# 'research' y 'tech_stack' corren en el MISMO superstep (fan-out desde START). Si
# cada sub-grafo devolviera al padre TODAS sus claves compartidas (company,
# search_feedback...), habría DOS escrituras de 'company' en el mismo paso y
# LangGraph fallaría con InvalidUpdateError (un canal LastValue admite 1 valor/paso).
# Limitando la salida a SOLO el resultado, cada sub-grafo escribe una clave distinta
# y no chocan. Los datos de ENTRADA (company, search_feedback) siguen llegando porque
# el esquema de entrada es el estado completo; solo filtramos lo que SALE.
class ResearchOut(TypedDict):
    research: str


class TechStackOut(TypedDict):
    tech_stack: str


def search_one(payload: dict) -> dict:
    """Nodo TRABAJADOR del map-reduce (común a ambos sub-grafos).

    Recibe como estado EXACTAMENTE el payload que le pasó su Send (aquí: {"query": ...}),
    no el estado completo del sub-grafo. Hace UNA búsqueda y devuelve sus textos.
    Como devolvemos la clave 'search_results' (que tiene reducer operator.add), el
    resultado se CONCATENA con el de los demás trabajadores en vez de pisarlo.
    """
    return {"search_results": _tavily_contents(payload["query"])}


def fan_out_research(state: ResearchState) -> list[Send]:
    """MAP del sub-grafo 'research': construye las queries y lanza un Send por cada una.

    Devolver una lista de Send hace que LangGraph cree N copias paralelas de
    'search_one'. Esta función NO modifica el estado; solo decide el trabajo a repartir.
    """
    company = state["company"]
    # search_feedback solo trae valor cuando venimos de un "re-buscar"; si no, "".
    extra = state.get("search_feedback") or ""
    if extra:
        print(f"  [re-buscando información general — '{extra}']")

    # Tres ángulos distintos para cubrir "qué hacen" + "noticias" + "producto/negocio".
    queries = [
        f"{company} company overview what they do {extra}".strip(),
        f"{company} recent news 2025 {extra}".strip(),
        f"{company} products business model funding {extra}".strip(),
    ]
    return [Send("search_one", {"query": q}) for q in queries]


def synthesize_research(state: ResearchState) -> dict:
    """REDUCE del sub-grafo 'research': junta todo lo buscado y lo resume con el LLM.

    Se ejecuta UNA sola vez, cuando TODOS los 'search_one' han terminado (fan-in).
    Lee 'search_results' (ya concatenado por el reducer) y 'company'.
    """
    content = "\n".join(state["search_results"])
    summary = llm.invoke(
        f"Resume en 150 palabras qué hace '{state['company']}' y sus noticias más "
        f"relevantes del último año. Sé concreto, sin frases genéricas.\n\n{content}"
    )
    # Devolvemos SOLO la clave que cruza al padre. .content es el texto de la respuesta.
    return {"research": summary.content}


def fan_out_tech(state: TechStackState) -> list[Send]:
    """MAP del sub-grafo 'tech_stack': tres señales para inferir el stack real.
    1. Blog de ingeniería.  2. Ofertas de empleo.  3. StackShare.
    """
    company = state["company"]
    extra = state.get("search_feedback") or ""
    if extra:
        print(f"  [re-buscando stack tecnológico — '{extra}']")

    queries = [
        f"{company} engineering blog tech stack architecture {extra}".strip(),
        f"{company} jobs software engineer requirements technologies {extra}".strip(),
        f"{company} site:stackshare.io OR {company} stackshare technologies",
    ]
    return [Send("search_one", {"query": q}) for q in queries]


def synthesize_tech(state: TechStackState) -> dict:
    """REDUCE del sub-grafo 'tech_stack': extrae el stack CONCRETO de lo buscado."""
    combined = "\n\n".join(state["search_results"])
    stack = llm.invoke(
        f"Extrae el stack tecnológico CONCRETO de '{state['company']}' a partir de esta "
        f"información. Lista: lenguajes, frameworks, cloud, bases de datos, CI/CD, "
        f"herramientas de observabilidad. Si no encuentras algo concreto, di "
        f"explícitamente qué sección está vacía. No inventes nada.\n\n{combined}"
    )
    return {"tech_stack": stack.content}


def _build_search_subgraph(state_schema, output_schema, fan_out, synthesize):
    """Fábrica que monta y COMPILA un sub-grafo de búsqueda map-reduce.

    Recibe el esquema de estado, el esquema de SALIDA (qué claves devuelve al padre),
    la función de fan-out (MAP) y la de síntesis (REDUCE), y devuelve el grafo ya
    compilado, listo para usarse como un nodo.
    Lo compilamos SIN checkpointer a propósito: heredará el del grafo padre.
    """
    b = StateGraph(state_schema, output_schema=output_schema)
    b.add_node("search_one", search_one)
    b.add_node("synthesize", synthesize)
    # add_conditional_edges desde START con una función que devuelve Sends = fan-out (MAP).
    # El tercer argumento (["search_one"]) declara a qué nodo(s) pueden ir esos Send.
    b.add_conditional_edges(START, fan_out, ["search_one"])
    # Cuando TODOS los 'search_one' terminan, se sigue UNA vez a 'synthesize' (fan-in / REDUCE).
    b.add_edge("search_one", "synthesize")
    b.add_edge("synthesize", END)
    return b.compile()


# Compilamos los dos sub-grafos de búsqueda.
research_subgraph = _build_search_subgraph(ResearchState, ResearchOut, fan_out_research, synthesize_research)
tech_stack_subgraph = _build_search_subgraph(TechStackState, TechStackOut, fan_out_tech, synthesize_tech)


def route_after_search(state: InterviewState) -> str:
    """Función de enrutado (en el grafo PADRE): decide a qué fase ir DESPUÉS de buscar.

    Las funciones de enrutado NO modifican el estado; solo LEEN el estado y
    devuelven el NOMBRE (string) del siguiente nodo. Aquí volvemos a la fase que
    pidió la búsqueda. Si no hay 'search_return', por defecto vamos a las preguntas.
    """
    return state.get("search_return") or "generate_questions_node"


# ---------------------------------------------------------------------------
# 4. Preguntas: generación (LLM) + revisión (interrupt) en nodos separados
# ---------------------------------------------------------------------------

def generate_questions_node(state: InterviewState) -> dict:
    """Genera (o edita) las 3 preguntas. NO contiene interrupt(), por eso es
    SEGURO que el LLM viva aquí (ver "idea clave 1" en la cabecera).

    Tiene tres modos:
    - EDICIÓN DIRIGIDA: si hay feedback + preguntas previas -> cambia solo lo
      pedido y deja el resto IGUAL.
    - DESDE CERO: primera vez, o después de re-buscar datos.
    - AUTO-REINTENTO: si detecta frases genéricas, reintenta hasta 3 veces solo
      (sin molestar al humano).
    """
    feedback = state.get("questions_feedback")
    previous = state.get("previous_questions")

    # Contexto común que damos al modelo en cualquiera de los modos.
    contexto = (
        f"Empresa: {state['company']}\n"
        f"Información general: {state['research']}\n"
        f"Stack tecnológico: {state['tech_stack']}"
    )

    avoid_note = ""   # nota extra que añadiremos si hay que evitar frases genéricas
    candidate = ""    # aquí guardaremos las preguntas generadas
    # Bucle de hasta 3 intentos para esquivar las frases genéricas.
    for _ in range(3):
        if feedback and previous:
            # --- MODO EDICIÓN DIRIGIDA ---
            # Le pasamos las preguntas actuales y le pedimos cambiar SOLO lo pedido.
            prompt = f"""{contexto}

Estas son las preguntas actuales:
{previous}

El humano pidió este cambio: "{feedback}".

Aplica EXCLUSIVAMENTE ese cambio:
- Identifica la(s) pregunta(s) a la(s) que se refiere y modifícala(s).
- Las demás preguntas devuélvelas EXACTAMENTE IGUAL, palabra por palabra, sin reescribirlas.
- Conserva la numeración 1, 2, 3 y el mismo orden.

Reglas para cualquier pregunta nueva o modificada:
{QUESTION_RULES}{avoid_note}

Formato de respuesta:
1. [pregunta]
2. [pregunta]
3. [pregunta]
"""
        else:
            # --- MODO DESDE CERO ---
            # Si hay feedback (p.ej. tras re-buscar), lo añadimos como nota.
            nota = f"\nTen en cuenta esta petición del usuario: {feedback}\n" if feedback else ""
            prompt = f"""{contexto}
{nota}
Genera exactamente 3 preguntas para hacer en una entrevista con esta empresa.

REGLAS ESTRICTAS:
{QUESTION_RULES}{avoid_note}

Formato de respuesta:
1. [pregunta]
2. [pregunta]
3. [pregunta]
"""
        # Llamamos al modelo y guardamos el texto.
        candidate = llm.invoke(prompt).content
        # Si NO hay frases genéricas, salimos del bucle (preguntas válidas).
        if not _has_generic(candidate):
            break
        # Si las había, preparamos una nota más dura para el siguiente intento.
        avoid_note = (
            "\n- IMPORTANTE: la versión anterior contenía frases genéricas "
            "prohibidas. Evítalas por completo."
        )

    # Guardamos las preguntas. Limpiamos los feedbacks ya consumidos:
    # - questions_feedback: ya lo aplicamos.
    # - search_feedback: ya lo usó el sub-grafo de búsqueda (si veníamos de re-buscar).
    return {"smart_questions": candidate, "questions_feedback": None, "search_feedback": None}


def review_questions_node(state: InterviewState) -> dict:
    """Nodo de REVISIÓN de preguntas. Aquí está el interrupt() (la pausa).
    No llama al LLM salvo para clasificar el feedback. Solo pausa, muestra las
    preguntas y, según lo que diga el humano, decide a dónde ir.
    """
    # interrupt(...) PAUSA el grafo aquí y devuelve el diccionario a quien invocó
    # el grafo (lo mostramos en run_agent). Cuando se reanuda con Command(resume=X),
    # esta misma llamada interrupt(...) DEVUELVE X. Por eso 'human_feedback' acaba
    # conteniendo la respuesta del humano.
    human_feedback = interrupt({
        "mensaje": (
            f"Preguntas generadas para {state['company']}.\n"
            "Responde 'ok' para continuar, escribe un retoque de redacción, "
            "o pide buscar más datos (p.ej. 'vuelve a buscar su stack')."
        ),
        "preguntas": state["smart_questions"],
    })

    fb = str(human_feedback).strip()  # normalizamos la respuesta (quita espacios)

    # CASO 1: el humano aprueba -> seguimos hacia el sub-grafo de briefing.
    if fb.lower() == "ok":
        return {"route": "briefing", "questions_feedback": None,
                "previous_questions": None}

    # Si NO es "ok", clasificamos qué tipo de petición es.
    intent = _classify_feedback(fb)

    # CASO 2: es una simple edición de texto -> regeneramos preguntas con los
    # MISMOS datos, haciendo edición dirigida sobre las preguntas mostradas.
    if intent == "edit":
        return {
            "route": "generate_questions_node",
            "smart_questions": None,
            "previous_questions": state["smart_questions"],   # base EXACTA de la edición
            "questions_feedback": fb,
        }

    # CASO 3: pide datos nuevos (research o tech_stack) -> vamos al sub-grafo de
    # búsqueda y luego volvemos a generar las preguntas. 'intent' ya es exactamente
    # el nombre del nodo de sub-grafo: "research" o "tech_stack".
    return {
        "route": intent,
        "smart_questions": None,
        "previous_questions": None,        # se regeneran desde cero con los datos nuevos
        "questions_feedback": fb,          # se usa como nota orientativa
        "search_return": "generate_questions_node",  # tras buscar, volver aquí
        "search_feedback": fb,             # afina la query de búsqueda
    }


def route_after_questions_review(state: InterviewState) -> str:
    """Enrutado tras la revisión de preguntas: simplemente devolvemos la decisión
    ('route') que ya calculó review_questions_node."""
    return state["route"]


# ---------------------------------------------------------------------------
# 5. Briefing: SUB-GRAFO con generación (LLM) + revisión (interrupt)
# ---------------------------------------------------------------------------
# El briefing es un sub-grafo que encapsula el ciclo "redacta + aprueba":
#
#     START ─► generate_briefing ─► review_briefing ──(edición)──┐
#                  ▲                      │                       │
#                  └──────────────────────┘  (bucle interno)      │
#                                         │ ok / re-buscar        │
#                                         ▼                       │
#                                        END  (sale al padre con  │
#                                              'route' decidido)  │
#
# El bucle de EDICIÓN es interno (no sale del sub-grafo). En cambio "ok" y
# "re-buscar" SALEN al padre: dejan la decisión en state['route'] y el grafo padre
# la usa para ir a write_file_node o a los sub-grafos de búsqueda.

class BriefingState(TypedDict):
    company: str                        # entra desde el padre
    research: str                       # entra desde el padre
    tech_stack: str                     # entra desde el padre
    smart_questions: Optional[str]      # entra desde el padre

    briefing: Optional[str]             # candidato o aprobado (cruza al padre)
    previous_briefing: Optional[str]    # base para la edición dirigida
    briefing_feedback: Optional[str]    # el cambio pedido sobre el briefing

    route: Optional[str]                # decisión que leerá el padre al salir
    search_return: Optional[str]        # a dónde volver tras re-buscar (cruza al padre)
    search_feedback: Optional[str]      # afina la query si se re-busca (cruza al padre)


def _full_briefing_prompt(state: BriefingState) -> str:
    """Construye el prompt para montar el briefing COMPLETO desde cero.
    Lo separamos en una función porque se usa en el primer montaje y también
    después de re-buscar datos."""
    return f"""
Monta un briefing de entrevista para '{state['company']}' en Markdown.
Usa exactamente estas secciones:

## Qué hacen
(2-3 frases concretas, sin marketing)

## Noticias recientes
(bullet points con fechas si las tienes)

## Stack tecnológico
(lista organizada por categoría)

## Preguntas para hacer
(las 3 preguntas, numeradas)

## Red flags / cosas a investigar más
(1-2 cosas que quedaron sin confirmar o que llaman la atención)

Empieza directamente por '## Qué hacen'. NO añadas un título de nivel 1 (#)
y NO envuelvas la respuesta en un bloque de código (```).

Datos disponibles:
RESEARCH: {state['research']}
STACK: {state['tech_stack']}
PREGUNTAS: {state['smart_questions']}
"""


def generate_briefing_node(state: BriefingState) -> dict:
    """Genera (o edita) el briefing. NO contiene interrupt() (mismo motivo que en
    las preguntas). Si hay feedback + briefing previo -> edición dirigida; si no,
    monta el briefing entero (también tras re-buscar datos)."""
    feedback = state.get("briefing_feedback")
    previous = state.get("previous_briefing")

    if feedback and previous:
        # --- EDICIÓN DIRIGIDA del briefing ---
        prompt = f"""Este es el briefing actual en Markdown:

{previous}

El humano pidió este cambio: "{feedback}".

Devuelve el briefing COMPLETO en Markdown aplicando EXCLUSIVAMENTE ese cambio:
- Modifica solo la sección o el contenido al que se refiere la petición.
- El resto del briefing debe quedar EXACTAMENTE IGUAL, palabra por palabra.
- Conserva las mismas secciones y el mismo orden.
- NO añadas un título de nivel 1 (#) ni envuelvas la respuesta en un bloque de código (```).
"""
    else:
        # --- MONTAJE COMPLETO ---
        prompt = _full_briefing_prompt(state)

    # Llamamos al LLM y quitamos el posible envoltorio ``` con el helper.
    briefing_text = _strip_code_fence(llm.invoke(prompt).content)
    return {"briefing": briefing_text, "briefing_feedback": None, "search_feedback": None}


def review_briefing_node(state: BriefingState) -> dict:
    """Nodo de REVISIÓN del briefing (dentro del sub-grafo). Aquí está el segundo
    interrupt(); como el sub-grafo hereda el checkpointer del padre, la pausa
    burbujea hasta el invoke() de arriba igual que la de las preguntas.
    'ok' -> salir al padre para guardar; 'editar' -> bucle interno; 'datos' -> salir
    al padre a re-buscar."""
    human_feedback = interrupt({
        "mensaje": (
            "Briefing final generado. Responde 'ok' para guardarlo en un .md, "
            "pide un retoque, o pide buscar más datos (p.ej. 'busca de nuevo sus frameworks')."
        ),
        "briefing": state["briefing"],
    })

    fb = str(human_feedback).strip()

    # CASO 1: aprobado -> el padre irá a escribir el archivo.
    if fb.lower() == "ok":
        return {"route": "write_file_node", "briefing_feedback": None,
                "previous_briefing": None}

    intent = _classify_feedback(fb)

    # CASO 2: edición de texto -> bucle INTERNO del sub-grafo (vuelve a generate_briefing).
    # Marcamos route="generate_briefing"; el router interno lo interpreta como "no salgas".
    if intent == "edit":
        return {
            "route": "generate_briefing",
            "briefing": None,
            "previous_briefing": state["briefing"],   # base EXACTA de la edición
            "briefing_feedback": fb,
        }

    # CASO 3: datos nuevos -> SALIR al padre a re-buscar y luego volver al briefing.
    # 'intent' es "research" o "tech_stack" (nombres de los sub-grafos del padre).
    return {
        "route": intent,
        "briefing": None,
        "previous_briefing": None,         # se reconstruye con los datos nuevos
        "briefing_feedback": None,
        "search_return": "briefing",       # tras buscar, volver a ESTE sub-grafo
        "search_feedback": fb,             # afina la query de búsqueda
    }


def route_inside_briefing(state: BriefingState) -> str:
    """Router INTERNO del sub-grafo de briefing tras la revisión.
    Solo la edición se queda dentro (bucle a generate_briefing); cualquier otra
    decisión SALE del sub-grafo (END) y el padre lee state['route']."""
    return "generate_briefing" if state["route"] == "generate_briefing" else END


def _build_briefing_subgraph():
    """Monta y compila el sub-grafo de briefing (genera + revisa con interrupt).
    Sin checkpointer: hereda el del padre, necesario para que el interrupt() funcione."""
    b = StateGraph(BriefingState)
    b.add_node("generate_briefing", generate_briefing_node)
    b.add_node("review_briefing", review_briefing_node)
    b.add_edge(START, "generate_briefing")
    b.add_edge("generate_briefing", "review_briefing")
    # Tras revisar: o bucle interno de edición, o salida (END) con la decisión en 'route'.
    b.add_conditional_edges("review_briefing", route_inside_briefing,
                            {"generate_briefing": "generate_briefing", END: END})
    return b.compile()


briefing_subgraph = _build_briefing_subgraph()


def route_after_briefing(state: InterviewState) -> str:
    """Enrutado en el PADRE tras el sub-grafo de briefing: devolvemos la decisión
    'route' que dejó review_briefing al salir ('write_file_node', 'research' o 'tech_stack')."""
    return state["route"]


# ---------------------------------------------------------------------------
# 6. Escritura del archivo Markdown (tras la aprobación final)
# ---------------------------------------------------------------------------

def write_file_node(state: InterviewState) -> dict:
    """Guarda el briefing aprobado en briefings/briefing-<empresa>-<timestamp>.md."""
    # 'slug': versión del nombre de la empresa apta para un nombre de archivo
    # (minúsculas y sin caracteres raros). re.sub reemplaza todo lo que no sea
    # letra/número por guiones; .strip("-") quita guiones sobrantes en los extremos.
    slug = re.sub(r"[^a-z0-9]+", "-", state["company"].lower()).strip("-") or "empresa"
    # Marca de tiempo para que cada ejecución cree un archivo distinto.
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Carpeta de salida 'briefings/' JUNTO a este archivo .py (no donde se ejecute).
    # __file__ es la ruta de este script; dirname obtiene su carpeta.
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "briefings")
    os.makedirs(out_dir, exist_ok=True)  # crea la carpeta si no existe
    path = os.path.join(out_dir, f"briefing-{slug}-{timestamp}.md")

    # Cabecera del documento: un título de nivel 1 y la fecha de generación.
    header = f"# Briefing de entrevista — {state['company']}\n\n"
    header += f"_Generado el {datetime.datetime.now():%Y-%m-%d %H:%M}_\n\n---\n\n"

    # Escribimos el archivo en UTF-8 (importante para acentos y caracteres especiales).
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + state["briefing"])

    # Guardamos la ruta en el estado para poder mostrarla al final.
    return {"briefing_path": path}


# ---------------------------------------------------------------------------
# 7. Construcción del grafo PADRE
# ---------------------------------------------------------------------------
# Aquí "cableamos" los nodos y las flechas. Los sub-grafos compilados
# (research_subgraph, tech_stack_subgraph, briefing_subgraph) se añaden como
# nodos NORMALES: el padre no necesita saber qué hacen por dentro.

# StateGraph recibe la forma del estado para saber qué claves existen.
builder = StateGraph(InterviewState)

# Registramos cada nodo con un NOMBRE (string). Los tres primeros son SUB-GRAFOS.
builder.add_node("research", research_subgraph)                    # sub-grafo map-reduce
builder.add_node("tech_stack", tech_stack_subgraph)               # sub-grafo map-reduce
builder.add_node("generate_questions_node", generate_questions_node)
builder.add_node("review_questions_node", review_questions_node)
builder.add_node("briefing", briefing_subgraph)                   # sub-grafo (genera + revisa)
builder.add_node("write_file_node", write_file_node)

# --- Flechas (edges) ---

# Desde START salen DOS flechas a la vez -> los sub-grafos 'research' y 'tech_stack'
# corren EN PARALELO (fan-out). Ambos terminan y luego se sigue a preguntas (fan-in).
builder.add_edge(START, "research")
builder.add_edge(START, "tech_stack")

# Flechas CONDICIONALES tras buscar: el destino depende de route_after_search, que
# devuelve a qué fase volver. El diccionario mapea ese nombre -> nodo destino real.
_search_targets = {
    "generate_questions_node": "generate_questions_node",
    "briefing": "briefing",
}
builder.add_conditional_edges("research", route_after_search, _search_targets)
builder.add_conditional_edges("tech_stack", route_after_search, _search_targets)

# Generar preguntas -> revisar preguntas (flecha simple, siempre va aquí).
builder.add_edge("generate_questions_node", "review_questions_node")
# Tras revisar preguntas, el destino depende de la decisión 'route':
builder.add_conditional_edges(
    "review_questions_node",
    route_after_questions_review,
    {
        "generate_questions_node": "generate_questions_node",  # retoque de redacción
        "research": "research",                                # re-buscar datos generales
        "tech_stack": "tech_stack",                            # re-buscar stack
        "briefing": "briefing",                                # aprobado -> al briefing
    },
)

# Tras el sub-grafo de briefing, el destino depende de la decisión 'route' que dejó
# al salir (la edición ya se resolvió DENTRO del sub-grafo, no llega aquí):
builder.add_conditional_edges(
    "briefing",
    route_after_briefing,
    {
        "research": "research",                                # re-buscar datos generales
        "tech_stack": "tech_stack",                            # re-buscar stack
        "write_file_node": "write_file_node",                  # aprobado -> guardar
    },
)

# Tras escribir el archivo, terminamos.
builder.add_edge("write_file_node", END)

# El checkpointer guarda el estado entre pausas. Sin esto, interrupt() no podría
# reanudar donde lo dejó. MemorySaver lo guarda en memoria (se pierde al cerrar
# el programa; para algo persistente se usaría una base de datos).
# NOTA: solo el grafo PADRE lleva checkpointer; los sub-grafos lo heredan, por eso
# el interrupt() dentro del sub-grafo de briefing funciona igual que el de preguntas.
memory = MemorySaver()

# compile() convierte el "plano" (builder) en un grafo EJECUTABLE.
graph = builder.compile(checkpointer=memory)


# ---------------------------------------------------------------------------
# 8. Ejecución con human-in-the-loop (el bucle que habla con la persona)
# ---------------------------------------------------------------------------

def run_agent(company: str, thread_id: str = "entrevista-1"):
    """Lanza el agente y gestiona las pausas de forma interactiva por consola.

    thread_id identifica esta "conversación". El checkpointer guarda el estado
    asociado a ese id, de modo que las reanudaciones continúan el mismo hilo.
    """
    # 'config' le dice a LangGraph bajo qué hilo guardar/leer el estado.
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n{'='*60}")
    print(f"  Investigando: {company}")
    print(f"{'='*60}\n")

    # Estado inicial: rellenamos todas las claves. Lo importante:
    # - company: lo que investigaremos.
    # - search_return: en el arranque, tras buscar vamos a generar preguntas.
    # - el resto en None/"" porque aún no hay nada.
    initial_state: InterviewState = {
        "company": company,
        "research": "",
        "tech_stack": "",
        "smart_questions": None,
        "previous_questions": None,
        "questions_feedback": None,
        "briefing": None,
        "previous_briefing": None,
        "briefing_feedback": None,
        "route": None,
        "search_return": "generate_questions_node",
        "search_feedback": None,
        "briefing_path": None,
    }

    # Primera ejecución: el grafo corre hasta el PRIMER interrupt() (la primera
    # pausa, que será la revisión de preguntas).
    result = graph.invoke(initial_state, config=config)

    # Mientras el resultado contenga "__interrupt__", significa que el grafo está
    # PAUSADO esperando una respuesta humana. Iteramos hasta que ya no haya pausa
    # (es decir, hasta que el grafo llegue a END).
    while result.get("__interrupt__"):
        # __interrupt__ es una lista; tomamos el primero y su .value es justo el
        # diccionario que pasamos a interrupt(...) en el nodo de revisión.
        interrupt_data = result["__interrupt__"][0].value

        print("\n" + "-"*60)
        print(interrupt_data.get("mensaje", ""))
        print("-"*60)

        # Mostramos las preguntas o el briefing, según en qué pausa estemos.
        if "preguntas" in interrupt_data:
            print("\n" + interrupt_data["preguntas"])
        elif "briefing" in interrupt_data:
            print("\n" + interrupt_data["briefing"])

        print("\n" + "-"*60)
        # Pedimos la respuesta por teclado.
        human_input = input("Tu respuesta ('ok' para aprobar, o escribe qué cambiar): ").strip()

        # REANUDAMOS el grafo. Command(resume=human_input) hace que la llamada a
        # interrupt(...) que dejó el grafo en pausa DEVUELVA 'human_input', y el
        # nodo de revisión continúa desde ahí. El grafo seguirá hasta la próxima
        # pausa o hasta END.
        result = graph.invoke(Command(resume=human_input), config=config)

    # Si salimos del bucle, el grafo terminó (llegó a END).
    print("\n" + "="*60)
    print("  BRIEFING FINAL")
    print("="*60)
    print(result.get("briefing", "No se generó briefing."))

    # Mostramos dónde quedó guardado el archivo .md.
    path = result.get("briefing_path")
    if path:
        print(f"\n✓ Briefing guardado en: {path}")

    return result


# ---------------------------------------------------------------------------
# 9. Punto de entrada
# ---------------------------------------------------------------------------
# Este bloque solo se ejecuta si lanzamos el archivo directamente
# (python interview_agent.py ...), no si se importa desde otro módulo.

if __name__ == "__main__":
    import sys  # para leer argumentos de la línea de comandos

    # sys.argv[0] es el nombre del script; sys.argv[1] sería el primer argumento.
    # Si pasas un nombre de empresa, se usa; si no, por defecto "Stripe".
    # Ejemplo de uso:  python interview_agent.py "Vercel"
    company = sys.argv[1] if len(sys.argv) > 1 else "Stripe"
    run_agent(company)
