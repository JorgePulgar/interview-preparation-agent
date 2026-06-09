"""
Genera un diagrama LIMPIO y organizado del grafo del agente.

Uso:
  python show_graph.py

Genera:
  - graph.mmd  : el diagrama en texto Mermaid
  - graph.png  : imagen del diagrama (requiere internet; render de mermaid.ink)

Abre graph.png en VSCode (doble clic). El .mmd se puede previsualizar con la
extensión "Markdown Preview Mermaid Support".

Nota: este diagrama está organizado a mano (agrupado por fases con subgrafos)
para que se lea bien. Si cambias la estructura del grafo en interview_agent.py,
actualiza también este diagrama. Para una versión 100% automática (pero con peor
layout) puedes usar: from interview_agent import graph; graph.get_graph().draw_mermaid()
"""

from langchain_core.runnables.graph_mermaid import draw_mermaid_png

# Nota: sin emojis a propósito. El render de mermaid.ink usa una URL con el
# diagrama en base64; los emojis (3-4 bytes c/u) la inflan y la petición falla.
# Los acentos del español sí funcionan.
MERMAID = """flowchart TD
    START([START]):::se

    %% --- Fila de investigación: dos sub-grafos lado a lado (izq / der) ---
    subgraph RES["research · sub-grafo (map-reduce)"]
        direction TB
        rfan["fan-out · 1 Send por query"]:::map
        rw1["search_one"]
        rw2["search_one"]
        rw3["search_one"]
        rsyn["synthesize · resume noticias"]:::map
        rfan --> rw1 & rw2 & rw3 --> rsyn
    end

    subgraph TEC["tech_stack · sub-grafo (map-reduce)"]
        direction TB
        tfan["fan-out · 1 Send por query"]:::map
        tw1["search_one"]
        tw2["search_one"]
        tw3["search_one"]
        tsyn["synthesize · extrae el stack"]:::map
        tfan --> tw1 & tw2 & tw3 --> tsyn
    end

    subgraph QST["Preguntas · grafo padre"]
        direction TB
        genq["generate_questions_node<br/><i>genera / edita</i>"]
        revq{{"review_questions_node<br/>interrupt() · pausa"}}
        genq --> revq
        revq -. "editar" .-> genq
    end

    subgraph BRF["briefing · sub-grafo"]
        direction TB
        genb["generate_briefing<br/><i>genera / edita</i>"]
        revb{{"review_briefing<br/>interrupt() · pausa"}}
        genb --> revb
        revb -. "editar (bucle interno)" .-> genb
    end

    write["write_file_node<br/><i>guarda el .md</i>"]
    DONE([END]):::se

    %% --- Espina dorsal vertical (caja debajo de caja) ---
    START --> RES
    START --> TEC
    RES --> QST
    TEC --> QST
    QST -- "ok" --> BRF
    BRF -- "ok" --> write
    write --> DONE

    %% Nota: los bucles de feedback (editar / re-buscar) NO se dibujan aquí.
    %% Son aristas que suben de rango y crean ciclos; mezclados con esta espina
    %% vertical, el motor de layout (dagre) descuadra todas las cajas. Por eso van
    %% en un SEGUNDO diagrama (graph-loops). El de edición interno (editar) sí cabe
    %% porque es local a cada caja.

    classDef se fill:#bfb6fc,stroke:#6c5ce7,color:#111,font-weight:bold;
    classDef default fill:#f2f0ff,stroke:#b9aef7,color:#111;
    classDef map fill:#ffe9c7,stroke:#f0a23b,color:#111;
    style RES fill:#eef7ff,stroke:#7fb3ff
    style TEC fill:#eef7ff,stroke:#7fb3ff
    style QST fill:#fff6ed,stroke:#ffb870
    style BRF fill:#eefcf0,stroke:#86d99a
"""

# Segundo diagrama: SOLO los bucles de feedback (qué pasa cuando NO respondes "ok"
# en una revisión). Va aparte porque estos bucles, dibujados sobre la espina
# vertical de arriba, rompen el layout. Aquí, aislados y en horizontal, se leen bien.
MERMAID_LOOPS = """flowchart LR
    rev{{"Revisión<br/>(preguntas o briefing)<br/>interrupt() · pausa"}}:::rev
    gen["Generar de nuevo<br/>la MISMA fase"]
    search["research / tech_stack<br/>(volver a BUSCAR datos)"]:::map

    rev -. "editar<br/>(retoque de redacción)" .-> gen
    rev -. "re-buscar<br/>(pides datos nuevos)" .-> search
    search -. "vuelve a la fase<br/>que pidió la búsqueda" .-> gen
    gen -- "muestra el<br/>resultado nuevo" --> rev

    classDef rev fill:#ede7ff,stroke:#7c5cff,color:#111;
    classDef map fill:#ffe9c7,stroke:#f0a23b,color:#111;
    classDef default fill:#f2f0ff,stroke:#b9aef7,color:#111;
"""


def _render(mermaid: str, stem: str):
    """Escribe <stem>.mmd y, si hay internet, <stem>.png."""
    with open(f"{stem}.mmd", "w", encoding="utf-8") as f:
        f.write(mermaid)
    print(f"Guardado: {stem}.mmd")
    try:
        png = draw_mermaid_png(mermaid, max_retries=5, retry_delay=2.0)
        with open(f"{stem}.png", "wb") as f:
            f.write(png)
        print(f"Guardado: {stem}.png  (ábrelo en VSCode)")
    except Exception as e:
        print(f"\nNo se pudo generar {stem}.png ({type(e).__name__}): {e}")
        print(f"Aun así tienes {stem}.mmd para previsualizar en VSCode.")


_render(MERMAID, "graph")              # camino principal (espina vertical)
_render(MERMAID_LOOPS, "graph-loops")  # bucles de feedback (editar / re-buscar)
