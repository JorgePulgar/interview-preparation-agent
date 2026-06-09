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

    %% --- Re-búsqueda (feedback que necesita datos nuevos) ---
    QST -. "re-buscar" .-> RES
    QST -. "re-buscar" .-> TEC
    BRF -. "re-buscar" .-> RES
    BRF -. "re-buscar" .-> TEC
    RES -. "vuelve tras re-buscar" .-> BRF
    TEC -. "vuelve tras re-buscar" .-> BRF

    classDef se fill:#bfb6fc,stroke:#6c5ce7,color:#111,font-weight:bold;
    classDef default fill:#f2f0ff,stroke:#b9aef7,color:#111;
    classDef map fill:#ffe9c7,stroke:#f0a23b,color:#111;
    style RES fill:#eef7ff,stroke:#7fb3ff
    style TEC fill:#eef7ff,stroke:#7fb3ff
    style QST fill:#fff6ed,stroke:#ffb870
    style BRF fill:#eefcf0,stroke:#86d99a
"""

with open("graph.mmd", "w", encoding="utf-8") as f:
    f.write(MERMAID)
print("Guardado: graph.mmd")

try:
    png = draw_mermaid_png(MERMAID, max_retries=5, retry_delay=2.0)
    with open("graph.png", "wb") as f:
        f.write(png)
    print("Guardado: graph.png  (ábrelo en VSCode)")
except Exception as e:
    print(f"\nNo se pudo generar el PNG ({type(e).__name__}): {e}")
    print("Aun así tienes graph.mmd para previsualizar en VSCode.")
