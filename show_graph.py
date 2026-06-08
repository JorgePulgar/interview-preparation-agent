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

    subgraph RES["Investigación · en paralelo"]
        direction LR
        research["research_node<br/><i>qué hacen · noticias</i>"]
        tech["tech_stack_node<br/><i>stack · frameworks</i>"]
    end

    subgraph QST["Preguntas"]
        direction LR
        genq["generate_questions_node<br/><i>genera / edita</i>"]
        revq{{"review_questions_node<br/>interrupt() · pausa"}}
    end

    subgraph BRF["Briefing"]
        direction LR
        genb["generate_briefing_node<br/><i>genera / edita</i>"]
        revb{{"review_briefing_node<br/>interrupt() · pausa"}}
    end

    write["write_file_node<br/><i>guarda el .md</i>"]
    DONE([END]):::se

    %% --- Camino principal ---
    START --> research
    START --> tech
    research --> genq
    tech --> genq
    genq --> revq
    revq -- "ok" --> genb
    genb --> revb
    revb -- "ok" --> write
    write --> DONE

    %% --- Bucles de edición (mismo dato, solo reescribe) ---
    revq -. "editar" .-> genq
    revb -. "editar" .-> genb

    %% --- Re-búsqueda (feedback que necesita datos nuevos) ---
    revq -. "re-buscar" .-> research
    revq -. "re-buscar" .-> tech
    revb -. "re-buscar" .-> research
    revb -. "re-buscar" .-> tech
    research -. "vuelve" .-> genb
    tech -. "vuelve" .-> genb

    classDef se fill:#bfb6fc,stroke:#6c5ce7,color:#111,font-weight:bold;
    classDef default fill:#f2f0ff,stroke:#b9aef7,color:#111;
    style RES fill:#eef7ff,stroke:#7fb3ff
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
