"""
Inspección de ejecución: ¿de verdad corrieron los SUB-GRAFOS y el MAP-REDUCE?
=============================================================================
Un run normal del agente produce la misma salida que la versión antigua, así que
"salió un briefing" NO prueba que las piezas nuevas (sub-grafos + búsqueda en
paralelo con Send) funcionaron. Este script lo hace VISIBLE.

Cómo: ejecuta el grafo con stream() en lugar de invoke():
  - stream_mode="updates" -> emite un evento cada vez que TERMINA un nodo.
  - subgraphs=True         -> cada evento trae el NAMESPACE del sub-grafo en el
                              que ocurrió (una tupla). Los nodos del grafo PADRE
                              traen namespace vacío ().

Qué mirar en la salida:
  - MAP-REDUCE corrió  -> verás 'node=search_one' TRES veces bajo el namespace
                          ('research:...') y otras TRES bajo ('tech_stack:...').
                          6 trabajadores = el fan-out con Send ocurrió.
                          (La versión antigua, secuencial, no tenía 'search_one'.)
  - SUB-GRAFOS corrieron -> esos eventos tienen namespace NO vacío
                          (p.ej. ('research:<uuid>',), ('briefing:<uuid>',)).
                          Los nodos del padre (generate_questions_node,
                          write_file_node) salen con ns=().
  - REDUCE corrió 1 vez -> 'node=synthesize' aparece UNA sola vez por cada
                          namespace de búsqueda, DESPUÉS de sus 3 'search_one'
                          (eso es el fan-in).

Uso:
  python inspect_run.py            # usa "Vercel" por defecto
  python inspect_run.py "Stripe"

Nota: esto hace una ejecución REAL -> necesita el .env con las claves de Azure
OpenAI y Tavily, igual que interview_agent.py. Aprueba automáticamente ambas
pausas con 'ok' (no es interactivo); el objetivo es ver el RECORRIDO de nodos,
no revisar el contenido.
"""

import sys
import interview_agent as m


def main(company: str):
    # thread_id propio para no chocar con tus runs interactivos.
    config = {"configurable": {"thread_id": "inspect-1"}}

    # Estado inicial: todas las claves a None y luego rellenamos lo mínimo.
    initial_state = {k: None for k in m.InterviewState.__annotations__}
    initial_state.update(
        company=company,
        research="",
        tech_stack="",
        search_return="generate_questions_node",
    )

    def drive(payload):
        """Avanza el grafo hasta la próxima pausa (interrupt) o el final,
        imprimiendo cada nodo que termina junto con su namespace de sub-grafo."""
        for ns, update in m.graph.stream(
            payload, config=config, stream_mode="updates", subgraphs=True
        ):
            # 'update' es {nombre_nodo: dict_de_cambios}. Sacamos el nombre.
            # En un paso de map-reduce puede venir más de una clave; las listamos.
            for node in update:
                # ns=() -> grafo padre.  ns=('research:uuid',) -> dentro del sub-grafo.
                ambito = "PADRE" if not ns else "SUB-GRAFO"
                print(f"  [{ambito:<9}] ns={str(ns):<30} node={node}")

    print(f"\n{'='*70}\n  Inspeccionando ejecución para: {company}\n{'='*70}")

    print("\n--- 1) arranque -> hasta la 1ª pausa (revisión de preguntas) ---")
    drive(initial_state)

    print("\n--- 2) resume 'ok' (aprobar preguntas) -> hasta la pausa del briefing ---")
    drive(m.Command(resume="ok"))

    print("\n--- 3) resume 'ok' (aprobar briefing) -> hasta END (escribe el .md) ---")
    drive(m.Command(resume="ok"))

    print(f"\n{'='*70}")
    print("Comprueba arriba:")
    print("  - 'search_one' x3 bajo ('research:...')  y  x3 bajo ('tech_stack:...')")
    print("  - 'synthesize' x1 por cada uno (fan-in del reduce)")
    print("  - namespaces NO vacíos = los sub-grafos corrieron de verdad")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    company = sys.argv[1] if len(sys.argv) > 1 else "Vercel"
    main(company)
