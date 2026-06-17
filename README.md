<!--
  GitHub repo README for Interview Research Agent — English version
  Repo: github.com/JorgePulgar/interview-preparation-agent
  Place this file as README.md at the root of the repo.
-->

<p align="right"><sub><b>English</b> · <a href="./README.es.md">Español</a></sub></p>

<h1 align="center">Interview Research Agent</h1>

<p align="center">
  <b>A LangGraph agent that researches a company before an interview and writes a Markdown briefing.</b><br>
  Human-in-the-loop at every gate · two levels of parallel web search · run inspection that proves the graph runs as designed.<br>
  <sub>Python · LangGraph · Azure OpenAI (Azure AI Foundry) · Tavily</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/LangGraph-0.2+-1C3C3C" alt="LangGraph">
  <img src="https://img.shields.io/badge/Azure_OpenAI-Foundry-0078D4?logo=microsoftazure&logoColor=white" alt="Azure OpenAI">
  <img src="https://img.shields.io/badge/Tavily-web_search-FF6154" alt="Tavily">
</p>

---

> 🇪🇸 *También disponible en español: [README.es.md](README.es.md)*

## TL;DR

- **What it does**: give it a company name, it researches the company + its real tech stack, drafts 3 concrete interview questions, assembles a Markdown briefing, and saves it to `briefings/`.
- **Human-in-the-loop**: the graph pauses (`interrupt()`) twice — once on the questions, once on the briefing. You approve with `ok`, ask for a wording edit, or ask it to **search the web again**.
- **Two levels of parallelism**: `research` and `tech_stack` run as parallel subgraphs; inside each, 3 web queries fan out at once via the map-reduce `Send` pattern. 6 workers in flight.
- **Generate and review are separate nodes** — a deliberate choice so that resuming after a pause doesn't re-run the LLM and silently change what you already approved.
- **Hallucination guard on the stack**: the prompt tells the model to name which sections are empty rather than invent a stack it didn't find.
- **`inspect_run.py`** streams the run with `subgraphs=True` so you can verify the subgraphs and the parallel fan-out actually happened — a normal run looks identical to a sequential one, so "a briefing came out" proves nothing.

## Project summary

This is an agent built with **LangGraph**. An agent here is not magic: it's a graph of steps (nodes) joined by edges, with a shared state dictionary flowing between them. Some nodes call an LLM (Azure OpenAI) or a web search tool (Tavily); others just **pause** so a human can review the output and say `ok` or ask for changes.

The interesting part isn't the briefing — it's the wiring. The graph composes three compiled **subgraphs** as if they were plain nodes, runs two of them in parallel, fans each one out into parallel web searches with map-reduce, and keeps the human in control at two interrupt gates. The design deals with a sharp LangGraph edge: when a graph resumes after `interrupt()`, the paused node re-runs from the top — so any LLM call placed before the pause would regenerate content on every resume. The fix is structural, not a workaround: generation and review live in different nodes.

---

## Table of Contents

- [What it does](#what-it-does)
- [Key features](#key-features)
- [Architecture](#architecture)
- [Key design decisions](#key-design-decisions)
- [Proving the subgraphs and map-reduce actually run](#proving-the-subgraphs-and-map-reduce-actually-run)
- [Technology stack](#technology-stack)
- [Requirements](#requirements)
- [Local setup](#local-setup)
- [Configuration (environment variables)](#configuration-environment-variables)
- [Usage](#usage)
- [Project structure](#project-structure)
- [Known limitations](#known-limitations)

---

## What it does

From a company name:

1. **Researches in parallel** what the company does + its recent news (`research` subgraph) and its real **tech stack** (`tech_stack` subgraph).
2. **Generates 3 concrete interview questions** — nothing generic — and **pauses** for you to review them.
3. **Assembles the final briefing** in Markdown and **pauses** again for your sign-off.
4. On approval, **writes the `.md`** to `briefings/briefing-<company>-<timestamp>.md`.

At each pause you can:

- type `ok` to approve;
- ask for a **wording edit** (e.g. *"swap question 2 for an AI one"*) — only that changes, the rest stays byte-for-byte identical;
- ask for **fresh data** (e.g. *"search their frameworks again"*) — the agent **actually searches** the web again and rebuilds from the updated info.

The difference between "wording edit" and "fresh data" is decided by an LLM classifier, so *"shorten the news section"* never triggers a search, while *"re-check their stack"* does.

[↑ Back to top](#table-of-contents)

---

## Key features

- **Human-in-the-loop at two gates**: `interrupt()` on the questions and on the briefing. The run halts, shows you the artifact, and waits for `ok` / edit / re-search.
- **Two-level parallelism**: (A) `research` ‖ `tech_stack` as siblings from `START`; (B) inside each, 3 `search_one` workers run at once. `briefing` runs later, alone, no internal parallelism.
- **Map-reduce search with `Send`**: a fan-out function returns one `Send` per query; workers concatenate into a `search_results` channel via an `Annotated[list[str], operator.add]` reducer; a single `synthesize` node summarizes once all workers finish.
- **Intent-routed feedback**: an LLM classifier sorts every rejection into `edit`, `research`, or `tech_stack`, and the graph routes accordingly — cheap rewrite vs. real re-search.
- **Generic-phrase guard on questions**: a blacklist (`día a día`, `cultura`, `valores`, …) is checked after generation; if a banned phrase appears, the node silently retries up to 3 times before showing you anything.
- **Hallucination-safe stack extraction**: the prompt instructs the model to state which sections are empty instead of inventing technologies it couldn't find.

[↑ Back to top](#table-of-contents)

---

## Architecture

![Agent graph](graph.png)

There is a **parent graph** (the main agent). Three of its nodes aren't plain functions — they're smaller compiled graphs (subgraphs) that own a whole phase:

| Component | Type | Role |
|---|---|---|
| `research` | subgraph (map-reduce) | Searches "what they do" + recent news, in 3 parallel queries, then summarizes. |
| `tech_stack` | subgraph (map-reduce) | Infers the real stack from engineering blog + job posts + StackShare, in 3 parallel queries. |
| `generate_questions_node` | node (LLM) | Drafts/edits the 3 questions. No `interrupt()` here, on purpose. |
| `review_questions_node` | node (interrupt) | Pauses for human review of the questions; classifies the feedback. |
| `briefing` | subgraph (generate + review) | Drafts the briefing and pauses for sign-off; edit loop is internal. |
| `write_file_node` | node | Writes the approved briefing to `briefings/`. |

**Request flow.** `START` fans out to `research` and `tech_stack` in parallel. Both finish, then the graph generates questions and pauses at `review_questions_node`. On `ok` it enters the `briefing` subgraph, which drafts and pauses again. On `ok` there, `write_file_node` saves the `.md` and the graph reaches `END`. At either pause, a "fresh data" request routes back to the search subgraphs and then returns to the phase that asked (tracked by `search_return`).

In one sentence: *`research` and `tech_stack` investigate in parallel (each firing its searches in parallel); when you approve the questions, the `briefing` subgraph drafts the document on its own and pauses for your sign-off.*

[↑ Back to top](#table-of-contents)

---

## Key design decisions

1. **Generate and review are separate nodes.** In LangGraph, when the graph resumes after `interrupt()`, the paused node re-runs from the top. If the LLM call lived before the pause, it would regenerate content on every resume and what you approved would not be what gets saved. So one node **generates** (calls the LLM) and another **reviews** (only pauses and reads state). On resume, nothing is regenerated.

2. **Intent-routed feedback.** A small classifier (`_classify_feedback`, one LLM call) decides whether your request is `edit` (rewrite with existing data — cheap, no search), `research`, or `tech_stack` (search again). The search nodes store a `search_return` so the graph knows which phase to come back to afterwards — the questions or the briefing.

3. **Subgraphs communicate by state-key name.** `research`, `tech_stack`, and `briefing` are compiled graphs used as nodes inside the parent. Keys that exist in **both** parent and child cross at entry/exit; keys that exist only in the child (like `search_results`) stay internal. They're compiled **without a checkpointer** so they inherit the parent's — which is why the `interrupt()` living inside the `briefing` subgraph bubbles up to the top-level `invoke()` exactly like the questions one does.

4. **Map-reduce with `Send` for parallel search.** Each search subgraph doesn't run its queries in a row:
   - **MAP** — a fan-out function returns a list of `Send(...)`, one per query; LangGraph spawns N copies of the `search_one` worker and runs them at once.
   - **REDUCE** — every worker writes to `search_results`, a channel with an `Annotated[list[str], operator.add]` reducer that **concatenates** instead of overwriting. On fan-in, a single `synthesize` node summarizes with the LLM.
   - Each subgraph declares an `output_schema` (only its result key) so that when `research` and `tech_stack` run in the same superstep, they don't both try to write shared keys (`company`, …) and trip LangGraph's `InvalidUpdateError`.

[↑ Back to top](#table-of-contents)

---

## Proving the subgraphs and map-reduce actually run

A normal run produces the same output a sequential version would, so "a briefing came out" proves nothing. `inspect_run.py` runs the graph with `stream(..., subgraphs=True)` and prints each node with its namespace:

```powershell
python inspect_run.py "Vercel"
```

What to look for in the output:

- `search_one` **×3** under `('research:...')` and **×3** under `('tech_stack:...')` → the `Send` fan-out happened (6 workers). If their events appear **interleaved** across the two namespaces, that's evidence they ran in parallel.
- `synthesize` **×1** per namespace → the reduce fan-in.
- **Non-empty namespaces** (`('briefing:...')`, etc.) vs. `ns=()` for the parent's own nodes → the subgraphs really ran.

[↑ Back to top](#table-of-contents)

---

## Technology stack

**Language & runtime**
- Python 3.10+

**Orchestration**
- LangGraph (`StateGraph`, subgraphs, `Send` map-reduce, `interrupt()` / `Command`, `MemorySaver` checkpointer)

**Model & search**
- Azure OpenAI / Azure AI Foundry via `langchain-openai` (`AzureChatOpenAI`, e.g. `gpt-4o`, `temperature=0.3`)
- Tavily web search via `langchain-tavily` (`TavilySearch`, `max_results=3`)

**Config**
- `python-dotenv` for loading `.env`

[↑ Back to top](#table-of-contents)

---

## Requirements

- Python 3.10+
- An **Azure OpenAI / Foundry** resource with a chat-model *deployment* (e.g. `gpt-4o`)
- A **Tavily API key** (https://tavily.com)

---

## Local setup

```powershell
# 1. Clone
git clone https://github.com/JorgePulgar/interview-preparation-agent.git
cd interview-preparation-agent

# 2. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate          # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Configuration (environment variables)

Copy the template and fill in your real keys:

```powershell
copy .env.example .env               # Windows
# cp .env.example .env                # macOS / Linux
```

Edit `.env`:

```dotenv
# --- Azure OpenAI (Azure AI Foundry) ---
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/   # or .services.ai.azure.com
AZURE_OPENAI_API_KEY=...
OPENAI_API_VERSION=2024-10-21        # the API version, NOT the model version
AZURE_OPENAI_DEPLOYMENT=gpt-4o       # the DEPLOYMENT NAME, not the model name

# --- Tavily (web search) ---
TAVILY_API_KEY=tvly-...
```

> ⚠️ Two common mistakes:
> - `OPENAI_API_VERSION` is the **API** version (`2024-10-21`), not the model version (`2024-11-20`). Use whatever your *deployment* shows.
> - `AZURE_OPENAI_DEPLOYMENT` is the **name you gave the deployment** in Foundry, which may differ from the model name.
>
> `.env` is in `.gitignore`: **never commit your keys.**

---

## Usage

```powershell
python interview_agent.py "Stripe"
```

If you pass no name, it defaults to `Stripe`. The agent pauses in the console; reply `ok` or type what to change. When it finishes, it prints the path of the generated `.md` in `briefings/`.

---

## Project structure

```
.
├── interview_agent.py   # the agent (parent graph + subgraphs + map-reduce) — commented in Spanish
├── inspect_run.py       # runs with stream(subgraphs=True) to expose subgraphs and map-reduce
├── requirements.txt     # dependencies
├── .env.example         # environment-variable template
├── graph.drawio         # graph diagram (editable in draw.io)
├── graph.png            # diagram image (exported from graph.drawio)
└── briefings/           # output: generated .md files (git-ignored)
```

---

## Known limitations

- **The feedback classifier uses the LLM**, so it's reliable but not perfect. If an ambiguous request is misread, rephrase it (e.g. *"search again..."* to force a re-search).
- **`MemorySaver` keeps state in memory** — it's lost when the program exits. A persistent setup would swap in a database-backed checkpointer.
- **Code comments are in Spanish.** The source is heavily commented as a teaching artifact for a Spanish-speaking team; the README is bilingual but the inline comments are not.
- **No automated test suite yet.** Correctness of the graph wiring is verified manually via `inspect_run.py` rather than with `pytest`.

[↑ Back to top](#table-of-contents)
