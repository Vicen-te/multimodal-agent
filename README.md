---
title: Multimodal Agent
emoji: 🤖
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.22.0
app_file: app.py
pinned: false
license: mit
short_description: Multimodal agent over images and docs (RAG + vision)
tags:
  - llm-agent
  - langgraph
  - rag
  - multimodal
  - computer-vision
  - ollama
  - evaluation
---

# Multimodal Agent

![CI](https://github.com/Vicen-te/multimodal-agent/actions/workflows/ci.yml/badge.svg)

A multimodal LLM agent that reasons over **images** and **documentation**. Upload
an image, ask a question, or both, and the agent decides whether to analyze the
image, search the docs, or both, then reflects on its answer before sending it.

> **Live demo:** deploy your own in minutes with the [Hugging Face Spaces](#deploy-to-hugging-face-spaces) steps below.

It combines three areas in one system:

- **LLM agents** — a [LangGraph](multimodal_agent/agent/graph.py) ReAct loop with a reflection step.
- **Computer vision** — a [vision tool](multimodal_agent/providers/vision.py) backed by a local multimodal model (Ollama).
- **RAG** — [hybrid retrieval](multimodal_agent/rag/retriever.py) (dense + BM25) fused with reciprocal rank fusion.

## Architecture

```
            User (image + text)
                    |
            Gradio frontend (app.py)
                    | stream
            LangGraph agent  <-----------------+
                    |                           |
        +-----------+-----------+               |
        |                       |               |
   Vision tool             RAG tool             |
   (Ollama VL)        (hybrid + RRF)            |
        |                       |               |
        +-----------+-----------+               |
                    |                           |
                 Reflection node ---------------+
              (critique -> revise or finish)
```

The agent's reasoning stays in **text**: the image is handed to a vision tool
that returns a description, which flows back into the loop like any other tool
result. This keeps the loop easy to debug and lets vision compose with document
search.

## How it works

1. The user message (text and/or image) becomes the initial graph state. The
   image is stored as base64 and never enters the text model's context directly.
2. The **agent node** asks the text model (`qwen3.5`) what to do. It can call
   `AnalyzeImage`, `SearchDocs`, both, or answer directly.
3. The **tool node** executes the calls: `AnalyzeImage` runs the vision model on
   the stored image; `SearchDocs` runs hybrid retrieval and returns citable
   chunks.
4. When the agent produces a final answer that relied on a tool, the
   **reflection node** critiques it *against the tool outputs*. First a
   deterministic check rejects any citation that was not actually retrieved (and
   any search whose result went uncited) with no extra model call; then, if that
   passes, the model reviews faithfulness against the retrieved sources and image
   analysis. It either finishes or routes back for one grounded revision. Plain
   conversational replies skip reflection.
5. Tokens stream to the Gradio UI.

## Run locally

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) running locally with the models pulled:

```bash
ollama pull qwen3.5:4b      # text / routing / reflection model
ollama pull qwen3-vl:8b     # vision model
```

> Model names are configurable (see `.env.example`). Any Ollama vision model
> works, e.g. `llava` or `moondream`; set `VISION_MODEL` accordingly.

### Install and launch

```bash
python -m venv .venv
. .venv/Scripts/activate        # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
cp .env.example .env            # then edit if needed

python app.py
```

The embedding model (`all-MiniLM-L6-v2`) downloads from Hugging Face on first
run. The corpus is ingested in memory at startup, so the first launch takes a
little longer.

## Evaluation

A small eval set lives in [data/eval/cases.jsonl](data/eval/cases.jsonl) covering
vision-only, docs-only, both, and no-tool cases, plus adversarial ones where the
right answer is to refuse: a question the corpus does not cover, an image that
was never attached, and a detail that is absent from the picture. A set where
every case is answerable cannot tell a grounded agent from a confident one. Two
negative controls go further: they are impossible by design -- a house drawing
whose image is deliberately withheld, a deployment guide the corpus does not
contain -- and are reported outside the headline metrics, where failing is the
correct outcome and a passing control would flag a lenient judge. It
measures **tool routing
accuracy** (did the agent pick the right tool?), **exact routing** (did it also
avoid calling tools it did not need?), **answer quality** (LLM-as-judge against a
rubric), and **citation integrity** (does every cited id come from a document the
run actually retrieved?). Quality and citations move independently: a rubric
rewards content, and an answer can be correct while citing nothing.

```bash
python data/eval/make_images.py             # generate the sample images
python -m multimodal_agent.evals.run_evals  # run the agent over every case
```

Results are written to `eval_results.md` as a table with routing accuracy,
average answer score, and reflection rate. Two flags open up the run:

```bash
python -m multimodal_agent.evals.run_evals --detailed  # every answer in full
python -m multimodal_agent.evals.run_evals --compare   # what reflection changed
```

`--detailed` adds each answer with the sources it retrieved, the critique that
triggered any revision, and the judge's verdict. `--compare` also grades the
draft the agent wrote *before* reflection — reflection only runs after an answer
exists, so that draft is what the run would have returned with the node off, and
scoring both sides measures the node instead of assuming it helps.

### Results

Over the 20 cases, with `qwen3.5:4b` and `qwen3-vl:8b` on Ollama and reflection
enabled:

| Metric | Result |
|--------|--------|
| Cases | 20 (6 docs, 6 image, 3 both, 2 no-tool, 3 adversarial) |
| Routing accuracy | 95% |
| Exact routing (no extra tools) | 90% |
| Average answer score | 5.00 / 5 |
| Citation integrity | 20/20 clean, from 17/20 before reflection |
| Answers reviewed / rewritten | 90% / 15% |
| Negative controls | 2/2 failing, as designed |

The three adversarial cases are the ones worth reading. Asked how many people are
in a bar chart, the agent answers that there are none rather than inventing them.
Asked about LoRA, which the corpus does not cover, its first draft wandered and
never said so; the citation check forced a revision and the rewrite states
plainly that no retrieved source mentions LoRA, which is the whole difference
between a grounded agent and a confident one. The third is a genuine failure: an
image the user says they attached but did not, where the agent calls the vision
tool anyway on the strength of the claim, then recovers and answers correctly.

That failure is why routing is also measured exactly, and the run shows the same
pattern elsewhere -- a docs question that picked up a spare vision call. The
reflection node's other contribution is citations: three answers used a search
without citing it, and rewriting them took citation integrity from 17/20 to
20/20. Every one of those rewrites came from the deterministic citation check,
which costs no model call; the LLM reviewer read the remaining drafts and asked
for no revision at all.

## Testing

The agent logic, RAG, and reflection loop are covered by offline unit tests that
stub the models, so they run with no Ollama and no model downloads:

```bash
pip install -r requirements-dev.txt
pytest
```

These same light dependencies run in CI on every push and pull request
(see [.github/workflows/ci.yml](.github/workflows/ci.yml)).

## Configuration

All settings come from environment variables (see [.env.example](.env.example)):

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_PROVIDER` | `ollama` | Chat and vision backend: `ollama`, `gemini`, or `openai` (any OpenAI-compatible API) |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama endpoint |
| `TEXT_MODEL` | `qwen3.5:4b` | Routing / reflection / answer model |
| `VISION_MODEL` | `qwen3-vl:8b` | Vision-language model |
| `GOOGLE_API_KEY` | empty | Gemini key, free from aistudio.google.com |
| `GEMINI_TEXT_MODEL` | `gemini-3.6-flash` | Gemini text model |
| `GEMINI_VISION_MODEL` | `gemini-3.6-flash` | Gemini vision model |
| `OPENAI_API_KEY` | empty | Key for the OpenAI-compatible backend |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Endpoint that picks the provider (Groq, OpenRouter, Mistral...) |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model for text and vision on that backend |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence-Transformers embedder |
| `RAG_TOP_K` | `4` | Parent chunks returned per search |
| `RRF_K` | `60` | Reciprocal rank fusion constant |
| `ENABLE_REFLECTION` | `true` | Toggle the reflection node |
| `MAX_REFLECTIONS` | `1` | Max revision loops |
| `LANGFUSE_*` | empty | Optional tracing (auto-enabled when keys set) |

## Deploy to Hugging Face Spaces

1. Create a new Space, SDK **Gradio** (the free tier works; no GPU needed).
2. Push this repository (the entry point is `app.py`).
3. In the Space settings add the secret `GOOGLE_API_KEY` and the variable
   `LLM_PROVIDER=gemini`: Ollama is not available on a Space, so the hosted demo
   runs on the Gemini backend instead.
4. Spaces installs `requirements.txt` and serves the app.

Visitors can also bring their own key in the UI ("Use your own API key or
model"): a Gemini key, or a key for any OpenAI-compatible provider (OpenAI,
Groq, OpenRouter, Mistral, DeepSeek, xAI, Anthropic) picked by base URL, with
the model of their choice. The key is used only for their requests and never
stored, and since every Gemini model has its own free quota, switching model is
another way to keep the demo working when one runs dry.

## Project structure

```
multimodal_agent/
  config.py            # settings from env
  providers/           # ollama chat + vision, sentence-transformers embedder
  rag/                 # chunking, hybrid store, RRF retriever, ingest, corpus
  agent/               # state, tool schemas, prompts, graph, runner
  evals/               # dataset, LLM judge, harness
app.py                 # Gradio frontend (HF Spaces entry point)
data/eval/             # eval cases + sample image generator
tests/                 # offline unit tests (stubbed models)
COMMITS.md             # the commit-by-commit build guide
```

## Stack

LangGraph · Qwen3-VL / Qwen3.5 (Ollama) · Sentence-Transformers · rank-bm25 ·
Gradio · Langfuse · Hugging Face Spaces

## License

MIT — see [LICENSE](LICENSE).
