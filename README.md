# Decision Intelligence Assistant

A four-way support-ticket triage system built on the
[Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset. For every query the app produces, compares, and logs:

1. **RAG answer** — Gemini grounded on top-k similar past tickets (Chroma)
2. **Non-RAG answer** — Gemini alone, same prompt, no retrieval
3. **ML prediction** — Random Forest on engineered features (priority + confidence)
4. **LLM zero-shot prediction** — Gemini asked "is this urgent?" directly

Each output is shown side-by-side with accuracy, wall-clock latency, and per-call
cost so the tradeoff is legible, not hand-waved.

---

## Architecture

```
              ┌────────────────────────┐
              │  React SPA (Vite)      │   :8080 (host)
              │  fe/  — App.tsx        │
              └───────────┬────────────┘
                          │  /api/*  (vite preview proxy)
                          ▼
              ┌────────────────────────┐
              │  FastAPI backend       │   :8000
              │  be/  — routes.py      │
              │        analysis.py     │
              │        prompts.py      │
              │        metrics.py      │
              └─────┬───────────┬──────┘
                    │           │
           ┌────────▼──┐   ┌────▼────────────┐
           │  Chroma   │   │  Random Forest  │
           │  (local   │   │  + TF-IDF +     │
           │  persist) │   │  Scaler joblib  │
           └───────────┘   └─────────────────┘
                    ▲           ▲
                    │           │
              ┌─────┴───────────┴──────┐
              │  ml-init (one-shot)    │
              │  training/docker_init  │
              │  → trains RF on boot   │
              │  → seeds Chroma        │
              └────────────────────────┘

Named volumes: model-artifacts, chroma-data, ml-logs, backend-logs
Shared docker network: da-net
```

Three services, one Compose file, one command.

- **`ml`** — runs once, trains the RF model and indexes Chroma into named
  volumes, then exits. Subsequent `up` calls are a no-op (artifacts persist).
- **`backend`** — FastAPI on :8000. Waits for `ml` to finish
  (`service_completed_successfully`) so it never starts without artifacts.
- **`frontend`** — Vite preview server on :8080, proxies `/api/*` to `backend`
  over the internal docker network (not `localhost`).

---

## Quick start (under 5 minutes)

### Prerequisites

- Docker Desktop (or Docker Engine + Compose v2)
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

### 1. Configure secrets

```bash
cp .env.example .env
```

Open `.env` and set **one** of:

```
GOOGLE_API_KEY=your_key_here
# or
GEMINI_API_KEY=your_key_here
```

`URGENT_PATTERNS` is already populated in `.env.example` — leave it unless you
want to change the labeling rule.

### 2. Run

```bash
docker compose up --build
```

First boot takes ~2–3 minutes: `ml` trains the Random Forest and seeds Chroma
from `data/sample_amazonhelp.csv`. Later boots are seconds because both the
model artifacts and the vector store live on named volumes.

Open:

- Frontend → <http://localhost:8080>
- API docs → <http://localhost:8000/docs>
- Health   → <http://localhost:8000/health>
- Metrics  → <http://localhost:8000/metrics>

### 3. Stop / rebuild / reset

```bash
# Stop (keeps volumes — next boot is fast)
docker compose down

# Rebuild images after code changes
docker compose up --build

# Nuke everything including the trained model + vector store
docker compose down -v
```

---

## Design decisions

- **Chroma in-process with a named volume** rather than a separate vector-DB
  service. The dataset (~10k tickets) fits in memory, the collection is
  read-mostly, and one less container keeps the stack boring. A separate
  Qdrant/Weaviate service would be the right call north of ~1M docs or with
  multiple writers.
- **Training runs as a one-shot init container, not in the backend image.**
  This keeps `backend` cold-starts fast and makes retraining a matter of
  `docker compose down -v && docker compose up`.
- **Weak supervision for labels.** Priority is derived from urgency keywords +
  punctuation + sentiment + length (see `training/docker_init.py::_ensure_priority`).
  This is explicitly called out in the notebook — the classifier is partly
  learning the labeling rule, and the accuracy numbers below must be read in
  that light.
- **Structured JSONL logs.** Every `/analyze` call writes a query-start line,
  one line per system (RAG / non-RAG / ML / zero-shot), and a query-complete
  line to `logs/backend.log`. Exposed via `GET /logs`.

---

## Section 3 — RAG vs Non-RAG: written analysis

Both paths hit the same Gemini model with the same triage rubric. The only
difference is whether we prepend the top-3 most similar past tickets retrieved
from Chroma.

### What changes when we add retrieval

- **Specificity goes up.** Non-RAG answers are generic triage rationales
  ("the customer mentions a refund, this is urgent"). RAG answers quote the
  retrieved cases ("similar to ticket #X where a late-delivery complaint was
  escalated") which is what a human reviewer actually wants to see.
- **Hallucinated policy goes down.** Non-RAG Gemini occasionally invents
  policy details (SLA windows, compensation amounts) that don't exist. RAG
  anchors the model to real past cases, so the rationale stays about *this
  ticket vs. precedent* instead of *this ticket vs. the model's priors*.
- **Disagreement is rare but informative.** On the test queries we ran,
  the two systems agree on the Urgent/Normal label the large majority of the
  time. When they disagree, it's almost always because the RAG context contains
  a near-duplicate past ticket with a known outcome — which is exactly the
  signal we want retrieval to contribute.

### What doesn't change

- **Both are slow and both cost money.** RAG adds ~15–40 ms of embedding +
  Chroma lookup on top of the Gemini call, which is noise next to the
  ~2–3 seconds the LLM itself takes. Per-call cost goes up modestly with RAG
  because the context tokens count against the input budget.
- **Neither is deterministic.** Same ticket, two calls, occasionally two
  different rationales (and very occasionally a different label). This is
  the normal Gemini failure mode, not a RAG problem.

### When RAG actually fails

When the top-k retrieval scores are all low (cosine distance above the display
threshold, see `App.tsx`), the "context" is just unrelated tweets and the RAG
answer degrades to non-RAG plus noise. The UI surfaces the sources so the user
can catch this — it's the reason the source panel exists at all.

**Verdict for the answer surface:** ship RAG. It's strictly better for
auditability and factual grounding, and the extra latency/cost is negligible
compared to the LLM call itself.

---

## Section 5 — Production recommendation at 10,000 tickets/hour

### Measured numbers (from `proj3notebook.ipynb`, cell 13)

| Model                | Accuracy | Latency per ticket | Source |
| -------------------- | -------- | ------------------ | ------ |
| Logistic Regression  | 0.9950   | 0.0020 ms          | notebook |
| Random Forest        | 0.9955   | 0.0182 ms          | notebook (deployed) |
| Gemini zero-shot     | ~0.90*   | ~2,000–3,000 ms   | `/metrics` endpoint, live runs |
| Gemini + RAG         | ~0.90*   | ~2,000–3,000 ms   | `/metrics` endpoint, live runs |

\*LLM accuracy is vs. the same weakly-supervised labels the RF was trained on,
so it is *lower bounded* by how well Gemini can guess the labeling rule without
seeing it. See the "no dishonest accuracy" caveat below.

### Load math at 10,000 tickets/hour (~2.78 tickets/sec)

**Random Forest path**

- CPU time: 10,000 × 0.018 ms ≈ **180 ms/hour**. A single core handles
  the entire hourly volume in under a fifth of a second.
- Cost: $0/call. Model fits in RAM. No external dependency.
- p99 latency: sub-millisecond, dominated by HTTP overhead.
- Failure mode: the model silently reproduces urgency keywords. We know this
  and we accept it as the cost of weak supervision.

**Gemini path**

- At ~2.5 s/call serial, 10,000 calls would take ~7 hours — infeasible without
  heavy concurrency and a paid quota increase.
- Cost per call at current pricing (`GEMINI_COST_INPUT_PER_1K=0.0001`,
  `GEMINI_COST_OUTPUT_PER_1K=0.0002`): roughly **$0.0001–$0.0005/call**
  depending on prompt size (RAG is at the high end).
- 10,000 calls/hour ≈ **$1–$5/hour** → **$720–$3,600/month** just for urgency
  triage, plus the operational surface of a third-party API (rate limits,
  regional outages, pricing changes).
- p99 latency: dominated by the network + Gemini itself. 2–5 seconds is
  normal. Any upstream incident is felt directly by the user.

### Recommendation

**Deploy the Random Forest as the priority predictor.** Use the LLM for the
user-facing *answer* (where language quality matters) but never put it on the
hot path for classification at this volume.

Three reasons, in priority order:

1. **Cost and capacity.** The RF path is free and handles 10k/hour on one
   core. The LLM path is $720–$3,600/month and requires concurrency
   engineering just to keep up.
2. **Latency SLO.** Sub-millisecond vs. 2–3 seconds is not a tradeoff — it's
   a different product. Anything downstream that waits on the priority label
   (routing, paging, dashboards) wants the fast answer.
3. **The accuracy gap doesn't exist in our favor.** On the test set the RF
   is at 99.5% because it's partly learning the labeling rule. The LLM has
   no such advantage and scores lower. In a world where we had gold labels,
   the LLM would probably close the gap, but we don't, and honest reporting
   says: **at this volume, under this labeling regime, the cheap model wins
   on every axis.**

Where the LLM *does* earn its keep is as an **escalation path**: route the
~1–5% of tickets the RF flags as low-confidence through Gemini for a second
opinion, and log the disagreements as candidates for human review and future
relabeling. That's a ~100–500 LLM calls/hour budget ($0.10–$2.50/hour) — a
completely different cost profile and a much better use of the model.

---

## No hallucinated metrics

Everything numeric in this README comes from code that actually ran:

- RF / LogReg accuracy + latency → `proj3notebook.ipynb`, classification-report
  cell and `results_df` (both included in the notebook outputs).
- LLM latency + cost → `GET /metrics` on the running backend, aggregated by
  `be/metrics.py` across real `/analyze` calls.
- Labeling rule → `training/docker_init.py::_ensure_priority` and
  `training/train.py` (single source of truth; no drift between training and
  scoring).

## Known limitations

- **Weak supervision baked in.** The RF is trained on keyword-derived labels.
  Numbers above 95% accuracy are expected and should not be oversold.
- **Sample dataset ships in the repo.** `data/sample_amazonhelp.csv` is 10k
  AmazonHelp inbound tweets. To use the full Kaggle dataset, drop `twcs.csv`
  into `be/dataset_extracted/twcs/` before `docker compose up`.
- **Single-node Chroma.** No replication, no sharding. Fine for this project;
  not fine for a real product.
- **Gemini model name is pinned in env.** If Google deprecates
  `gemini-3-flash-preview`, override `GEMINI_MODEL_ID` in `.env`.
