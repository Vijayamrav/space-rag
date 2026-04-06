# AstroRAG

AstroRAG is a research assistant for space science. You point it at a topic, it goes and finds real academic papers, reads them, and then lets you have a conversation about what they say. Every answer it gives is grounded in the actual papers it has read, with citations back to the source so you can verify anything it tells you.

It is not a chatbot that makes things up. If the papers don't cover something, it will tell you that.

---

## The idea behind it

Most AI assistants answer questions from their training data, which means the answers are frozen in time and you have no idea where the information came from. AstroRAG works differently. You feed it papers first, then ask questions. The answers come directly from those papers, with the exact source cited inline. If you ask about something the papers don't cover, it says so rather than guessing.

The system pulls papers from arXiv, Semantic Scholar, and OpenAlex simultaneously, ranks them by citation quality and recency, downloads the full PDFs, reads them, and stores the content in a vector database. When you ask a question, it searches that database using a hybrid of semantic similarity and keyword matching, then passes the most relevant passages to an LLM with strict instructions to only use what it was given.

---

## What you need before starting

- Python 3.11 or higher
- Node.js 18 or higher
- A free Pinecone account at [pinecone.io](https://pinecone.io) — you need an API key and you need to note your region
- An OpenRouter account at [openrouter.ai](https://openrouter.ai) — you need an API key, and a small amount of credits (queries are cheap, typically fractions of a cent each)

---

## Getting it running

### 1. Get the code

```bash
git clone <your-repo-url>
cd astrorag
```

### 2. Set up Python

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Mac or Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Set up your environment variables

Create a `.env` file in the root of the project with the following:

```env
OPENROUTER_API_KEY=your_openrouter_key_here
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_INDEX_NAME=space-rag
PINECONE_ENVIRONMENT=us-east-1
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
EMBEDDING_DIM=768
```

The Pinecone index will be created automatically when you first start the server. You do not need to create it manually.

### 4. Set up the frontend

```bash
cd frontend
npm install
```

### 5. Start the backend

From the project root:

```bash
uvicorn app.main:app --reload --port 8000
```

### 6. Start the frontend

From the `frontend` folder:

```bash
npm run dev
```

Open your browser at `http://localhost:5173` and you should see the interface.

---

## Using it for the first time

The system starts empty. Before you can ask questions, you need to give it some papers to read. This is called ingestion.

The easiest way to get started is to ingest by topic category. For example, to load papers about general relativity:

```bash
curl -X POST http://localhost:8000/ingest/categories \
  -H "Content-Type: application/json" \
  -d '{"categories": ["gr-qc"], "max_results": 5}'
```

This will go fetch papers, download their PDFs, read them, and store everything in Pinecone. Watch the server logs to see it working. Once it finishes, go to the UI and start asking questions.

You can also ingest by free-text query if you have something specific in mind:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"query": "james webb telescope exoplanet atmospheres", "max_results": 10}'
```

---

## The available paper categories

When ingesting by category, these are the arXiv categories the system knows about:

| Category | Topic |
|---|---|
| astro-ph.EP | Exoplanets |
| astro-ph.HE | High Energy Astrophysics — black holes, neutron stars |
| astro-ph.GA | Galaxies and Quasars |
| astro-ph.CO | Cosmology and Dark Matter |
| astro-ph.SR | Stellar and Solar Physics |
| astro-ph.IM | Instrumentation and Methods |
| gr-qc | General Relativity and Quantum Cosmology |
| hep-ph | High Energy Physics |

---

## How the search works

When you ask a question, the system does two things at once. It converts your question into a vector and searches Pinecone for semantically similar passages (dense search). At the same time it does a keyword-based BM25 search against the same index (sparse search). The results from both are blended together — by default 75% semantic, 25% keyword — and the top matches are passed to the LLM.

This hybrid approach means it handles both conceptual questions ("what is the role of dark matter in galaxy formation") and specific technical queries ("what did paper 1905.00038 say about the Hubble constant") reasonably well.

---

## Asking questions via the API directly



```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How does JWST detect molecules in exoplanet atmospheres?", "top_k": 5}'
```

The response includes the answer, the model used, and the source chunks with arXiv IDs and relevance scores so you can trace every claim back to its paper.

---

## Configuration reference

Everything is controlled through environment variables. The defaults are sensible but you can override any of them.

| Variable | Default | What it does |
|---|---|---|
| OPENROUTER_API_KEY | required | Your OpenRouter key |
| OPENROUTER_BASE_URL | https://openrouter.ai/api/v1 | OpenRouter endpoint |
| MODEL_NAME | meta-llama/llama-3.1-8b-instruct | The LLM used for generating answers |
| PINECONE_API_KEY | required | Your Pinecone key |
| PINECONE_INDEX_NAME | space-rag | Name of the vector index |
| PINECONE_ENVIRONMENT | us-east-1 | Pinecone region |
| EMBEDDING_MODEL | sentence-transformers/all-mpnet-base-v2 | Model used to embed text |
| EMBEDDING_DIM | 768 | Must match the embedding model output |
| CHUNK_SIZE | 512 | Characters per text chunk |
| CHUNK_OVERLAP | 64 | Overlap between chunks |
| TOP_K | 5 | Chunks retrieved per query |

---

## Note

Ingestion takes time. Downloading PDFs, extracting text, embedding hundreds of chunks — it is not instant. For 5 papers expect a few minutes. For 50 papers expect longer. The server logs will show you exactly what is happening.

Only arXiv papers get fully processed. Papers from Semantic Scholar or OpenAlex that do not have an arXiv ID are skipped at the PDF download stage. Those sources are still useful because they contribute citation counts used for ranking.

The Pinecone index is created automatically. If it does not exist when the server starts, it gets created with the right settings. If you ever need to change the embedding model or dimension, you will need to delete the index from the Pinecone console and let it recreate — the dimension is fixed at creation time.

The LLM will not make things up. The system prompt explicitly instructs it to only use the retrieved passages and to flag anything the context does not cover. If you ask about something that has not been ingested, it will tell you rather than hallucinate an answer.

