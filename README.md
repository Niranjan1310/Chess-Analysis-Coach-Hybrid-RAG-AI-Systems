# ♟️ Chess Analysis Coach: RAG + Stockfish

A chess coach that finds your mistakes with Stockfish and explains why they were bad in plain English, using retrieved chess knowledge and an LLM, with citations.

---

## 🤔 What does it do in simple words?

Paste a PGN (or import games from Lichess). The app:
- **Finds every inaccuracy, mistake, and blunder** using Stockfish.
- **Looks up relevant opening theory, strategy notes, and similar master positions.**
- **Explains each flagged move:** what went wrong, the better move, and one concept to learn.

It doesn't just say "blunder". It tells you what the engine saw, what you should have played, and which idea you were missing, with the sources it used.

---

## 🛡️ How does it avoid making things up?

Chess LLMs are known to invent moves and evaluations. This project is built around preventing that:
- **The LLM never judges move quality.** Stockfish is ground truth and is always passed into the prompt.
- **The LLM only explains.** It may use only the engine data and retrieved context. If context is missing, it says so.
- **Every explanation cites sources** (chunk IDs/titles).
- **Move-faithfulness guard.** Every move in an explanation is parsed with `python-chess` and checked against the engine line, the played move, and the game. Unlisted or illegal moves trigger a retry, then a fallback to an engine-only explanation.
- **Statistics are computed by code, not the LLM.** Weakness rates and similar-position counts come from SQL/Python. The LLM only writes the summary around them.
- **"I don't know" behavior.** If retrieval scores are below a threshold, the chat says the knowledge base has no good answer.

*(For developers, the architecture and setup instructions are below.)*

---

## 🏗️ Architecture

```text
┌────────────────┐
│  PGN / Lichess │
└───────┬────────┘
        ▼
┌────────────────┐      ┌────────────────────┐
│   PGN Parser   │─────▶│  Stockfish (UCI)   │
│  FEN + moves   │      │  eval, best line   │
└───────┬────────┘      └─────────┬──────────┘
        │                         ▼
        │               ┌────────────────────┐
        │               │  Mistake Detector  │
        │               │  cp loss → class   │
        │               └─────────┬──────────┘
        ▼                         ▼
        (flagged moves only)
┌──────────────────────────────────────────┐
│ Position Describer (opening, structure,  │
│ king safety, material, phase)            │
└───────────────────┬──────────────────────┘
                    ▼
┌──────────────────────────────────────────┐
│ Hybrid Retriever                         │
│ Chroma vectors + BM25, RRF fusion        │
└───────────────────┬──────────────────────┘
                    ▼
┌──────────────────────────────────────────┐
│ Generator (Gemini via LLMClient)         │
│ + move-faithfulness check                │
└───────────────────┬──────────────────────┘
                    ▼
┌──────────────────────────────────────────┐
│ FastAPI backend → HTML/JS frontend       │
└──────────────────────────────────────────┘
```

---

## 🌟 Features

### ♞ Game Analysis
- Upload a PGN and get every move classified (inaccuracy / mistake / blunder)
- Mate scores handled correctly (a missed mate counts as a blunder)
- Evals stored from White's point of view; sign flipped for Black
- Full-game analysis runs as a background job (never in the request thread)
- Results cached by `(position, move)` to save LLM calls

### 📚 Hybrid RAG
- Vector search (`BAAI/bge-small-en-v1.5`) + keyword search (BM25), merged with Reciprocal Rank Fusion
- Metadata filters: `source`, `type`, `eco`, `phase`, `rating_band`
- Vector-only, BM25-only, and hybrid modes, so retrieval can be evaluated
- Knowledge base: opening data (ECO), strategy notes, master-game positions

### 🔍 Similar Position Search
- "Positions like this were usually won by doing X"
- Move counts and score % computed by code from indexed master games
- Shows how many positions matched; says "not enough similar positions" below 5

### 📈 Weakness Summary
- Error rates per phase, opening, piece type, and game context, computed in SQL
- Needs a minimum number of games and moves per category before flagging
- Retrieves study material for the top weaknesses

### 💬 Ask-a-Question Chat
- Answers from the knowledge base with citations
- Last 3 to 4 turns kept as memory
- Calls Stockfish for questions about a specific position

---

## 🛠️ Tech Stack

| Layer | Technologies |
|---|---|
| **Backend** | Python 3.11+, FastAPI, Uvicorn, Pydantic v2 |
| **Chess** | python-chess, Stockfish (UCI) |
| **Vector DB** | Chroma (persistent, local) |
| **Embeddings** | sentence-transformers `BAAI/bge-small-en-v1.5` |
| **Keyword search** | rank-bm25 |
| **LLM** | Gemini API behind a thin `LLMClient` wrapper |
| **Database** | SQLite via SQLAlchemy |
| **Background jobs** | FastAPI `BackgroundTasks` |
| **Frontend** | Plain HTML + CSS + vanilla JS, chessboard.js + chess.js from CDN |
| **Tests** | pytest |

---

## 🚀 How to Run

### Prerequisites
- Python 3.11+ with pip
- [Stockfish](https://stockfishchess.org/download/) (any x86-64 build)
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)

### Step 1: Install Backend Dependencies
```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

### Step 2: Configure .env
Create a `.env` file in `backend/` and fill in:
```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-3.8-flash # check ai.google.dev for current model names
STOCKFISH_PATH=C:/path/to/stockfish.exe
STOCKFISH_DEPTH=14
INACCURACY_THRESHOLD=50
MISTAKE_THRESHOLD=100
BLUNDER_THRESHOLD=300
```
> [!NOTE]
> *Never commit `.env`.*

### Step 3: Build the Knowledge Base (once)
From the repo root:
```powershell
python -m ingestion.load_openings --data-dir data/openings --persist-dir data/chroma
python -m ingestion.load_articles --data-dir data/notes --persist-dir data/chroma
python -m ingestion.build_position_index --persist-dir data/chroma
```
*The embedding model downloads on first use.*

### Step 4: Start the Server
```powershell
cd backend
python -m uvicorn app.main:app --reload --port 8000
```
- App: http://localhost:8000
- API docs: http://localhost:8000/docs

### Step 5: Run the Tests
```powershell
cd backend
pytest -v
```
The real-Stockfish test is skipped unless `STOCKFISH_PATH` is set in the shell:
```powershell
$env:STOCKFISH_PATH = "C:/path/to/stockfish.exe"
pytest -v -k mate_in_one
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/analyze` | Start analysis of a PGN, returns a `job_id` |
| GET | `/api/analyze/{job_id}` | Job status and result |
| POST | `/api/similar` | Similar master positions and statistics |
| GET | `/api/weaknesses?player=<name>` | Player weakness summary and study recommendations |
| POST | `/api/chat` | Ask a chess question with RAG citations |
| GET | `/api/health` | Health check |

### Example: Analysis result (illustrative)
```json
{
  "move_number": 14,
  "side": "black",
  "played": "Bxh2+",
  "classification": "blunder",
  "eval_before_cp": -20,
  "eval_after_cp": 380,
  "best_move": "Be7",
  "best_line": ["Be7", "Re1", "O-O"],
  "explanation": "Taking on h2 loses a piece because the bishop is trapped after Kxh2 ...",
  "concept": "Greek gift sacrifice requires a follow-up attacker",
  "sources": ["notes/greek-gift#2", "eco/C00"]
}
```

---

## 📁 Project Structure

```text
chess-rag/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── api/          # route modules (analysis, chat, similar, weakness, games)
│   │   ├── core/         # engine, pgn_parser, mistake_detector, position_describer
│   │   ├── rag/          # embeddings, vector_store, retriever, prompts, generator
│   │   ├── services/     # analysis_service, lichess_client, weakness_service
│   │   ├── models/       # SQLAlchemy models
│   │   └── schemas/      # Pydantic schemas
│   ├── tests/
│   └── pyproject.toml
├── ingestion/            # knowledge-base loaders
├── data/                 # gitignored
├── frontend/             # index.html, style.css, app.js
├── evals/                # rag_eval.py, eval_set.json
├── docker-compose.yml
└── README.md
```

---

## 🔬 How Analysis Works

1. `pgn_parser` splits the game into positions (FEN + move + move number).
2. `engine` evaluates each position at the configured depth.
3. `mistake_detector` converts mate scores to large centipawn values and classifies each move by centipawn loss.
4. For flagged moves only, `position_describer` writes a text description of the position.
5. `retriever` runs hybrid search and returns the top-k chunks.
6. `generator` builds the prompt from the engine data, the position description, and the chunks.
7. The LLM output is validated move by move. Failures retry once, then fall back to an engine-only explanation.
8. The result is saved and returned as JSON.

---

## 📊 Evaluation

All numbers below are empirically measured by running `python -m evals.rag_eval` against the benchmark dataset in `evals/eval_set.json`.

### 1. Retrieval Ablation Benchmarks

Comparing the individual retrieval components against the unified hybrid fusion system:

| Retrieval Architecture | Recall@5 | MRR | nDCG@5 | Latency (p50) | Latency (p95) | Key Characteristic |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **Dense Vector (Baseline)** | 1.000 | 0.792 | 0.817 | 5.48 ms | 11.08 ms | Captures conceptual similarity; misses exact chess notation tokens |
| **Sparse BM25 Only** | 0.900 | 0.867 | 0.859 | 0.41 ms | 0.97 ms | Exact keyword precision (ECO codes, moves); misses conceptual rephrasing |
| **Hybrid (RRF Fusion)** | **1.000** | **0.935** | **0.926** | **6.96 ms** | **9.21 ms** | **+18.1% MRR & +13.3% nDCG@5 gain over dense baseline** |

### 2. Classification & Guardrail Metrics (Precision / Recall / F1)

Treating safety and mistake detection as classification tasks:

| Component / Task | Precision | Recall | F1-Score | Accuracy | Evaluation Details |
|---|:---:|:---:|:---:|:---:|---|
| **Out-of-Scope Refusal Guardrail** | **1.000** | **1.000** | **1.000** | **100.0%** | Tested across 16 queries (8 non-chess vs 8 in-scope chess queries) |
| **Move Mistake Classifier** | **1.000** | **1.000** | **1.000 (Macro)**| **100.0%** | Evaluated on Blunder, Mistake, Inaccuracy, and Good centipawn loss |
| **Move Faithfulness Guardrail** | — | — | — | **100.0%** | Deterministic `python-chess` legal validation on generated candidate moves |

### 3. System & Generation Metrics

- **Citation Rate:** `100.0%` of coach explanations contain direct chunk citations (e.g. `[opening_B12]`, `[notes/pawn_structure]`).
- **Engine Analysis Speed:** ~45 ms per move at Stockfish depth 14 (~2.5s for a 40-move game).
- **Asynchronous Execution:** Background workers (`FastAPI BackgroundTasks`) ensure API response time remains `< 50ms`.

---

## 🧭 Build Status

| Phase | Scope | Status |
|---|---|:---:|
| 1 | Backend skeleton, Stockfish wrapper, PGN parser, mistake detector | ✅ |
| 2 | Position describer, ingestion, Chroma store, hybrid retriever | ✅ |
| 3 | Prompts, generator, analysis service, `/api/analyze` | ✅ |
| 4 | Basic frontend (chessboard.js + step-through analysis) | ✅ |
| 5 | Game history, Lichess import, weakness summary | ✅ |
| 6 | Similar position search | ✅ |
| 7 | Ask-a-question chat | ✅ |
| 8 | Evals, Docker, deployment documentation | ✅ |

*151/151 unit and integration tests passing.*

---

## 🐳 Docker Deployment

Run the complete stack (FastAPI server + embedded Stockfish + static frontend) with Docker Compose:

```bash
docker-compose up --build
```
Navigate to `http://localhost:8000`.

---

## ⚠️ Limitations

- Explanations are only as good as the retrieved knowledge base and the engine depth used.
- Similar-position statistics show what strong players tended to do, not what is objectively best. Engine eval remains the ground truth.
- Evaluation sets are small and hand-built. Results are indicative, not definitive.

---

## 📜 Data Sources and Licenses

- Opening data: [lichess-org/chess-openings](https://github.com/lichess-org/chess-openings)
- Master games and puzzles: Lichess open database
- Strategy notes: self-written or public domain

*Respect each source's license. No copyrighted books are scraped.*
