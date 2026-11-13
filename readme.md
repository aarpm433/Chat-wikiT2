# Chat-wikiT: Local Architectural Blueprint & Execution Engine

**Chat-wikiT** is a privacy-first, zero-cloud Conversational Knowledge Engine designed to run high-precision Retrieval-Augmented Generation (RAG) over structured wiki datasets entirely on resource-constrained edge hardware.

By co-locating ultra-lightweight embedded vector tables, dependency-free C-runtime embedding models, and aggressively quantized Small Language Models (SLMs) within a unified process space, Chat-wikiT achieves sub-second token generation and context retrieval under strict system memory budgets (4 GB to 8 GB total RAM).

## Architecture Overview

```
                          [ Wiki Data Source ]
                         (terraria.wiki.gg, etc.)
                                    │
                                    ▼
                         [ MediaWiki Scraper ]
                         (Incremental MD5 Check)
                                    │
                                    ▼
                         [ Structured Markdown ]
                           (./wiki_data/*.md)
                                    │
                                    ▼
                     [ Ingestion Pipeline & Chunker ]
                       (400 W / 80 Overlap Windows)
                                    │
       ┌────────────────────────────┴────────────────────────────┐
       ▼                                                         ▼
[ FTS5 Indexing ]                                     [ fasttextembed C Engine ]
(BM25 Lexical Tokens)                                (BGE-Small-EN 384-d Vectors)
       │                                                         │
       └────────────────────────────┬────────────────────────────┘
                                    ▼
                      [ Unified SQLite Database ]
                         (chat_wikiT.sqlite)
                                    │
 ┌──────────────────────────────────┴──────────────────────────────────┐
 │                                                                     │
 ▼                                                                     ▼
[ User Query ] ──► [ Lexical BM25 (FTS5) ]        [ Semantic KNN (sqlite-vec) ]
                        (Top 20 Chunks)                 (Top 20 Vectors)
                               │                               │
                               └───────────────┬───────────────┘
                                               ▼
                                 [ Reciprocal Rank Fusion ]
                                        (RRF k=60)
                                               │
                                               ▼
                                  [ Exact-Match Fallback ]
                                 (Force Key Technical Terms)
                                               │
                                               ▼
                                   [ Context Window Assembly ]
                                               │
                                               ▼
                                   [ llama.cpp SLM Engine ]
                                (Llama-3.2-3B / 8-bit KV Cache)
                                               │
                                               ▼
                                   [ Real-Time Stream Token ]

```

## Key Performance Features

1. **Zero-Infrastructural Storage Overhead**

   Bypasses client-server vector database servers (such as Pinecone, Chroma server, or Milvus) in favor of a single unified SQLite file using `sqlite-vec` and `FTS5`.

2. **C-Runtime Embedding Acceleration**

   Employs `fasttextembed` hardcoded for BGE-Small-EN-v1.5 (384-dimensional vectors). Avoids PyTorch or ONNX dependencies, reducing memory overhead to $\sim 90\text{ MB}$ of RAM.

3. **Hybrid RRF Search & Exact-Match Guardrail**

   Executes dense vector similarity search and sparse BM25 lexical keyword matching in parallel, combining ranks via Reciprocal Rank Fusion ($k=60$). An exact-match safety check preserves numeric item IDs, craft recipes, and specialized terminology.

4. **Quantized Memory Footprint Management**

   Configured specifically for `llama.cpp` using 8-bit quantized Key-Value (KV) caching (`type_k="q8_0"`, `type_v="q8_0"`), capping active context memory overhead and preventing system thrashing on 4 GB–8 GB consumer hardware.

## Memory Allocation Targets

| Subsystem Component | 4 GB Profile (Strict Budget) | 8 GB Profile (Performance Balanced) | 
| ----- | ----- | ----- | 
| **Operating System & Base OS Overhead** | 1.00 GB | 1.50 GB | 
| **Embedding Engine (`fasttextembed`)** | 0.09 GB ($\sim 90\text{ MB}$) | 0.09 GB ($\sim 90\text{ MB}$) | 
| **Vector & Scalar Storage Index** | 0.05 GB (sqlite-vec) | 0.20 GB (sqlite-vec / LanceDB) | 
| **SLM Foundation Weights** | 1.80 GB (Llama 3.2 1B / Qwen3 1.7B Q4_K_M) | 2.50 GB (Llama 3.2 3B Q4_K_M / Qwen2.5 3B Q5_K_M) | 
| **Active KV Cache (Context Buffer)** | 0.50 GB ($4\text{K}$ tokens, Q8_0 KV) | 1.20 GB ($8\text{K}$ tokens, Q8_0 KV) | 
| **Unallocated Buffer / System Headroom** | 0.56 GB | 2.51 GB | 

## Retrieval Mathematics

### Reciprocal Rank Fusion (RRF)

To merge lexical scoring (unbounded BM25 scores) and vector distances (bounded cosine metrics) without scale normalization distortion, Chat-wikiT implements Reciprocal Rank Fusion:

$$
RRF(d \in D) = \sum_{m \in M} \frac{1}{k + r_m(d)}
$$

Where:

* $M$ is the set of retrieval systems (FTS5 lexical and sqlite-vec semantic).

* $r_m(d)$ represents the 1-based ordinal rank position of document chunk $d$ in retrieval system $m$.

* $k$ is set to $60$ to minimize the influence of extreme rank outliers.

### Exact-Match Fallback Logic

If document $d$ contains an exact substring match for query string $q$ in the top FTS5 lexical result set, it is forcibly injected into ordinal position 1 of the final context array $C$:

$$
\text{if } q \subseteq \text{text}(d_{\text{top\_lexical}}) \implies C_1 \leftarrow d_{\text{top\_lexical}}
$$

## Project Structure

```
Chat-WikiT/
├── wiki_data/                 # Raw markdown extracts generated by scraper
├── models/                    # Quantized GGUF model storage directory
│   └── Llama-3.2-3B-Instruct-Q4_K_M.gguf
├── chat_wikiT.sqlite          # Unified SQLite database (FTS5 + vec0 virtual tables)
├── scrape_terraria_wiki.py    # MediaWiki API scraping script with auto-retry
├── chat_wikit.py              # Main ingestion, RRF hybrid search, and SLM engine
├── requirements.txt           # Python dependency declarations
└── README.md                  # System architecture documentation

```

## Quick Start & Installation

### 1. Prerequisites & System Setup

Ensure Python 3.10+ is installed on your machine. Clone the repository and initialize your environment:

```
git clone https://github.com/your-username/Chat-WikiT.git
cd Chat-WikiT

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

```

### 2. Install Dependencies

#### Standard CPU Installation:

```
pip install -r requirements.txt

```

#### Apple Silicon (Metal Acceleration) Installation:

For fast token generation on Apple M-Series chips, compile `llama-cpp-python` with Metal support:

```
CMAKE_ARGS="-DGGML_METAL=on" pip install --force-reinstall llama-cpp-python --no-cache-dir
pip install sqlite-vec fasttextembed requests

```

#### NVIDIA GPU (CUDA Acceleration) Installation:

```
CMAKE_ARGS="-DGGML_CUDA=on" pip install --force-reinstall llama-cpp-python --no-cache-dir
pip install sqlite-vec fasttextembed requests

```

## Model Acquisition

Chat-wikiT uses GGUF-formatted small language models.

### Download Meta Llama 3.2 3B Instruct (Recommended)

Run `curl` from inside your project directory to download the 4-bit quantized model weights directly:

```
mkdir -p models
curl -L -o models/Llama-3.2-3B-Instruct-Q4_K_M.gguf "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf?download=true"

```

*Expected File Size:* `~2.02 GB`

## Execution Workflow

### Step 1: Scrape Wiki Knowledge Base

Populate `./wiki_data` with structured Markdown content using the MediaWiki API pipeline:

```
python terraria_ingest.py

```

*Features:* Uses script-relative paths, automatically skips previously downloaded pages, and handles transient network drops with backoff retries.

### Step 2: Ingest & Query

Execute the main application pipeline:

```
python chat_wikit.py

```

Upon launch:

1. **Directory Scanning:** Automatically converts `.md` files in `./wiki_data/` into 400-word sliding chunks with 80-word overlaps.

2. **State Indexing:** Computes MD5 content hashes to bypass unchanged files on subsequent launches.

3. **Database Population:** Populates FTS5 full-text indices and `sqlite-vec` 384-d vector embeddings in `chat_wikiT.sqlite`.

4. **SLM Initialization:** Loads model weights into RAM once and starts an interactive prompt:

```
Checking wiki files for ingestion...
Ingestion complete. Processed 0 updated file(s).
Loading language model into RAM...
Model ready.

Ask Chat-wikiT (or 'exit' to quit): What ingredients are required to craft the Zenith sword?

--- Answer ---
To craft the Zenith in Terraria, you must combine 10 different swords at a Mythril or Orichalcum Anvil:
1. Terra Blade
2. Meowmere
3. Star Wrath
4. Influx Waver
5. The Horseman's Blade
6. Seedler
7. Starfury
8. Bee Keeper
9. Enchanted Sword
10. Copper Shortsword

```

## Configuration & Tuning Parameters

All primary execution settings can be adjusted inside `chat_wikit.py`:

```
# System Paths
DB_PATH = "chat_wikiT.sqlite"
MODEL_PATH = "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
WIKI_DIR = "./wiki_data"

# Chunking Dynamics
CHUNK_SIZE = 400    # Sliding window token/word target
CHUNK_OVERLAP = 80  # Boundary retention overlap

# RRF Search Settings
RRF_K = 60          # Rank smoothing parameter
TOP_N_CONTEXT = 5   # Maximum context chunks injected into prompt

# Llama-cpp Memory Caching
CONTEXT_SIZE = 4096 # Active context token window limit
KV_QUANT = "q8_0"   # 8-bit quantized Key-Value cache format

```

## Troubleshooting & FAQ

### 1. FTS5 Query Syntax Operational Error

* **Issue:** Queries containing special characters (hyphens, quotes, or colons) crash SQLite FTS5.

* **Fix:** Chat-wikiT includes `sanitize_fts_query()`, which wraps individual query tokens in escaped quotes to prevent parser exceptions.

### 2. Slow Second Query Execution

* **Issue:** System spends several seconds reloading weights on every prompt.

* **Fix:** Ensure the `Llama(...)` instance is loaded once in `__main__` and passed to `ask(query, llm)`, keeping model memory warm.

### 3. Out-Of-Memory (OOM) Errors

* **Fix:** Lower `CONTEXT_SIZE` from `4096` to `2048` or switch base models to Meta Llama 3.2 1B (`Llama-3.2-1B-Instruct-Q4_K_M.gguf`).

## License

Distributed under the MIT License. See `LICENSE` for details.