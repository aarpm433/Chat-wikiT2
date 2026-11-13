import os
import re
import hashlib
import sqlite3

import sqlite_vec
from sqlite_vec import serialize_float32
from fastembed import TextEmbedding
from llama_cpp import Llama

# ------------------------------------------------------------------
# PATH & ENVIRONMENT CONFIGURATION
# ------------------------------------------------------------------
def find_project_root(start):
    """Walk up from `start` until a directory containing wiki_data is found."""
    d = start
    while True:
        if os.path.isdir(os.path.join(d, "wiki_data")):
            return d
        parent = os.path.dirname(d)
        if parent == d:          # reached filesystem root
            return start         # fall back to the script's folder
        d = parent

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = find_project_root(SCRIPT_DIR)

WIKI_DIR = os.path.join(PROJECT_ROOT, "wiki_data")
DB_PATH = os.path.join(PROJECT_ROOT, "chat_wikiT_v2.sqlite") 
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")

EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # 384 dimensions
EMBED_DIM = 384
CHUNK_WORDS = 250
CHUNK_OVERLAP = 50
TOP_N = 4

_embedder = None


def get_embedder():
    """Lazy-load the embedding model (downloads ~130MB on first run)."""
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name=EMBED_MODEL)
    return _embedder


def embed_documents(texts):
    """Embeds a list of passages in batches; returns sqlite-vec blobs."""
    return [serialize_float32(v.tolist()) for v in get_embedder().embed(texts, batch_size=32)]


def embed_query(text):
    """Embeds a search query (BGE uses a special query prefix, handled by query_embed)."""
    vec = next(iter(get_embedder().query_embed(text)))
    return serialize_float32(vec.tolist())


def connect(path):
    conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


# ==========================================
# 1. DATABASE & INGESTION PIPELINE
# ==========================================

def chunk_text(text, title="", chunk_size=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    """Sliding-window word chunks. The page title is prepended to every
    chunk after the first so each chunk keeps its context."""
    words = text.split()
    step = max(1, chunk_size - overlap)
    chunks = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if not chunk.strip():
            continue
        if i > 0 and title:
            chunk = f"[{title}] {chunk}"
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks


def setup_database(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS file_states (
            filepath TEXT PRIMARY KEY,
            md5_hash TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            filepath TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(filepath)")
    # Both virtual tables use rowid == chunk_id so they can be joined cheaply.
    conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS wiki_fts USING fts5(text)")
    conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS wiki_vectors USING vec0(embedding float[{EMBED_DIM}])")
    conn.commit()


def delete_file_chunks(conn, filepath):
    ids = [r[0] for r in conn.execute("SELECT chunk_id FROM chunks WHERE filepath = ?", (filepath,))]
    for cid in ids:
        conn.execute("DELETE FROM wiki_fts WHERE rowid = ?", (cid,))
        conn.execute("DELETE FROM wiki_vectors WHERE rowid = ?", (cid,))
    conn.execute("DELETE FROM chunks WHERE filepath = ?", (filepath,))


def ingest_wiki_data(db_path, wiki_directory):
    print(f"Checking wiki files in '{wiki_directory}' for ingestion...")
    if not os.path.exists(wiki_directory):
        os.makedirs(wiki_directory, exist_ok=True)
        print(f"Created empty directory at '{wiki_directory}'. Add .md files there.")
        return

    conn = connect(db_path)
    setup_database(conn)

    paths = []
    for root, _, files in os.walk(wiki_directory):
        for file in files:
            if file.endswith((".md", ".txt")):
                paths.append(os.path.join(root, file))

    # Remove data for files that no longer exist on disk
    on_disk = set(paths)
    for (fp,) in conn.execute("SELECT filepath FROM file_states").fetchall():
        if fp not in on_disk:
            delete_file_chunks(conn, fp)
            conn.execute("DELETE FROM file_states WHERE filepath = ?", (fp,))
    conn.commit()

    ingested = 0
    for n, filepath in enumerate(paths, start=1):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        current_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
        row = conn.execute("SELECT md5_hash FROM file_states WHERE filepath = ?", (filepath,)).fetchone()
        if row and row[0] == current_hash:
            continue

        first_line = content.split("\n", 1)[0]
        title = first_line.lstrip("# ").strip() if first_line.startswith("#") else os.path.splitext(os.path.basename(filepath))[0]

        chunks = chunk_text(content, title=title)
        if not chunks:
            continue

        vectors = embed_documents(chunks)  # one batched call per file

        delete_file_chunks(conn, filepath)  # avoid duplicates when a file changed
        for chunk, vec in zip(chunks, vectors):
            cur = conn.execute("INSERT INTO chunks (filepath) VALUES (?)", (filepath,))
            cid = cur.lastrowid
            conn.execute("INSERT INTO wiki_fts (rowid, text) VALUES (?, ?)", (cid, chunk))
            conn.execute("INSERT INTO wiki_vectors (rowid, embedding) VALUES (?, ?)", (cid, vec))

        conn.execute("INSERT OR REPLACE INTO file_states (filepath, md5_hash) VALUES (?, ?)",
                     (filepath, current_hash))
        conn.commit()  # commit per file so an interrupted run can resume
        ingested += 1
        if ingested % 25 == 0:
            print(f"  ...{ingested} files ingested ({n}/{len(paths)} scanned)")

    conn.close()
    print(f"Ingestion complete. Processed {ingested} new/updated file(s).")


# ==========================================
# 2. HYBRID RETRIEVAL ENGINE
# ==========================================

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "of", "to", "in", "on", "for", "and", "or",
    "what", "which", "who", "how", "do", "does", "i", "me", "my", "it", "its", "with", "that",
    "this", "can", "you", "get", "from", "at", "by", "be", "as",
}


def sanitize_fts_query(query_string):
    """Builds a safe FTS5 query: quoted terms joined with OR, stopwords removed."""
    words = re.findall(r"\w+", query_string.lower())
    keep = [w for w in words if w not in STOPWORDS] or words
    return " OR ".join(f'"{w}"' for w in keep)


def reciprocal_rank_fusion(lexical_results, semantic_results, k=60):
    scores, by_id = {}, {}
    for results in (lexical_results, semantic_results):
        for rank, r in enumerate(results, start=1):
            by_id[r["id"]] = r
            scores[r["id"]] = scores.get(r["id"], 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [by_id[i] for i, _ in ordered]


def hybrid_search(conn, query_string, top_n=TOP_N):
    # --- BM25 lexical search ---
    lexical = []
    fts_query = sanitize_fts_query(query_string)
    if fts_query:
        try:
            rows = conn.execute(
                "SELECT rowid, text FROM wiki_fts WHERE wiki_fts MATCH ? ORDER BY rank LIMIT 20",
                (fts_query,),
            ).fetchall()
            lexical = [{"id": r[0], "text": r[1]} for r in rows]
        except sqlite3.OperationalError:
            pass

    # --- Vector KNN search (vec0 table has no text column, so fetch it from FTS) ---
    qvec = embed_query(query_string)
    knn = conn.execute(
        "SELECT rowid, distance FROM wiki_vectors WHERE embedding MATCH ? AND k = 20 ORDER BY distance",
        (qvec,),
    ).fetchall()
    semantic = []
    if knn:
        ids = [r[0] for r in knn]
        placeholders = ",".join("?" * len(ids))
        text_by_id = dict(conn.execute(
            f"SELECT rowid, text FROM wiki_fts WHERE rowid IN ({placeholders})", ids
        ).fetchall())
        semantic = [{"id": i, "text": text_by_id[i]} for i in ids if i in text_by_id]

    final = reciprocal_rank_fusion(lexical, semantic)[:top_n]

    # Exact-phrase safety net
    if lexical and query_string.lower() in lexical[0]["text"].lower():
        if lexical[0]["id"] not in {c["id"] for c in final}:
            final = ([lexical[0]] + final)[:top_n]
    return final


# ==========================================
# 3. QUERY & STREAMING RESPONSE
# ==========================================

def ask(query_string, llm):
    conn = connect(DB_PATH)
    try:
        context_chunks = hybrid_search(conn, query_string)
        for i, c in enumerate(context_chunks, 1):
            print(f"\n[DEBUG Source {i}] {c['text'][:300]}...")
    finally:
        conn.close()

    if not context_chunks:
        print("No relevant context found in the database.\n")
        return

    context_text = "\n\n".join(f"[Source {i}]\n{c['text']}" for i, c in enumerate(context_chunks, 1))

    messages = [
        {"role": "system", "content": (
            "You are a helpful wiki assistant. Answer using only the context provided. "
            "If the context does not contain the answer, that's too vague.\n\n"
            f"Context:\n{context_text}"
        )},
        {"role": "user", "content": query_string},
    ]

    print("\n--- Answer ---")
    stream = llm.create_chat_completion(messages=messages, max_tokens=512, temperature=0.2, stream=True)
    for part in stream:
        delta = part["choices"][0]["delta"].get("content")
        if delta:
            print(delta, end="", flush=True)
    print("\n")


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    ingest_wiki_data(DB_PATH, WIKI_DIR)

    print(f"Loading language model from '{MODEL_PATH}'...")
    llm_instance = Llama(
        model_path=MODEL_PATH,
        n_ctx=8192,
        n_gpu_layers=-1, # use Metal on Apple Silicon
        verbose=False,
    )
    print("Model ready.\n")

    while True:
        try:
            user_query = input("Ask Chat-wikiT (or 'exit' to quit): ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not user_query:
            continue
        if user_query.lower() in ("exit", "quit"):
            break
        ask(user_query, llm_instance)