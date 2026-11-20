"""Exact contexts of benchmark answers, stored next to the results.

The judge must grade an answer against the chunks the generator actually saw. Re-retrieving later is only the same
context if chunking, top_k, ordering and embeddings are all unchanged, so each benchmark row records a hash of its
context and the chunks themselves go into a content-addressed store (one line per distinct context: every model in
a run shares the same context per question).
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List

from .schemas import ChunkResponse

STORE_NAME = "contexts.jsonl"


def context_hash(chunks: List[ChunkResponse]) -> str:
    """Order matters: [Source N] numbers follow the prompt order."""
    raw = json.dumps([[c.metadata.chunk_id, c.text] for c in chunks], ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def load_store(results_dir: Path) -> Dict[str, List[ChunkResponse]]:
    path = Path(results_dir) / STORE_NAME
    if not path.exists():
        return {}
    store = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            entry = json.loads(line)
            store[entry["hash"]] = [ChunkResponse(**c) for c in entry["chunks"]]
    return store


def save_contexts(results_dir: Path, contexts: Dict[str, List[ChunkResponse]]) -> Dict[str, str]:
    """Appends contexts not yet in the store; returns question id → context hash."""
    path = Path(results_dir) / STORE_NAME
    known = set(load_store(results_dir))
    hashes = {qid: context_hash(chunks) for qid, chunks in contexts.items()}
    new = {h: contexts[qid] for qid, h in hashes.items() if h not in known}
    if new:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for h, chunks in new.items():
                f.write(json.dumps({"hash": h, "chunks": [c.model_dump() for c in chunks]}, ensure_ascii=False) + "\n")
    return hashes
