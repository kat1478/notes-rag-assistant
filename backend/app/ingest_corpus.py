# Run from the backend directory:  python -m app.ingest_corpus --corpus v2
# Rebuilds one corpus' Chroma collection from its notes directory (*.txt, *.md; README files are skipped).
# For the corpus the app searches (CHROMA_COLLECTION) it also rebuilds the TF-IDF fallback (data/processed/chunks.json).

import argparse
import logging

from .chunking import chunk_text
from .config import CHROMA_COLLECTION, CORPORA
from .storage import replace_chunks
from .vector_store import get_client, get_collection, upsert_chunks, use_collection

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def ingest(corpus: str) -> int:
    notes_dir, collection = CORPORA[corpus]
    use_collection(collection)
    try:
        get_client().delete_collection(collection)
    except Exception:
        pass  # first ingest: nothing to delete
    get_collection()
    all_chunks = []
    files = sorted(p for p in notes_dir.iterdir() if p.suffix in (".txt", ".md") and not p.name.lower().startswith("readme"))
    for path in files:
        chunks = chunk_text(path.read_text(encoding="utf-8"), path.name)
        upsert_chunks(chunks, path.name)
        all_chunks.extend(chunks)
    if collection == CHROMA_COLLECTION:
        replace_chunks(all_chunks)  # keep the TF-IDF fallback on the same corpus as the vector store
        logger.info("Rebuilt the TF-IDF fallback corpus (data/processed/chunks.json)")
    logger.info(f"Corpus {corpus}: {len(all_chunks)} chunks from {len(files)} files in collection '{collection}'")
    return len(all_chunks)


def main():
    parser = argparse.ArgumentParser(description="Rebuild the Chroma collection of an evaluation corpus.")
    parser.add_argument("--corpus", choices=sorted(CORPORA), required=True)
    ingest(parser.parse_args().corpus)


if __name__ == "__main__":
    main()
