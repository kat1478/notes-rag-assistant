# Run from the backend directory:  python -m app.ingest_corpus --corpus v2
# Rebuilds one corpus' Chroma collection from its notes directory (*.txt only; README files are skipped).

import argparse
import logging

from .chunking import chunk_text
from .config import CORPORA
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
    total = 0
    for path in sorted(notes_dir.glob("*.txt")):
        chunks = chunk_text(path.read_text(encoding="utf-8"), path.name)
        upsert_chunks(chunks, path.name)
        total += len(chunks)
    logger.info(f"Corpus {corpus}: {total} chunks in collection '{collection}'")
    return total


def main():
    parser = argparse.ArgumentParser(description="Rebuild the Chroma collection of an evaluation corpus.")
    parser.add_argument("--corpus", choices=sorted(CORPORA), required=True)
    ingest(parser.parse_args().corpus)


if __name__ == "__main__":
    main()
