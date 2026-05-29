# Run this script from the backend directory using: python -m app.run_retrieval_eval

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import List

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from .evaluation import load_evaluation_dataset, run_retrieval_evaluation
from .config import DATA_DIR
from .chunking import chunk_text
from .vector_store import get_client, get_collection, COLLECTION_NAME, upsert_chunks

def reset_and_reingest_notes():
    """Deletes existing collection in Chroma and ingests all student notes."""
    notes_dir = DATA_DIR / "notes"
    if not notes_dir.exists() or not notes_dir.is_dir():
        logger.error(f"Notes directory not found at {notes_dir}")
        sys.exit(1)
        
    client = get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info(f"Deleted Chroma collection '{COLLECTION_NAME}'")
    except Exception as e:
        logger.info(f"No existing collection '{COLLECTION_NAME}' to delete or delete failed: {e}")
        
    # Recreate collection
    get_collection()
    logger.info(f"Created new Chroma collection '{COLLECTION_NAME}'")
    
    # List and ingest all txt/md files
    notes_files = list(notes_dir.glob("*.txt")) + list(notes_dir.glob("*.md"))
    if not notes_files:
        logger.warning(f"No .txt or .md files found in {notes_dir}")
        return
        
    for file_path in notes_files:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        file_name = file_path.name
        chunks = chunk_text(text, file_name)
        if chunks:
            upsert_chunks(chunks, file_name)
            logger.info(f"Ingested {file_name}: created {len(chunks)} chunks")
        else:
            logger.warning(f"No chunks created for {file_name}")

def save_results_to_csv(results: List[dict], output_path: Path):
    """Saves evaluation results to CSV file matching the required schema."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question_id", "question", "category", "expected_file", "top_k",
        "retrieved_files", "retrieved_chunk_ids", "retrieval_hit",
        "first_hit_correct", "num_results_returned", "notes"
    ]
    
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            writer.writerow({k: res.get(k, "") for k in fieldnames})
            
    logger.info(f"Saved retrieval evaluation results to {output_path}")

def calculate_metrics(results: List[dict]):
    """Calculates and prints summary metrics for retrieval evaluation."""
    # Group results by top_k
    by_k = {}
    for res in results:
        k = res["top_k"]
        if k not in by_k:
            by_k[k] = []
        by_k[k].append(res)
        
    print("\n" + "="*80)
    print("  RETRIEVAL EVALUATION SUMMARY METRICS")
    print("="*80)
    
    # Table headers
    print(f"{'top_k':<5} | {'category':<25} | {'total_queries':<13} | {'hits':<5} | {'hit_rate':<8} | {'first_hit_correct':<17}")
    print("-" * 80)
    
    for k in sorted(by_k.keys()):
        k_results = by_k[k]
        
        # Filter queries where expected_file is specified
        valid_queries = [r for r in k_results if r["expected_file"]]
        total = len(valid_queries)
        if total == 0:
            continue
            
        # Overall metrics for this k
        hits = sum(1 for r in valid_queries if r["retrieval_hit"] == "True")
        first_correct = sum(1 for r in valid_queries if r["first_hit_correct"] == "True")
        
        hit_rate = hits / total
        first_correct_rate = first_correct / total
        
        print(f"k={k:<3} | {'OVERALL (grounded only)':<25} | {total:<13} | {hits:<5} | {hit_rate:.2%} | {first_correct_rate:.2%}")
        
        # Group by category
        by_cat = {}
        for r in valid_queries:
            cat = r["category"]
            if cat not in by_cat:
                by_cat[cat] = []
            by_cat[cat].append(r)
            
        for cat, cat_res in sorted(by_cat.items()):
            cat_total = len(cat_res)
            cat_hits = sum(1 for r in cat_res if r["retrieval_hit"] == "True")
            cat_first_correct = sum(1 for r in cat_res if r["first_hit_correct"] == "True")
            
            cat_hit_rate = cat_hits / cat_total
            cat_first_correct_rate = cat_first_correct / cat_total
            
            print(f"k={k:<3} | {cat:<25} | {cat_total:<13} | {cat_hits:<5} | {cat_hit_rate:.2%} | {cat_first_correct_rate:.2%}")
        print("-" * 80)

def main():
    parser = argparse.ArgumentParser(description="Run retrieval evaluation for semantic search.")
    parser.add_argument(
        "--reset-and-reingest", 
        action="store_true", 
        help="Reset the Chroma collection and re-ingest all student notes before evaluating."
    )
    parser.add_argument(
        "--dataset-path", 
        type=str, 
        default=str(DATA_DIR / "eval" / "evaluation_dataset.json"),
        help="Path to the evaluation dataset JSON file."
    )
    parser.add_argument(
        "--output-path", 
        type=str, 
        default=str(DATA_DIR / "eval" / "retrieval_eval_results.csv"),
        help="Path to save the evaluation results CSV."
    )
    
    args = parser.parse_args()
    
    if args.reset_and_reingest:
        logger.info("Resetting collection and re-ingesting student notes...")
        reset_and_reingest_notes()
        
    dataset_path = Path(args.dataset_path)
    output_path = Path(args.output_path)
    
    logger.info(f"Loading dataset from {dataset_path}")
    try:
        dataset = load_evaluation_dataset(dataset_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
        
    logger.info(f"Loaded {len(dataset)} evaluation queries.")
    
    # We evaluate for top_k values 1, 3, and 5
    top_k_values = [1, 3, 5]
    logger.info(f"Running retrieval evaluation for top_k={top_k_values}")
    
    results = run_retrieval_evaluation(dataset, top_k_values)
    
    save_results_to_csv(results, output_path)
    calculate_metrics(results)

if __name__ == "__main__":
    main()
