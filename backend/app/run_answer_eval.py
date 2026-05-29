# Run this script from the backend directory using: python -m app.run_answer_eval

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import List

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from .evaluation import load_evaluation_dataset, run_answer_evaluation
from .config import DATA_DIR

def save_results_to_csv(results: List[dict], output_path: Path):
    """Saves answer evaluation results to CSV file matching the required schema."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question_id", "question", "category", "should_answer", "expected_file",
        "answer_text", "retrieved_files", "source_reference_present",
        "answered_when_expected", "correct_refusal", "answer_correct",
        "answer_grounded", "hallucination", "notes", "retrieval_mode",
        "generation_mode", "top_k_used", "response_time_ms", "used_fallback",
        "retrieved_chunk_ids"
    ]
    
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            writer.writerow({k: res.get(k, "") for k in fieldnames})
            
    logger.info(f"Saved answer evaluation results to {output_path}")

def calculate_and_print_metrics(results: List[dict]):
    """Calculates and prints summary statistics for answer quality evaluation."""
    total = len(results)
    if total == 0:
        logger.warning("No results to summarize.")
        return
        
    # Mode splits
    ollama_count = sum(1 for r in results if r["generation_mode"] == "ollama")
    fallback_count = sum(1 for r in results if r["generation_mode"] == "extractive_fallback")
    
    # Grounded / In-scope queries
    grounded_queries = [r for r in results if r["should_answer"] == "True"]
    grounded_total = len(grounded_queries)
    grounded_answered = sum(1 for r in grounded_queries if r["answered_when_expected"] == "True")
    
    # Out-of-scope queries
    oos_queries = [r for r in results if r["should_answer"] == "False"]
    oos_total = len(oos_queries)
    oos_refused = sum(1 for r in oos_queries if r["correct_refusal"] == "True")
    
    # Citations (Source reference present)
    citation_present = sum(1 for r in results if r["source_reference_present"] == "True")
    
    # Response times
    avg_time = sum(r["response_time_ms"] for r in results) / total
    
    avg_ollama_time = 0.0
    if ollama_count > 0:
        avg_ollama_time = sum(r["response_time_ms"] for r in results if r["generation_mode"] == "ollama") / ollama_count
        
    avg_fallback_time = 0.0
    if fallback_count > 0:
        avg_fallback_time = sum(r["response_time_ms"] for r in results if r["generation_mode"] == "extractive_fallback") / fallback_count
        
    print("\n" + "="*80)
    print("  ANSWER QUALITY EVALUATION SUMMARY METRICS (top_k={})".format(results[0]["top_k_used"]))
    print("="*80)
    print(f"Total queries evaluated:            {total}")
    print("-" * 80)
    print(f"Generation Mode Breakdown:")
    print(f"  - LLM (Ollama):                   {ollama_count} ({ollama_count/total:.1%})")
    print(f"  - Extractive Fallback:            {fallback_count} ({fallback_count/total:.1%})")
    print("-" * 80)
    print(f"Grounded (In-Scope) Queries:")
    print(f"  - Total grounded queries:         {grounded_total}")
    if grounded_total > 0:
        print(f"  - Answered as expected:           {grounded_answered} ({grounded_answered/grounded_total:.1%})")
    print("-" * 80)
    print(f"Out-of-Scope Queries:")
    print(f"  - Total out-of-scope queries:     {oos_total}")
    if oos_total > 0:
        print(f"  - Correctly refused:              {oos_refused} ({oos_refused/oos_total:.1%})")
    print("-" * 80)
    print(f"Source Citations (Answer text citation check):")
    print(f"  - Answers with citations present:  {citation_present} ({citation_present/total:.1%})")
    print("-" * 80)
    print(f"Latency / Performance:")
    print(f"  - Average overall response time:  {avg_time:.1f} ms")
    if ollama_count > 0:
        print(f"  - Average LLM generation time:    {avg_ollama_time:.1f} ms")
    if fallback_count > 0:
        print(f"  - Average fallback run time:      {avg_fallback_time:.1f} ms")
    print("="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Run answer quality evaluation for local RAG stack.")
    parser.add_argument(
        "--dataset-path", 
        type=str, 
        default=str(DATA_DIR / "eval" / "evaluation_dataset.json"),
        help="Path to the evaluation dataset JSON file."
    )
    parser.add_argument(
        "--output-path", 
        type=str, 
        default=str(DATA_DIR / "eval" / "answer_eval_results.csv"),
        help="Path to save the answer evaluation results CSV."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Value of k to use for semantic retrieval."
    )
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    output_path = Path(args.output_path)
    top_k = args.top_k
    
    logger.info(f"Loading dataset from {dataset_path}")
    try:
        dataset = load_evaluation_dataset(dataset_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
        
    logger.info(f"Loaded {len(dataset)} evaluation queries. Starting answer evaluation with top_k={top_k}...")
    
    results = run_answer_evaluation(dataset, top_k)
    
    save_results_to_csv(results, output_path)
    calculate_and_print_metrics(results)

if __name__ == "__main__":
    main()
