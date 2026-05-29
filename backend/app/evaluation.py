import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from .schemas import ChunkResponse
from .retrieval import perform_search
from .config import DATA_DIR

logger = logging.getLogger(__name__)

def load_evaluation_dataset(dataset_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Loads the evaluation dataset from the specified path or the default location."""
    if dataset_path is None:
        dataset_path = DATA_DIR / "eval" / "evaluation_dataset.json"
        
    if not dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at {dataset_path}")
        
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_retrieval_evaluation_for_item(
    item: Dict[str, Any], 
    top_k: int
) -> Dict[str, Any]:
    """Runs semantic search retrieval for a single evaluation item and computes metrics."""
    question_id = item.get("id", "")
    question = item.get("question", "")
    category = item.get("category", "")
    expected_file = item.get("expected_file", "")
    notes = item.get("notes", "")
    
    # Run the retriever using perform_search (uses Chroma by default)
    # This executes semantic retrieval without invoking LLM or /ask logic
    results: List[ChunkResponse] = perform_search(question, top_k=top_k, use_semantic=True)
    
    retrieved_files = [chunk.metadata.file_name for chunk in results]
    retrieved_chunk_ids = [chunk.metadata.chunk_id for chunk in results]
    num_results_returned = len(results)
    
    # Calculate retrieval_hit and first_hit_correct
    if not expected_file:
        # Out-of-scope or other query type where no file is expected
        retrieval_hit = "N/A"
        first_hit_correct = "N/A"
    else:
        # Check if expected file is in retrieved files
        is_hit = any(f.lower() == expected_file.lower() for f in retrieved_files)
        retrieval_hit = "True" if is_hit else "False"
        
        # Check if first hit is correct
        if retrieved_files:
            is_first_correct = retrieved_files[0].lower() == expected_file.lower()
            first_hit_correct = "True" if is_first_correct else "False"
        else:
            first_hit_correct = "False"
            
    # Format list fields as semicolon or comma-separated strings for CSV readability
    retrieved_files_str = ";".join(retrieved_files)
    retrieved_chunk_ids_str = ";".join(retrieved_chunk_ids)
    
    return {
        "question_id": question_id,
        "question": question,
        "category": category,
        "expected_file": expected_file,
        "top_k": top_k,
        "retrieved_files": retrieved_files_str,
        "retrieved_chunk_ids": retrieved_chunk_ids_str,
        "retrieval_hit": retrieval_hit,
        "first_hit_correct": first_hit_correct,
        "num_results_returned": num_results_returned,
        "notes": notes
    }

def run_retrieval_evaluation(
    dataset: List[Dict[str, Any]], 
    top_k_values: List[int]
) -> List[Dict[str, Any]]:
    """Runs retrieval evaluation on the full dataset for a list of top_k values."""
    eval_results = []
    for top_k in top_k_values:
        for item in dataset:
            res = run_retrieval_evaluation_for_item(item, top_k)
            eval_results.append(res)
    return eval_results
