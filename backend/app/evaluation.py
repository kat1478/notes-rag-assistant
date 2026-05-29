import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from .schemas import ChunkResponse
from .retrieval import perform_search
from .answering import generate_answer
from .config import DATA_DIR
from .vector_store import get_collection

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
            
    # Format list fields as semicolon-separated strings for CSV readability
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

def is_refusal_response(answer_text: str) -> bool:
    """Determines if the answer text contains common refusal phrases."""
    refusal_phrases = [
        "don't have enough information",
        "do not have enough information",
        "insufficient information",
        "nie znaleziono w notatkach informacji",
        "nie mam wystarczających informacji",
        "nie posiadam informacji"
    ]
    answer_lower = answer_text.lower()
    return any(phrase in answer_lower for phrase in refusal_phrases)

def detect_source_ref_in_text(answer_text: str, retrieved_files: List[str]) -> bool:
    """Scans the answer text for source citations (bracketed notation or file names)."""
    answer_lower = answer_text.lower()
    
    # Look for explicit bracketed citations like [Source 1], [1], [source 1]
    if "[source" in answer_lower or "source " in answer_lower or "[1]" in answer_lower or "[2]" in answer_lower or "[3]" in answer_lower:
        return True
        
    # Look for mentions of the retrieved files in the answer text
    for f in retrieved_files:
        if f and f.lower() in answer_lower:
            return True
            
    return False

def run_answer_evaluation_for_item(
    item: Dict[str, Any], 
    top_k: int
) -> Dict[str, Any]:
    """Executes semantic search and LLM answering for a single item, timing and evaluating it."""
    question_id = item.get("id", "")
    question = item.get("question", "")
    category = item.get("category", "")
    expected_file = item.get("expected_file", "")
    should_answer_val = item.get("should_answer", True)
    notes = item.get("notes", "")
    
    start_time = time.time()
    
    # 1. Retrieval
    retrieval_mode = "semantic"
    used_fallback_retrieval = False
    try:
        results = perform_search(question, top_k=top_k, use_semantic=True)
    except Exception:
        results = perform_search(question, top_k=top_k, use_semantic=False)
        retrieval_mode = "tfidf_fallback"
        used_fallback_retrieval = True
        
    # Explicitly check for TF-IDF fallback if collection is empty
    if retrieval_mode == "semantic":
        try:
            coll = get_collection()
            if coll.count() == 0:
                retrieval_mode = "tfidf_fallback"
                used_fallback_retrieval = True
        except Exception:
            retrieval_mode = "tfidf_fallback"
            used_fallback_retrieval = True
            
    retrieved_files = [chunk.metadata.file_name for chunk in results]
    retrieved_chunk_ids = [chunk.metadata.chunk_id for chunk in results]
    
    # 2. Generation
    answer_text = generate_answer(question, results)
    
    end_time = time.time()
    response_time_ms = int((end_time - start_time) * 1000)
    
    # Detect if generation fallback was used (Ollama failure / extractive context printed)
    used_fallback_gen = False
    generation_mode = "ollama"
    if answer_text.startswith("LLM generation failed."):
        used_fallback_gen = True
        generation_mode = "extractive_fallback"
        
    used_fallback = "True" if (used_fallback_retrieval or used_fallback_gen) else "False"
    
    # Calculate source_reference_present on answer_text only
    source_ref_present = "True" if detect_source_ref_in_text(answer_text, retrieved_files) else "False"
    
    # Determine if refusal response
    refused = is_refusal_response(answer_text)
    
    # Compute answered_when_expected (for should_answer=True questions)
    if should_answer_val:
        is_non_empty = bool(answer_text.strip())
        answered_when_expected = "True" if (is_non_empty and not refused) else "False"
    else:
        answered_when_expected = "N/A"
        
    # Compute correct_refusal (for should_answer=False questions)
    if not should_answer_val:
        correct_refusal = "True" if refused else "False"
    else:
        correct_refusal = "N/A"
        
    retrieved_files_str = ";".join(retrieved_files)
    retrieved_chunk_ids_str = ";".join(retrieved_chunk_ids)
    
    # Clean up newlines in answer text for CSV formatting safety
    clean_answer = answer_text.replace("\n", " ").replace("\r", " ")
    
    return {
        "question_id": question_id,
        "question": question,
        "category": category,
        "should_answer": "True" if should_answer_val else "False",
        "expected_file": expected_file,
        "answer_text": clean_answer,
        "retrieved_files": retrieved_files_str,
        "source_reference_present": source_ref_present,
        "answered_when_expected": answered_when_expected,
        "correct_refusal": correct_refusal,
        "answer_correct": "TODO_MANUAL",
        "answer_grounded": "TODO_MANUAL",
        "hallucination": "TODO_MANUAL",
        "notes": notes,
        "retrieval_mode": retrieval_mode,
        "generation_mode": generation_mode,
        "top_k_used": top_k,
        "response_time_ms": response_time_ms,
        "used_fallback": used_fallback,
        "retrieved_chunk_ids": retrieved_chunk_ids_str
    }

def run_answer_evaluation(
    dataset: List[Dict[str, Any]], 
    top_k: int
) -> List[Dict[str, Any]]:
    """Runs answer quality evaluation on the full dataset for a given top_k."""
    eval_results = []
    for item in dataset:
        res = run_answer_evaluation_for_item(item, top_k)
        eval_results.append(res)
    return eval_results
