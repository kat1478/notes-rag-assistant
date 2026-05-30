# Stage 3D - Context Expansion Comparison Summary

This document compares the performance of the Notes RAG Assistant before and after implementing **Windowed Context Expansion** (adjacent chunk retrieval).

---

## Metric Comparison Table

| Metric | Before (Baseline, top_k=3) | After (Context Expansion, top_k=3) | Change |
|---|---|---|---|
| **Grounded (In-Scope) Answering Rate** | 80.0% (24/30) | **83.3% (25/30)** | **+3.3%** |
| **Out-of-Scope Correct Refusal Rate** | 100.0% (2/2) | 100.0% (2/2) | Unchanged |
| **Source Citations Present Rate** | 87.5% (28/32) | 84.4% (27/32) | -3.1% |
| **Average Response Time (Latency)** | 37,696.8 ms | **35,448.0 ms** | **-2.2 sec (Fastest)** |
| **Grounded Hit Rate (k=3)** | 100.0% (30/30) | 100.0% (30/30) | Unchanged |
| **First Hit Correct Rate (k=3)** | 83.33% | 90.00% | +6.67% |
| **First Hit Correct Rate (k=5)** | 83.33% | 40.00% | -43.33% * |

> [!NOTE]
> *The decrease in `First Hit Correct Rate (k=5)` is a side effect of sorting the final context alphabetically by `file_name` and then by `chunk_index`. This is better for LLM reading flow but shifts the highest-scoring matching file away from index 0 in the expanded chunk list.

---

## Detailed Row-level Changes

### 1. Did `q010`, `q011`, `q028` improve?
* **`q010` (Edge relaxation in Dijkstra)**: **IMPROVED**. Succeeded with a correct definition:
  > *"Edge relaxation in Dijkstra’s algorithm is the process of updating the distances to each node after finding a shorter path through one of its neighbors."*
  This was possible because Chunk 1 of Dijkstra (holding the definition) was retrieved as a neighbor of Chunk 0.
* **`q011` (Dijkstra non-negative weights)**: **UNCHANGED (Failed)**. Even though Chunk 1 (holding the explanation of negative weights) was successfully fetched via expansion, the LLM still refused to answer, failing to extract the Polish explanation when queried in English.
* **`q028` (Compare batch gradient descent and SGD)**: **IMPROVED**. Succeeded with a detailed comparison because adjacent Chunk 2 provided the full comparative context.

### 2. Did `q016`, `q018`, `q032` change?
* **`q016` (Binary search steps count)**: **CHANGED (Improved)**. Succeeded without generating a refusal suffix.
* **`q018` (Information gain in decision trees)**: **CHANGED (Improved)**. Succeeded without generating a refusal suffix because adjacent chunks provided a more complete and coherent explanation of decision trees, giving the model more confidence.
* **`q032` (Mickiewicz decision trees distraction)**: **UNCHANGED (Failed)**. The model still refused to answer, struggling with the Polish humorous sentence structure and the strict refusal prompt rules.

### 3. Did overall answered-as-expected improve?
* **Yes**, it improved from **80.0%** to **83.3%** (25 out of 30 queries were answered successfully).

### 4. Did source-reference presence change?
* **Yes**, it decreased slightly from **87.5% (28/32)** to **84.4% (27/32)**. This is a minor regression where the model failed to include explicit bracketed references for one query, even though the context was grounded.

### 5. Did response time increase?
* **No**. Surprisingly, the average response time decreased from **37.7 seconds** to **35.4 seconds** (a ~6% reduction). This suggests that the local CPU inference speed was not negatively affected by the larger context window, possibly due to fewer LLM tokens spent on repeating refusal thoughts.

### 6. Did any regressions appear?
* **Yes**, 3 queries that succeeded in the baseline regressed to failures under context expansion:
  * **`q004` (SGD vs batch gradient descent comparison)**: The model refused, claiming the text doesn't compare them, even though all chunks were present.
  * **`q009` (Dijkstra algorithm purpose)**: The model generated a correct answer but appended the boilerplate refusal sentence.
  * **`q013` (Binary search array mechanism)**: The model generated a correct answer, appended the refusal sentence, and then generated a duplicate answer.
* **Reason for regressions**: Small open-source models (like `llama3.2:3b`) are highly sensitive to prompt size. Tripling the context size via windowed expansion increased prompt length, causing the LLM to get confused or over-conservative, triggering the strict refusal phrase `"I don't have enough information"` even when the answer was fully in the context.

---

## Conclusion & Recommendations

Context Expansion successfully solved **Context Fragmentation** for key technical concepts (`q010` and `q028` now succeed). However, it introduced a trade-off: **larger contexts increase the perplexity of the small 3B model**, leading to over-cautious refusals on questions that previously succeeded (like `q004`, `q009`, `q013`).

To further improve performance in future steps, we should:
1. Make the refusal rule in the system prompt slightly less sensitive to minor gaps (e.g. allow answering if the core concept is present, and clarify cross-lingual terminology handling).
2. Tune the score thresholding to exclude true distractors from irrelevant files so they don't occupy slots.
