# Notes RAG Assistant - Pure Validation Step Summary Report

This report summarizes the performance of the current best Notes RAG Assistant configuration (Stage 3 best settings: `top_k = 3` with neighbor context expansion, simplified prompt, and deterministic `temperature = 0.0` using `llama3.2:3b`) evaluated on a fresh set of Polish notes (`notatka_11` through `notatka_15`) and 17 validation questions.

---

## 1. Key Metrics Summary

| Metric | Automated Rate | Manual (Reviewed) Rate | Notes |
| :--- | :---: | :---: | :--- |
| **Retrieval Hit Rate (at $k=3$)** | 81.3% (13/16) | 81.3% (13/16) | Failed on all 3 queries for `notatka_11_indukcja.txt`. |
| **Grounded Answered-as-Expected** | 81.3% (13/16) | 75.0% (12/16) | `val003` answered with wrong file; `val008` & `val012` suffered chunk misses. |
| **Out-of-Scope Correct Refusal** | 0.0% (0/1) | 100% (1/1) | Automated checker failed to match the Polish refusal phrase. |
| **Citation Presence Rate (Answered)**| 84.6% (11/13) | 84.6% (11/13) | 2 queries answered correctly but lacked bracketed citations. |
| **Citation Presence Rate (Overall)** | 64.7% (11/17) | 64.7% (11/17) | Includes refused queries. |

---

## 2. Failure Analysis & Key Insights

The validation run on fresh notes revealed two distinct, highly instructive RAG failure modes:

### A. Cross-Lingual Retrieval Generalization Failures (Spelling-Dependent Embedding Space)
The embedding model used is `all-MiniLM-L6-v2` (an English-centric model). The system successfully retrieved documents when technical terms shared spelling roots (cognates/borrowed words) between English (questions) and Polish (notes), such as:
* *Bayes* (Bayesowski)
* *Markov* (Markowa)
* *Q-Learning*
* *holdout*
* *cross-validation* (k-CV)
* *MSE/MAE*

However, retrieval failed for **all three queries** targeting `notatka_11_indukcja.txt` because the translations of key concepts had no lexical or spelling overlap:
* *Goal of inductive learning* $\rightarrow$ *Celem indukcyjnego uczenia się* (missed)
* *Representation/preference bias* $\rightarrow$ *Obciążenie reprezentacji/preferencji* (missed)
* *Hypothesis space* $\rightarrow$ *Przestrzeń wersji/modeli* (missed, instead matched `n atrybutach` in `notatka_13_bayes.txt`)

This highlights that lightweight monolingual English embedding models cannot reliably generalise to translated technical notes unless technical terminology overlaps orthographically.

### B. Context Fragmentation & Neighbor Expansion Limits
The neighbor context expansion configuration retrieves the immediate predecessor ($index - 1$) and successor ($index + 1$) of any chunk returned by the primary semantic search query. This is a 1-hop expansion. We observed two failures where this was insufficient:

1. **`val008` (Naive Bayes missing values)**:
   * Expected information was located in **Chunk 2** of `notatka_13_bayes.txt`.
   * The primary semantic search returned **Chunk 0**.
   * Neighbor expansion retrieved **Chunk 1** (neighbor of Chunk 0).
   * Since Chunk 1 was not a primary hit, its neighbor (Chunk 2) was not expanded. The critical information was missed, causing the LLM to refuse to answer.
2. **`val012` (Episodic tasks difference)**:
   * Expected comparison was located in **Chunk 3** of `notatka_14_wzmacnianie.txt`.
   * Primary search returned **Chunk 0** and **Chunk 1**.
   * Neighbor expansion retrieved **Chunk 2** (neighbor of Chunk 1).
   * Chunk 3 was missed because it is 2 hops away from the primary hits. The LLM was forced to qualify its answer as not directly present, though it attempted to reason out the comparison.

This shows that 1-hop neighbor expansion remains vulnerable to context fragmentation when files are split into multiple smaller chunks (e.g. 500 characters) and the relevant fact lies beyond the 1-hop boundary.

---

## 3. Detailed Query-by-Query Results

| Query ID | Question | Category | Expected File | Hit? | Answered? | Citation? | Notes |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| `val001` | What is the goal of inductive learning? | direct_fact | `notatka_11_indukcja.txt` | No | No | No | Refused correctly based on retrieved context, but retrieval failed. |
| `val002` | What is the difference between representation bias and preference bias? | comparison | `notatka_11_indukcja.txt` | No | No | No | Refused correctly based on retrieved context, but retrieval failed. |
| `val003` | How large is the hypothesis space for Boolean conjunctions with n attributes? | direct_fact | `notatka_11_indukcja.txt` | No | Yes | Yes | **Hallucination/Misdirection**: Retried Bayes note and answered $2^n$ instead of $3^n+1$. |
| `val004` | How does holdout help estimate model quality? | explanation | `notatka_12_jakosc.txt` | Yes | Yes | Yes | Correctly answered with source reference `[Source 5]`. |
| `val005` | What is the difference between k-fold cross-validation and holdout? | comparison | `notatka_12_jakosc.txt` | Yes | Yes | Yes | Correctly answered with source reference `[Source 2]`. |
| `val006` | Why does MSE penalize large errors more strongly than MAE? | explanation | `notatka_12_jakosc.txt` | Yes | Yes | Yes | Correctly answered with source reference `[Source 2]`. |
| `val007` | What is the core assumption behind the naive Bayes classifier? | direct_fact | `notatka_13_bayes.txt` | Yes | Yes | Yes | Correctly answered with citation `[notatka_13_bayes.txt]`. |
| `val008` | How does naive Bayes handle missing attribute values? | direct_fact | `notatka_13_bayes.txt` | Yes | No | No | **Refused**: Retrieval hit the file, but missed Chunk 2 due to neighbor limits. |
| `val009` | Compare naive Bayes with decision trees. | comparison | `notatka_13_bayes.txt` | Yes | Yes | Yes | Correctly answered with citations `[Source 1], [Source 5]`. |
| `val010` | What is a Markov Decision Process in reinforcement learning? | direct_fact | `notatka_14_wzmacnianie.txt` | Yes | Yes | Yes | Correctly answered with citation `[notatka_14_wzmacnianie.txt]`. |
| `val011` | What is the Markov property? | direct_fact | `notatka_14_wzmacnianie.txt` | Yes | Yes | Yes | Correctly answered with citation `[Source 4]`. |
| `val012` | What is the difference between episodic tasks “to success” and “to failure”? | comparison | `notatka_14_wzmacnianie.txt` | Yes | Yes | Yes | **Partial Chunk Miss**: Answered with disclaimer after reasoning. |
| `val013` | What makes Q-Learning an off-policy algorithm? | explanation | `notatka_15_qlearning.txt` | Yes | Yes | No | **Citation Miss**: Correct answer, but failed to include bracketed citation. |
| `val014` | Compare Q-Learning with Sarsa. | comparison | `notatka_15_qlearning.txt` | Yes | Yes | Yes | Correctly answered with citation `[Source 4]`. |
| `val015` | What problem is solved by epsilon-greedy in Q-Learning? | explanation | `notatka_15_qlearning.txt` | Yes | Yes | No | **Citation Miss**: Correct answer, but failed to include bracketed citation. |
| `val016` | Who wrote “Pan Tadeusz”? | distraction_resistance | `notatka_11_indukcja.txt` | Yes | Yes | Yes | **Distraction Hit**: Note contained distraction text, LLM answered correctly. |
| `val017` | How much is a reduced cinema ticket? | out_of_scope | N/A | N/A | No | No | **Correct Refusal**: Output correctly refused in Polish, but automated check missed it. |
