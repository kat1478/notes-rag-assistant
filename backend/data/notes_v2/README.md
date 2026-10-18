# Corpus v2

Frozen evaluation corpus for model benchmarks (dataset v2, 2026-10-17).

- `notatka_*.txt`, `student_note_*.txt` — the 11 notes of corpus v1, copied unchanged from `../notes/`.
- `syn_*.txt` — **synthetic** notes written by an LLM (Claude) for this benchmark. Topics are deliberately
  close to the v1 notes (Bellman-Ford next to Dijkstra, BM25 next to TF-IDF, Sarsa next to Q-Learning, …)
  so retrieval has realistic competitors, and some adjacent facts are deliberately left out so that
  near-topic out-of-scope questions can test refusal. Not a source of truth — written for evaluation only.

Do not edit files here: results are only comparable on the exact corpus they ran on. Changes → corpus v3.
