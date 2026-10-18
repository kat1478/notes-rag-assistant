# Stage 3C - Failure Review

## Contents
_Line numbers of each section:_
- L22 — Row-by-Row Failure Analysis
  - L24 — 1. `q010`
  - L41 — 2. `q011`
  - L57 — 3. `q016`
  - L72 — 4. `q018`
  - L87 — 5. `q028`
  - L102 — 6. `q032`
- L117 — Classification of Failure Patterns
- L134 — Proposed Primary Improvement
  - L136 — Windowed Context Expansion (Chunk Merging)
  - L144 — Why this should be tested first:


This document reviews the 6 failed in-scope queries from the Stage 3C answer quality evaluation and classifies their failure patterns.

---

## Row-by-Row Failure Analysis

### 1. `q010`
* **Question**: "What is edge relaxation in Dijkstra’s algorithm?"
* **Grounded Answer Expected**: Yes (`should_answer`: True).
* **Expected File**: `notatka_studenta_03_dijkstra.txt`
* **Generated Answer**:
  > *I don't have enough information to answer that based on the notes. The context only provides information about Dijkstra's algorithm from Source 2 [notatka_studenta_03_dijkstra.txt], but it does not mention "edge relaxation".*
* **Retrieved Files**: `notatka_studenta_01_gradient.txt;notatka_studenta_03_dijkstra.txt;student_note_04_binary_search.txt`
* **Retrieved Chunk IDs**: `b7a5e204-be28-4a71-88dc-cde98a40f449;7a2c12be-323f-4ddb-8133-fc78575ea413;11689b13-0b05-45e0-9f58-2667ed4dc086`
* **Chunk Excerpt**: The only retrieved Dijkstra chunk was Chunk 0, which ends with:
  > *"...Do zarządzania tymi węzłami najlepiej użyć kolejki priorytetowej, żeby zawsze mieć najmniejszą wartość na wierzchu. Bardzo ważnym elementem jest relaksacja krawędzi."*
* **Expected Answer Present in Chunks?**: No. The term "relaksacja krawędzi" is mentioned at the end of Chunk 0, but its actual definition is in **Chunk 1**:
  > *"Polega to na tym, że jeśli znajdziemy nową drogę do wierzchołka v przez u..."*
  Chunk 1 was not retrieved.
* **Likely Failure Source**: **Context Fragmentation & Dilution**. The definition of the term was separated into a different chunk due to character limits, and the retriever failed to fetch Chunk 1. Meanwhile, 2 slots were occupied by irrelevant gradient/binary search distractors.

---

### 2. `q011`
* **Question**: "Why does Dijkstra’s algorithm require non-negative edge weights?"
* **Grounded Answer Expected**: Yes (`should_answer`: True).
* **Expected File**: `notatka_studenta_03_dijkstra.txt`
* **Generated Answer**:
  > *I don't have enough information to answer that based on the notes. (Source: None of the provided context mentions anything about Dijkstra's algorithm requiring non-negative edge weights.)*
* **Retrieved Files**: `notatka_studenta_01_gradient.txt;notatka_studenta_03_dijkstra.txt;student_note_06_relu.txt`
* **Retrieved Chunk IDs**: `b7a5e204-be28-4a71-88dc-cde98a40f449;7a2c12be-323f-4ddb-8133-fc78575ea413;4bdd583a-9cf7-4e5c-bc34-134117586e71`
* **Chunk Excerpt**: Similar to `q010`, only Chunk 0 of Dijkstra was retrieved.
* **Expected Answer Present in Chunks?**: No. The explanation of non-negative weights is in **Chunk 1**:
  > *"...Dijkstra działa poprawnie tylko wtedy, gdy wagi krawędzi są nieujemne. Jeśli pojawią się ujemne wagi, algorytm może się pogubić..."*
  Chunk 1 was not retrieved.
* **Likely Failure Source**: **Context Fragmentation & Dilution**. The answer is in Chunk 1, but the retriever only retrieved Chunk 0. The other 2 slots were diluted by Gradient and ReLU distractors.

---

### 3. `q016`
* **Question**: "In the notes’ example, how many steps does binary search need to find 7 in [1, 3, 5, 7, 9]?"
* **Grounded Answer Expected**: Yes (`should_answer`: True).
* **Expected File**: `student_note_04_binary_search.txt`
* **Generated Answer**:
  > *Based on Source 2 [student_note_04_binary_search.txt], the correct answer is that binary search needs only 2 steps to find 7 in the array [1, 3, 5, 7, 9]. [Source 2] I don't have enough information to answer that based on the notes.*
* **Retrieved Files**: `student_note_04_binary_search.txt;student_note_04_binary_search.txt;student_note_04_binary_search.txt`
* **Retrieved Chunk IDs**: `ef6c8a79-02c2-46c2-bf41-79744c52b2cd;c6de49dd-ccee-44e9-a88f-4d323b9bf3d9;11689b13-0b05-45e0-9f58-2667ed4dc086`
* **Chunk Excerpt**: Chunks 0, 1, and 2 were all retrieved. Chunk 1 and Chunk 2 contain the example:
  > *"...For example, if you have an array [1, 3, 5, 7, 9] and want to find 7, you start with the middle (5). Since 7 > 5, you look at the right side [7, 9], find the new middle (7), and you're done in just two steps."*
* **Expected Answer Present in Chunks?**: Yes, fully present.
* **Likely Failure Source**: **LLM Over-conservatism**. The LLM correctly reasoned and wrote the answer, but appended the boilerplate refusal sentence because the prompt instructions are extremely strict, or because it noticed the distractor info in the same file (e.g. capital of Australia) and became over-cautious.

---

### 4. `q018`
* **Question**: "What is information gain in decision trees?"
* **Grounded Answer Expected**: Yes (`should_answer`: True).
* **Expected File**: `notatka_studenta_05_drzewa.txt`
* **Generated Answer**:
  > *According to Source 1 [notatka_studenta_05_drzewa.txt], Information Gain (Zysk Informacyjny) is calculated by measuring the decrease in entropy after making a split. The goal is to choose the split that results in the greatest information gain. [Source 1] I don't have enough information to answer that based on the notes.*
* **Retrieved Files**: `notatka_studenta_05_drzewa.txt;student_note_02_tfidf.txt;student_note_02_tfidf.txt`
* **Retrieved Chunk IDs**: `e279a376-8c5a-4ddf-93ae-4c43fca91d05;8aa270e4-635b-4527-af4d-b0f1b8ad7560;59e40df8-71df-47d0-ae69-046a4d8edad5`
* **Chunk Excerpt**: Chunk 1 of decision trees was retrieved:
  > *"...Na tej podstawie liczymy Zysk Informacyjny (Information Gain), czyli sprawdzamy, o ile spadła entropia po dokonaniu podziału. Wybieramy ten podział, który daje największy zysk..."*
* **Expected Answer Present in Chunks?**: Yes.
* **Likely Failure Source**: **LLM Over-conservatism / Context Dilution**. The LLM answered correctly but appended the refusal phrase because the other two slots were filled with completely irrelevant TF-IDF context, which diluted the prompt and made the LLM insecure.

---

### 5. `q028`
* **Question**: "Compare batch gradient descent and SGD based on the notes."
* **Grounded Answer Expected**: Yes (`should_answer`: True).
* **Expected File**: `notatka_studenta_01_gradient.txt`
* **Generated Answer**:
  > *Based on the provided context, here's a comparison between batch gradient descent and SGD: Batch gradient descent is not mentioned as an option in the context. However, SGD (stochastic gradient descent) is compared to using a "czysty spadek gradientu" (pure gradient descent), which is described as "bo to wolne" (because it's free)... I don't have enough information to answer that based on the notes.*
* **Retrieved Files**: `notatka_studenta_01_gradient.txt;notatka_studenta_01_gradient.txt;student_note_06_relu.txt`
* **Retrieved Chunk IDs**: `5bf6c804-ce88-444a-87d7-47c40fb721d8;b7a5e204-be28-4a71-88dc-cde98a40f449;fe8894ff-d53b-4ea9-89b0-1eef0594d069`
* **Chunk Excerpt**: Chunk 1 of Gradient was retrieved:
  > *"...W praktyce rzadko używamy czystego spadku gradientu na całym zbiorze danych (batch), bo to wolne. Częściej stosuje się SGD (stochastyczny spadek gradientu), który aktualizuje parametry po każdym przykładzie..."*
* **Expected Answer Present in Chunks?**: Yes.
* **Likely Failure Source**: **Bilingual Term Mismatch & Fragmentation**. The query asks for "batch gradient descent" in English, but the note uses Polish "czystego spadku gradientu na całym zbiorze danych (batch)". The LLM failed to map the English term to the Polish term. Also, Chunk 2 (which contains the end of the sentence about SGD) was not retrieved, and 1 slot was wasted on a ReLU distractor.

---

### 6. `q032`
* **Question**: "Did Adam Mickiewicz use decision trees to write Pan Tadeusz?"
* **Grounded Answer Expected**: Yes (`should_answer`: True).
* **Expected File**: `notatka_studenta_05_drzewa.txt`
* **Generated Answer**:
  > *I don't have enough information to answer that based on the notes. The provided context only discusses the concept of decision trees and their application in data analysis, but it does not mention Adam Mickiewicz or his writing style. [Source 1]*
* **Retrieved Files**: `notatka_studenta_05_drzewa.txt;notatka_studenta_05_drzewa.txt;notatka_studenta_05_drzewa.txt`
* **Retrieved Chunk IDs**: `f7efd80f-c390-4880-90f8-028261b3aa49;f70e14d3-2bae-4305-9393-6fdaf2b4781d;e279a376-8c5a-4ddf-93ae-4c43fca91d05`
* **Chunk Excerpt**: All three chunks of decision trees were retrieved. Chunk 2 contains:
  > *"...Adam Mickiewicz napisał 'Pana Tadeusza', ale nie sądzę, żeby używał do tego drzew decyzyjnych."*
* **Expected Answer Present in Chunks?**: Yes, fully present in Chunk 2.
* **Likely Failure Source**: **LLM Reasoning / Complex Polish Syntax**. The LLM did not parse the humorous Polish sentence correctly or was overwhelmed by the negative/hypothetical phrasing combined with strict refusal instructions.

---

## Classification of Failure Patterns

1. **Context Fragmentation (Split Boundaries)**:
   - *Impacted queries*: `q010`, `q011`, `q028`
   - *Description*: Relevant details are split across chunk boundaries (e.g. terms in Chunk 0, definition in Chunk 1). The retriever only pulls one chunk, leading to incomplete context.
2. **Context Dilution (Irrelevant Distractor Chunks)**:
   - *Impacted queries*: `q010`, `q011`, `q018`, `q028`
   - *Description*: Irrelevant chunks from other files occupy `top_k=3` slots, displacing adjacent relevant chunks and confusing the LLM.
3. **LLM Over-conservatism**:
   - *Impacted queries*: `q016`, `q018`
   - *Description*: The LLM answers correctly but appends the boilerplate refusal text due to the extreme strictness of prompt rule 2.
4. **Cross-Lingual Terminology Alignment**:
   - *Impacted queries*: `q028`, `q032`
   - *Description*: Mismatches between English query terms ("batch gradient descent", "Adam Mickiewicz") and Polish context terms ("czysty spadek... (batch)", Polish sentence structure).

---

## Proposed Primary Improvement

### Windowed Context Expansion (Chunk Merging)
We propose to implement **Context Expansion** during retrieval:
When the semantic retriever fetches the top `k` chunks:
1. For each retrieved chunk, check its metadata (`file_name` and `chunk_index`).
2. Fetch the immediately adjacent chunks from the same file (e.g., `chunk_index - 1` and `chunk_index + 1`).
3. Combine and sort the chunks by `chunk_index` to form a contiguous block of text for each retrieved document.
4. Deduplicate any overlapping chunks to ensure the context remains clean.

### Why this should be tested first:
- **Directly targets the main failure pattern**: Fragmentation and Dilution represent the source of failure for 4 out of the 6 cases (`q010`, `q011`, `q028` directly; `q018` indirectly via distractor dilution).
- **Feasible and Lightweight**: Since our chunk metadata contains `file_name` and `chunk_index`, context expansion can be implemented purely inside `retrieval.py` without rewriting chunking algorithms or retraining models.
- **Maintains Grounding**: This increases the richness of the grounded context without changing the core LLM prompt rules, keeping the system extremely safe against hallucinations.
