# Refusal classifier calibration (2026-11-05)

Blind human labels for 60 saved benchmark answers: does the answer **refuse**, answer **partially** (answers part and
says the rest is not in the notes), or **answer** (without saying anything is missing — even if wrong)?

- `items.json` — what the labeller saw (question + answer, no model, no detector output); `label.html` — the page.
- `key.json` — per item: stratum (`bucket`), stratum size in the pool, source run, `should_answer`, regex verdict.
  The sample is stratified over 1467 unique saved answers, so pool-level error rates re-weight by `bucket_size`.
- `labels.json` — first pass. `labels_revised.json` — after re-checking 7 disputed items (all 7 changed to the
  classifier's label: wrong answers had been labelled "partial"). Both are kept; scores are reported for both.

Scored by `python -m app.compare_refusal_detectors` → `../refusal_detectors.json`.
