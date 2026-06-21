# Evaluation question set

Phase 6 will populate this folder with `healthcare_policy_eval.json`.

Schema (each item):

```json
{
  "question": "What is the timeline to file a first-level appeal of a denied claim?",
  "expected_answer": "Within 60 calendar days of the denial notice.",
  "expected_document": "appeal_process_policy.txt",
  "expected_keywords": ["60 days", "first-level appeal", "denial notice"]
}
```

The dataset is small on purpose: enough to catch obvious regressions in
retrieval / grounding without burning tokens on every CI run.
