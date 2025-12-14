# Example Workflow: Few-Shot Learning for Question Classification

This document demonstrates a complete workflow for collecting and using few-shot examples.

## Step 1: Run Batch Processing with LLM Judge

First, run a batch of questions using the LLM judge mode to collect high-quality examples:

```bash
python ama_kbqa/agents/kqapro_agent/batch_runner.py \
  --n_questions 50 \
  --seed 42 \
  --postprocessing_mode llm_judge
```

**What happens:**
1. The agent processes 50 random questions
2. Each answer is evaluated by an LLM judge
3. The judge assesses correctness and argumentation quality (1-5 score)
4. High-quality examples (score ≥ 4) are automatically exported to this directory

**Expected output:**
```
[OK] Saved LLM judgments to: batch_results/Batch001/judgments.json
[OK] Average Argumentation Score: 4.2/5

[INFO] Exporting high-quality examples as few-shot examples...
[OK] Exported 23 new few-shot examples:
  - Count: 5 examples
  - Verify: 4 examples
  - QueryAttr: 7 examples
  - QueryRelation: 4 examples
  - SelectAmong: 3 examples
```

## Step 2: Review Exported Examples

Check the quality of exported examples:

```bash
# View examples for a specific question type
cat db/datasets/kqapro/fewshot-examples/Count.json

# Or use any JSON viewer
```

**Example content:**
```json
[
  {
    "question": "How many boroughs does New York City have?",
    "reasoning": "This question uses the phrase 'How many', which is a definitive indicator of a Count question. It expects a numeric answer.",
    "correct_qtype": "Count",
    "lesson_learned": "The phrase 'How many' is a strong signal for Count questions, even when the entity type varies.",
    "argumentation_score": 5,
    "exported_at": "2025-01-15T14:23:45.123456"
  }
]
```

## Step 3: Run Subsequent Batches

Now run another batch. The `QtypePrediction` tool will automatically use the examples:

```bash
python ama_kbqa/agents/kqapro_agent/batch_runner.py \
  --n_questions 100 \
  --seed 123
```

**What happens:**
1. The `QtypePrediction` tool loads examples from the JSON files
2. Examples are injected into the classification prompt
3. The LLM uses these examples for few-shot learning
4. Classification accuracy should improve

## Step 4: Monitor Improvement

Compare accuracy before and after:

```bash
# Before (without examples)
Accuracy Rate: 72.5%

# After (with 20-30 examples)
Accuracy Rate: 85.3%  # Expected improvement
```

Check the logs to confirm examples are being loaded:

```
[kqapro_server.log]
2025-01-15 14:30:00 | INFO | Loaded 5 few-shot examples for Count
2025-01-15 14:30:00 | INFO | Loaded 4 few-shot examples for Verify
2025-01-15 14:30:00 | INFO | Loaded total of 23 few-shot examples across all question types
```

## Step 5: Iterative Improvement

Continue the cycle:

1. **Run more batches** → Collect more examples
2. **Review and curate** → Remove poor examples, add manual ones
3. **Test improvements** → Measure accuracy gains
4. **Repeat** → Continuously improve

## Manual Example Addition

You can also manually add high-quality examples:

1. Open the appropriate JSON file (e.g., `QueryAttr.json`)
2. Add your example:

```json
[
  {
    "question": "What is the population of Tokyo?",
    "reasoning": "This question asks for a specific attribute (population) of a named entity (Tokyo) without any qualifying context like time or place.",
    "correct_qtype": "QueryAttr",
    "lesson_learned": "Questions asking for single attributes (population, height, date) of a specific entity are QueryAttr, not QueryAttrQualifier.",
    "argumentation_score": 5,
    "exported_at": "2025-01-15T15:00:00.000000"
  }
]
```

3. Save and validate the JSON

## Advanced: Targeted Collection

Focus on specific question types that need improvement:

```bash
# Run a large batch to collect diverse examples
python ama_kbqa/agents/kqapro_agent/batch_runner.py \
  --n_questions 200 \
  --postprocessing_mode llm_judge
```

Then manually review and keep only the best examples for underrepresented types.

## Measuring Impact

Track these metrics:

### Before Few-Shot Examples
```
Question Type Distribution:
  Count               : 15  (Accuracy: 60.0%)
  Verify              : 12  (Accuracy: 75.0%)
  QueryAttr           : 23  (Accuracy: 65.2%)
  Overall Accuracy    : 68.5%
```

### After Few-Shot Examples (20-30 examples)
```
Question Type Distribution:
  Count               : 15  (Accuracy: 86.7%)  ↑ +26.7%
  Verify              : 12  (Accuracy: 91.7%)  ↑ +16.7%
  QueryAttr           : 23  (Accuracy: 82.6%)  ↑ +17.4%
  Overall Accuracy    : 85.3%                  ↑ +16.8%
```

## Best Practices

1. **Start small**: Begin with 5-10 examples per type
2. **Diverse examples**: Include edge cases and common patterns
3. **Quality over quantity**: Better to have 5 perfect examples than 10 mediocre ones
4. **Regular updates**: Refresh examples as the dataset or model changes
5. **Document lessons**: Ensure `lesson_learned` field captures key insights

## Troubleshooting

### Low Export Counts

If few examples are being exported:
- Lower `min_argumentation_score` temporarily (default: 4)
- Run larger batches (100+ questions)
- Check if the judge model is too strict

### No Accuracy Improvement

If accuracy doesn't improve:
- Verify examples are actually loading (check logs)
- Ensure examples are high-quality and diverse
- Check if examples match the types of questions being tested
- Consider using more examples (increase max_per_type)

### Degraded Performance

If accuracy gets worse:
- Review exported examples for errors
- Remove low-quality examples manually
- Ensure examples aren't contradictory
- Check prompt length isn't exceeding limits

## Integration with CI/CD

Automate example collection:

```bash
# Weekly cron job to refresh examples
0 2 * * 0 python ama_kbqa/agents/kqapro_agent/batch_runner.py \
  --n_questions 500 \
  --postprocessing_mode llm_judge \
  --seed $(date +%s)
```

## Summary

The few-shot learning system creates a virtuous cycle:

1. **Collect** → Run batches with LLM judge
2. **Curate** → High-quality examples auto-export
3. **Apply** → QtypePrediction uses examples
4. **Improve** → Better classifications
5. **Repeat** → More good examples collected

This leads to continuous, automatic improvement in classification accuracy.
