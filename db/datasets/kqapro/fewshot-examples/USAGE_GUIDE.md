# Few-Shot Examples Usage Guide

## Overview

The `QtypePrediction` tool now supports loading curated few-shot examples to improve question type classification accuracy. These examples are automatically extracted from high-quality batch processing results using the LLM judge.

## How It Works

### 1. Automatic Collection (via batch_runner.py)

When running batch processing with `--postprocessing_mode llm_judge`:

```bash
python ama_kbqa/agents/kqapro_agent/batch_runner.py --n_questions 50 --postprocessing_mode llm_judge
```

The system will automatically:
- Evaluate each classification using an LLM judge
- Score the argumentation quality (1-5)
- Export high-quality examples (score ≥ 4) to this directory
- Merge with existing examples, keeping the top 10 per question type

### 2. Automatic Loading (in kqapro_server.py)

When the `QtypePrediction` tool is called:
- It loads up to 10 examples per question type from JSON files in this directory
- Examples are injected into the classification prompt
- The LLM uses these examples for few-shot learning
- If no examples exist, the tool gracefully continues without them

## File Structure

Each question type has its own JSON file:

```
fewshot-examples/
├── Count.json                  # Examples for Count questions
├── Verify.json                 # Examples for Verify questions
├── SelectBetween.json          # Examples for SelectBetween questions
├── SelectAmong.json            # Examples for SelectAmong questions
├── QueryAttr.json              # Examples for QueryAttr questions
├── QueryAttrQualifier.json     # Examples for QueryAttrQualifier questions
├── QueryRelation.json          # Examples for QueryRelation questions
├── QueryRelationQualifier.json # Examples for QueryRelationQualifier questions
└── QueryName.json              # Examples for QueryName questions
```

## Example Format

Each JSON file contains an array of example objects:

```json
[
  {
    "question": "How many districts does Berlin have?",
    "reasoning": "This question explicitly asks 'How many', which is a clear indicator of a Count question type. It expects a numeric answer representing the quantity of districts.",
    "correct_qtype": "Count",
    "lesson_learned": "Questions starting with 'How many' almost always indicate a Count question, regardless of the specific entity being counted.",
    "argumentation_score": 5,
    "exported_at": "2025-01-15T10:30:00.000000"
  },
  {
    "question": "Is Berlin the capital of Germany?",
    "reasoning": "This is a yes/no question that can be verified as true or false. The question starts with 'Is' and seeks a boolean answer.",
    "correct_qtype": "Verify",
    "lesson_learned": "Questions starting with 'Is', 'Was', 'Did', 'Does' that form complete statements are typically Verify questions.",
    "argumentation_score": 5,
    "exported_at": "2025-01-15T10:35:00.000000"
  }
]
```

### Field Descriptions

- **question**: The original question text
- **reasoning**: Explanation from the LLM judge about why this classification is correct
- **correct_qtype**: The verified correct question type
- **lesson_learned**: Key insight or pattern from the `suggested_improvement` field
- **argumentation_score**: Quality score (1-5) from the LLM judge
- **exported_at**: ISO timestamp when this example was exported

## Manual Curation

You can also manually add or edit examples:

1. Open the appropriate JSON file (e.g., `Count.json`)
2. Add your example following the format above
3. Ensure the JSON is valid
4. Save the file

**Note**: The system automatically keeps only the top 10 examples per type (sorted by argumentation_score).

## Quality Criteria

Examples are only exported if they meet these criteria:
- ✓ Answer was correct (matches gold answer)
- ✓ Argumentation score ≥ 4 (good to excellent)
- ✓ Question type is known (not "Unknown")
- ✓ Not a duplicate of an existing example

## Monitoring Example Quality

After each batch run with `llm_judge` mode, you'll see output like:

```
[INFO] Exporting high-quality examples as few-shot examples...
[OK] Exported 12 new few-shot examples:
  - Count: 3 examples
  - Verify: 2 examples
  - QueryAttr: 4 examples
  - QueryRelation: 3 examples
```

## Benefits

- **Improved Accuracy**: The LLM learns from real examples with explanations
- **Pattern Recognition**: Curated lessons help identify subtle patterns
- **Continuous Improvement**: Examples accumulate and improve over time
- **Domain-Specific Learning**: Examples are specific to the KQAPro dataset

## Best Practices

1. **Run Regular Batches**: Periodically run batches with `llm_judge` mode to collect examples
2. **Review Examples**: Manually review exported examples for quality
3. **Diverse Coverage**: Ensure all question types have good examples
4. **Update as Needed**: Remove outdated or incorrect examples manually
5. **Balance Quantity**: Aim for 5-10 high-quality examples per type

## Technical Details

### Loading Implementation (kqapro_server.py)

```python
def load_fewshot_examples(max_per_type: int = 10) -> str:
    """Load and format few-shot examples from JSON files"""
    # Loads examples from this directory
    # Returns formatted string for prompt injection
```

### Export Implementation (batch_runner.py)

```python
def export_fewshot_examples_from_judgments(
    results: List[Dict[str, Any]],
    output_dir: Path = None,
    min_argumentation_score: int = 4
) -> Dict[str, int]:
    """Extract and save high-quality examples from batch results"""
    # Filters by accuracy and argumentation score
    # Merges with existing examples
    # Keeps top 10 per type
```

## Troubleshooting

### No Examples Being Exported

**Problem**: After running with `llm_judge`, no examples are exported.

**Solutions**:
- Check that answers are correct (accuracy must be True)
- Verify argumentation scores are ≥ 4
- Ensure question types are properly classified
- Check logs for error messages

### Examples Not Loading

**Problem**: QtypePrediction doesn't seem to use examples.

**Solutions**:
- Verify JSON files exist in the correct directory
- Check JSON syntax is valid
- Look for errors in `kqapro_server.log`
- Ensure FEWSHOT_EXAMPLES_DIR path is correct

### Duplicate Examples

**Problem**: Same question appears multiple times.

**Solutions**:
- The system prevents duplicates based on question text
- If duplicates exist, manually edit the JSON file
- The system keeps only top 10 per type when exporting

## Future Enhancements

Potential improvements:
- Automatic deduplication based on semantic similarity
- Example versioning and history tracking
- Web UI for example management
- Cross-validation of example quality
- Export examples to different formats (YAML, CSV)
