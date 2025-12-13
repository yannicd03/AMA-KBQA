# Few-Shot Examples for Question Type Classification

This directory contains curated examples for each question type to improve classification accuracy.

## File Structure

Each question type has its own JSON file:
- `Count.json`
- `Verify.json`
- `SelectBetween.json`
- `SelectAmong.json`
- `QueryAttr.json`
- `QueryAttrQualifier.json`
- `QueryRelation.json`
- `QueryRelationQualifier.json`
- `QueryName.json`

## Example Format

Each JSON file contains an array of examples with the following structure:

```json
[
  {
    "question": "How many people live in Berlin?",
    "reasoning": "This question asks for a COUNT of entities (people). The key phrase 'How many' indicates a counting operation.",
    "correct_qtype": "Count",
    "lesson_learned": "Questions starting with 'How many' are almost always Count type, regardless of the entity being counted."
  }
]
```

## Generating Examples

These examples are automatically curated from batch_runner.py output using the LLM judge's:
- `correctness_reasoning` - Why the classification was correct/incorrect
- `argumentation_quality` - Analysis of the reasoning process
- `suggested_improvement` - Lessons learned for future classifications

## Usage

The QtypePrediction tool loads up to 10 examples from the relevant question type file(s) to improve classification accuracy through few-shot learning.
