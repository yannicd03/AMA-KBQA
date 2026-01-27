# KQAPro Dataset Structure

This document describes the structure and format of the KQAPro dataset used in the AMA-KBQA system.

## Overview

KQAPro (Knowledge-graph Question Answering with Programs) is a large-scale dataset for complex question answering over knowledge graphs. It contains:

- **~16,960 entities** (excluding concepts)
- **794 concepts** (hierarchical type system)
- **~1.6M RDF triples** after conversion
- **~94K question-answer pairs** across train/val/test splits

## File Locations

```
db/datasets/kqapro/
├── kb.json                    # Knowledge base (76 MB)
├── kb.nt                      # RDF N-Triples format (1.6M lines)
├── train.json                 # Training questions (85 MB)
├── val.json                   # Validation questions (11 MB)
├── test.json                  # Test questions (3.2 MB)
├── convert_kb_to_nt.py        # JSON to N-Triples converter
└── fewshot-examples/          # Few-shot examples for question types
    ├── Count.json
    ├── Verify.json
    ├── Select.json
    ├── SelectAmong.json
    ├── SelectBetween.json
    ├── QueryAttr.json
    ├── QueryAttrQualifier.json
    ├── QueryRelation.json
    └── QueryRelationQualifier.json
```

## Knowledge Base Structure (kb.json)

The knowledge base JSON contains two main sections: `concepts` and `entities`.

### Concepts

Concepts form a hierarchical type system. Each concept has:

```json
{
  "concepts": {
    "<concept_id>": {
      "name": "Human",
      "instanceOf": ["<parent_concept_id>", ...]
    }
  }
}
```

### Entities

Entities are the main knowledge graph nodes with attributes and relations:

```json
{
  "entities": {
    "<entity_id>": {
      "name": "Albert Einstein",
      "instanceOf": ["<concept_id>", ...],
      "attributes": [...],
      "relations": [...]
    }
  }
}
```

### Attributes

Attributes are properties with typed values:

```json
{
  "key": "date_of_birth",
  "value": {
    "type": "date",
    "value": "1879-03-14"
  },
  "qualifiers": {}
}
```

**Value Types:**

| Type | Example | Description |
|------|---------|-------------|
| `string` | `"physicist"` | Plain text values |
| `quantity` | `{"value": "8848.86", "unit": "metre"}` | Numeric with unit |
| `date` | `"1879-03-14"` | ISO date format |
| `year` | `"1905"` | Year only |

### Relations

Relations connect entities to other entities:

```json
{
  "predicate": "place_of_birth",
  "object": "<entity_id>",
  "direction": "forward",
  "qualifiers": {}
}
```

**Direction:**
- `forward`: Subject → Object (e.g., Einstein → Ulm)
- `backward`: Object → Subject (inverse relation)

### Qualifiers

Both attributes and relations can have qualifiers (metadata about the statement):

```json
{
  "qualifiers": {
    "start_time": [
      {"type": "year", "value": "1905"}
    ],
    "end_time": [
      {"type": "year", "value": "1915"}
    ]
  }
}
```

## RDF/N-Triples Structure

The `kb.nt` file contains RDF triples in N-Triples format, one triple per line.

### Namespace Prefixes

```sparql
PREFIX ex:   <http://kqapro.org/entity/>
PREFIX prop: <http://kqapro.org/property/>
PREFIX attr: <http://kqapro.org/attribute/>
PREFIX qual: <http://kqapro.org/qualifier/>
PREFIX unit: <http://kqapro.org/unit/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
```

### Triple Patterns

| Pattern | Example |
|---------|---------|
| **Entity Labels** | `ex:Albert_Einstein rdfs:label "Albert Einstein"` |
| **Entity Types** | `ex:Albert_Einstein rdf:type ex:Human` |
| **Simple Attributes** | `ex:Albert_Einstein attr:date_of_birth "1879-03-14"^^xsd:date` |
| **Quantity Attributes** | Uses blank node with `rdf:value` and `unit:unit` |
| **Relations** | `ex:Albert_Einstein prop:place_of_birth ex:Ulm` |
| **Qualified Statements** | Reified using `rdf:Statement` nodes |

### Example: Quantity Attribute with Unit

```turtle
ex:Mount_Everest attr:height _:bnode1 .
_:bnode1 rdf:value "8848.86"^^xsd:decimal .
_:bnode1 unit:unit unit:metre .
```

### Example: Qualified Relation

```turtle
_:stmt1 rdf:type rdf:Statement .
_:stmt1 rdf:subject ex:Einstein .
_:stmt1 rdf:predicate prop:employer .
_:stmt1 rdf:object ex:ETH_Zurich .
_:stmt1 qual:start_time "1912"^^xsd:gYear .
_:stmt1 qual:end_time "1914"^^xsd:gYear .
```

## Question-Answer Dataset Structure

### Training/Validation Format

Each question includes full annotations:

```json
{
  "question": "What is the capital of France?",
  "sparql": "SELECT ?x WHERE { ex:France prop:capital ?x }",
  "program": {
    "function": "Find",
    "inputs": ["France"],
    "dependencies": []
  },
  "choices": ["Paris", "London", "Berlin", ...],
  "answer": "Paris"
}
```

**Fields:**

| Field | Description |
|-------|-------------|
| `question` | Natural language question |
| `sparql` | Executable SPARQL query |
| `program` | Functional program representation |
| `choices` | 10 answer choices (for multiple choice) |
| `answer` | Ground truth answer |

### Test Format

Test questions only include `question` and `choices` (no gold answer for evaluation).

## Question Types

KQAPro categorizes questions into 9 types:

| Type | Description | Example |
|------|-------------|---------|
| `Count` | Counting questions | "How many children does X have?" |
| `Verify` | Yes/No verification | "Is X a scientist?" |
| `Select` | Selection from choices | "Which one is the capital?" |
| `SelectAmong` | Multiple-choice | "Which of these are European?" |
| `SelectBetween` | Binary comparison | "Who is older, A or B?" |
| `QueryAttr` | Simple attribute lookup | "What is X's birth date?" |
| `QueryAttrQualifier` | Attribute with qualifier | "When did X graduate from Y?" |
| `QueryRelation` | Relation query | "Who is X's spouse?" |
| `QueryRelationQualifier` | Relation with qualifier | "Where did X work in 1905?" |

## Qdrant Vector Schema

Entities and relations are embedded for semantic search:

### kqapro-entities Collection

```json
{
  "vector": "float[4096]",
  "payload": {
    "original_id": "Q12345",
    "node_type": "entity|concept",
    "name": "Albert Einstein",
    "instanceOf": ["Human", "Physicist"],
    "attributes": [...],
    "relations": [...]
  }
}
```

### kqapro-relations Collection

```json
{
  "vector": "float[4096]",
  "payload": {
    "predicate": "place_of_birth",
    "type": "relation_schema"
  }
}
```

## Statistics

| Metric | Value |
|--------|-------|
| Entities | ~16,960 |
| Concepts | 794 |
| RDF Triples | ~1.6M |
| Training Questions | ~94K |
| Validation Questions | ~12K |
| Test Questions | ~12K |
| Unique Predicates | ~1,000 |
| Entity Embeddings | ~47K |
| Relation Embeddings | ~1K |

## References

- [KQAPro Paper](https://arxiv.org/abs/2007.03875)
- [KQAPro GitHub](https://github.com/shijx12/KQAPro_Baselines)
