"""WikiKGQA 2026 challenge support.

Self-contained module for the WikiKGQA challenge (ISWC 2026): loading the
extended QALD-JSON datasets, scoring with Macro QALD F1 over executed answer
sets, and writing valid submission files. Kept separate from the KQAPro/SciQA
benchmark harness because the scoring semantics differ (exact answer-set F1
versus LLM-judged semantic correctness).
"""
