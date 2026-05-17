from ama_kbqa.agents.kqapro_agent.agent import KQAProAgent
from ama_kbqa.agents.sciqa_agent.agent import SciQAAgent
from ama_kbqa.framework.base_agent import BaseKBQAAgent
from ama_kbqa.server.sciqa_server import (
    _looks_numeric_value,
    _payload_node_type_matches,
    _schema_display_value,
    _schema_usage_hint,
    _short_orkg_term,
)


def _agent() -> KQAProAgent:
    return object.__new__(KQAProAgent)


def test_kqapro_extracts_code_constraint():
    constraints = _agent()._extract_exact_attribute_constraints(
        "Which type of sport has IAB code 543?"
    )

    assert {"attribute_name": "IAB code", "value": "543"} in constraints


def test_kqapro_extracts_official_name_constraint():
    constraints = _agent()._extract_exact_attribute_constraints(
        "I would like to know the army that has official name Land Force Command"
    )

    assert {"attribute_name": "official name", "value": "Land Force Command"} in constraints


def test_kqapro_extracts_birth_date_constraint():
    constraints = _agent()._extract_exact_attribute_constraints(
        "When did John Powell born 1936-03-10 die?"
    )

    assert {"attribute_name": "date of birth", "value": "1936-03-10"} in constraints


def test_kqapro_normalizes_iswc_typo():
    constraints = _agent()._extract_exact_attribute_constraints(
        "What name is Massachusetts known under ISCW T-011.363.815-5?"
    )

    assert {"attribute_name": "ISWC", "value": "T-011.363.815-5"} in constraints


def test_agent_analysis_context_overrides_accept_query_argument():
    kqapro = _agent()
    kqapro.use_fewshot = False
    context = kqapro._build_analysis_context(
        "Query",
        [],
        [],
        query="Which type of sport has IAB code 543?",
    )

    assert "EXACT ATTRIBUTE CONSTRAINTS DETECTED" in context
    assert "IAB code = 543" in context

    sciqa = object.__new__(SciQAAgent)
    context = sciqa._build_analysis_context("Factoid", [], [], query="Any query")

    assert "Question Type" in context


def test_final_answer_cleanup_strips_think_blocks():
    answer = _agent()._finalize_answer_text(
        "<think>internal reasoning</think>\nLand Force Command",
        "Query",
    )

    assert answer == "Land Force Command"


def test_verify_answer_cleanup_normalizes_affirmative_statement():
    answer = _agent()._finalize_answer_text(
        "Reno's population is greater than 100.",
        "Verify",
    )

    assert answer == "yes"


def test_verify_answer_cleanup_normalizes_negative_statement():
    answer = _agent()._finalize_answer_text(
        "The condition is not satisfied.",
        "Verify",
    )

    assert answer == "no"


def test_fast_path_extractors_parse_answer_values():
    node_id, name = BaseKBQAAgent._extract_first_match_identity(
        '{"matches": [{"original_id": "Q42", "name": "Douglas Adams"}]}'
    )
    assert node_id == "Q42"
    assert name == "Douglas Adams"

    attr_values = BaseKBQAAgent._extract_attribute_values(
        '{"values": [{"value": "1952-03-11"}, {"value": "42", "unit": "year"}]}'
    )
    assert attr_values == ["1952-03-11", "42 year"]

    relation_ids = BaseKBQAAgent._extract_relation_ids(
        '{"triples": [{"related_id": "Q1"}, {"related_id": "Q1"}, {"related_id": "Q2"}]}'
    )
    assert relation_ids == ["Q1", "Q2"]

    labels = BaseKBQAAgent._extract_batch_labels(
        '{"resolved": {"Q1": "Universe", "Q2": "Earth"}}'
    )
    assert labels == {"Q1": "Universe", "Q2": "Earth"}


def test_sciqa_payload_node_type_filter_accepts_normalized_payload_type():
    assert _payload_node_type_matches("comparison", "Comparison")
    assert _payload_node_type_matches("orkgc:ResearchField", "Research Field")
    assert not _payload_node_type_matches("paper", "Comparison")


def test_sciqa_schema_helpers_compact_values_and_paths():
    binding = {
        "obj": {"value": "http://orkg.org/orkg/resource/R43133"},
        "objLabel": {"value": "Installed capacity"},
        "nestedValue": {"value": "367.570798339843756"},
    }

    assert _short_orkg_term("http://orkg.org/orkg/predicate/P43133") == "P43133"
    assert _schema_display_value(binding, "obj", "objLabel", "nestedValue") == "367.570798339843756"
    assert _looks_numeric_value("n=54 patients")
    assert not _looks_numeric_value("Heat sector")

    hint = _schema_usage_hint(
        "R153801",
        "P43133",
        intermediate_predicate="P43135",
    )
    assert 'intermediate_predicate="P43135"' in hint
    assert 'value_predicate="P43133"' in hint
