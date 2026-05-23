from ama_kbqa.agents.kqapro_agent.agent import KQAProAgent
from ama_kbqa.agents.sciqa_agent.agent import SciQAAgent
from ama_kbqa.framework.base_agent import BaseKBQAAgent
from ama_kbqa.postprocessing import evaluate_accuracy_simple, numeric_answer_equivalent
from ama_kbqa.server.sciqa_server import (
    AggregateComparisonValues,
    DiagnoseComparisonAggregation,
    _build_comparison_aggregation_diagnostics,
    _lexical_candidate_token_sets,
    _lexical_label_score,
    _looks_numeric_value,
    _looks_rollup_label,
    _normalize_aggregation_name,
    _parse_numeric_value,
    _payload_node_type_matches,
    _rollup_candidate_value,
    _rollup_intermediate_candidates,
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


def test_sciqa_lexical_label_score_handles_extra_query_words():
    assert _lexical_label_score("text summarization before 2002", "Summarization before 2002") > 0.9
    assert _lexical_label_score("summarization comparison", "Extractive Text Summarization") == 0


def test_sciqa_lexical_token_sets_allow_one_context_word():
    token_sets = _lexical_candidate_token_sets({"text", "summarization", "before", "2002"})

    assert ("2002", "before", "summarization", "text") in token_sets
    assert ("2002", "before", "summarization") in token_sets
    assert _lexical_candidate_token_sets({"token1", "token2", "token3", "token4", "token5", "token6", "token7"}) == []


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
    assert "intermediate_filter_value" in hint

    path_hint = _schema_usage_hint(
        "R150337",
        "P37668",
        intermediate_path=["P37586", "P37675"],
    )
    assert 'intermediate_path="P37586,P37675"' in path_hint
    assert 'value_predicate="P37668"' in path_hint


def test_sciqa_aggregate_tool_exposes_intermediate_filter_parameter():
    params = AggregateComparisonValues.parameters["properties"]
    required = AggregateComparisonValues.parameters.get("required", [])

    assert "intermediate_filter_value" in params
    assert "intermediate_filter_match" in params
    assert "intermediate_path" in params
    assert "group_by_path" in params
    assert "group_by_intermediate" in params
    assert "value_predicate" not in required
    assert "comparison_id" not in required


def test_sciqa_diagnostics_helper_reports_denominator_candidates():
    rows = [
        {"contrib": "C1", "intermediate": "Solar", "group": None, "value": "10"},
        {"contrib": "C1", "intermediate": "Wind", "group": None, "value": "20"},
        {"contrib": "C2", "intermediate": "Solar", "group": None, "value": "30"},
    ]

    diagnostics = _build_comparison_aggregation_diagnostics(
        rows,
        value_parser="leading_number",
        scope_contribution_count=3,
    )

    assert diagnostics["population"]["matched_rows"] == 3
    assert diagnostics["population"]["distinct_contributions_with_values"] == 2
    assert diagnostics["population"]["contributions_without_matching_value"] == 1
    assert diagnostics["denominator_candidates"]["row_level"]["avg"] == 20
    assert diagnostics["denominator_candidates"]["contribution_sum_level"]["avg"] == 30
    assert diagnostics["denominator_candidates"]["contribution_mean_level"]["avg"] == 22.5
    assert "intermediate_label_mean_level" in diagnostics["denominator_candidates"]
    assert diagnostics["warnings"]


def test_sciqa_rollup_intermediate_candidates_surface_total_rows():
    rows = [
        {"contrib": "C1", "intermediate": "all sources", "value": "10"},
        {"contrib": "C2", "intermediate": "all sources", "value": "20"},
        {"contrib": "C1", "intermediate": "wind power", "value": "5"},
    ]

    candidates = _rollup_intermediate_candidates(rows)

    assert _looks_rollup_label("all sources")
    assert not _looks_rollup_label("wind power")
    assert candidates[0]["intermediate_filter_value"] == "all sources"
    assert candidates[0]["numeric_summary"]["avg"] == 15
    assert candidates[0]["distinct_contributions"] == 2
    assert _rollup_candidate_value(candidates[0], "avg") == 15
    assert _rollup_candidate_value(candidates[0], "count") == 2


def test_sciqa_diagnostics_tool_schema_exposes_generic_parameters():
    params = DiagnoseComparisonAggregation.parameters["properties"]

    assert "intermediate_filter_value" in params
    assert "comparison_ids" in params
    assert "value_parser" in params
    assert _parse_numeric_value("n=54", "embedded_number") == 54
    assert _normalize_aggregation_name("frequency") == "mode_top"
    assert _normalize_aggregation_name("mean") == "avg"


def test_numeric_answer_equivalence_allows_benchmark_rounding_noise():
    gold = "367.570798339843756"
    predicted = (
        'The average installed capacity in "Greenhouse Gas Reduction Scenarios '
        'for Germany" is 367.57 GW across 25 scenario contributions.'
    )

    assert numeric_answer_equivalent(predicted, gold)
    assert evaluate_accuracy_simple(predicted, gold, "Non-Factoid\nCount")
    assert not numeric_answer_equivalent("Heat sector 8", "Heat sector 8")
