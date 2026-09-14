"""Tests for the live graph panel's normalisers (frontend/utils/live_graph_data.py).

Two properties matter here and neither is obvious from the code:

1. **Shape coverage.** The journal's `verified_facts` list is written by a
   dozen different tools in at least seven different shapes (documented at
   `kqapro_server.py:1804`), and `found_values` mixes scalars, lists, and
   dicts that may be a neighbour entity or a literal. Every shape gets its
   own test, because the failure mode of a missed shape is silent: the graph
   just renders fewer edges.
2. **Tolerance.** All of this input is produced by an LLM-driven agent
   against live servers. A malformed fact, a non-JSON tool result or a
   `None` where a string belongs must be skipped, never raised: an exception
   here would take down the chat page mid-run.

The tool-result fixtures are built by instantiating the real server response
models and dumping them, so a schema change in `kqapro_server.py` /
`sciqa_server.py` breaks these tests instead of silently emptying the panel.
Three tools return a plain `dict` rather than a Pydantic model
(`GetNodeSummary`, `GetRelationTargets`, `GetResourceDetails`); their
fixtures are hand-written from the dict literals in the server source and
name the line they mirror.
"""

from __future__ import annotations

import json

from ama_kbqa.frontend.utils.live_graph_data import (
    GraphData,
    GraphNode,
    answer_node_ids,
    apply_caps,
    graph_from_journal_state,
    graph_from_tool_result,
    graph_from_trace,
    source_for_agent,
    to_vis_payload,
)
from ama_kbqa.server import kqapro_server as kqa
from ama_kbqa.server import sciqa_server as sci


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _edge_triples(graph: GraphData) -> set[tuple[str, str, str]]:
    return {(e.src, e.label, e.dst) for e in graph.edges.values()}


def _labels(graph: GraphData) -> dict[str, str]:
    return {n.id: n.label for n in graph.nodes.values()}


# ---------------------------------------------------------------------------
# source_for_agent
# ---------------------------------------------------------------------------

class TestSourceForAgent:
    def test_maps_every_spelling_of_the_specialists(self):
        for name in ("kqapro_agent", "KQAProAgent", "KQAPro"):
            assert source_for_agent(name) == "kqapro"
        for name in ("sciqa_agent", "SciQAAgent", "SciQA", "ORKG"):
            assert source_for_agent(name) == "sciqa"

    def test_unknown_and_orchestrator_callers_are_unattributed(self):
        assert source_for_agent("Orchestrator (Router)") == ""
        assert source_for_agent(None) == ""
        assert source_for_agent("") == ""

    def test_accepts_an_agent_object_via_its_name(self):
        class _Agent:
            name = "sciqa_agent"

        assert source_for_agent(_Agent()) == "sciqa"


# ---------------------------------------------------------------------------
# Journal: visited_nodes + found_values
# ---------------------------------------------------------------------------

class TestJournalVisitedNodes:
    def test_visited_nodes_become_labelled_entities(self):
        graph = graph_from_journal_state(
            {"visited_nodes": {"Q937": "Albert Einstein"}}, source="kqapro"
        )
        node = graph.nodes["Q937"]
        assert node.kind == "entity"
        assert node.label == "Albert Einstein"
        assert node.source == "kqapro"

    def test_placeholder_label_counts_as_unlabelled(self):
        graph = graph_from_journal_state(
            {"visited_nodes": {"Q1": "(resolving label)", "Q2": "Q2"}},
            source="kqapro",
        )
        assert _labels(graph) == {"Q1": "Q1", "Q2": "Q2"}
        assert "label pending" in graph.nodes["Q1"].title

    def test_long_labels_are_truncated_but_kept_whole_in_the_tooltip(self):
        long_label = "A very long entity label that will not fit on a node at all"
        graph = graph_from_journal_state(
            {"visited_nodes": {"Q1": long_label}}, source="kqapro"
        )
        node = graph.nodes["Q1"]
        assert len(node.label) == 40 and node.label.endswith("…")
        assert long_label in node.title

    def test_empty_state_is_empty_graph(self):
        assert graph_from_journal_state({}, source="kqapro").is_empty()


class TestJournalFoundValues:
    def test_scalar_value_becomes_a_literal_leaf(self):
        graph = graph_from_journal_state(
            {"visited_nodes": {"Q1": "Berlin"}, "found_values": {"Q1": {"population": 3700000}}},
            source="kqapro",
        )
        assert ("Q1", "population", "lit:Q1:population:0") in _edge_triples(graph)
        assert graph.nodes["lit:Q1:population:0"].kind == "literal"

    def test_list_of_scalars_becomes_one_literal_per_item(self):
        graph = graph_from_journal_state(
            {"found_values": {"Q1": {"population": ["100", "200"]}}}, source="kqapro"
        )
        assert graph.nodes["lit:Q1:population:0"].label == "100"
        assert graph.nodes["lit:Q1:population:1"].label == "200"

    def test_dict_item_with_related_id_becomes_an_entity_edge(self):
        # GetRelationDetails' found_values shape (kqapro_server.py:3203).
        graph = graph_from_journal_state(
            {
                "visited_nodes": {"Q1": "Berlin"},
                "found_values": {"Q1": {"country": [{"value": "Germany", "related_id": "Q183"}]}},
            },
            source="kqapro",
        )
        assert graph.nodes["Q183"].kind == "entity"
        assert graph.nodes["Q183"].label == "Germany"
        assert ("Q1", "country", "Q183") in _edge_triples(graph)

    def test_dict_item_with_unit_keeps_the_unit_in_the_label(self):
        graph = graph_from_journal_state(
            {"found_values": {"Q1": {"area": [{"value": "891.7", "unit": "square_kilometre"}]}}},
            source="kqapro",
        )
        assert graph.nodes["lit:Q1:area:0"].label == "891.7 square_kilometre"

    def test_id_shaped_scalar_becomes_an_entity_not_a_literal(self):
        graph = graph_from_journal_state(
            {"found_values": {"Q1": {"country": "Q183"}}}, source="kqapro"
        )
        assert graph.nodes["Q183"].kind == "entity"
        assert ("Q1", "country", "Q183") in _edge_triples(graph)

    def test_underscore_keys_are_tooltip_only(self):
        # SciQA's GetResourceDetails writes {"_type": [...]} (sciqa_server.py:1746).
        graph = graph_from_journal_state(
            {"found_values": {"R1": {"_type": ["Paper"]}}}, source="sciqa"
        )
        assert set(graph.nodes) == {"R1"}
        assert "_type" in graph.nodes["R1"].title
        assert not graph.edges

    def test_literal_children_are_capped_per_attribute(self):
        graph = graph_from_journal_state(
            {"found_values": {"Q1": {"alias": [str(i) for i in range(30)]}}},
            source="kqapro",
        )
        literals = [n for n in graph.nodes.values() if n.kind == "literal"]
        assert len(literals) == 8

    def test_relation_to_keys_are_predicate_names_not_values(self):
        # GetRelationBetween: found_values[subj]["relation_to_<obj>"] = predicate
        state = {
            "visited_nodes": {"Q1": "Berlin", "Q183": "Germany"},
            "found_values": {"Q1": {"relation_to_Q183": "country"}},
            "verified_facts": [
                {"subject": "Q1", "relation": "country", "related_id": "Q183",
                 "direction": "forward", "source": "GetRelationBetween"},
            ],
        }
        graph = graph_from_journal_state(state, source="kqapro")
        assert graph.stats()["literals"] == 0
        assert _edge_triples(graph) == {("Q1", "country", "Q183")}

    def test_sparql_result_buckets_are_not_entities(self):
        # RunSPARQL parks rows under a synthetic key (kqapro_server.py:3712)
        # that is not an entity id at all.
        graph = graph_from_journal_state(
            {"found_values": {"sparql_result_1": {"query": "SELECT ...", "result_count": 3}}},
            source="kqapro",
        )
        assert graph.is_empty()


# ---------------------------------------------------------------------------
# Journal: verified_facts shapes
# ---------------------------------------------------------------------------

class TestVerifiedFactShapes:
    def test_subject_predicate_object_entity(self):
        graph = graph_from_journal_state(
            {
                "visited_nodes": {"Q1": "Berlin", "Q2": "Germany"},
                "verified_facts": [
                    {"subject": "Q1", "predicate": "country", "object": "Q2", "source": "tool"}
                ],
            },
            source="kqapro",
        )
        assert ("Q1", "country", "Q2") in _edge_triples(graph)

    def test_subject_predicate_object_literal(self):
        graph = graph_from_journal_state(
            {
                "visited_nodes": {"Q1": "Berlin"},
                "verified_facts": [
                    {"subject": "Q1", "predicate": "population", "object": "3700000"}
                ],
            },
            source="kqapro",
        )
        literal = graph.nodes["lit:Q1:population:0"]
        assert literal.kind == "literal" and literal.label == "3700000"
        assert ("Q1", "population", "lit:Q1:population:0") in _edge_triples(graph)

    def test_relation_related_id_forward(self):
        graph = graph_from_journal_state(
            {
                "verified_facts": [
                    {
                        "subject": "Q1", "relation": "country", "related_id": "Q183",
                        "direction": "forward", "source": "GetRelationDetails",
                    }
                ]
            },
            source="kqapro",
        )
        assert ("Q1", "country", "Q183") in _edge_triples(graph)

    def test_relation_related_id_is_reversed_for_backward_directions(self):
        # GetRelationDetails writes "backward" (kqapro_server.py:3177);
        # GetRelationBetween writes "inverse_only" (kqapro_server.py:5077).
        for direction in ("backward", "inverse_only", "reverse"):
            graph = graph_from_journal_state(
                {
                    "verified_facts": [
                        {
                            "subject": "Q1", "relation": "country",
                            "related_id": "Q183", "direction": direction,
                        }
                    ]
                },
                source="kqapro",
            )
            assert ("Q183", "country", "Q1") in _edge_triples(graph), direction

    def test_subject_attribute_value_with_unit(self):
        graph = graph_from_journal_state(
            {
                "verified_facts": [
                    {"subject": "Q1", "attribute": "area", "value": "891.7",
                     "unit": "square_kilometre"}
                ]
            },
            source="kqapro",
        )
        assert graph.nodes["lit:Q1:area:0"].label == "891.7 square_kilometre"
        assert ("Q1", "area", "lit:Q1:area:0") in _edge_triples(graph)

    def test_subject_predicate_objects_list_is_capped_at_five(self):
        # ExploreNeighborhood's shape (kqapro_server.py:3342).
        graph = graph_from_journal_state(
            {
                "verified_facts": [
                    {
                        "subject": "Q1",
                        "predicate": "cast member",
                        "objects": [{"value": f"Q{i}", "type": "uri"} for i in range(10, 20)],
                        "source": "ExploreNeighborhood",
                    }
                ]
            },
            source="kqapro",
        )
        assert len(graph.edges) == 5
        assert ("Q1", "cast member", "Q10") in _edge_triples(graph)

    def test_relation_qualifiers_becomes_an_edge(self):
        graph = graph_from_journal_state(
            {
                "verified_facts": [
                    {
                        "type": "relation_qualifiers", "node": "Q1",
                        "relation": "spouse", "target": "Q2",
                        "qualifiers": ["start time", "end time"],
                    }
                ]
            },
            source="kqapro",
        )
        assert ("Q1", "spouse", "Q2") in _edge_triples(graph)
        assert "start time" in graph.nodes["Q1"].title

    def test_edge_qualifiers_adds_a_tooltip_line_only(self):
        graph = graph_from_journal_state(
            {
                "verified_facts": [
                    {
                        "type": "edge_qualifiers", "node": "Q1",
                        "attribute": "population", "qualifiers": ["point in time"],
                    }
                ]
            },
            source="kqapro",
        )
        assert set(graph.nodes) == {"Q1"}
        assert not graph.edges
        assert "point in time" in graph.nodes["Q1"].title

    def test_qualifier_value_adds_a_tooltip_line_only(self):
        graph = graph_from_journal_state(
            {
                "verified_facts": [
                    {
                        "type": "qualifier_value", "subject": "Q1",
                        "predicate": "spouse", "target": "Q2",
                        "qualifier": "start time", "value_count": 1,
                    }
                ]
            },
            source="kqapro",
        )
        assert set(graph.nodes) == {"Q1"}
        assert not graph.edges

    def test_free_text_and_sparql_summaries_are_ignored(self):
        graph = graph_from_journal_state(
            {
                "verified_facts": [
                    {"fact": "Berlin is the capital of Germany", "source": "manual"},
                    {"raw": "something"},
                    {"query_type": "SPARQL", "variables": ["x"], "result_count": 3,
                     "results": [], "source": "RunSPARQL"},
                ]
            },
            source="kqapro",
        )
        assert graph.is_empty()

    def test_sciqa_shaped_facts_work_the_same(self):
        graph = graph_from_journal_state(
            {
                "visited_nodes": {"R1": "A paper"},
                "found_values": {"R1": {"P30": [{"id": "R2", "label": "Contribution 1"}]}},
                "verified_facts": [
                    {"subject": "R1", "predicate": "has_author", "object": "R99"}
                ],
                "kg_name": "ORKG",
            },
            source="sciqa",
        )
        assert graph.nodes["R2"].label == "Contribution 1"
        assert ("R1", "P30", "R2") in _edge_triples(graph)
        assert ("R1", "has_author", "R99") in _edge_triples(graph)
        assert all(n.source == "sciqa" for n in graph.nodes.values())


class TestMalformedJournalInput:
    def test_none_and_non_dict_states_do_not_raise(self):
        assert graph_from_journal_state(None, source="").is_empty()
        assert graph_from_journal_state("not a state", source="").is_empty()
        assert graph_from_journal_state({"visited_nodes": "nope"}, source="").is_empty()

    def test_fact_with_none_object_is_skipped_not_rendered(self):
        graph = graph_from_journal_state(
            {
                "visited_nodes": {"Q1": "Berlin"},
                "verified_facts": [
                    {"subject": "Q1", "predicate": "country", "object": None},
                    None,
                    "a bare string",
                    {"subject": None, "predicate": "x", "object": "Q2"},
                ],
            },
            source="kqapro",
        )
        assert set(graph.nodes) == {"Q1"}
        assert not graph.edges


# ---------------------------------------------------------------------------
# id heuristic
# ---------------------------------------------------------------------------

class TestIdHeuristic:
    def test_uri_values_are_shortened_but_keep_the_full_uri_in_the_tooltip(self):
        graph = graph_from_journal_state(
            {"found_values": {"Q1": {"country": "http://kqapro.org/entity/Q183"}}},
            source="kqapro",
        )
        assert "Q183" in graph.nodes
        assert "http://kqapro.org/entity/Q183" in graph.nodes["Q183"].title

    def test_orkg_curies_and_letter_prefixes_are_ids(self):
        graph = graph_from_journal_state(
            {"found_values": {"R1": {"p": ["orkgr:R55", "C12", "L7", "P31"]}}},
            source="sciqa",
        )
        assert {"R55", "C12", "L7", "P31"} <= set(graph.nodes)

    def test_plain_values_are_not_ids(self):
        graph = graph_from_journal_state(
            {"found_values": {"Q1": {"name": ["Berlin", "1871", "Q-not-an-id"]}}},
            source="kqapro",
        )
        assert all(
            n.kind == "literal"
            for n in graph.nodes.values()
            if n.id.startswith("lit:")
        )
        assert len([n for n in graph.nodes.values() if n.kind == "literal"]) == 3


# ---------------------------------------------------------------------------
# GraphData: merge, stats, caps
# ---------------------------------------------------------------------------

class TestMergeAndCaps:
    def test_entity_kind_wins_over_candidate(self):
        candidates = GraphData(nodes={"Q1": GraphNode("Q1", "Q1", "candidate", "kqapro")})
        visited = GraphData(nodes={"Q1": GraphNode("Q1", "Berlin", "entity", "kqapro")})
        assert candidates.merge(visited).nodes["Q1"].kind == "entity"
        assert visited.merge(candidates).nodes["Q1"].kind == "entity"

    def test_real_label_wins_over_id_as_label(self):
        bare = GraphData(nodes={"Q1": GraphNode("Q1", "Q1", "entity", "")})
        named = GraphData(nodes={"Q1": GraphNode("Q1", "Berlin", "entity", "kqapro")})
        assert bare.merge(named).nodes["Q1"].label == "Berlin"
        assert named.merge(bare).nodes["Q1"].label == "Berlin"

    def test_merge_keeps_the_known_source_and_the_original_order(self):
        first = GraphData(nodes={"Q1": GraphNode("Q1", "Q1", "entity", "", order=0)})
        second = GraphData(nodes={"Q1": GraphNode("Q1", "Berlin", "entity", "sciqa", order=5)})
        merged = first.merge(second)
        assert merged.nodes["Q1"].source == "sciqa"
        assert merged.nodes["Q1"].order == 0

    def test_merge_does_not_mutate_either_input(self):
        a = GraphData(nodes={"Q1": GraphNode("Q1", "Berlin")})
        b = GraphData(nodes={"Q2": GraphNode("Q2", "Germany")})
        a.merge(b)
        assert set(a.nodes) == {"Q1"} and set(b.nodes) == {"Q2"}

    def test_stats_counts_each_kind(self):
        graph = graph_from_journal_state(
            {
                "visited_nodes": {"Q1": "Berlin"},
                "found_values": {"Q1": {"population": "3700000"}},
            },
            source="kqapro",
        )
        stats = graph.stats()
        assert stats == {"entities": 1, "literals": 1, "candidates": 0, "edges": 1}

    def test_caps_keep_entities_then_literals_then_candidates(self):
        graph = GraphData(nodes={
            "cand": GraphNode("cand", "cand", "candidate", order=0),
            "lit:x": GraphNode("lit:x", "42", "literal", order=1),
            "Q1": GraphNode("Q1", "Berlin", "entity", order=2),
        })
        capped = apply_caps(graph, max_nodes=2)
        assert set(capped.nodes) == {"Q1", "lit:x"}
        assert capped.truncated_nodes == 1

    def test_caps_drop_edges_whose_endpoints_are_gone(self):
        graph = graph_from_journal_state(
            {
                "visited_nodes": {"Q1": "Berlin"},
                "found_values": {"Q1": {"population": ["1", "2", "3"]}},
            },
            source="kqapro",
        )
        capped = apply_caps(graph, max_nodes=1)
        assert set(capped.nodes) == {"Q1"}
        assert not capped.edges

    def test_caps_are_a_no_op_below_the_limits(self):
        graph = graph_from_journal_state({"visited_nodes": {"Q1": "Berlin"}}, source="kqapro")
        assert apply_caps(graph) is graph


# ---------------------------------------------------------------------------
# answer_node_ids
# ---------------------------------------------------------------------------

class TestAnswerNodeIds:
    def test_id_appearing_verbatim_matches(self):
        graph = graph_from_journal_state({"visited_nodes": {"Q937": "Q937"}}, source="kqapro")
        assert answer_node_ids(graph, "The entity is Q937.") == {"Q937"}

    def test_id_does_not_match_as_a_substring_of_a_longer_id(self):
        graph = graph_from_journal_state(
            {"visited_nodes": {"Q1": "Q1", "Q123": "Q123"}}, source="kqapro"
        )
        assert answer_node_ids(graph, "The entity is Q123.") == {"Q123"}

    def test_label_matches_on_word_boundaries_case_insensitively(self):
        graph = graph_from_journal_state(
            {"visited_nodes": {"Q1": "Berlin", "Q2": "Bern"}}, source="kqapro"
        )
        assert answer_node_ids(graph, "the capital is berlin") == {"Q1"}

    def test_short_labels_do_not_match(self):
        graph = graph_from_journal_state({"visited_nodes": {"Q1": "US"}}, source="kqapro")
        assert answer_node_ids(graph, "in the US") == set()

    def test_literal_value_matches_as_a_substring(self):
        graph = graph_from_journal_state(
            {"found_values": {"Q1": {"population": "3700000"}}}, source="kqapro"
        )
        assert "lit:Q1:population:0" in answer_node_ids(graph, "Berlin has 3700000 people.")

    def test_no_answer_means_no_highlight(self):
        graph = graph_from_journal_state({"visited_nodes": {"Q1": "Berlin"}}, source="kqapro")
        assert answer_node_ids(graph, None) == set()
        assert answer_node_ids(graph, "") == set()

    def test_highlighting_is_capped_at_fifty(self):
        labels = {f"Q{i}": f"Entity{i:03d}" for i in range(80)}
        graph = graph_from_journal_state({"visited_nodes": labels}, source="kqapro")
        answer = " ".join(labels.values())
        assert len(answer_node_ids(graph, answer)) == 50


# ---------------------------------------------------------------------------
# to_vis_payload
# ---------------------------------------------------------------------------

class TestToVisPayload:
    def test_groups_follow_kind_and_source(self):
        graph = GraphData(nodes={
            "Q1": GraphNode("Q1", "Berlin", "entity", "kqapro"),
            "R1": GraphNode("R1", "Paper", "entity", "sciqa"),
            "X1": GraphNode("X1", "Unknown", "entity", ""),
            "lit:x": GraphNode("lit:x", "42", "literal", "kqapro"),
            "C1": GraphNode("C1", "Candidate", "candidate", "kqapro"),
        })
        groups = {n["id"]: n["group"] for n in to_vis_payload(graph, highlight=set(), new_ids=set())["nodes"]}
        assert groups == {
            "Q1": "kqapro", "R1": "sciqa", "X1": "entity",
            "lit:x": "literal", "C1": "candidate",
        }

    def test_highlight_and_new_flags_are_set_per_node(self):
        graph = GraphData(nodes={
            "Q1": GraphNode("Q1", "Berlin"),
            "Q2": GraphNode("Q2", "Germany"),
        })
        payload = to_vis_payload(graph, highlight={"Q1"}, new_ids={"Q2"})
        flags = {n["id"]: (n["highlighted"], n["new"]) for n in payload["nodes"]}
        assert flags == {"Q1": (True, False), "Q2": (False, True)}

    def test_edges_use_vis_from_to_keys(self):
        graph = graph_from_journal_state(
            {"verified_facts": [{"subject": "Q1", "predicate": "country", "object": "Q2"}]},
            source="kqapro",
        )
        edge = to_vis_payload(graph, highlight=set(), new_ids=set())["edges"][0]
        assert edge["from"] == "Q1" and edge["to"] == "Q2"
        assert edge["id"] == "Q1|country|Q2"


# ---------------------------------------------------------------------------
# Tool-result extractors (fixtures from the real response models)
# ---------------------------------------------------------------------------

def _kqapro_search_json() -> str:
    return kqa.SearchResponse(
        matches=[
            kqa.NodeMatch(
                original_id="Q937", name="Albert Einstein", node_type="entity",
                relevance_score=0.91, available_attributes=["date of birth"],
                available_predicates=["country of citizenship"],
            ),
            kqa.NodeMatch(
                original_id="Q11879", name="Einstein (crater)", node_type="entity",
                relevance_score=0.72,
            ),
        ],
        result_count=2,
    ).model_dump_json()


def _sciqa_search_json() -> str:
    return sci.SearchResponse(
        matches=[
            sci.ResourceMatch(
                original_id="R12345", name="A COVID-19 detection paper",
                node_type="Paper", relevance_score=0.88,
                available_predicates=["P30"],
            )
        ],
        result_count=1,
    ).model_dump_json()


class TestSearchExtractors:
    def test_find_node_matches_become_candidates(self):
        graph = graph_from_tool_result(
            "FindNode", {"semantic_node_name": "Einstein"},
            _kqapro_search_json(), source="kqapro",
        )
        assert set(graph.nodes) == {"Q937", "Q11879"}
        assert all(n.kind == "candidate" for n in graph.nodes.values())
        assert graph.nodes["Q937"].label == "Albert Einstein"
        assert not graph.edges

    def test_lookup_entity_by_name_uses_the_same_shape(self):
        graph = graph_from_tool_result(
            "LookupEntityByName", {"entity_name": "Einstein"},
            _kqapro_search_json(), source="kqapro",
        )
        assert set(graph.nodes) == {"Q937", "Q11879"}

    def test_find_by_attribute_uses_the_same_shape(self):
        graph = graph_from_tool_result(
            "FindByAttribute", {"value": "UKE11", "attribute_name": "NUTS code"},
            _kqapro_search_json(), source="kqapro",
        )
        assert set(graph.nodes) == {"Q937", "Q11879"}

    def test_find_resource_matches_become_candidates(self):
        graph = graph_from_tool_result(
            "FindResource", {"semantic_query": "covid"},
            _sciqa_search_json(), source="sciqa",
        )
        node = graph.nodes["R12345"]
        assert node.kind == "candidate" and node.source == "sciqa"

    def test_lookup_resource_by_label_uses_the_same_shape(self):
        graph = graph_from_tool_result(
            "LookupResourceByLabel", {"label": "covid"},
            _sciqa_search_json(), source="sciqa",
        )
        assert set(graph.nodes) == {"R12345"}

    def test_candidates_are_capped_at_ten(self):
        payload = kqa.SearchResponse(
            matches=[
                kqa.NodeMatch(
                    original_id=f"Q{i}", name=f"Entity {i}",
                    node_type="entity", relevance_score=0.5,
                )
                for i in range(25)
            ],
            result_count=25,
        ).model_dump_json()
        graph = graph_from_tool_result("FindNode", {}, payload, source="kqapro")
        assert len(graph.nodes) == 10


class TestGetNodeSummaryExtractor:
    # GetNodeSummary returns a plain dict, not a Pydantic model; this mirrors
    # the documented return value at kqapro_server.py:2451-2463.
    _RESULT = json.dumps({
        "node_id": "Q54089",
        "name": "Barnstable County",
        "node_type": "entity",
        "attributes": {"population": ["215918", "214990"], "area": ["1000.5 square_kilometre"]},
        "relations": {"country": ["Q30"], "located in time zone": ["Q941"]},
        "summary_stats": {"attribute_count": 2, "relation_count": 2},
        "status": "Success",
    })

    def test_relations_become_neighbour_edges(self):
        graph = graph_from_tool_result("GetNodeSummary", {"node_id": "Q54089"},
                                       self._RESULT, source="kqapro")
        assert ("Q54089", "country", "Q30") in _edge_triples(graph)
        assert ("Q54089", "located in time zone", "Q941") in _edge_triples(graph)
        assert graph.nodes["Q30"].kind == "entity"
        # No journal label yet: the id stands in until one arrives.
        assert graph.nodes["Q30"].label == "Q30"

    def test_attributes_become_literal_leaves(self):
        graph = graph_from_tool_result("GetNodeSummary", {}, self._RESULT, source="kqapro")
        literals = {n.label for n in graph.nodes.values() if n.kind == "literal"}
        assert literals == {"215918", "214990", "1000.5 square_kilometre"}

    def test_base_node_keeps_its_name(self):
        graph = graph_from_tool_result("GetNodeSummary", {}, self._RESULT, source="kqapro")
        assert graph.nodes["Q54089"].label == "Barnstable County"


class TestGetRelationDetailsExtractor:
    def test_triples_become_edges_from_the_base_node(self):
        result = kqa.RelationDetailsResponse(
            node_id="Q54089",
            relation_name="country",
            triples=[
                {"related_id": "Q30", "related_uri": "http://kqapro.org/entity/Q30",
                 "direction": "forward"},
                {"related_id": "Q99", "related_uri": "http://kqapro.org/entity/Q99",
                 "direction": "backward"},
            ],
            status="Found 2 relation(s)",
        ).model_dump_json()
        graph = graph_from_tool_result(
            "GetRelationDetails",
            {"base_node_id": "Q54089", "relation_name": "country"},
            result, source="kqapro",
        )
        assert ("Q54089", "country", "Q30") in _edge_triples(graph)
        # "backward" means the stored triple points at the base node.
        assert ("Q99", "country", "Q54089") in _edge_triples(graph)

    def test_empty_triples_yield_only_the_base_node(self):
        result = kqa.RelationDetailsResponse(
            node_id="Q1", relation_name="country", triples=[], status="No relations found",
        ).model_dump_json()
        graph = graph_from_tool_result("GetRelationDetails", {}, result, source="kqapro")
        assert set(graph.nodes) == {"Q1"}
        assert not graph.edges


class TestSciqaExplorationExtractors:
    # GetRelationTargets / GetResourceDetails return hand-built JSON dicts,
    # not Pydantic models: these mirror sciqa_server.py:1864 and :1749.
    _TARGETS = json.dumps({
        "resource_id": "R12345",
        "predicate": "P30",
        "targets": [
            {"id": "R678", "label": "Contribution 1"},
            {"value": "2021"},
        ],
        "count": 2,
        "status": "Found 2 targets for P30",
    })

    _DETAILS = json.dumps({
        "resource_id": "R12345",
        "label": "A COVID-19 detection paper",
        "types": ["Paper"],
        "relations": {
            "P30": [{"id": "R678", "label": "Contribution 1"}],
            "P26": [{"value": "10.1234/abc"}],
        },
        "status": "Found 2 relation types for R12345",
    })

    def test_relation_targets_split_into_entities_and_literals(self):
        graph = graph_from_tool_result(
            "GetRelationTargets", {"resource_id": "R12345", "predicate": "P30"},
            self._TARGETS, source="sciqa",
        )
        assert ("R12345", "P30", "R678") in _edge_triples(graph)
        assert graph.nodes["R678"].label == "Contribution 1"
        literal_id = "lit:R12345:P30:1"
        assert graph.nodes[literal_id].label == "2021"
        assert ("R12345", "P30", literal_id) in _edge_triples(graph)

    def test_resource_details_reads_relations_and_types(self):
        graph = graph_from_tool_result(
            "GetResourceDetails", {"resource_id": "R12345"},
            self._DETAILS, source="sciqa",
        )
        assert graph.nodes["R12345"].label == "A COVID-19 detection paper"
        assert "Paper" in graph.nodes["R12345"].title
        assert ("R12345", "P30", "R678") in _edge_triples(graph)
        assert ("R12345", "P26", "lit:R12345:P26:0") in _edge_triples(graph)


class TestSparqlExtractors:
    def test_uri_bindings_become_entities_with_sibling_labels(self):
        result = kqa.SPARQLResponse(
            vars=["entity", "entityLabel", "count"],
            bindings=[
                {"entity": "http://kqapro.org/entity/Q937",
                 "entityLabel": "Albert Einstein", "count": "3"},
            ],
            raw_json={},
            result_count=1,
            returned_count=1,
        ).model_dump_json()
        graph = graph_from_tool_result("RunSPARQL", {"query": "SELECT ..."},
                                       result, source="kqapro")
        assert set(graph.nodes) == {"Q937"}
        assert graph.nodes["Q937"].label == "Albert Einstein"
        assert not graph.edges

    def test_sparql_nodes_are_capped(self):
        result = sci.SPARQLResponse(
            vars=["r"],
            bindings=[{"r": f"http://orkg.org/orkg/resource/R{i}"} for i in range(60)],
            raw_json={},
        ).model_dump_json()
        graph = graph_from_tool_result("RunORKGSPARQL", {}, result, source="sciqa")
        assert len(graph.nodes) == 25


class TestToolResultTolerance:
    def test_non_allow_listed_tools_yield_nothing(self):
        graph = graph_from_tool_result(
            "GetJournalSummary", {}, _kqapro_search_json(), source="kqapro"
        )
        assert graph.is_empty()

    def test_non_json_result_yields_nothing(self):
        assert graph_from_tool_result("FindNode", {}, "not json at all", source="").is_empty()
        assert graph_from_tool_result("FindNode", {}, "", source="").is_empty()
        assert graph_from_tool_result("FindNode", {}, None, source="").is_empty()

    def test_error_payload_yields_nothing(self):
        graph = graph_from_tool_result(
            "FindResource", {}, json.dumps({"error": "boom"}), source="sciqa"
        )
        assert graph.is_empty()

    def test_wrongly_shaped_payload_yields_nothing(self):
        graph = graph_from_tool_result(
            "GetNodeSummary", {}, json.dumps({"node_id": "Q1", "relations": "nope"}),
            source="kqapro",
        )
        assert set(graph.nodes) == {"Q1"}
        assert not graph.edges

    def test_json_array_result_yields_nothing(self):
        assert graph_from_tool_result("FindNode", {}, "[1, 2, 3]", source="").is_empty()


# ---------------------------------------------------------------------------
# graph_from_trace
# ---------------------------------------------------------------------------

def _event(span_id, parent, kind, name, *, attributes=None, payload=None):
    return {
        "trace_id": "T",
        "span_id": span_id,
        "parent_span_id": parent,
        "kind": kind,
        "name": name,
        "start_time_unix_nano": 0,
        "end_time_unix_nano": 1,
        "duration_ms": 1.0,
        "status": "ok",
        "is_event": False,
        "attributes": attributes or {},
        "payload": payload or {},
        "error": None,
    }


class TestGraphFromTrace:
    def test_single_agent_trace_uses_the_traces_own_agent_as_source(self):
        trace = {
            "trace_id": "T",
            "agent": "KQAPro",
            "events": [
                _event("s1", None, "agent_run", "KQAPRO"),
                _event("s2", "s1", "tool_call", "FindNode",
                       attributes={"tool_name": "FindNode"},
                       payload={"arguments": {}, "result": _kqapro_search_json()}),
            ],
            "journal_snapshots": [
                {"ts": 1.0, "trigger": "after:FindNode",
                 "state": {"visited_nodes": {"Q937": "Albert Einstein"}}},
            ],
        }
        graph = graph_from_trace(trace)
        assert graph.nodes["Q937"].source == "kqapro"
        # The journal upgrades the search candidate to a visited entity.
        assert graph.nodes["Q937"].kind == "entity"
        assert graph.nodes["Q11879"].kind == "candidate"

    def test_federated_trace_attributes_tool_calls_to_their_specialist(self):
        trace = {
            "trace_id": "T",
            "agent": "Orchestrator (Federated)",
            "events": [
                _event("orch", None, "agent_run", "ORCHESTRATOR"),
                _event("d1", "orch", "delegate", "kqapro_agent",
                       attributes={"sub_agent": "kqapro_agent"}),
                _event("d2", "orch", "delegate", "sciqa_agent",
                       attributes={"sub_agent": "sciqa_agent"}),
                _event("k1", "d1", "agent_run", "KQAPRO"),
                _event("k2", "k1", "tool_call", "FindNode",
                       payload={"arguments": {}, "result": _kqapro_search_json()}),
                _event("s1", "d2", "agent_run", "SCIQA"),
                _event("s2", "s1", "tool_call", "FindResource",
                       payload={"arguments": {}, "result": _sciqa_search_json()}),
            ],
            "journal_snapshots": [],
        }
        graph = graph_from_trace(trace)
        assert graph.nodes["Q937"].source == "kqapro"
        assert graph.nodes["R12345"].source == "sciqa"

    def test_only_the_last_snapshot_per_source_agent_is_merged(self):
        trace = {
            "agent": "Orchestrator (Router)",
            "events": [],
            "journal_snapshots": [
                {"state": {"visited_nodes": {"Q1": "Q1"}}, "source_agent": "kqapro_agent"},
                {"state": {"visited_nodes": {"Q1": "Berlin", "Q2": "Germany"}},
                 "source_agent": "kqapro_agent"},
                {"state": {"visited_nodes": {"R1": "Paper"}}, "source_agent": "sciqa_agent"},
            ],
        }
        graph = graph_from_trace(trace)
        assert set(graph.nodes) == {"Q1", "Q2", "R1"}
        assert graph.nodes["Q1"].label == "Berlin"
        assert graph.nodes["Q1"].source == "kqapro"
        assert graph.nodes["R1"].source == "sciqa"

    def test_kg_name_is_the_source_fallback_for_untagged_snapshots(self):
        trace = {
            "agent": "",
            "events": [],
            "journal_snapshots": [
                {"state": {"visited_nodes": {"R1": "Paper"}, "kg_name": "ORKG"}},
            ],
        }
        assert graph_from_trace(trace).nodes["R1"].source == "sciqa"

    def test_malformed_traces_never_raise(self):
        assert graph_from_trace(None).is_empty()
        assert graph_from_trace({}).is_empty()
        assert graph_from_trace({"events": "nope", "journal_snapshots": "nope"}).is_empty()
        assert graph_from_trace({"events": [None, {"kind": "tool_call"}]}).is_empty()
