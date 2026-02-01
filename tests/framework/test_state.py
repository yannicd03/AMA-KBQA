"""Tests for framework state management."""

import pytest
import json
from ama_kbqa.framework.state import JournalState, JournalManager


class TestJournalState:
    """Tests for JournalState dataclass."""

    def test_create_journal_state(self):
        """Test creating a JournalState."""
        state = JournalState(
            question_text="What is the capital of France?",
            question_type="QueryAttr",
            target_entities=["France"],
            target_attributes=["capital"],
            kg_name="KQAPro",
        )

        assert state.question_text == "What is the capital of France?"
        assert state.question_type == "QueryAttr"
        assert "France" in state.target_entities
        assert "capital" in state.target_attributes
        assert state.kg_name == "KQAPro"

    def test_journal_state_defaults(self):
        """Test JournalState default values."""
        state = JournalState()

        assert state.question_text == ""
        assert state.question_type == ""
        assert state.target_entities == []
        assert state.visited_nodes == {}
        assert state.found_values == {}
        assert state.verified_facts == []
        assert state.failed_attempts == []
        assert state.completed_steps == []
        assert state.current_plan == []
        assert state.partial_answer == ""

    def test_journal_state_to_dict(self):
        """Test serializing JournalState to dict."""
        state = JournalState(
            question_text="Test question",
            question_type="Count",
            visited_nodes={"Q123": "France"},
        )

        d = state.to_dict()
        assert d["question_text"] == "Test question"
        assert d["question_type"] == "Count"
        assert d["visited_nodes"]["Q123"] == "France"

    def test_journal_state_to_json(self):
        """Test serializing JournalState to JSON."""
        state = JournalState(
            question_text="Test question",
            question_type="Count",
        )

        json_str = state.to_json()
        parsed = json.loads(json_str)
        assert parsed["question_text"] == "Test question"

    def test_journal_state_from_dict(self):
        """Test creating JournalState from dict."""
        data = {
            "question_text": "Test question",
            "question_type": "QueryAttr",
            "visited_nodes": {"Q123": "Paris"},
            "found_values": {"Q123": {"capital": "Paris"}},
        }

        state = JournalState.from_dict(data)
        assert state.question_text == "Test question"
        assert state.visited_nodes["Q123"] == "Paris"
        assert state.found_values["Q123"]["capital"] == "Paris"

    def test_journal_state_to_summary_str(self):
        """Test generating summary string."""
        state = JournalState(
            question_text="What is the capital of France?",
            question_type="QueryAttr",
            kg_name="KQAPro",
            target_entities=["France"],
            target_attributes=["capital"],
            visited_nodes={"Q123": "France"},
            found_values={"Q123": {"capital": "Paris"}},
            completed_steps=["Found entity France (Q123)"],
        )

        summary = state.to_summary_str()

        assert "What is the capital of France?" in summary
        assert "QueryAttr" in summary
        assert "KQAPro" in summary
        assert "France" in summary
        assert "Q123" in summary
        assert "Paris" in summary
        assert "Found entity" in summary


class TestJournalManager:
    """Tests for JournalManager class."""

    def test_create_journal_manager(self):
        """Test creating a JournalManager."""
        manager = JournalManager(kg_name="KQAPro")

        assert manager.state.kg_name == "KQAPro"
        assert manager.state.question_text == ""

    def test_set_question(self):
        """Test setting the question."""
        manager = JournalManager()
        manager.set_question(
            text="What is the population of Paris?",
            qtype="QueryAttr",
            entities=["Paris"],
            attributes=["population"],
        )

        assert manager.state.question_text == "What is the population of Paris?"
        assert manager.state.question_type == "QueryAttr"
        assert "Paris" in manager.state.target_entities
        assert "population" in manager.state.target_attributes

    def test_add_visited_node(self):
        """Test adding a visited node."""
        manager = JournalManager()
        manager.add_visited_node("Q123", "Paris")
        manager.add_visited_node("Q456", "France")

        assert manager.state.visited_nodes["Q123"] == "Paris"
        assert manager.state.visited_nodes["Q456"] == "France"

    def test_add_found_value(self):
        """Test adding a found value."""
        manager = JournalManager()
        manager.add_found_value("Q123", "population", 2165000)
        manager.add_found_value("Q123", "country", "France")

        assert manager.state.found_values["Q123"]["population"] == 2165000
        assert manager.state.found_values["Q123"]["country"] == "France"

    def test_add_verified_fact(self):
        """Test adding a verified fact."""
        manager = JournalManager()
        manager.add_verified_fact(
            subject="Paris",
            predicate="located_in",
            obj="France",
            source="GetRelationDetails",
        )

        assert len(manager.state.verified_facts) == 1
        fact = manager.state.verified_facts[0]
        assert fact["subject"] == "Paris"
        assert fact["predicate"] == "located_in"
        assert fact["object"] == "France"
        assert fact["source"] == "GetRelationDetails"

    def test_add_failed_attempt(self):
        """Test adding a failed attempt."""
        manager = JournalManager()
        manager.add_failed_attempt("FindNode('Unknown') returned no results")

        assert len(manager.state.failed_attempts) == 1
        assert "Unknown" in manager.state.failed_attempts[0]

    def test_add_completed_step(self):
        """Test adding a completed step."""
        manager = JournalManager()
        manager.add_completed_step("Found entity Paris (Q123)")
        manager.add_completed_step("Retrieved population: 2165000")

        assert len(manager.state.completed_steps) == 2
        assert "Paris" in manager.state.completed_steps[0]

    def test_set_current_plan(self):
        """Test setting the current plan."""
        manager = JournalManager()
        manager.set_current_plan([
            "Find entity Paris",
            "Get population attribute",
            "Return answer",
        ])

        assert len(manager.state.current_plan) == 3
        assert "Find entity" in manager.state.current_plan[0]

    def test_set_partial_answer(self):
        """Test setting partial answer."""
        manager = JournalManager()
        manager.set_partial_answer("The population appears to be around 2 million")

        assert "2 million" in manager.state.partial_answer

    def test_reset(self):
        """Test resetting the manager."""
        manager = JournalManager(kg_name="KQAPro")
        manager.set_question("Test", "Count")
        manager.add_visited_node("Q123", "Test")

        manager.reset()

        assert manager.state.question_text == ""
        assert manager.state.visited_nodes == {}
        assert manager.state.kg_name == "KQAPro"  # Preserved

    def test_get_summary(self):
        """Test getting summary string."""
        manager = JournalManager(kg_name="KQAPro")
        manager.set_question("Test question?", "QueryAttr")
        manager.add_visited_node("Q123", "Test Entity")

        summary = manager.get_summary()

        assert "Test question?" in summary
        assert "Q123" in summary

    def test_get_json(self):
        """Test getting JSON string."""
        manager = JournalManager()
        manager.set_question("Test", "Count")

        json_str = manager.get_json()
        parsed = json.loads(json_str)

        assert parsed["question_text"] == "Test"

    def test_load_from_dict(self):
        """Test loading state from dict."""
        manager = JournalManager()
        manager.load_from_dict({
            "question_text": "Loaded question",
            "question_type": "Verify",
            "visited_nodes": {"Q999": "Loaded Node"},
        })

        assert manager.state.question_text == "Loaded question"
        assert manager.state.visited_nodes["Q999"] == "Loaded Node"

    def test_load_from_json(self):
        """Test loading state from JSON."""
        manager = JournalManager()
        json_str = '{"question_text": "JSON question", "question_type": "Count"}'
        manager.load_from_json(json_str)

        assert manager.state.question_text == "JSON question"

    def test_has_progress_since(self):
        """Test checking for progress."""
        manager = JournalManager()
        manager.set_question("Test", "Count")

        summary1 = manager.get_summary()

        # No change
        assert not manager.has_progress_since(summary1)

        # Add progress
        manager.add_visited_node("Q123", "New Node")
        assert manager.has_progress_since(summary1)

    def test_get_stats(self):
        """Test getting statistics."""
        manager = JournalManager()
        manager.add_visited_node("Q1", "Node 1")
        manager.add_visited_node("Q2", "Node 2")
        manager.add_found_value("Q1", "attr1", "value1")
        manager.add_verified_fact("S", "P", "O")
        manager.add_completed_step("Step 1")
        manager.add_failed_attempt("Attempt 1")

        stats = manager.get_stats()

        assert stats["visited_nodes"] == 2
        assert stats["found_values"] == 1
        assert stats["verified_facts"] == 1
        assert stats["completed_steps"] == 1
        assert stats["failed_attempts"] == 1

    def test_get_visited_node_count(self):
        """Test counting visited nodes."""
        manager = JournalManager()
        assert manager.get_visited_node_count() == 0

        manager.add_visited_node("Q1", "Node 1")
        manager.add_visited_node("Q2", "Node 2")

        assert manager.get_visited_node_count() == 2

    def test_get_found_value_count(self):
        """Test counting found values."""
        manager = JournalManager()
        assert manager.get_found_value_count() == 0

        manager.add_found_value("Q1", "attr1", "v1")
        manager.add_found_value("Q1", "attr2", "v2")
        manager.add_found_value("Q2", "attr1", "v3")

        assert manager.get_found_value_count() == 3
