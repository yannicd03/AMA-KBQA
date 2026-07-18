"""
State management for KBQA agents.

Provides JournalState (Pydantic model) and JournalManager for tracking agent
progress, visited nodes, found values, and verified facts.
"""

from __future__ import annotations
from typing import Any, ClassVar, Dict, List, Optional
import json
from datetime import datetime

from pydantic import BaseModel, Field


class JournalState(BaseModel):
    """
    Represents the current state of an agent's investigation journal.

    This is the single source of truth used by both the MCP server scratchpad
    and the framework's JournalManager.
    """
    question_text: str = Field(default="", description="The original question being answered")
    question_type: str = Field(default="", description="Question type: Count, Verify, SelectBetween, etc.")
    target_entities: list[str] = Field(default_factory=list, description="Entity names we're looking for")

    visited_nodes: dict[str, str] = Field(
        default_factory=dict, description="Map of {node_id: node_name} already explored")
    found_values: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Map of {entity_id: {attribute: value}} storing all discovered values")
    verified_facts: list[dict] = Field(default_factory=list, description="Verified facts with structure")
    failed_attempts: list[str] = Field(
        default_factory=list, description="Track what didn't work to avoid repeating")

    current_plan: list[str] = Field(default_factory=list, description="Step-by-step plan for remaining steps")
    completed_steps: list[str] = Field(default_factory=list, description="Steps that have been completed")

    partial_answer: str = Field(default="", description="Intermediate answer being constructed")

    kg_name: str = Field(default="", description="Knowledge graph name (e.g., KQAPro)")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    MAX_COMPLETED_STEPS: ClassVar[int] = 20
    MAX_FAILED_ATTEMPTS: ClassVar[int] = 10

    def add_completed_step(self, step: str) -> None:
        """Append a completed step, capping to last MAX_COMPLETED_STEPS."""
        self.completed_steps.append(step)
        if len(self.completed_steps) > self.MAX_COMPLETED_STEPS:
            self.completed_steps = self.completed_steps[-self.MAX_COMPLETED_STEPS:]

    def add_failed_attempt(self, attempt: str) -> None:
        """Append a failed attempt, capping to last MAX_FAILED_ATTEMPTS."""
        self.failed_attempts.append(attempt)
        if len(self.failed_attempts) > self.MAX_FAILED_ATTEMPTS:
            self.failed_attempts = self.failed_attempts[-self.MAX_FAILED_ATTEMPTS:]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def to_str(self) -> str:
        """Enhanced visualization with better structure and readability (for MCP server)."""
        lines = ["=" * 70]
        lines.append("SCRATCHPAD STATE")
        lines.append("=" * 70)

        if self.question_type:
            lines.append(f"Question Type: {self.question_type}")
        if self.target_entities:
            lines.append(f"Target Entities: {', '.join(self.target_entities)}")

        if self.visited_nodes:
            lines.append(f"\nEXPLORED NODES ({len(self.visited_nodes)}):")
            for node_id, node_name in list(self.visited_nodes.items())[:5]:
                lines.append(f"  \u2022 {node_name} ({node_id})")
            if len(self.visited_nodes) > 5:
                lines.append(f"  ... and {len(self.visited_nodes) - 5} more")

        if self.found_values:
            lines.append("\nDISCOVERED VALUES:")
            for entity_id, attrs in self.found_values.items():
                entity_name = self.visited_nodes.get(entity_id, entity_id)
                lines.append(f"  {entity_name}:")
                for attr_name, attr_data in attrs.items():
                    if isinstance(attr_data, list) and attr_data:
                        for val_item in attr_data[:3]:
                            if isinstance(val_item, dict):
                                val_str = val_item.get("value", "?")
                                unit_str = val_item.get("unit", "")
                                lines.append(f"    - {attr_name}: {val_str} {unit_str}".strip())
                            else:
                                lines.append(f"    - {attr_name}: {val_item}")
                    else:
                        lines.append(f"    - {attr_name}: {attr_data}")

        if self.completed_steps:
            lines.append(f"\nCOMPLETED STEPS ({len(self.completed_steps)}):")
            for step in self.completed_steps[-3:]:
                lines.append(f"  \u2713 {step}")

        if self.current_plan:
            lines.append("\nNEXT STEPS:")
            for i, step in enumerate(self.current_plan[:3], 1):
                lines.append(f"  {i}. {step}")

        if self.failed_attempts:
            lines.append(f"\nFAILED ATTEMPTS ({len(self.failed_attempts)}):")
            for attempt in self.failed_attempts[-2:]:
                lines.append(f"  \u2717 {attempt}")

        if self.partial_answer:
            lines.append(f"\nPARTIAL ANSWER: {self.partial_answer}")

        lines.append(
            f"\nSTATS: {len(self.visited_nodes)} nodes, {len(self.found_values)} entities with data, {len(self.completed_steps)} steps done")

        lines.append("=" * 70)
        return "\n".join(lines)

    def to_summary_str(self) -> str:
        """Generate a formatted summary string for the LLM (framework format)."""
        lines = []

        lines.append(f"Question: {self.question_text}")
        lines.append(f"Question Type: {self.question_type}")
        lines.append(f"Knowledge Graph: {self.kg_name}")
        lines.append("")

        if self.target_entities:
            lines.append("Target Entities:")
            for entity in self.target_entities:
                lines.append(f"  - {entity}")
            lines.append("")

        if self.visited_nodes:
            lines.append("Visited Nodes:")
            for node_id, label in self.visited_nodes.items():
                lines.append(f"  - {node_id}: {label}")
            lines.append("")

        if self.found_values:
            lines.append("Found Values:")
            for entity_id, attrs in self.found_values.items():
                lines.append(f"  {entity_id}:")
                for attr_name, value in attrs.items():
                    lines.append(f"    - {attr_name}: {value}")
            lines.append("")

        if self.verified_facts:
            lines.append("Verified Facts:")
            for fact in self.verified_facts:
                subject = fact.get("subject", "?")
                predicate = fact.get("predicate", "?")
                obj = fact.get("object", "?")
                source = fact.get("source", "unknown")
                lines.append(f"  - [{subject}] --{predicate}--> [{obj}] (from: {source})")
            lines.append("")

        if self.completed_steps:
            lines.append("Completed Steps:")
            for i, step in enumerate(self.completed_steps, 1):
                lines.append(f"  {i}. {step}")
            lines.append("")

        if self.failed_attempts:
            lines.append("Failed Attempts:")
            for attempt in self.failed_attempts:
                lines.append(f"  - {attempt}")
            lines.append("")

        if self.partial_answer:
            lines.append("Partial Answer:")
            lines.append(f"  {self.partial_answer}")
            lines.append("")

        return "\n".join(lines)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JournalState":
        """Create from dictionary."""
        return cls(
            question_text=data.get("question_text", ""),
            question_type=data.get("question_type", ""),
            target_entities=data.get("target_entities", []),
            visited_nodes=data.get("visited_nodes", {}),
            found_values=data.get("found_values", {}),
            verified_facts=data.get("verified_facts", []),
            failed_attempts=data.get("failed_attempts", []),
            completed_steps=data.get("completed_steps", []),
            current_plan=data.get("current_plan", []),
            partial_answer=data.get("partial_answer", ""),
            kg_name=data.get("kg_name", ""),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
        )


class JournalManager:
    """
    Manager for agent journal state with a clean update API.

    Provides methods for updating different aspects of the journal
    state and generating summaries.
    """

    def __init__(self, kg_name: str = ""):
        """Initialize the journal manager."""
        self._state = JournalState(kg_name=kg_name)

    @property
    def state(self) -> JournalState:
        """Get the current journal state."""
        return self._state

    def reset(self) -> None:
        """Reset the journal to a fresh state."""
        kg_name = self._state.kg_name
        self._state = JournalState(kg_name=kg_name)

    def set_question(
        self,
        text: str,
        qtype: str,
        entities: Optional[List[str]] = None,
    ) -> None:
        """Set the current question being answered."""
        self._state.question_text = text
        self._state.question_type = qtype
        self._state.target_entities = entities or []
        self._update_timestamp()

    def add_visited_node(self, node_id: str, label: str) -> None:
        """Record a visited node."""
        self._state.visited_nodes[node_id] = label
        self._update_timestamp()

    def add_found_value(
        self,
        node_id: str,
        attribute: str,
        value: Any,
    ) -> None:
        """Record a found attribute value."""
        if node_id not in self._state.found_values:
            self._state.found_values[node_id] = {}
        self._state.found_values[node_id][attribute] = value
        self._update_timestamp()

    def add_verified_fact(
        self,
        subject: str,
        predicate: str,
        obj: Any,
        source: str = "tool",
    ) -> None:
        """Record a verified fact."""
        self._state.verified_facts.append({
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "source": source,
            "timestamp": datetime.now().isoformat(),
        })
        self._update_timestamp()

    def add_failed_attempt(self, description: str) -> None:
        """Record a failed attempt (capped)."""
        self._state.add_failed_attempt(description)
        self._update_timestamp()

    def add_completed_step(self, description: str) -> None:
        """Record a completed step (capped)."""
        self._state.add_completed_step(description)
        self._update_timestamp()

    def set_current_plan(self, steps: List[str]) -> None:
        """Set the current plan steps."""
        self._state.current_plan = steps
        self._update_timestamp()

    def set_partial_answer(self, answer: str) -> None:
        """Set the partial answer."""
        self._state.partial_answer = answer
        self._update_timestamp()

    def get_summary(self) -> str:
        """Get a formatted summary of the journal state."""
        return self._state.to_summary_str()

    def get_json(self) -> str:
        """Get the journal state as JSON."""
        return self._state.to_json()

    def load_from_dict(self, data: Dict[str, Any]) -> None:
        """Load state from a dictionary."""
        self._state = JournalState.from_dict(data)

    def load_from_json(self, json_str: str) -> None:
        """Load state from a JSON string."""
        data = json.loads(json_str)
        self.load_from_dict(data)

    def _update_timestamp(self) -> None:
        """Update the last modified timestamp."""
        self._state.updated_at = datetime.now().isoformat()

    def has_progress_since(self, previous_summary: str) -> bool:
        """Check if progress has been made since a previous summary."""
        current_summary = self.get_summary()
        return current_summary != previous_summary

    def get_visited_node_count(self) -> int:
        """Get the number of visited nodes."""
        return len(self._state.visited_nodes)

    def get_found_value_count(self) -> int:
        """Get the total number of found values."""
        return sum(len(attrs) for attrs in self._state.found_values.values())

    def get_verified_fact_count(self) -> int:
        """Get the number of verified facts."""
        return len(self._state.verified_facts)

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about the journal state."""
        return {
            "visited_nodes": self.get_visited_node_count(),
            "found_values": self.get_found_value_count(),
            "verified_facts": self.get_verified_fact_count(),
            "completed_steps": len(self._state.completed_steps),
            "failed_attempts": len(self._state.failed_attempts),
        }
