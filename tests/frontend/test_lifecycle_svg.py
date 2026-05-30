"""Tests for the inline lifecycle SVG renderer."""

from __future__ import annotations

from ama_kbqa.frontend.utils.lifecycle_svg import (
    LIFECYCLE_EDGES,
    LIFECYCLE_NODES,
    NODES_BY_ID,
    render_lifecycle_svg,
)


class TestRenderSkeleton:
    def test_returns_svg_with_viewbox(self):
        out = render_lifecycle_svg()
        assert out.startswith("<svg"), out[:80]
        assert "viewBox=" in out
        assert out.endswith("</svg>")

    def test_every_node_appears_in_output(self):
        out = render_lifecycle_svg()
        for node in LIFECYCLE_NODES:
            assert f'data-id="{node.id}"' in out, f"missing node {node.id}"

    def test_all_nodes_idle_by_default(self):
        out = render_lifecycle_svg()
        for node in LIFECYCLE_NODES:
            assert f'data-id="{node.id}" data-state="idle"' in out, node.id

    def test_phase_titles_present(self):
        out = render_lifecycle_svg()
        for title in ("Pre-Agent Hook", "Main-Agent Loop", "Post-Agent Hook"):
            assert title in out


class TestActiveAndVisited:
    def test_active_node_gets_active_state(self):
        out = render_lifecycle_svg({"main_llm_reason"}, set())
        assert 'data-id="main_llm_reason" data-state="active"' in out

    def test_visited_node_gets_visited_state(self):
        out = render_lifecycle_svg(set(), {"pre_classifier"})
        assert 'data-id="pre_classifier" data-state="visited"' in out

    def test_active_takes_precedence_over_visited(self):
        out = render_lifecycle_svg(
            {"main_llm_reason"},
            {"main_llm_reason", "pre_classifier"},
        )
        assert 'data-id="main_llm_reason" data-state="active"' in out
        assert 'data-id="pre_classifier" data-state="visited"' in out

    def test_only_named_node_is_active(self):
        out = render_lifecycle_svg({"main_llm_reason"}, set())
        for node in LIFECYCLE_NODES:
            expected = "active" if node.id == "main_llm_reason" else "idle"
            assert f'data-id="{node.id}" data-state="{expected}"' in out, node.id

    def test_caption_appears_when_provided(self):
        out = render_lifecycle_svg(set(), set(), current_label="llm_call · gpt-4o")
        assert "llm_call · gpt-4o" in out


class TestEdgesAndStructure:
    def test_every_edge_references_known_nodes(self):
        for e in LIFECYCLE_EDGES:
            assert e.from_id in NODES_BY_ID, e.from_id
            assert e.to_id in NODES_BY_ID, e.to_id

    def test_arrowhead_marker_defined(self):
        out = render_lifecycle_svg()
        assert 'id="lifecycle-arrowhead"' in out
        assert 'marker-end="url(#lifecycle-arrowhead)"' in out

    def test_dashed_edges_present_in_output(self):
        out = render_lifecycle_svg()
        # The Lessons-Learned → Strategy-Inject feedback loop is dashed.
        assert "edge dashed" in out

    def test_loop_back_edge_is_addressable(self):
        out = render_lifecycle_svg()
        assert 'data-edge-id="loop_back"' in out

    def test_active_edge_gets_state_and_blue_arrowhead(self):
        # Idle render: no edge references the active marker or carries the
        # active state, even though the marker itself is always defined.
        idle = render_lifecycle_svg()
        assert 'marker-end="url(#lifecycle-arrowhead-active)"' not in idle
        assert 'data-state="active"' not in idle

        out = render_lifecycle_svg(active_edge_ids={"loop_back"})
        # The active arrowhead marker is defined and referenced.
        assert 'id="lifecycle-arrowhead-active"' in out
        assert 'marker-end="url(#lifecycle-arrowhead-active)"' in out
        # The loop-back path itself is flagged active.
        assert 'data-state="active"' in out

    def test_inactive_edges_have_no_active_state(self):
        # Lighting only the loop-back edge must not flag any other edge active.
        out = render_lifecycle_svg(active_edge_ids={"loop_back"})
        assert out.count('marker-end="url(#lifecycle-arrowhead-active)"') == 1
