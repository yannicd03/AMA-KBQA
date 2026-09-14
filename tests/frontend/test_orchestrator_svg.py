"""Tests for the inline orchestrator (multi-agent) SVG renderer."""

from __future__ import annotations

from ama_kbqa.frontend.utils.orchestrator_svg import (
    ORCH_EDGES,
    ORCH_NODES,
    ORCH_NODES_BY_ID,
    render_orchestrator_svg,
)


class TestSkeleton:
    def test_returns_svg_with_viewbox(self):
        out = render_orchestrator_svg()
        assert out.startswith("<svg"), out[:80]
        assert "viewBox=" in out
        assert out.endswith("</svg>")
        # Reuses the lifecycle CSS namespace plus the orchestrator marker class.
        assert 'class="lifecycle-svg orchestrator-svg"' in out

    def test_every_node_appears(self):
        out = render_orchestrator_svg()
        for node in ORCH_NODES:
            assert f'data-id="{node.id}"' in out, f"missing node {node.id}"

    def test_orchestrator_and_subagent_nodes_present(self):
        out = render_orchestrator_svg()
        for nid in (
            "orch_user", "orch_probe", "orch_dispatch", "orch_combine",
            "sub_kqapro", "sub_sciqa",
            "sub_kqapro_pre", "sub_kqapro_main", "sub_kqapro_post",
            "sub_sciqa_pre", "sub_sciqa_main", "sub_sciqa_post",
        ):
            assert f'data-id="{nid}"' in out, nid

    def test_subagent_captions_present(self):
        out = render_orchestrator_svg()
        assert ">KQAPro<" in out
        assert ">SciQA<" in out

    def test_phase_band_title_present(self):
        assert "Orchestrator" in render_orchestrator_svg()

    def test_all_idle_by_default(self):
        out = render_orchestrator_svg()
        for node in ORCH_NODES:
            assert f'data-id="{node.id}" data-state="idle"' in out, node.id


class TestStates:
    def test_active_node_gets_active_state(self):
        out = render_orchestrator_svg({"sub_kqapro"}, set())
        assert 'data-id="sub_kqapro" data-state="active"' in out

    def test_visited_node_gets_visited_state(self):
        out = render_orchestrator_svg(set(), {"orch_probe"})
        assert 'data-id="orch_probe" data-state="visited"' in out

    def test_active_takes_precedence_over_visited(self):
        out = render_orchestrator_svg({"sub_kqapro"}, {"sub_kqapro"})
        assert 'data-id="sub_kqapro" data-state="active"' in out

    def test_dispatched_specialist_lit_other_stays_idle(self):
        # KQAPro ran; SciQA was never dispatched → it stays idle (inactive).
        out = render_orchestrator_svg(set(), {"sub_kqapro", "sub_kqapro_main"})
        assert 'data-id="sub_kqapro" data-state="visited"' in out
        assert 'data-id="sub_sciqa" data-state="idle"' in out

    def test_mini_lights_for_current_phase(self):
        out = render_orchestrator_svg({"sub_sciqa", "sub_sciqa_main"}, set())
        assert 'data-id="sub_sciqa_main" data-state="active"' in out
        assert 'data-id="sub_sciqa_pre" data-state="idle"' in out

    def test_caption_appears_when_provided(self):
        out = render_orchestrator_svg(current_label="completed in 2.1s")
        assert "completed in 2.1s" in out


class TestStructure:
    def test_edges_reference_real_geometry(self):
        # Every edge is a polyline with at least two points.
        for e in ORCH_EDGES:
            assert len(e.points) >= 2, e.id

    def test_dispatch_and_return_edges_exist(self):
        ids = {e.id for e in ORCH_EDGES}
        assert {"dispatch_kqapro", "dispatch_sciqa", "return_kqapro", "return_sciqa"} <= ids

    def test_container_nodes_are_containers(self):
        assert ORCH_NODES_BY_ID["sub_kqapro"].shape == "container"
        assert ORCH_NODES_BY_ID["sub_sciqa"].shape == "container"

    def test_arrowhead_markers_defined(self):
        out = render_orchestrator_svg()
        assert 'id="lifecycle-arrowhead"' in out
        assert 'marker-end="url(#lifecycle-arrowhead)"' in out


class TestFederatedDispatch:
    """Federated dispatch can light BOTH specialist containers at once (the
    docstring previously claimed the undispatched one always stays idle —
    only true for router/single dispatch)."""

    def test_both_specialists_active_at_once(self):
        out = render_orchestrator_svg({"sub_kqapro", "sub_sciqa"}, set())
        assert 'data-id="sub_kqapro" data-state="active"' in out
        assert 'data-id="sub_sciqa" data-state="active"' in out

    def test_both_specialist_minis_active_at_once(self):
        out = render_orchestrator_svg(
            {"sub_kqapro", "sub_kqapro_main", "sub_sciqa", "sub_sciqa_main"}, set(),
        )
        assert 'data-id="sub_kqapro_main" data-state="active"' in out
        assert 'data-id="sub_sciqa_main" data-state="active"' in out

    def test_dispatch_and_return_edges_light_when_active(self):
        out = render_orchestrator_svg(
            active_edge_ids={"dispatch_kqapro", "dispatch_sciqa"},
        )
        assert 'data-edge-id="dispatch_kqapro" data-state="active"' in out
        assert 'data-edge-id="dispatch_sciqa" data-state="active"' in out
        # Not requested → not marked active.
        assert 'data-edge-id="return_kqapro" data-state="active"' not in out

    def test_default_mode_aria_label_says_nothing_about_federation(self):
        out = render_orchestrator_svg()
        assert "federated" not in out.lower()
        assert 'aria-label="Orchestrator multi-agent flow"' in out

    def test_federated_mode_mentions_federation_in_aria_label(self):
        out = render_orchestrator_svg(mode="federated")
        assert "federated" in out.lower()
        assert out.startswith('<svg class="lifecycle-svg orchestrator-svg"')

    def test_router_mode_is_equivalent_to_default(self):
        assert render_orchestrator_svg(mode="router") == render_orchestrator_svg()

    def test_renders_fine_with_both_specialists_active_and_federated_mode(self):
        # End-to-end smoke: nothing about combining "both active" + the
        # federated label breaks the renderer.
        out = render_orchestrator_svg(
            {"sub_kqapro", "sub_kqapro_main", "sub_sciqa", "sub_sciqa_main", "orch_combine"},
            {"orch_user", "orch_probe", "orch_dispatch"},
            current_label="fusing answers…",
            active_edge_ids={"dispatch_kqapro", "dispatch_sciqa"},
            mode="federated",
        )
        assert out.startswith("<svg") and out.endswith("</svg>")
        for nid in ("sub_kqapro", "sub_sciqa", "orch_combine"):
            assert f'data-id="{nid}" data-state="active"' in out
        assert "fusing answers…" in out
