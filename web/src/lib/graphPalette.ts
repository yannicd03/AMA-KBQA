// Graph colours, mirroring the group styling in
// ama_kbqa/frontend/utils/graph_html.py, recoloured to the KIT palette:
// KQAPro = KIT blue, SciQA = KIT green, answer nodes = KIT orange.
import type { NodeGroup } from "../api";

export interface Swatch {
  background: string;
  border: string;
}

export interface GraphPalette {
  groups: Record<NodeGroup, Swatch>;
  answer: Swatch;
  newBorder: string;
  accent: string;
  font: string;
  fontLiteral: string;
  edge: string;
  edgeFont: string;
  canvas: string;
}

export const LIGHT_PALETTE: GraphPalette = {
  groups: {
    kqapro: { background: "#e3e8f4", border: "#4664aa" },
    sciqa: { background: "#d9f0eb", border: "#009682" },
    entity: { background: "#eceeed", border: "#9aa5a1" },
    literal: { background: "#f5f7f6", border: "#d0d7d4" },
    candidate: { background: "#ffffff", border: "#8a9dcb" },
  },
  answer: { background: "#fbe9c4", border: "#df9b1b" },
  newBorder: "#d9a300",
  accent: "#009682",
  font: "#26302e",
  fontLiteral: "#56625f",
  edge: "#a9b3b0",
  edgeFont: "#5f6b68",
  canvas: "#ffffff",
};

export const DARK_PALETTE: GraphPalette = {
  groups: {
    kqapro: { background: "#243152", border: "#8ea3dc" },
    sciqa: { background: "#123d37", border: "#2fc4ab" },
    entity: { background: "#2a3230", border: "#77837f" },
    literal: { background: "#1f2524", border: "#48524f" },
    candidate: { background: "#161b1a", border: "#7f94c9" },
  },
  answer: { background: "#4a3610", border: "#f0b43c" },
  newBorder: "#f2c230",
  accent: "#2fc4ab",
  font: "#e3e9e7",
  fontLiteral: "#a9b4b1",
  edge: "#5c6865",
  edgeFont: "#9aa6a3",
  canvas: "#121716",
};

export const LEGEND: { key: NodeGroup | "answer"; label: string; shape: "dot" | "box" | "dashed" }[] = [
  { key: "kqapro", label: "KQAPro", shape: "dot" },
  { key: "sciqa", label: "SciQA", shape: "dot" },
  { key: "literal", label: "Literal", shape: "box" },
  { key: "candidate", label: "Candidate", shape: "dashed" },
  { key: "answer", label: "In answer", shape: "dot" },
];
