// Graph node/edge titles arrive as small server-escaped HTML fragments
// (live_graph_data._make_title: `<br>` line breaks, `<i>`/`<b>` emphasis).
// vis-network renders a *string* title as plain text, which would show the
// tags literally, so we turn the fragment into an element ourselves. Only a
// tiny allow-list of inline tags survives; everything else becomes text, so
// this can never introduce markup the server did not intend.

const ALLOWED = new Set(["B", "I", "EM", "STRONG", "BR", "SPAN"]);

function copySafe(src: Node, dst: Node): void {
  src.childNodes.forEach((child) => {
    if (child.nodeType === Node.TEXT_NODE) {
      dst.appendChild(document.createTextNode(child.textContent ?? ""));
    } else if (child.nodeType === Node.ELEMENT_NODE) {
      const el = child as Element;
      if (ALLOWED.has(el.tagName)) {
        const clean = document.createElement(el.tagName.toLowerCase());
        copySafe(el, clean);
        dst.appendChild(clean);
      } else {
        copySafe(el, dst);
      }
    }
  });
}

export function titleElement(fragment: string | null | undefined): HTMLElement | undefined {
  if (!fragment) return undefined;
  // DOMParser builds an inert document: no scripts run, no resources load.
  const doc = new DOMParser().parseFromString(`<body>${fragment}</body>`, "text/html");
  const out = document.createElement("div");
  out.className = "graph-tip";
  copySafe(doc.body, out);
  return out;
}
