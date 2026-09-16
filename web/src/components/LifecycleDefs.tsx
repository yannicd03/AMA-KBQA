// Every server-rendered figure repeats the same <marker> ids
// (lifecycle-arrowhead, -rev, -active). With several figures on one page,
// `url(#lifecycle-arrowhead)` resolves to the first element with that id in
// the document, and a figure inside a collapsed <details> is not rendered, so
// its markers can vanish for every other figure. One always-rendered copy at
// the top of the document makes the references resolve reliably.
export function LifecycleDefs() {
  return (
    <svg className="lifecycle-svg lifecycle-defs" aria-hidden="true" focusable="false" width="0" height="0">
      <defs>
        <marker id="lifecycle-arrowhead" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" className="edge-arrow" />
        </marker>
        <marker id="lifecycle-arrowhead-rev" viewBox="0 0 10 10" refX="1" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M 10 0 L 0 5 L 10 10 z" className="edge-arrow" />
        </marker>
        <marker id="lifecycle-arrowhead-active" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" className="edge-arrow-active" />
        </marker>
      </defs>
    </svg>
  );
}
