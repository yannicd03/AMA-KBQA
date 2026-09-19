// The assistant's mark: a four-node knowledge-graph glyph in KIT green. While
// the agent works, a pulse travels along its edges (the one piece of ambient
// motion in the UI; disabled under prefers-reduced-motion).

interface MarkProps {
  size?: number;
  thinking?: boolean;
  className?: string;
}

export function Mark({ size = 28, thinking = false, className = "" }: MarkProps) {
  return (
    <span
      className={`mark${thinking ? " mark--thinking" : ""} ${className}`}
      style={{ width: size, height: size }}
      aria-hidden="true"
    >
      <svg viewBox="0 0 32 32" width={size} height={size} focusable="false">
        <g className="mark__edges">
          <path d="M9 22.5 15.5 9" />
          <path d="M15.5 9 23.5 17" />
          <path d="M9 22.5 23.5 17" />
          <path d="M15.5 9 16 20.2" />
        </g>
        <g className="mark__pulse">
          <path d="M9 22.5 15.5 9 23.5 17 9 22.5" pathLength={100} />
        </g>
        <circle className="mark__node" cx="9" cy="22.5" r="3.1" />
        <circle className="mark__node" cx="15.5" cy="9" r="3.1" />
        <circle className="mark__node" cx="23.5" cy="17" r="3.1" />
        <circle className="mark__node mark__node--core" cx="16" cy="20.2" r="2.1" />
      </svg>
    </span>
  );
}
