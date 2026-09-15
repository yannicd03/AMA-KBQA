// Small inline icon set (24px grid, stroke = currentColor). No icon font or
// CDN: booth networks are unreliable, so everything ships in the bundle.
import type { ReactNode, SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement> & { size?: number };

function Svg({ size = 20, children, ...rest }: IconProps & { children: ReactNode }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.75}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...rest}
    >
      {children}
    </svg>
  );
}

export const IconArrowUp = (p: IconProps) => (
  <Svg {...p} strokeWidth={2.2}>
    <path d="M12 19V5" />
    <path d="m5.5 11.5 6.5-6.5 6.5 6.5" />
  </Svg>
);
export const IconChevronDown = (p: IconProps) => (
  <Svg {...p}>
    <path d="m6 9 6 6 6-6" />
  </Svg>
);
export const IconChevronRight = (p: IconProps) => (
  <Svg {...p}>
    <path d="m9 6 6 6-6 6" />
  </Svg>
);
export const IconCheck = (p: IconProps) => (
  <Svg {...p}>
    <path d="m5 12.5 4.5 4.5L19 7.5" />
  </Svg>
);
export const IconClose = (p: IconProps) => (
  <Svg {...p}>
    <path d="M6 6l12 12M18 6 6 18" />
  </Svg>
);
export const IconMenu = (p: IconProps) => (
  <Svg {...p}>
    <path d="M4 7h16M4 12h16M4 17h10" />
  </Svg>
);
export const IconSidebar = (p: IconProps) => (
  <Svg {...p}>
    <rect x="3.5" y="4.5" width="17" height="15" rx="3" />
    <path d="M9.5 4.5v15" />
  </Svg>
);
export const IconPanelRight = (p: IconProps) => (
  <Svg {...p}>
    <rect x="3.5" y="4.5" width="17" height="15" rx="3" />
    <path d="M14.5 4.5v15" />
  </Svg>
);
export const IconNewChat = (p: IconProps) => (
  <Svg {...p}>
    <path d="M12 4.5H7a2.5 2.5 0 0 0-2.5 2.5v10A2.5 2.5 0 0 0 7 19.5h10a2.5 2.5 0 0 0 2.5-2.5v-5" />
    <path d="M17.6 3.9a1.9 1.9 0 0 1 2.7 2.7L13 13.9l-3.4.7.7-3.4z" />
  </Svg>
);
export const IconHelp = (p: IconProps) => (
  <Svg {...p}>
    <circle cx="12" cy="12" r="8.5" />
    <path d="M9.6 9.4a2.5 2.5 0 0 1 4.8.9c0 1.7-2.4 2.2-2.4 3.7" />
    <circle cx="12" cy="17" r=".6" fill="currentColor" />
  </Svg>
);
export const IconSliders = (p: IconProps) => (
  <Svg {...p}>
    <path d="M4 7h9M17 7h3M4 17h3M11 17h9" />
    <circle cx="15" cy="7" r="2" />
    <circle cx="9" cy="17" r="2" />
  </Svg>
);
export const IconCopy = (p: IconProps) => (
  <Svg {...p}>
    <rect x="8.5" y="8.5" width="11" height="11" rx="2.5" />
    <path d="M15.5 8.5V6.5a2 2 0 0 0-2-2h-7a2 2 0 0 0-2 2v7a2 2 0 0 0 2 2h2" />
  </Svg>
);
export const IconGraph = (p: IconProps) => (
  <Svg {...p}>
    <circle cx="6" cy="17" r="2.2" />
    <circle cx="12" cy="6.5" r="2.2" />
    <circle cx="18" cy="15" r="2.2" />
    <path d="m7.1 15.1 3.8-6.6M13.3 8.4l3.5 4.8M8.2 16.7l7.6-1.3" />
  </Svg>
);
export const IconAlert = (p: IconProps) => (
  <Svg {...p}>
    <circle cx="12" cy="12" r="8.5" />
    <path d="M12 7.8v5" />
    <circle cx="12" cy="16.2" r=".6" fill="currentColor" />
  </Svg>
);
export const IconFit = (p: IconProps) => (
  <Svg {...p}>
    <path d="M4.5 9V6a1.5 1.5 0 0 1 1.5-1.5h3M15 4.5h3A1.5 1.5 0 0 1 19.5 6v3M19.5 15v3a1.5 1.5 0 0 1-1.5 1.5h-3M9 19.5H6A1.5 1.5 0 0 1 4.5 18v-3" />
  </Svg>
);
export const IconRefresh = (p: IconProps) => (
  <Svg {...p}>
    <path d="M19 12a7 7 0 1 1-2.1-5" />
    <path d="M19.5 4.5V8H16" />
  </Svg>
);

// ── Suggestion icons (Material Symbols names from AGENT_SUGGESTIONS) ────────

const SUGGESTION_ICONS: Record<string, (p: IconProps) => ReactNode> = {
  movie: (p) => (
    <Svg {...p}>
      <path d="M4 11h16v7a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2z" />
      <path d="m3.6 7.9 14.9-4 .9 3.3-14.9 4z" />
      <path d="m8 6.8 2.6 2.4M12.6 5.6l2.6 2.4" />
    </Svg>
  ),
  science: (p) => (
    <Svg {...p}>
      <path d="M9.5 3.5h5M10.5 3.5v5.3L5.2 17.6A2 2 0 0 0 6.9 20.5h10.2a2 2 0 0 0 1.7-2.9l-5.3-8.8V3.5" />
      <path d="M7.6 14.5h8.8" />
    </Svg>
  ),
  location_on: (p) => (
    <Svg {...p}>
      <path d="M12 20.5s-6.5-5.6-6.5-10.6a6.5 6.5 0 0 1 13 0c0 5-6.5 10.6-6.5 10.6z" />
      <circle cx="12" cy="10" r="2.3" />
    </Svg>
  ),
  music_note: (p) => (
    <Svg {...p}>
      <circle cx="8" cy="17.5" r="2.8" />
      <path d="M10.8 17.5V4.5l7 2.2v3.4l-7-2.2" />
    </Svg>
  ),
  biotech: (p) => (
    <Svg {...p}>
      <path d="M5 20.5h14M9 17.5h6M12 17.5a5 5 0 0 0 4.6-7" />
      <path d="m9.2 4.2 3.6 2.1-3.2 5.5-3.6-2.1z" />
      <path d="m10.4 2.8 3.2 1.9" />
    </Svg>
  ),
  article: (p) => (
    <Svg {...p}>
      <rect x="4.5" y="3.5" width="15" height="17" rx="2.5" />
      <path d="M8.5 8.5h7M8.5 12h7M8.5 15.5h4" />
    </Svg>
  ),
  hub: (p) => (
    <Svg {...p}>
      <circle cx="12" cy="12" r="2.4" />
      <circle cx="12" cy="4.5" r="1.6" />
      <circle cx="18.5" cy="16" r="1.6" />
      <circle cx="5.5" cy="16" r="1.6" />
      <path d="M12 6.1v3.5M13.9 13.3l3.2 1.9M10.1 13.3l-3.2 1.9" />
    </Svg>
  ),
};

export function SuggestionIcon({ icon, ...p }: IconProps & { icon: string | null }) {
  const render = (icon && SUGGESTION_ICONS[icon]) || null;
  if (render) return <>{render(p)}</>;
  return (
    <Svg {...p}>
      <path d="M12 4.5v3M12 16.5v3M4.5 12h3M16.5 12h3M7 7l1.8 1.8M15.2 15.2 17 17M17 7l-1.8 1.8M8.8 15.2 7 17" />
    </Svg>
  );
}
