/** The inspector's three tabs (README §04): no floating popovers for primary
 * controls — starting points and recent settings are tabs beside Settings. */
export type InspectorTab = "settings" | "starters" | "recent";

export const INSPECTOR_TABS: readonly { id: InspectorTab; label: string }[] = [
  { id: "settings", label: "Settings" },
  { id: "starters", label: "Starters" },
  { id: "recent", label: "Recent" },
];
