import type { IconName } from "@ui/icons";

/** Browser destinations keep their stable URLs while sharing visible names. */
export interface Workspace {
  name: string;
  label: string;
  icon: IconName;
  path: string;
  match: string[];
  keywords: string[];
}
export const WORKSPACES: Workspace[] = [
  {
    name: "create",
    label: "New image",
    icon: "create",
    path: "/create",
    match: ["create", "mesh-workflow"],
    keywords: ["create", "generate", "compose", "clip", "3-D"],
  },
  {
    name: "queue",
    label: "Queue",
    icon: "list",
    path: "/queue",
    match: ["queue"],
    keywords: ["jobs", "activity", "progress"],
  },
  {
    name: "library",
    label: "My images",
    icon: "library",
    path: "/library",
    match: ["library"],
    keywords: ["library", "gallery", "prints"],
  },
  {
    name: "models",
    label: "Styles",
    icon: "models",
    path: "/models",
    match: ["models"],
    keywords: ["models", "catalog"],
  },
  {
    name: "machines",
    label: "Machines",
    icon: "machines",
    path: "/machines",
    match: ["machines", "host-detail"],
    keywords: ["hosts", "gpu"],
  },
];
export const SETTINGS_DESTINATION: Workspace = {
  name: "settings",
  label: "Settings",
  icon: "settings",
  path: "/settings",
  match: ["settings"],
  keywords: ["preferences"],
};
export function workspaceLabel(name: string): string {
  return WORKSPACES.find((workspace) => workspace.name === name)?.label ?? name;
}
