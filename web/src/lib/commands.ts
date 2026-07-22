import { theme, themeFamily } from "./theme";
import type { ModelInfoExtended } from "../types";

/*
 * ⌘K command registry (spec §06 system affordances). Commands are pure data +
 * a `run` thunk so they unit-test without a Vue runtime: navigation, model
 * selection, and downloads flow through the injected `CommandContext`, while
 * the Theme commands mutate the shared `lib/theme` refs directly.
 */

export interface Command {
  id: string;
  /** Short uppercase group label (spec voice: table headers UPPERCASE). */
  section: string;
  label: string;
  keywords?: string[];
  run: () => void;
}

export interface CommandContext {
  /** Navigate to a route path. */
  go: (path: string) => void;
  /** Select a model for the next generation and open Create. */
  runModel: (name: string) => void;
  /** Open the downloads center. */
  openDownloads: () => void;
  /** Start a fresh print on Create. */
  newPrint: () => void;
}

const GO: Array<{
  id: string;
  label: string;
  path: string;
  keywords?: string[];
}> = [
  {
    id: "go-library",
    label: "Library",
    path: "/library",
    keywords: ["gallery", "prints"],
  },
  {
    id: "go-create",
    label: "Create",
    path: "/create",
    keywords: ["generate", "compose"],
  },
  { id: "go-models", label: "Models", path: "/models", keywords: ["catalog"] },
  {
    id: "go-machines",
    label: "Machines",
    path: "/machines",
    keywords: ["hosts", "gpu"],
  },
  { id: "go-settings", label: "Settings", path: "/settings" },
  {
    id: "go-sequence",
    label: "Compose sequence",
    path: "/create?mode=sequence",
    keywords: ["chain", "video"],
  },
  {
    id: "go-library-images",
    label: "Browse image prints",
    path: "/library?type=images",
    keywords: ["gallery", "photos"],
  },
  {
    id: "go-library-video",
    label: "Browse video prints",
    path: "/library?type=video",
    keywords: ["gallery", "clips"],
  },
  {
    id: "go-models-installed",
    label: "Installed models",
    path: "/models?tab=installed",
    keywords: ["local"],
  },
  {
    id: "go-models-discover",
    label: "Discover models",
    path: "/models?tab=discover",
    keywords: ["catalog", "pull"],
  },
  {
    id: "go-add-machine",
    label: "Add machine",
    path: "/machines?add=1",
    keywords: ["connect", "host", "remote"],
  },
  {
    id: "go-config",
    label: "Engine configuration",
    path: "/settings",
    keywords: ["server", "defaults"],
  },
  {
    id: "go-expansion",
    label: "Expansion settings",
    path: "/settings",
    keywords: ["prompt", "llm"],
  },
];

/** Installed models become "Run <name>" commands, loaded lazily on open. */
export function modelCommands(
  models: readonly ModelInfoExtended[],
  ctx: CommandContext,
): Command[] {
  return models
    .filter((m) => m.downloaded)
    .map((m) => ({
      id: `model-${m.name}`,
      section: "Model",
      label: `Run ${m.name}`,
      run: () => ctx.runModel(m.name),
    }));
}

/** The fixed commands: navigation, quick actions, and theme controls. */
export function baseCommands(ctx: CommandContext): Command[] {
  const go: Command[] = GO.map((g) => ({
    id: g.id,
    section: "Go to",
    label: g.label,
    keywords: g.keywords,
    run: () => ctx.go(g.path),
  }));

  const actions: Command[] = [
    {
      id: "action-new-print",
      section: "Action",
      label: "New print",
      run: () => ctx.newPrint(),
    },
    {
      id: "action-downloads",
      section: "Action",
      label: "Open downloads",
      run: () => ctx.openDownloads(),
    },
  ];

  const themes: Command[] = [
    {
      id: "theme-family-mold",
      section: "Theme",
      label: "Mold theme",
      run: () => {
        themeFamily.value = "mold";
      },
    },
    {
      id: "theme-family-safelight",
      section: "Theme",
      label: "Safelight theme",
      run: () => {
        themeFamily.value = "safelight";
      },
    },
    {
      id: "theme-appearance-dark",
      section: "Theme",
      label: "Dark appearance",
      run: () => {
        theme.value = "dark";
      },
    },
    {
      id: "theme-appearance-light",
      section: "Theme",
      label: "Light appearance",
      run: () => {
        theme.value = "light";
      },
    },
    {
      id: "theme-appearance-system",
      section: "Theme",
      label: "System appearance",
      run: () => {
        theme.value = "system";
      },
    },
  ];

  return [...go, ...actions, ...themes];
}

function scoreText(query: string, text: string): number {
  const q = query.toLowerCase();
  const value = text.toLowerCase();
  if (value.startsWith(q)) return 3;
  if (value.split(/[\s\-_.:/]+/).some((word) => word.startsWith(q))) return 2;
  if (value.includes(q)) return 1;
  return 0;
}

/** Ranked prefix > word-start > substring matching, stable for ties. */
export function filterCommands(
  commands: readonly Command[],
  query: string,
): Command[] {
  const q = query.trim().toLowerCase();
  if (!q) return [...commands];
  return commands
    .map((command, index) => ({
      command,
      index,
      score: Math.max(
        scoreText(q, command.label),
        ...(command.keywords ?? []).map((keyword) => scoreText(q, keyword)),
      ),
    }))
    .filter((match) => match.score > 0)
    .sort((a, b) => b.score - a.score || a.index - b.index)
    .map((match) => match.command);
}
