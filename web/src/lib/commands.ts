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

const GO: Array<{ id: string; label: string; path: string }> = [
  { id: "go-gallery", label: "Gallery", path: "/" },
  { id: "go-create", label: "Create", path: "/create" },
  { id: "go-models", label: "Models", path: "/models" },
  { id: "go-machines", label: "Machines", path: "/machines" },
  { id: "go-settings", label: "Settings", path: "/settings" },
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

/** Case-insensitive substring match across section + label. */
export function filterCommands(
  commands: readonly Command[],
  query: string,
): Command[] {
  const q = query.trim().toLowerCase();
  if (!q) return [...commands];
  return commands.filter((c) =>
    `${c.section} ${c.label}`.toLowerCase().includes(q),
  );
}
