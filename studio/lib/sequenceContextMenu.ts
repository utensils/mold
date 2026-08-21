/**
 * Right-click entries for the sequence bench — ONE builder for every
 * surface. Desktop renders them through its app-wide context-menu store and
 * web through its own inline menu, so the types below are deliberately
 * structurally compatible with desktop's `MenuEntry` (label / action /
 * disabled / danger, plus a bare `{ separator: true }`) rather than
 * importing it: `studio/` must never depend on an application shell.
 *
 * Labels, order, and the disabled rules live here so the two surfaces
 * cannot drift — in particular the two-clip floor `removeClip` enforces in
 * the store, which a menu that offers an enabled Remove would only report
 * as a silent no-op.
 */

export interface SequenceMenuSeparator {
  separator: true;
}

export interface SequenceMenuItem {
  label: string;
  action: () => void;
  disabled?: boolean;
  danger?: boolean;
}

export type SequenceMenuEntry = SequenceMenuItem | SequenceMenuSeparator;

export const isSequenceMenuSeparator = (
  entry: SequenceMenuEntry,
): entry is SequenceMenuSeparator => "separator" in entry;

const SEPARATOR: SequenceMenuSeparator = { separator: true };

export interface ClipContextState {
  /** Zero-based position of the right-clicked clip. */
  index: number;
  count: number;
  maxStages: number;
  /** A rendered stage exists for this clip, so playback is meaningful. */
  canPlay: boolean;
  /** The rail is read-only (submitting, or an explicitly disabled bench). */
  locked: boolean;
}

export interface ClipContextActions {
  play: () => void;
  duplicate: () => void;
  insertBefore: () => void;
  insertAfter: () => void;
  moveTo: (index: number) => void;
  remove: () => void;
}

export function clipContextEntries(
  state: ClipContextState,
  actions: ClipContextActions,
): SequenceMenuEntry[] {
  const { index, count, maxStages, canPlay, locked } = state;
  const atCap = locked || count >= maxStages;
  const first = locked || index <= 0;
  const last = locked || index >= count - 1;
  const entries: SequenceMenuEntry[] = [];
  if (canPlay) {
    entries.push({ label: "Play clip", action: actions.play, disabled: false });
  }
  entries.push(
    { label: "Duplicate clip", action: actions.duplicate, disabled: atCap },
    {
      label: "Insert clip before",
      action: actions.insertBefore,
      disabled: atCap,
    },
    {
      label: "Insert clip after",
      action: actions.insertAfter,
      disabled: atCap,
    },
    SEPARATOR,
    {
      label: "Move to start",
      action: () => actions.moveTo(0),
      disabled: first,
    },
    {
      label: "Move left",
      action: () => actions.moveTo(index - 1),
      disabled: first,
    },
    {
      label: "Move right",
      action: () => actions.moveTo(index + 1),
      disabled: last,
    },
    {
      label: "Move to end",
      action: () => actions.moveTo(count - 1),
      disabled: last,
    },
    SEPARATOR,
    {
      label: "Remove clip",
      action: actions.remove,
      // A sequence keeps a two-clip floor; the store refuses below it.
      disabled: locked || count <= 2,
      danger: true,
    },
  );
  return entries;
}

export interface RailContextState {
  count: number;
  maxStages: number;
  locked: boolean;
  /** The draft is complete enough for `POST /api/generate/chain/validate`. */
  canValidate: boolean;
}

export interface RailContextActions {
  addClip: () => void;
  validate: () => void;
  importToml: () => void;
  exportToml: () => void;
  copyToml: () => void;
  clear: () => void;
}

export function railContextEntries(
  state: RailContextState,
  actions: RailContextActions,
): SequenceMenuEntry[] {
  const { count, maxStages, locked, canValidate } = state;
  return [
    {
      label: "Add clip",
      action: actions.addClip,
      disabled: locked || count >= maxStages,
    },
    SEPARATOR,
    {
      label: "Validate plan",
      action: actions.validate,
      disabled: !canValidate,
    },
    SEPARATOR,
    // File tools stay available on a draft that cannot be generated — an
    // export is how a broken sequence gets fixed elsewhere.
    { label: "Import TOML…", action: actions.importToml, disabled: false },
    { label: "Export TOML", action: actions.exportToml, disabled: false },
    { label: "Copy TOML", action: actions.copyToml, disabled: false },
    SEPARATOR,
    {
      label: "Clear sequence",
      action: actions.clear,
      disabled: locked,
      danger: true,
    },
  ];
}
