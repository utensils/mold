import { describe, expect, it, vi } from "vitest";
import {
  clipContextEntries,
  railContextEntries,
  isSequenceMenuSeparator,
  type SequenceMenuEntry,
} from "./sequenceContextMenu";

function labels(entries: SequenceMenuEntry[]): string[] {
  return entries.flatMap((entry) =>
    isSequenceMenuSeparator(entry) ? [] : [entry.label],
  );
}

function item(entries: SequenceMenuEntry[], label: string) {
  const found = entries.find(
    (entry) => !isSequenceMenuSeparator(entry) && entry.label === label,
  );
  if (!found || isSequenceMenuSeparator(found)) {
    throw new Error(`no menu item labelled ${label}`);
  }
  return found;
}

function clipActions() {
  return {
    play: vi.fn(),
    duplicate: vi.fn(),
    insertBefore: vi.fn(),
    insertAfter: vi.fn(),
    moveTo: vi.fn(),
    remove: vi.fn(),
  };
}

describe("clipContextEntries", () => {
  it("lists the clip actions in one pinned order", () => {
    const entries = clipContextEntries(
      { index: 1, count: 3, maxStages: 16, canPlay: true, locked: false },
      clipActions(),
    );
    expect(labels(entries)).toEqual([
      "Play clip",
      "Duplicate clip",
      "Insert clip before",
      "Insert clip after",
      "Move to start",
      "Move left",
      "Move right",
      "Move to end",
      "Remove clip",
    ]);
    // Separators after the add group and after the move group.
    expect(entries.filter(isSequenceMenuSeparator)).toHaveLength(2);
    expect(entries[4]).toEqual({ separator: true });
    expect(entries[9]).toEqual({ separator: true });
  });

  it("omits Play clip when the clip has no cached media", () => {
    const entries = clipContextEntries(
      { index: 0, count: 2, maxStages: 16, canPlay: false, locked: false },
      clipActions(),
    );
    expect(labels(entries)).not.toContain("Play clip");
  });

  it("disables the add group at the stage cap", () => {
    const entries = clipContextEntries(
      { index: 0, count: 4, maxStages: 4, canPlay: false, locked: false },
      clipActions(),
    );
    expect(item(entries, "Duplicate clip").disabled).toBe(true);
    expect(item(entries, "Insert clip before").disabled).toBe(true);
    expect(item(entries, "Insert clip after").disabled).toBe(true);
  });

  it("disables the move actions at the bounds", () => {
    const first = clipContextEntries(
      { index: 0, count: 3, maxStages: 16, canPlay: false, locked: false },
      clipActions(),
    );
    expect(item(first, "Move to start").disabled).toBe(true);
    expect(item(first, "Move left").disabled).toBe(true);
    expect(item(first, "Move right").disabled).toBe(false);
    expect(item(first, "Move to end").disabled).toBe(false);

    const last = clipContextEntries(
      { index: 2, count: 3, maxStages: 16, canPlay: false, locked: false },
      clipActions(),
    );
    expect(item(last, "Move right").disabled).toBe(true);
    expect(item(last, "Move to end").disabled).toBe(true);
    expect(item(last, "Move left").disabled).toBe(false);
  });

  it("moves to the resolved index for each move action", () => {
    const actions = clipActions();
    const entries = clipContextEntries(
      { index: 2, count: 5, maxStages: 16, canPlay: false, locked: false },
      actions,
    );
    item(entries, "Move to start").action();
    item(entries, "Move left").action();
    item(entries, "Move right").action();
    item(entries, "Move to end").action();
    expect(actions.moveTo.mock.calls).toEqual([[0], [1], [3], [4]]);
  });

  it("keeps the two-clip floor on Remove clip", () => {
    const two = clipContextEntries(
      { index: 0, count: 2, maxStages: 16, canPlay: false, locked: false },
      clipActions(),
    );
    const remove = item(two, "Remove clip");
    expect(remove.danger).toBe(true);
    expect(remove.disabled).toBe(true);

    const three = clipContextEntries(
      { index: 0, count: 3, maxStages: 16, canPlay: false, locked: false },
      clipActions(),
    );
    expect(item(three, "Remove clip").disabled).toBe(false);
  });

  it("disables every mutation while the rail is locked", () => {
    const entries = clipContextEntries(
      { index: 1, count: 3, maxStages: 16, canPlay: true, locked: true },
      clipActions(),
    );
    for (const label of [
      "Duplicate clip",
      "Insert clip before",
      "Insert clip after",
      "Move to start",
      "Move left",
      "Move right",
      "Move to end",
      "Remove clip",
    ]) {
      expect(item(entries, label).disabled).toBe(true);
    }
    // Playing a rendered clip is not a mutation.
    expect(item(entries, "Play clip").disabled).toBe(false);
  });

  it("runs the matching action", () => {
    const actions = clipActions();
    const entries = clipContextEntries(
      { index: 1, count: 3, maxStages: 16, canPlay: true, locked: false },
      actions,
    );
    item(entries, "Play clip").action();
    item(entries, "Duplicate clip").action();
    item(entries, "Insert clip before").action();
    item(entries, "Insert clip after").action();
    item(entries, "Remove clip").action();
    expect(actions.play).toHaveBeenCalledTimes(1);
    expect(actions.duplicate).toHaveBeenCalledTimes(1);
    expect(actions.insertBefore).toHaveBeenCalledTimes(1);
    expect(actions.insertAfter).toHaveBeenCalledTimes(1);
    expect(actions.remove).toHaveBeenCalledTimes(1);
  });
});

function railActions() {
  return {
    addClip: vi.fn(),
    validate: vi.fn(),
    importToml: vi.fn(),
    exportToml: vi.fn(),
    copyToml: vi.fn(),
    clear: vi.fn(),
  };
}

describe("railContextEntries", () => {
  it("lists the bench actions in one pinned order", () => {
    const entries = railContextEntries(
      { count: 2, maxStages: 16, locked: false, canValidate: true },
      railActions(),
    );
    expect(labels(entries)).toEqual([
      "Add clip",
      "Validate plan",
      "Import TOML…",
      "Export TOML",
      "Copy TOML",
      "Clear sequence",
    ]);
    expect(entries.filter(isSequenceMenuSeparator)).toHaveLength(3);
  });

  it("disables Add clip at the stage cap and while locked", () => {
    expect(
      item(
        railContextEntries(
          { count: 4, maxStages: 4, locked: false, canValidate: true },
          railActions(),
        ),
        "Add clip",
      ).disabled,
    ).toBe(true);
    expect(
      item(
        railContextEntries(
          { count: 2, maxStages: 4, locked: true, canValidate: true },
          railActions(),
        ),
        "Add clip",
      ).disabled,
    ).toBe(true);
  });

  it("disables Validate plan when the draft cannot be validated", () => {
    const entries = railContextEntries(
      { count: 2, maxStages: 16, locked: false, canValidate: false },
      railActions(),
    );
    expect(item(entries, "Validate plan").disabled).toBe(true);
    // File tools stay available on an invalid draft.
    expect(item(entries, "Export TOML").disabled).toBe(false);
    expect(item(entries, "Copy TOML").disabled).toBe(false);
    expect(item(entries, "Import TOML…").disabled).toBe(false);
  });

  it("marks Clear sequence dangerous and locks it while submitting", () => {
    const clear = item(
      railContextEntries(
        { count: 2, maxStages: 16, locked: true, canValidate: false },
        railActions(),
      ),
      "Clear sequence",
    );
    expect(clear.danger).toBe(true);
    expect(clear.disabled).toBe(true);
  });

  it("runs the matching action", () => {
    const actions = railActions();
    const entries = railContextEntries(
      { count: 2, maxStages: 16, locked: false, canValidate: true },
      actions,
    );
    item(entries, "Add clip").action();
    item(entries, "Validate plan").action();
    item(entries, "Import TOML…").action();
    item(entries, "Export TOML").action();
    item(entries, "Copy TOML").action();
    item(entries, "Clear sequence").action();
    expect(actions.addClip).toHaveBeenCalledTimes(1);
    expect(actions.validate).toHaveBeenCalledTimes(1);
    expect(actions.importToml).toHaveBeenCalledTimes(1);
    expect(actions.exportToml).toHaveBeenCalledTimes(1);
    expect(actions.copyToml).toHaveBeenCalledTimes(1);
    expect(actions.clear).toHaveBeenCalledTimes(1);
  });
});
