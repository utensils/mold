import { describe, expect, it, vi } from "vitest";
import { baseCommands, filterCommands, type CommandContext } from "./commands";

const ctx: CommandContext = {
  go: vi.fn(),
  runModel: vi.fn(),
  openDownloads: vi.fn(),
  newPrint: vi.fn(),
};

describe("web command registry", () => {
  it("covers workspace destinations and focused sub-surfaces", () => {
    expect(baseCommands(ctx).length).toBeGreaterThanOrEqual(20);
  });

  it("ranks prefix matches ahead of word-start and substring matches", () => {
    const commands = [
      { id: "substring", section: "Go", label: "Regenerate", run: vi.fn() },
      { id: "word", section: "Go", label: "New generation", run: vi.fn() },
      { id: "prefix", section: "Go", label: "Generate now", run: vi.fn() },
    ];
    expect(
      filterCommands(commands, "gen").map((command) => command.id),
    ).toEqual(["prefix", "word", "substring"]);
  });

  it("matches command synonyms", () => {
    const result = filterCommands(baseCommands(ctx), "gallery");
    expect(result.some((command) => command.id === "go-library")).toBe(true);
  });
});
