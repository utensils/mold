import { describe, expect, it, vi } from "vitest";
import type { ModelInfoExtended } from "../types";
import {
  baseCommands,
  catalogCommands,
  filterCommands,
  modelCommands,
  type CommandContext,
} from "./commands";

const ctx: CommandContext = {
  go: vi.fn(),
  runModel: vi.fn(),
  installModel: vi.fn(),
  openDownloads: vi.fn(),
  newPrint: vi.fn(),
};

const model = (name: string, family: string, downloaded = true) =>
  ({ name, family, downloaded }) as ModelInfoExtended;

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

describe("web model commands", () => {
  it("skips models that are not downloaded", () => {
    const rows = modelCommands(
      [model("flux-dev:q4", "flux"), model("sdxl:fp16", "sdxl", false)],
      ctx,
    );
    expect(rows.map((r) => r.id)).toEqual(["model-flux-dev:q4"]);
  });

  it("routes a model on another machine through runModel with that host", () => {
    const runModel = vi.fn();
    const rows = modelCommands(
      [model("ltx2-distilled", "ltx2")],
      {
        ...ctx,
        runModel,
      },
      {
        ownersFor: () => ["bender"],
        targetHostId: "origin",
        hostLabel: (id) => (id === "bender" ? "bender" : id),
      },
    );
    rows[0]!.run();
    expect(runModel).toHaveBeenCalledWith("ltx2-distilled", "bender");
    expect(rows[0]!.hint).toBe("on bender");
  });

  it("treats the auto and capable sentinels as no switch at all", () => {
    for (const targetHostId of ["auto", "capable"]) {
      const rows = modelCommands([model("ltx2-distilled", "ltx2")], ctx, {
        ownersFor: () => ["bender"],
        targetHostId,
      });
      expect(rows[0]!.hint).toBe("ltx2");
    }
  });

  it("ranks a model name typed in full above a nav row that merely contains it", () => {
    const commands = [
      ...baseCommands(ctx),
      ...modelCommands([model("flux-dev:q4", "flux")], ctx),
    ];
    const [first] = filterCommands(commands, "flux-dev");
    expect(first?.id).toBe("model-flux-dev:q4");
  });
});

describe("web catalog commands", () => {
  const entry = {
    id: "hf:org/qwen",
    name: "Qwen Image",
    family: "qwen-image",
    source: "hf",
    installed: false,
  };

  it("routes an install through installModel with the raw catalog id", () => {
    const installModel = vi.fn();
    const rows = catalogCommands([entry], { ...ctx, installModel });
    expect(rows[0]).toMatchObject({
      id: "install-hf:org/qwen",
      section: "Install",
      label: "Install Qwen Image",
    });
    rows[0]!.run();
    expect(installModel).toHaveBeenCalledWith("hf:org/qwen", "Qwen Image");
  });

  it("drops entries the fleet already installed", () => {
    expect(
      catalogCommands([entry], ctx, { installedNames: ["hf:org/qwen"] }),
    ).toEqual([]);
  });
});
