import { describe, expect, it } from "vitest";
import {
  buildInstallModelCommands,
  buildUseModelCommands,
  resolveModelHost,
  shouldSearchCatalog,
  CATALOG_SEARCH_LIMIT,
  CATALOG_SEARCH_MIN_QUERY,
} from "./modelSearch";

const label = (id: string) =>
  ({ halcyon: "halcyon", bender: "bender" })[id] ?? id;

describe("resolveModelHost", () => {
  it("needs no switch when routing is automatic", () => {
    expect(
      resolveModelHost(["bender"], "auto", {
        autoTargetIds: ["auto", "capable"],
      }),
    ).toEqual({
      switchToHostId: null,
      ownerHostId: "bender",
    });
  });

  it("needs no switch when the pinned target already owns the model", () => {
    expect(resolveModelHost(["halcyon", "bender"], "halcyon")).toEqual({
      switchToHostId: null,
      ownerHostId: "halcyon",
    });
  });

  it("switches to the first owner when the pinned target lacks the model", () => {
    expect(resolveModelHost(["bender", "halcyon"], "plato")).toEqual({
      switchToHostId: "bender",
      ownerHostId: "bender",
    });
  });

  it("reports no owner when nobody has the model", () => {
    expect(resolveModelHost([], "halcyon")).toEqual({
      switchToHostId: null,
      ownerHostId: null,
    });
  });

  it("treats a null target as automatic routing", () => {
    expect(resolveModelHost(["bender"], null)).toEqual({
      switchToHostId: null,
      ownerHostId: "bender",
    });
  });
});

describe("buildUseModelCommands", () => {
  const ownersFor = (name: string) =>
    name === "flux-dev:q4" ? ["bender"] : ["halcyon"];

  it("labels a reachable model without a host hint", () => {
    const [cmd] = buildUseModelCommands([{ name: "sdxl", family: "sdxl" }], {
      ownersFor,
      targetHostId: "halcyon",
      hostLabel: label,
    });
    expect(cmd).toMatchObject({
      kind: "use",
      model: "sdxl",
      label: "Use sdxl",
      hint: "sdxl",
      switchToHostId: null,
    });
  });

  it("announces the machine it would switch to", () => {
    const [cmd] = buildUseModelCommands(
      [{ name: "flux-dev:q4", family: "flux" }],
      {
        ownersFor,
        targetHostId: "halcyon",
        hostLabel: label,
      },
    );
    expect(cmd?.switchToHostId).toBe("bender");
    expect(cmd?.hint).toBe("on bender");
  });

  it("prefers a loaded hint over the family when no switch is needed", () => {
    const [cmd] = buildUseModelCommands(
      [{ name: "sdxl", family: "sdxl", is_loaded: true }],
      { ownersFor, targetHostId: "halcyon", hostLabel: label },
    );
    expect(cmd?.hint).toBe("loaded");
  });

  it("shows the human title but keeps the opaque id for requests", () => {
    const [cmd] = buildUseModelCommands(
      [{ name: "cv:12345", display_name: "Dreamy Mix", family: "sdxl" }],
      { ownersFor, targetHostId: "halcyon", hostLabel: label },
    );
    expect(cmd?.label).toBe("Use Dreamy Mix");
    expect(cmd?.model).toBe("cv:12345");
  });

  it("matches on tokens split out of the id, the family, and the title", () => {
    const [cmd] = buildUseModelCommands(
      [{ name: "flux-dev:q4", display_name: "FLUX Dev", family: "flux" }],
      { ownersFor, targetHostId: "bender", hostLabel: label },
    );
    expect(cmd?.keywords).toEqual(
      expect.arrayContaining(["flux-dev:q4", "flux", "dev", "q4", "model"]),
    );
  });

  it("still emits a command when no machine claims the model yet", () => {
    const cmds = buildUseModelCommands([{ name: "sdxl" }], {
      ownersFor: () => [],
      targetHostId: "halcyon",
      hostLabel: label,
    });
    expect(cmds).toHaveLength(1);
    expect(cmds[0]?.switchToHostId).toBeNull();
  });

  it("keeps ids stable and unique per model", () => {
    const cmds = buildUseModelCommands(
      [{ name: "sdxl" }, { name: "flux-dev:q4" }],
      { ownersFor, targetHostId: "halcyon", hostLabel: label },
    );
    expect(cmds.map((c) => c.id)).toEqual(["model-sdxl", "model-flux-dev:q4"]);
  });
});

describe("buildInstallModelCommands", () => {
  const entries = [
    { id: "cv:1", name: "Dreamy Mix", source: "civitai", family: "sdxl" },
    {
      id: "hf:org/qwen",
      name: "Qwen Image",
      source: "hf",
      family: "qwen-image",
    },
  ];

  it("offers an install for a model nobody has", () => {
    const [cmd] = buildInstallModelCommands(entries, { installedNames: [] });
    expect(cmd).toMatchObject({
      kind: "install",
      id: "install-cv:1",
      model: "cv:1",
      displayName: "Dreamy Mix",
      label: "Install Dreamy Mix",
      hint: "not installed · civitai",
    });
  });

  it("drops entries already installed somewhere, matched on the catalog id", () => {
    const cmds = buildInstallModelCommands(entries, {
      installedNames: ["cv:1"],
    });
    expect(cmds.map((c) => c.model)).toEqual(["hf:org/qwen"]);
  });

  it("drops entries whose server-reported name is installed", () => {
    const cmds = buildInstallModelCommands(entries, {
      installedNames: ["Qwen Image"],
    });
    expect(cmds.map((c) => c.model)).toEqual(["cv:1"]);
  });

  it("honours the catalog's own installed flag", () => {
    const cmds = buildInstallModelCommands(
      [{ ...entries[0]!, installed: true }, entries[1]!],
      { installedNames: [] },
    );
    expect(cmds.map((c) => c.model)).toEqual(["hf:org/qwen"]);
  });

  it("caps the rows so catalog results never bury the commands", () => {
    const many = Array.from({ length: 20 }, (_, i) => ({
      id: `cv:${i}`,
      name: `Model ${i}`,
    }));
    expect(buildInstallModelCommands(many, {})).toHaveLength(
      CATALOG_SEARCH_LIMIT,
    );
    expect(buildInstallModelCommands(many, { limit: 2 })).toHaveLength(2);
  });

  it("omits the source when the catalog does not report one", () => {
    const [cmd] = buildInstallModelCommands([{ id: "cv:1", name: "Mix" }], {});
    expect(cmd?.hint).toBe("not installed");
  });
});

describe("shouldSearchCatalog", () => {
  it("waits for enough of a query to be worth a round trip", () => {
    expect(shouldSearchCatalog("")).toBe(false);
    expect(shouldSearchCatalog("f")).toBe(false);
    expect(shouldSearchCatalog("fl")).toBe(true);
    expect(CATALOG_SEARCH_MIN_QUERY).toBe(2);
  });

  it("ignores surrounding whitespace", () => {
    expect(shouldSearchCatalog("  f  ")).toBe(false);
    expect(shouldSearchCatalog("  flux  ")).toBe(true);
  });
});
