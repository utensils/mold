import { describe, expect, it } from "vitest";
import { router } from "./router";

describe("router — five-destination IA", () => {
  it("serves the five workspaces with their titlebar names", () => {
    for (const [path, title] of [
      ["/create", "Create"],
      ["/library", "Library"],
      ["/models", "Models"],
      ["/machines", "Machines"],
      ["/settings", "Settings"],
    ] as const) {
      expect(router.resolve(path).meta.title).toBe(title);
    }
  });

  it("roots the app at Create", async () => {
    await router.push("/");
    expect(router.currentRoute.value.path).toBe("/create");
  });

  it("authors chains inside Create at /create/chain, still titled Create", () => {
    const resolved = router.resolve("/create/chain");
    expect(resolved.name).toBe("chains");
    expect(resolved.meta.title).toBe("Create");
  });

  it("keeps RunPod as a literal segment under Machines, ahead of host detail", () => {
    expect(router.resolve("/machines/runpod").name).toBe("runpod");
    expect(router.resolve("/machines/hal9000-7680").name).toBe("host-detail");
  });

  it("still serves the standalone Jobs route (absorbed into Machines nav)", () => {
    expect(router.resolve("/jobs").name).toBe("jobs");
  });
});

describe("router — legacy redirects", () => {
  it.each([
    ["/generate", "/create", "create"],
    ["/gallery", "/library", "library"],
    ["/chains", "/create/chain", "chains"],
    ["/runpod", "/machines/runpod", "runpod"],
  ])("folds %s into %s", async (from, to, name) => {
    await router.push(from);
    expect(router.currentRoute.value.path).toBe(to);
    expect(router.currentRoute.value.name).toBe(name);
  });

  it("redirects the legacy /hosts/:id link to /machines/:id", async () => {
    await router.push("/hosts/hal9000-7680");
    expect(router.currentRoute.value.path).toBe("/machines/hal9000-7680");
    expect(router.currentRoute.value.name).toBe("host-detail");
  });

  it("carries a legacy /gallery?host= deep-link's query onto /library", async () => {
    await router.push("/gallery?host=hal9000-7680");
    expect(router.currentRoute.value.path).toBe("/library");
    expect(router.currentRoute.value.query.host).toBe("hal9000-7680");
  });

  it("sends the retired /history path to the Library history drawer", async () => {
    await router.push("/history");
    expect(router.currentRoute.value.path).toBe("/library");
    expect(router.currentRoute.value.query.panel).toBe("history");
  });
});

describe("router — persisted-route restore", () => {
  // Boot restore replays the saved path with `router.replace`, which runs the
  // same redirects — a window last on the old Gallery reopens on Library,
  // never a dead 404.
  it("restoring an old persisted path lands on its new home", async () => {
    await router.replace("/gallery");
    expect(router.currentRoute.value.path).toBe("/library");
    await router.replace("/generate");
    expect(router.currentRoute.value.path).toBe("/create");
  });
});
