import { describe, expect, it, vi } from "vitest";
import { router } from "./router";

// Router behavior is the authority under test: the mapping from a path to a
// route's path, name, query and title. Loading a real view module would put
// its whole transform graph — every store, primitive and studio contract it
// pulls — inside a 5 s navigation timeout, which is why Create was already
// stubbed here. Under the full desktop suite (454 files sharing the workers)
// the same cost made `/jobs -> /machines` time out, so stub EVERY lazily
// imported destination: each view has its own suite that mounts it for real.
vi.mock("./views/GenerateView.vue", () => ({ default: { template: "<div />" } }));
vi.mock("./views/LibraryView.vue", () => ({ default: { template: "<div />" } }));
vi.mock("./views/ModelsView.vue", () => ({ default: { template: "<div />" } }));
vi.mock("./views/MachinesView.vue", () => ({ default: { template: "<div />" } }));
vi.mock("./views/RunPodView.vue", () => ({ default: { template: "<div />" } }));
vi.mock("./views/HostDetailView.vue", () => ({ default: { template: "<div />" } }));
vi.mock("./views/SettingsView.vue", () => ({ default: { template: "<div />" } }));

describe("router — five-destination IA", () => {
  it("serves the six destinations with their plain-English titles", () => {
    for (const [path, title] of [
      ["/create", "New image"],
      ["/queue", "Queue"],
      ["/library", "My images"],
      ["/models", "Styles"],
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

  it("folds the retired chain composer route into Create's sequence output", async () => {
    // Sequence is an Output setting of Create now, not a place — both the
    // nested /create/chain route and the legacy /chains path deep-link into
    // Create with the sequence output preselected.
    for (const legacy of ["/create/chain", "/chains"]) {
      await router.push(legacy);
      expect(router.currentRoute.value.path).toBe("/create");
      expect(router.currentRoute.value.name).toBe("create");
      expect(router.currentRoute.value.query.output).toBe("sequence");
    }
  });

  it("keeps RunPod as a literal segment under Machines, ahead of host detail", () => {
    expect(router.resolve("/machines/runpod").name).toBe("runpod");
    expect(router.resolve("/machines/hal9000-7680").name).toBe("host-detail");
  });

  it("keeps the Machines title on a machine's pane and on Rent a GPU", () => {
    expect(router.resolve("/machines/hal9000-7680").meta.title).toBe("Machines");
    expect(router.resolve("/machines/runpod").meta.title).toBe("Machines");
  });

  it("retires standalone Jobs into the Queue", async () => {
    await router.push("/jobs");
    expect(router.currentRoute.value.path).toBe("/queue");
    expect(router.currentRoute.value.name).toBe("queue");
  });
});

describe("router — legacy redirects", () => {
  it.each([
    ["/generate", "/create", "create"],
    ["/gallery", "/library", "library"],
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
