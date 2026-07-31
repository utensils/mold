import { beforeEach, describe, expect, it, vi } from "vitest";
import { ref } from "vue";
import type { RoutableHost } from "../lib/hostRouting";
import type { InstallTarget } from "./useModelInstallTargets";

const mockHosts = ref<RoutableHost[]>([]);
const mockOwners = ref<Record<string, string[]>>({});
const mockKnown = ref<Record<string, boolean>>({});

vi.mock("./useHostRouting", () => ({
  useHostRouting: () => ({
    hosts: mockHosts,
    modelOwnerIds: (name: string) => mockOwners.value[name] ?? [],
    inventoryKnown: (id: string) => mockKnown.value[id] ?? true,
  }),
}));

const mockStartDownload = vi.fn().mockResolvedValue(undefined);
vi.mock("./useCatalog", () => ({
  useCatalog: () => ({ startDownload: mockStartDownload }),
}));

const mockHostModelDownload = vi.fn().mockResolvedValue({ queued: 1 });
vi.mock("../components/machines/hostClient", () => ({
  hostModelDownload: (...args: unknown[]) => mockHostModelDownload(...args),
}));

import { useModelInstallTargets } from "./useModelInstallTargets";
import {
  addHost,
  HOSTS_STORAGE_KEY,
  ORIGIN_HOST_ID,
} from "../lib/hostRegistry";

function host(id: string, over: Partial<RoutableHost> = {}): RoutableHost {
  return {
    id,
    label: id === ORIGIN_HOST_ID ? "this server" : id,
    url: id === ORIGIN_HOST_ID ? "" : `http://${id}:7680`,
    status: "ready",
    queueDepth: 0,
    gpu: null,
    ...over,
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
  mockHosts.value = [host(ORIGIN_HOST_ID)];
  mockOwners.value = {};
  mockKnown.value = {};
  useModelInstallTargets().cancel();
});

describe("planFor", () => {
  it("keeps offering an install while any reachable machine lacks the model", () => {
    mockHosts.value = [host(ORIGIN_HOST_ID), host("studio")];
    mockOwners.value = { "cv:1": [ORIGIN_HOST_ID] };

    const plan = useModelInstallTargets().planFor("cv:1", true);

    expect(plan.canInstall).toBe(true);
    expect(plan.label).toBe("Pull");
    // Install targets lead; the machine that already has it degrades to repair.
    expect(plan.targets.map((t) => [t.host.id, t.action])).toEqual([
      ["studio", "install"],
      [ORIGIN_HOST_ID, "repair"],
    ]);
  });

  it("degrades to Repair only when every reachable machine owns it", () => {
    mockHosts.value = [host(ORIGIN_HOST_ID), host("studio")];
    mockOwners.value = { "cv:1": [ORIGIN_HOST_ID, "studio"] };

    const plan = useModelInstallTargets().planFor("cv:1", true);

    expect(plan.canInstall).toBe(false);
    expect(plan.label).toBe("Repair");
    expect(plan.targets.every((t) => t.action === "repair")).toBe(true);
  });

  it("treats an origin-installed catalog row as owned even before models land", () => {
    mockKnown.value = { [ORIGIN_HOST_ID]: false };

    expect(useModelInstallTargets().planFor("cv:1", true).label).toBe("Repair");
    expect(useModelInstallTargets().planFor("cv:1", false).label).toBe("Pull");
  });

  it("never claims a machine whose inventory has not landed is missing it", () => {
    mockHosts.value = [host(ORIGIN_HOST_ID), host("studio")];
    mockOwners.value = { "cv:1": [ORIGIN_HOST_ID] };
    mockKnown.value = { studio: false };

    const plan = useModelInstallTargets().planFor("cv:1", true);

    expect(plan.canInstall).toBe(false);
    expect(plan.targets.map((t) => t.host.id)).toEqual([ORIGIN_HOST_ID]);
  });

  it("ignores machines that are not reachable", () => {
    mockHosts.value = [
      host(ORIGIN_HOST_ID),
      host("studio", { status: "error" }),
      host("laptop", { status: "connecting" }),
    ];

    const plan = useModelInstallTargets().planFor("cv:1", false);

    expect(plan.targets.map((t) => t.host.id)).toEqual([ORIGIN_HOST_ID]);
  });
});

describe("chooseInstallTarget", () => {
  it("acts immediately without a picker for a single-host registry", async () => {
    const targets = useModelInstallTargets();
    const choice = await targets.chooseInstallTarget({
      modelId: "cv:1",
      displayName: "Alpha",
    });

    expect(targets.pending.value).toBeNull();
    expect(choice).toEqual({
      kind: "target",
      target: { host: mockHosts.value[0], action: "install" },
    });
  });

  it("falls back to the origin when nothing is reachable yet", async () => {
    mockHosts.value = [host(ORIGIN_HOST_ID, { status: "connecting" })];
    const targets = useModelInstallTargets();

    const choice = await targets.chooseInstallTarget({
      modelId: "cv:1",
      displayName: "Alpha",
    });

    expect(targets.pending.value).toBeNull();
    expect(choice).toEqual({ kind: "target", target: null });
  });

  it("asks which machine when the plan has more than one target", async () => {
    mockHosts.value = [host(ORIGIN_HOST_ID), host("studio")];
    mockOwners.value = { "cv:1": [ORIGIN_HOST_ID] };
    const targets = useModelInstallTargets();

    const promise = targets.chooseInstallTarget({
      modelId: "cv:1",
      displayName: "Alpha",
      ownedByOrigin: true,
    });
    await Promise.resolve();

    expect(targets.pending.value).not.toBeNull();
    expect(targets.pending.value?.displayName).toBe("Alpha");
    expect(targets.pending.value?.targets).toHaveLength(2);

    targets.choose(targets.pending.value!.targets[0]);
    const choice = await promise;

    expect(targets.pending.value).toBeNull();
    expect(choice).toEqual({
      kind: "target",
      target: { host: mockHosts.value[1], action: "install" },
    });
  });

  it("reports a dismissed picker as cancelled", async () => {
    mockHosts.value = [host(ORIGIN_HOST_ID), host("studio")];
    mockOwners.value = { "cv:1": [ORIGIN_HOST_ID] };
    const targets = useModelInstallTargets();

    const promise = targets.chooseInstallTarget({
      modelId: "cv:1",
      displayName: "Alpha",
      ownedByOrigin: true,
    });
    await Promise.resolve();
    targets.cancel();

    expect(await promise).toEqual({ kind: "cancelled" });
    expect(targets.pending.value).toBeNull();
  });
});

describe("startDownloadOn", () => {
  it("keeps the origin on the catalog composable so downloads repaint", async () => {
    const targets = useModelInstallTargets();
    await targets.startDownloadOn(
      { host: host(ORIGIN_HOST_ID), action: "install" },
      "cv:1",
    );
    expect(mockStartDownload).toHaveBeenCalledWith("cv:1");
    expect(mockHostModelDownload).not.toHaveBeenCalled();
  });

  it("sends a manifest model name down the same origin path", async () => {
    // The Installed shelf is mostly manifest names, and the two download
    // endpoints each 400 on the other's id shape — so this must not hardcode
    // either route. `useCatalog().startDownload` owns that routing.
    await useModelInstallTargets().startDownloadOn(null, "flux-schnell:q8");
    expect(mockStartDownload).toHaveBeenCalledWith("flux-schnell:q8");
  });

  it("treats a null target as the origin path", async () => {
    await useModelInstallTargets().startDownloadOn(null, "cv:1");
    expect(mockStartDownload).toHaveBeenCalledWith("cv:1");
  });

  it("routes a remote target through that machine's authenticated client", async () => {
    const entry = addHost({
      url: "http://studio.local:7680",
      name: "Studio",
      apiKey: "sekret",
    });
    mockHosts.value = [host(ORIGIN_HOST_ID), host(entry.id)];

    await useModelInstallTargets().startDownloadOn(
      { host: mockHosts.value[1], action: "install" },
      "cv:1",
    );

    expect(mockStartDownload).not.toHaveBeenCalled();
    expect(mockHostModelDownload).toHaveBeenCalledWith(entry, "cv:1");
  });

  it("refuses to guess when the chosen machine left the registry", async () => {
    mockHosts.value = [host(ORIGIN_HOST_ID), host("ghost")];
    localStorage.removeItem(HOSTS_STORAGE_KEY);

    await expect(
      useModelInstallTargets().startDownloadOn(
        { host: mockHosts.value[1], action: "install" },
        "cv:1",
      ),
    ).rejects.toThrow(/ghost/);
    expect(mockStartDownload).not.toHaveBeenCalled();
  });
});

describe("queuedMessage", () => {
  it("says exactly what it says today on a single-host registry", () => {
    const targets = useModelInstallTargets();
    const originTarget: InstallTarget = {
      host: mockHosts.value[0],
      action: "install",
    };
    expect(targets.queuedMessage(originTarget)).toBe("Download queued");
    expect(targets.queuedMessage(null, "repair")).toBe("Repair queued");
  });

  it("names the machine once more than one is connected", () => {
    mockHosts.value = [host(ORIGIN_HOST_ID), host("studio")];
    const targets = useModelInstallTargets();
    expect(
      targets.queuedMessage({ host: mockHosts.value[1], action: "install" }),
    ).toBe("Download queued on studio");
    expect(
      targets.queuedMessage({ host: mockHosts.value[0], action: "repair" }),
    ).toBe("Repair queued on this server");
  });
});
