import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import type { ModelEntry, ServerCapabilities } from "../lib/api/types";
import {
  AUTHENTICATED_MINIMAX_H3_PROFILE_SHA256,
  authenticatedMiniMaxH3Capabilities,
} from "@studio/lib/minimaxH3Inventory.testFixtures";

vi.mock("../lib/ipc", () => ({
  inTauri: () => false,
  ipc: {
    appSettingsGet: vi.fn(),
    appSettingsSet: vi.fn(),
    secretGet: vi.fn(),
    secretSet: vi.fn(),
    testRemoteHost: vi.fn(),
  },
}));

const apiJsonTo = vi.fn();
vi.mock("../lib/api/client", () => ({
  apiJsonTo: (...a: unknown[]) => apiJsonTo(...a),
}));

import { useConnectionStore } from "./connection";
import { useHostModelsStore } from "./hostModels";
import { useHostsStore } from "./hosts";

function model(name: string, overrides: Partial<ModelEntry> = {}): ModelEntry {
  return {
    name,
    family: "flux",
    size_gb: 12,
    is_loaded: false,
    hf_repo: "black-forest-labs/FLUX.1-dev",
    default_steps: 28,
    default_guidance: 3.5,
    default_width: 1024,
    default_height: 1024,
    description: "",
    downloaded: true,
    ...overrides,
  };
}

function addExtra(id: string, url: string, status: "ready" | "error" = "ready") {
  useHostsStore().extras.push({
    id,
    label: id,
    url,
    apiKey: null,
    status,
    error: null,
    instanceId: null,
  });
}

beforeEach(() => {
  setActivePinia(createPinia());
  vi.clearAllMocks();
  const conn = useConnectionStore();
  conn.info = { mode: "local", baseUrl: "http://127.0.0.1:49152", apiKey: "k" };
  conn.status = "ready";
  apiJsonTo.mockResolvedValue([model("flux-dev:q8")]);
});

describe("hostModels store", () => {
  it("refresh() fans out to ready hosts only", async () => {
    addExtra("hal9000-7680", "http://hal9000:7680", "ready");
    addExtra("bender-7680", "http://bender:7680", "error");
    const store = useHostModelsStore();
    await store.refresh();
    const urls = apiJsonTo.mock.calls.map((c) => (c[0] as { baseUrl: string }).baseUrl);
    expect(urls).toContain("http://127.0.0.1:49152");
    expect(urls).toContain("http://hal9000:7680");
    expect(urls).not.toContain("http://bender:7680");
    expect(apiJsonTo.mock.calls.every((c) => c[1] === "/api/models")).toBe(true);
    expect(store.byHost["local"]?.entries.map((m) => m.name)).toEqual(["flux-dev:q8"]);
  });

  it("keeps a host's last entries and records the error when a fetch fails", async () => {
    addExtra("hal9000-7680", "http://hal9000:7680");
    const store = useHostModelsStore();
    await store.refresh();
    expect(store.byHost["hal9000-7680"]?.entries).toHaveLength(1);

    apiJsonTo.mockImplementation((target: { baseUrl: string }) =>
      target.baseUrl.includes("hal9000")
        ? Promise.reject(new Error("connection refused"))
        : Promise.resolve([model("flux-dev:q8")]),
    );
    await store.refresh(true);
    expect(store.byHost["hal9000-7680"]?.entries.map((m) => m.name)).toEqual(["flux-dev:q8"]);
    expect(store.byHost["hal9000-7680"]?.error).toContain("connection refused");
    expect(store.byHost["local"]?.error).toBeNull();
  });

  it("skips hosts fetched under 60s ago unless forced", async () => {
    const store = useHostModelsStore();
    await store.refresh();
    expect(apiJsonTo).toHaveBeenCalledTimes(1);

    await store.refresh();
    expect(apiJsonTo).toHaveBeenCalledTimes(1); // fresh — skipped

    await store.refresh(true);
    expect(apiJsonTo).toHaveBeenCalledTimes(2); // forced

    store.byHost["local"]!.fetchedAt = Date.now() - 61_000;
    await store.refresh();
    expect(apiJsonTo).toHaveBeenCalledTimes(3); // stale — refetched
  });

  it("unionInstalled dedups by name, keeps the first host's entry, collects hostIds", async () => {
    addExtra("hal9000-7680", "http://hal9000:7680");
    const store = useHostModelsStore();
    store.byHost["local"] = {
      entries: [
        model("flux-dev:q8", { description: "primary copy" }),
        model("real-esrgan-x4plus", { family: "real-esrgan" }), // not a generation model
        model("z-image:q8", { downloaded: false }), // not installed locally
      ],
      fetchedAt: Date.now(),
      error: null,
    };
    store.byHost["hal9000-7680"] = {
      entries: [model("flux-dev:q8", { description: "remote copy" }), model("z-image:q8")],
      fetchedAt: Date.now(),
      error: null,
    };

    const union = store.unionInstalled;
    const flux = union.find((m) => m.name === "flux-dev:q8");
    expect(flux?.hostIds).toEqual(["local", "hal9000-7680"]);
    expect(flux?.description).toBe("primary copy");
    const zimage = union.find((m) => m.name === "z-image:q8");
    expect(zimage?.hostIds).toEqual(["hal9000-7680"]);
    expect(union.some((m) => m.name === "real-esrgan-x4plus")).toBe(false);
  });

  it("removes rows restricted by each host's advertised model access policy", () => {
    const hosts = useHostsStore();
    hosts.capabilities.local = {
      gallery: { can_delete: true },
      model_access: {
        restrictions: [
          {
            code: "minimax_h3_authorization_required",
            family: "minimax-h3",
            message: "MiniMax H3 is not activated.",
            license_url: "https://example.test/license",
            authorization_url: "https://example.test/authorize",
          },
        ],
      },
    };
    const store = useHostModelsStore();
    store.byHost.local = {
      entries: [model("flux-dev:q8"), model("cv:123", { family: "MiniMaxH3" })],
      fetchedAt: Date.now(),
      error: null,
    };

    expect(store.modelsOn("local").map((entry) => entry.name)).toEqual(["flux-dev:q8"]);
    expect(store.installedOn("local").map((entry) => entry.name)).toEqual(["flux-dev:q8"]);
    expect(store.hostsFor("cv:123")).toEqual([]);
  });

  it("keeps and routes the exact authenticated private FL2VA row", () => {
    const h3 = "minimax-h3-fl2va:comfy-pruned-int8";
    const hosts = useHostsStore();
    hosts.capabilities.local = {
      gallery: { can_delete: true },
      ...authenticatedMiniMaxH3Capabilities(),
    } as ServerCapabilities;
    const store = useHostModelsStore();
    store.byHost.local = {
      entries: [
        model(h3, {
          family: "minimax-h3",
          generation_profile: {
            profile_hash: AUTHENTICATED_MINIMAX_H3_PROFILE_SHA256,
          } as never,
        }),
      ],
      fetchedAt: Date.now(),
      error: null,
    };

    expect(store.modelsOn("local").map((entry) => entry.name)).toEqual([h3]);
    expect(store.installedOn("local").map((entry) => entry.name)).toEqual([h3]);
    expect(store.hostsFor(h3)).toEqual(["local"]);
  });

  it("conservatively fills missing presentation metadata from duplicate host rows", () => {
    addExtra("hal9000-7680", "http://hal9000:7680");
    const store = useHostModelsStore();
    store.byHost["local"] = {
      entries: [
        model("shared-adapter", {
          description: "",
          kind: null,
          modality: null,
          nsfw: null,
          default_steps: 22,
        }),
      ],
      fetchedAt: Date.now(),
      error: null,
    };
    store.byHost["hal9000-7680"] = {
      entries: [
        model("shared-adapter", {
          description: "A mature portrait adapter.",
          kind: "lora",
          modality: "image",
          nsfw: true,
          default_steps: 40,
        }),
      ],
      fetchedAt: Date.now(),
      error: null,
    };

    expect(store.unionInstalled).toEqual([
      expect.objectContaining({
        name: "shared-adapter",
        hostIds: ["local", "hal9000-7680"],
        description: "A mature portrait adapter.",
        kind: "lora",
        modality: "image",
        nsfw: true,
        // Runtime defaults still come from the preferred first host.
        default_steps: 22,
      }),
    ]);
  });

  it("hostsFor() answers per-model availability", async () => {
    addExtra("hal9000-7680", "http://hal9000:7680");
    const store = useHostModelsStore();
    store.byHost["hal9000-7680"] = {
      entries: [model("z-image:q8")],
      fetchedAt: Date.now(),
      error: null,
    };
    expect(store.hostsFor("z-image:q8")).toEqual(["hal9000-7680"]);
    expect(store.hostsFor("never-heard-of-it")).toEqual([]);
  });

  it("resolves an explicit host's exact model row instead of the union's first row", () => {
    addExtra("hal9000-7680", "http://hal9000:7680");
    const store = useHostModelsStore();
    store.byHost.local = {
      entries: [model("flux-dev:q8", { default_steps: 20 })],
      fetchedAt: Date.now(),
      error: null,
    };
    store.byHost["hal9000-7680"] = {
      entries: [model("flux-dev:q8", { default_steps: 36 })],
      fetchedAt: Date.now(),
      error: null,
    };

    expect(store.installedEntryForTarget("flux-dev:q8", null)?.default_steps).toBe(20);
    expect(store.installedEntryForTarget("flux-dev:q8", "hal9000-7680")?.default_steps).toBe(36);
    expect(store.installedEntryForTarget("missing", "hal9000-7680")).toBeNull();
  });
});
