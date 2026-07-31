import { mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ref } from "vue";
import type { ModelInfoExtended } from "../../types";
import type { RoutableHost } from "../../lib/hostRouting";
import InstalledModelRow from "./InstalledModelRow.vue";

/* Multi-host install targeting — see useModelInstallTargets. */
const mockHosts = ref<RoutableHost[]>([]);
const mockOwners = ref<Record<string, string[]>>({});
vi.mock("../../composables/useHostRouting", () => ({
  useHostRouting: () => ({
    hosts: mockHosts,
    modelOwnerIds: (name: string) => mockOwners.value[name] ?? [],
    inventoryKnown: () => true,
  }),
}));

function host(id: string, over: Partial<RoutableHost> = {}): RoutableHost {
  return {
    id,
    label: id === "origin" ? "this server" : id,
    url: id === "origin" ? "" : `http://${id}:7680`,
    status: "ready",
    queueDepth: 0,
    gpu: null,
    ...over,
  };
}

beforeEach(() => {
  mockHosts.value = [host("origin")];
  mockOwners.value = {};
});

function makeModel(over: Partial<ModelInfoExtended> = {}): ModelInfoExtended {
  return {
    name: "flux-schnell:q8",
    family: "flux",
    size_gb: 12.3,
    is_loaded: false,
    last_used: null,
    hf_repo: "black-forest-labs/FLUX.1-schnell",
    downloaded: true,
    default_steps: 4,
    default_guidance: 0,
    default_width: 1024,
    default_height: 1024,
    description: "",
    ...over,
  };
}

describe("InstalledModelRow", () => {
  it("renders the model name and family · size", () => {
    const w = mount(InstalledModelRow, { props: { model: makeModel() } });
    expect(w.text()).toContain("flux-schnell:q8");
    expect(w.text()).toContain("flux");
    expect(w.text()).toContain("12.3 GB");
  });

  it("shows the inferred model type", () => {
    const checkpoint = mount(InstalledModelRow, {
      props: { model: makeModel() },
    });
    expect(checkpoint.get("[data-test=model-kind-badge]").text()).toBe(
      "Checkpoint",
    );

    const upscaler = mount(InstalledModelRow, {
      props: { model: makeModel({ family: "upscaler" }) },
    });
    expect(upscaler.get("[data-test=model-kind-badge]").text()).toBe(
      "Upscaler",
    );
  });

  it("honors additive kind and NSFW metadata from newer servers", () => {
    const model = {
      ...makeModel(),
      kind: "lora",
      nsfw: true,
    } as ModelInfoExtended;
    const w = mount(InstalledModelRow, { props: { model } });
    expect(w.get("[data-test=model-kind-badge]").text()).toBe("LoRA");
    expect(w.get("[data-test=model-nsfw-badge]").text()).toBe("18+ NSFW");
  });

  it("shows the ★ loaded badge only when the model is loaded", () => {
    const loaded = mount(InstalledModelRow, {
      props: { model: makeModel({ is_loaded: true }) },
    });
    expect(loaded.find("[data-test=loaded-badge]").exists()).toBe(true);
    expect(loaded.text()).toMatch(/★ loaded/);

    const idle = mount(InstalledModelRow, {
      props: { model: makeModel({ is_loaded: false }) },
    });
    expect(idle.find("[data-test=loaded-badge]").exists()).toBe(false);
  });

  it("emits open when the row is clicked", async () => {
    const w = mount(InstalledModelRow, { props: { model: makeModel() } });
    await w.find("[data-test=installed-row]").trigger("click");
    expect(w.emitted("open")).toBeTruthy();
  });

  it("adds no install control for a single-host registry", () => {
    const w = mount(InstalledModelRow, { props: { model: makeModel() } });
    expect(w.find("[data-test=install-elsewhere-btn]").exists()).toBe(false);
  });

  it("offers an install when another reachable machine lacks the model", async () => {
    mockHosts.value = [host("origin"), host("studio")];
    mockOwners.value = { "flux-schnell:q8": ["origin"] };
    const w = mount(InstalledModelRow, { props: { model: makeModel() } });

    const install = w.find("[data-test=install-elsewhere-btn]");
    expect(install.exists()).toBe(true);
    expect(install.attributes("title")).toContain("studio");

    await install.trigger("click");
    expect(w.emitted("install")).toBeTruthy();
    // The row action must not double as opening the drawer.
    expect(w.emitted("open")).toBeFalsy();
  });

  it("drops the install control once every reachable machine owns it", () => {
    mockHosts.value = [host("origin"), host("studio")];
    mockOwners.value = { "flux-schnell:q8": ["origin", "studio"] };
    const w = mount(InstalledModelRow, { props: { model: makeModel() } });
    expect(w.find("[data-test=install-elsewhere-btn]").exists()).toBe(false);
  });
});
