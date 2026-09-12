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

  /* Every row on this shelf is a style, so a "Checkpoint" badge on all of
   * them is noise that makes the rows that ARE something else harder to
   * spot. Only the exceptions carry a badge now. */
  it("badges only the rows that are not ordinary styles", () => {
    const checkpoint = mount(InstalledModelRow, {
      props: { model: makeModel() },
    });
    expect(checkpoint.find("[data-test=model-kind-badge]").exists()).toBe(
      false,
    );
    expect(checkpoint.text()).not.toContain("Checkpoint");

    const upscaler = mount(InstalledModelRow, {
      props: { model: makeModel({ family: "upscaler" }) },
    });
    expect(upscaler.get("[data-test=model-kind-badge]").text()).toBe(
      "Upscaler",
    );
  });

  /* The friendly name is what a person looks for; the runnable id is the
   * mono truth under it, never the row's headline. */
  it("leads with the style's friendly name and keeps the id in mono below", () => {
    const w = mount(InstalledModelRow, {
      props: {
        model: makeModel({
          name: "cv:8001",
          description: "Dreamy Photoreal",
        }),
      },
    });
    expect(w.get("[data-test=installed-row-name]").text()).toBe(
      "Dreamy Photoreal",
    );
    expect(w.get("[data-test=installed-row-id]").text()).toBe("cv:8001");
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

  it("shows a source glyph before the name, following modelSource", () => {
    const w = mount(InstalledModelRow, {
      props: {
        model: makeModel({
          name: "flux-dev:q8",
          hf_repo: "black-forest-labs/FLUX.1-dev",
        }),
      },
    });
    expect(w.find("svg[data-source='hf']").exists()).toBe(true);

    const civitai = mount(InstalledModelRow, {
      props: { model: makeModel({ name: "cv:8001", hf_repo: "" }) },
    });
    expect(civitai.find("svg[data-source='civitai']").exists()).toBe(true);

    const local = mount(InstalledModelRow, {
      props: { model: makeModel({ name: "my-lora.safetensors", hf_repo: "" }) },
    });
    expect(local.find("svg[data-source='local']").exists()).toBe(true);
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
    expect(install.attributes("aria-label")).toContain("studio");

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
