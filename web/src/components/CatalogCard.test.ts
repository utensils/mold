import { mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ref } from "vue";
import type { CatalogEntryWire } from "../types";
import type { RoutableHost } from "../lib/hostRouting";
import CatalogCard from "./CatalogCard.vue";

/*
 * The card's action label is multi-host: it stays a Pull while any reachable
 * machine lacks the model. Stub the routing poller so each test states exactly
 * which machines exist and which of them own the entry.
 */
const mockHosts = ref<RoutableHost[]>([]);
const mockOwners = ref<Record<string, string[]>>({});

vi.mock("../composables/useHostRouting", () => ({
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

const baseEntry: CatalogEntryWire = {
  id: "hf:a",
  name: "Alpha",
  family: "flux",
  supported: true,
  installed: false,
  source: "hf",
  source_id: "a",
  author: "alice",
  family_role: "finetune",
  sub_family: null,
  modality: "image",
  kind: "checkpoint",
  file_format: "safetensors",
  bundling: "separated",
  size_bytes: 6_000_000_000,
  download_count: 1234,
  rating: 4.7,
  likes: 0,
  nsfw: false,
  thumbnail_url: null,
  description: null,
  license: null,
  license_flags: null,
  tags: [],
  companions: [],
  download_recipe: { files: [], needs_token: null },
  primary_path: null,
  created_at: null,
  updated_at: null,
  added_at: 0,
};

describe("CatalogCard (discover)", () => {
  it("renders a friendly kind badge, name, and family · size", () => {
    const w = mount(CatalogCard, { props: { entry: baseEntry } });
    expect(w.get("[data-test=model-kind-badge]").text()).toBe("Checkpoint");
    expect(w.text()).toContain("Alpha");
    expect(w.text()).toContain("flux");
    expect(w.text()).toContain("6.0 GB");
  });

  it.each([
    ["checkpoint", "Checkpoint"],
    ["lora", "LoRA"],
    ["vae", "VAE"],
    ["text-encoder", "Text encoder"],
    ["tokenizer", "Tokenizer"],
    ["clip", "CLIP"],
    ["control-net", "ControlNet"],
  ] as const)("labels %s entries as %s", (kind, label) => {
    const w = mount(CatalogCard, {
      props: { entry: { ...baseEntry, kind } },
    });
    expect(w.get("[data-test=model-kind-badge]").text()).toBe(label);
  });

  it("shows a literal 18+ NSFW badge and includes it in accessible names", () => {
    const entry: CatalogEntryWire = {
      ...baseEntry,
      kind: "lora",
      nsfw: true,
    };
    const w = mount(CatalogCard, { props: { entry } });

    expect(w.get("[data-test=model-nsfw-badge]").text()).toBe("18+ NSFW");
    expect(w.get("[data-test=discover-card]").attributes("aria-label")).toMatch(
      /Alpha.*LoRA.*18\+ NSFW/i,
    );
    expect(w.get("[data-test=card-open]").attributes("aria-label")).toMatch(
      /Alpha.*LoRA.*18\+ NSFW/i,
    );
    expect(w.get("[data-test=details-btn]").attributes("aria-label")).toMatch(
      /Alpha.*LoRA.*18\+ NSFW/i,
    );
  });

  it("omits the mature-content badge for safe entries", () => {
    const w = mount(CatalogCard, { props: { entry: baseEntry } });
    expect(w.find("[data-test=model-nsfw-badge]").exists()).toBe(false);
  });

  it.each(["grid", "list"] as const)(
    "shows a nonblank description cleanly in %s layout",
    (layout) => {
      const w = mount(CatalogCard, {
        props: {
          entry: {
            ...baseEntry,
            description:
              "A cinematic portrait model with carefully tuned skin tones.",
          },
          layout,
        },
      });
      expect(w.get("[data-test=card-description]").text()).toContain(
        "A cinematic portrait model",
      );
      expect(w.get("[data-test=discover-card]").attributes("data-layout")).toBe(
        layout,
      );
    },
  );

  it("omits empty and whitespace-only descriptions", () => {
    const absent = mount(CatalogCard, { props: { entry: baseEntry } });
    const blank = mount(CatalogCard, {
      props: { entry: { ...baseEntry, description: "   " } },
    });
    expect(absent.find("[data-test=card-description]").exists()).toBe(false);
    expect(blank.find("[data-test=card-description]").exists()).toBe(false);
  });

  it("Pull button emits pull", async () => {
    const w = mount(CatalogCard, { props: { entry: baseEntry } });
    const pull = w.find("[data-test=pull-btn]");
    expect(pull.text()).toContain("Pull");
    await pull.trigger("click");
    expect(w.emitted("pull")).toBeTruthy();
  });

  it("Details button emits open", async () => {
    const w = mount(CatalogCard, { props: { entry: baseEntry } });
    await w.find("[data-test=details-btn]").trigger("click");
    expect(w.emitted("open")).toBeTruthy();
  });

  it("card name line emits open", async () => {
    const w = mount(CatalogCard, { props: { entry: baseEntry } });
    await w.find("[data-test=card-open]").trigger("click");
    expect(w.emitted("open")).toBeTruthy();
  });

  it("disables Pull with an unsupported tooltip", () => {
    const entry: CatalogEntryWire = { ...baseEntry, supported: false };
    const w = mount(CatalogCard, { props: { entry } });
    const pull = w.find("[data-test=pull-btn]");
    expect((pull.element as HTMLButtonElement).disabled).toBe(true);
    expect(pull.attributes("title")).toMatch(/unsupported/i);
  });

  it("labels Pull as Repair and shows an installed badge when installed", () => {
    const entry: CatalogEntryWire = { ...baseEntry, installed: true };
    const w = mount(CatalogCard, { props: { entry } });
    expect(w.find("[data-test=pull-btn]").text()).toContain("Repair");
    expect(w.text()).toMatch(/installed/i);
  });

  it("does not show an installed badge when not installed", () => {
    const w = mount(CatalogCard, { props: { entry: baseEntry } });
    expect(w.text()).not.toMatch(/installed/i);
  });

  it("still offers a Pull when a connected machine lacks an installed model", () => {
    // Installed here, absent there: collapsing that into one boolean is what
    // hid the install action for the machine that does not have it.
    mockHosts.value = [host("origin"), host("studio")];
    mockOwners.value = { "hf:a": ["origin"] };
    const entry: CatalogEntryWire = { ...baseEntry, installed: true };
    const w = mount(CatalogCard, { props: { entry } });

    expect(w.find("[data-test=pull-btn]").text()).toContain("Pull");
    expect(w.find("[data-test=pull-btn]").text()).not.toContain("Repair");
    // The origin still owns it, so the "installed" tag is still accurate.
    expect(w.text()).toMatch(/installed/i);
  });

  it("degrades to Repair once every reachable machine owns it", () => {
    mockHosts.value = [host("origin"), host("studio")];
    mockOwners.value = { "hf:a": ["origin", "studio"] };
    const entry: CatalogEntryWire = { ...baseEntry, installed: true };
    const w = mount(CatalogCard, { props: { entry } });

    expect(w.find("[data-test=pull-btn]").text()).toContain("Repair");
  });

  it("renders a lazy preview image when the entry has a thumbnail", () => {
    const entry: CatalogEntryWire = {
      ...baseEntry,
      thumbnail_url: "https://example.test/preview.jpeg",
    };
    const w = mount(CatalogCard, { props: { entry } });
    const img = w.get("[data-test=card-thumb]");
    expect(img.attributes("src")).toBe("https://example.test/preview.jpeg");
    expect(img.attributes("loading")).toBe("lazy");
    expect(img.attributes("decoding")).toBe("async");
    expect(w.find("[data-test=card-thumb-placeholder]").exists()).toBe(false);
  });

  it("normalizes a Civitai CDN thumbnail to one shared width derivative", () => {
    const entry: CatalogEntryWire = {
      ...baseEntry,
      thumbnail_url:
        "https://image.civitai.com/token/id/original=true/preview.jpeg",
    };
    const w = mount(CatalogCard, { props: { entry } });
    expect(w.get("[data-test=card-thumb]").attributes("src")).toBe(
      "https://image.civitai.com/token/id/width=512/preview.jpeg",
    );
  });

  it("renders the family placeholder and no image without a thumbnail", () => {
    const w = mount(CatalogCard, { props: { entry: baseEntry } });
    const placeholder = w.get("[data-test=card-thumb-placeholder]");
    expect(placeholder.text()).toContain("FLUX");
    expect(w.find("[data-test=card-thumb]").exists()).toBe(false);
  });

  it("falls back to the placeholder when the thumbnail fails to load", async () => {
    const entry: CatalogEntryWire = {
      ...baseEntry,
      thumbnail_url: "https://example.test/gone.jpeg",
    };
    const w = mount(CatalogCard, { props: { entry } });
    await w.get("[data-test=card-thumb]").trigger("error");
    expect(w.find("[data-test=card-thumb]").exists()).toBe(false);
    expect(w.find("[data-test=card-thumb-placeholder]").exists()).toBe(true);
  });

  it("hides the loading shimmer once the thumbnail loads", async () => {
    const entry: CatalogEntryWire = {
      ...baseEntry,
      thumbnail_url: "https://example.test/preview.jpeg",
    };
    const w = mount(CatalogCard, { props: { entry } });
    expect(w.find("[data-test=card-thumb-shimmer]").exists()).toBe(true);
    await w.get("[data-test=card-thumb]").trigger("load");
    expect(w.find("[data-test=card-thumb-shimmer]").exists()).toBe(false);
  });
});
