import { flushPromises, mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ref } from "vue";
import ModelDetailDrawer from "./ModelDetailDrawer.vue";
import type { CatalogEntryWire, ModelInfoExtended } from "../../types";
import type { ModelDetail, ModelVariant } from "../../composables/useCatalog";
import type { RoutableHost } from "../../lib/hostRouting";
import { useModelInstallTargets } from "../../composables/useModelInstallTargets";
import { addHost } from "../../lib/hostRegistry";

const makeEntry = (over: Partial<CatalogEntryWire> = {}): CatalogEntryWire => ({
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
  description: "A test model",
  license: "apache-2.0",
  license_flags: null,
  tags: [],
  trained_words: [],
  page_url: null,
  companions: [],
  companion_details: [],
  download_recipe: { files: [], needs_token: null },
  primary_path: null,
  created_at: null,
  updated_at: null,
  added_at: 0,
  ...over,
});

const makeModel = (
  over: Partial<ModelInfoExtended> = {},
): ModelInfoExtended => ({
  name: "flux-schnell:q8",
  family: "flux",
  size_gb: 12,
  is_loaded: false,
  last_used: null,
  hf_repo: "black-forest-labs/FLUX.1-schnell",
  downloaded: true,
  default_steps: 4,
  default_guidance: 0,
  default_width: 1024,
  default_height: 1024,
  description: "Fast FLUX",
  ...over,
});

const catalogDetail = (
  entry = makeEntry(),
  variants: ModelVariant[] = [{ id: entry.id, label: entry.file_format }],
): ModelDetail => ({ kind: "catalog", entry, variants });

const mockDetail = ref<ModelDetail | null>(null);
const mockDetailLoadingId = ref<string | null>(null);
const mockDetailError = ref<{ id: string; message: string } | null>(null);
const mockRetryDetail = vi.fn();
const mockCloseDetail = vi.fn();
const mockStartDownload = vi.fn();
const mockCanDownload = vi.fn((e: { supported: boolean }) => e.supported);
const mockLoad = vi.fn().mockResolvedValue(undefined);
const mockUnload = vi.fn().mockResolvedValue(undefined);
const mockDelete = vi
  .fn()
  .mockResolvedValue({ removed: [], kept: [], freed_bytes: 6_000_000_000 });

const mockAvailableManifests = ref<ModelInfoExtended[]>([]);
vi.mock("../../composables/useCatalog", () => ({
  useCatalog: () => ({
    availableManifests: mockAvailableManifests,
    detail: mockDetail,
    detailLoadingId: mockDetailLoadingId,
    detailError: mockDetailError,
    retryDetail: mockRetryDetail,
    closeDetail: mockCloseDetail,
    startDownload: mockStartDownload,
    canDownload: mockCanDownload,
    loadInstalled: mockLoad,
    unloadInstalled: mockUnload,
    deleteInstalled: mockDelete,
  }),
}));

const mockConfirm = vi.fn().mockResolvedValue(true);
const mockToast = vi.fn();
vi.mock("../../lib/toasts", () => ({
  requestConfirm: (opts: unknown) => mockConfirm(opts),
  toast: (...args: unknown[]) => mockToast(...args),
}));

/*
 * Multi-host install targeting. The drawer's action is only "Repair" once every
 * reachable machine owns the model; stub the routing poller so each test says
 * which machines exist and who owns what.
 */
const mockHosts = ref<RoutableHost[]>([]);
const mockOwners = ref<Record<string, string[]>>({});
const mockTargetModels = ref<ModelInfoExtended[]>([]);
vi.mock("../../composables/useHostRouting", () => ({
  useHostRouting: () => ({
    hosts: mockHosts,
    targetModels: mockTargetModels,
    modelOwnerIds: (name: string) => mockOwners.value[name] ?? [],
    inventoryKnown: () => true,
  }),
}));

const mockHostCatalogDownload = vi.fn().mockResolvedValue({ queued: 1 });
vi.mock("../machines/hostClient", () => ({
  hostModelDownload: (...args: unknown[]) => mockHostCatalogDownload(...args),
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

describe("ModelDetailDrawer", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockConfirm.mockResolvedValue(true);
    mockDetail.value = null;
    mockDetailLoadingId.value = null;
    mockDetailError.value = null;
    localStorage.clear();
    mockHosts.value = [host("origin")];
    mockOwners.value = {};
    mockAvailableManifests.value = [];
    mockTargetModels.value = [];
    useModelInstallTargets().cancel();
  });

  describe("never-blank states (G4)", () => {
    it("shows a loading state while a non-cached detail fetch is in flight", () => {
      mockDetailLoadingId.value = "cv:123";
      const w = mount(ModelDetailDrawer);
      expect(w.find("[data-test='detail-loading']").exists()).toBe(true);
    });

    it("shows a retryable error instead of a blank panel", async () => {
      mockDetailError.value = {
        id: "cv:123",
        message: "Couldn't load this model's details.",
      };
      const w = mount(ModelDetailDrawer);
      const err = w.find("[data-test='detail-error']");
      expect(err.exists()).toBe(true);
      expect(err.text()).toContain("Couldn't load");
      await w.get("[data-test='detail-retry']").trigger("click");
      expect(mockRetryDetail).toHaveBeenCalledWith("cv:123");
    });

    it("never crashes the render on a malformed entry (null download_recipe)", () => {
      mockDetail.value = catalogDetail(
        makeEntry({
          name: "Malformed Model",
          // Force the shape the type says can't happen but the wire sometimes
          // does — the drawer must render, not throw and blank out.
          download_recipe: null as unknown as {
            needs_token: null;
            files: [];
          },
        }),
      );
      const w = mount(ModelDetailDrawer);
      expect(w.find(".md").exists()).toBe(true);
      expect(w.text()).toContain("Malformed Model");
    });

    it("falls back to a terminal state when the detail has no renderable payload", () => {
      // The photographed bug: `detail` is set but carries nothing the drawer
      // can render, so every branch of the state chain misses and the open
      // panel paints blank. There must always be a last branch.
      mockDetail.value = {
        kind: "catalog",
        entry: null,
        variants: [],
      } as unknown as ModelDetail;
      const w = mount(ModelDetailDrawer);
      const fallback = w.find("[data-test='detail-unrenderable']");
      expect(fallback.exists()).toBe(true);
      expect(fallback.text().length).toBeGreaterThan(0);
    });

    it("falls back to a terminal state for an unknown detail kind", () => {
      mockDetail.value = { kind: "future-thing" } as unknown as ModelDetail;
      const w = mount(ModelDetailDrawer);
      expect(w.find("[data-test='detail-unrenderable']").exists()).toBe(true);
      expect(w.find("[data-test='detail-content']").exists()).toBe(false);
    });
  });

  describe("catalog (Discover)", () => {
    /* #1276: these checkpoints are 21-42 GB, and the only honest refusal used
       to arrive at submit time. The note has to be here, beside Pull, and it
       must never disable Pull — the model IS downloadable. */
    it("warns before the pull when the host cannot run the model", () => {
      const entry = makeEntry({
        id: "minimax-h3-ref2va:comfy-pruned-int8",
        supported: true,
      });
      mockAvailableManifests.value = [
        makeModel({
          name: "minimax-h3-ref2va:comfy-pruned-int8",
          downloaded: false,
          runtime_available: false,
          runtime_unavailable_reason:
            "MiniMax H3 reference-to-audio-video (Ref2VA) execution is not available in any released build.",
        }),
      ];
      mockDetail.value = catalogDetail(entry);
      const w = mount(ModelDetailDrawer);
      expect(w.get("[data-test=runtime-unavailable-note]").text()).toContain(
        "Ref2VA",
      );
      const pull = w.get("[data-test=pull-btn]");
      expect(pull.attributes("disabled")).toBeUndefined();
    });

    it("keeps quiet when another connected machine can run the model (#1276)", () => {
      // Pull can land on any reachable machine, so the origin's own answer
      // alone would be materially wrong wording on a mixed fleet.
      mockAvailableManifests.value = [
        makeModel({
          name: "minimax-h3-fl2va:comfy-pruned-int8",
          downloaded: false,
          runtime_available: false,
          runtime_unavailable_reason: "This build has no H3 engine.",
        }),
      ];
      mockTargetModels.value = [
        makeModel({
          name: "minimax-h3-fl2va:comfy-pruned-int8",
          runtime_available: true,
        }),
      ];
      mockDetail.value = catalogDetail(
        makeEntry({
          id: "minimax-h3-fl2va:comfy-pruned-int8",
          supported: true,
        }),
      );
      const w = mount(ModelDetailDrawer);
      expect(w.find("[data-test=runtime-unavailable-note]").exists()).toBe(
        false,
      );
    });

    it("says nothing about runtime for a model the host can run", () => {
      mockAvailableManifests.value = [
        makeModel({
          name: "cv:123",
          downloaded: false,
          runtime_available: true,
        }),
      ];
      mockDetail.value = catalogDetail(makeEntry({ id: "cv:123" }));
      const w = mount(ModelDetailDrawer);
      expect(w.find("[data-test=runtime-unavailable-note]").exists()).toBe(
        false,
      );
    });

    it("says nothing for a catalog id the host listing does not mention", () => {
      mockAvailableManifests.value = [];
      mockDetail.value = catalogDetail();
      const w = mount(ModelDetailDrawer);
      expect(w.find("[data-test=runtime-unavailable-note]").exists()).toBe(
        false,
      );
    });

    it("renders name, family, description, source and license", () => {
      mockDetail.value = catalogDetail();
      const w = mount(ModelDetailDrawer);
      expect(w.text()).toContain("Alpha");
      expect(w.text()).toContain("A test model");
      expect(w.text()).toContain("Hugging Face");
      expect(w.text()).toContain("apache-2.0");
    });

    it("renders model weights + full footprint stat tiles", () => {
      mockDetail.value = catalogDetail(
        makeEntry({
          size_bytes: 6_000_000_000,
          download_recipe: {
            needs_token: null,
            files: [
              {
                url: "https://hf.example/model",
                dest: "flux/model.safetensors",
                sha256: null,
                size_bytes: 8_000_000_000,
              },
            ],
          },
          companion_details: [
            {
              name: "flux2-te-9b",
              kind: "text-encoder",
              repo: "x",
              size_bytes: 16_000_000_000,
            },
            {
              name: "flux2-vae",
              kind: "vae",
              repo: "y",
              size_bytes: 168_000_000,
            },
          ],
        }),
      );
      const w = mount(ModelDetailDrawer);
      // Model weights = SIZE (weights only).
      expect(w.find("[data-test=tile-model-weights]").text()).toBe("6.0 GB");
      expect(w.get("[data-test=tile-model-weights-label]").text()).toBe(
        "Checkpoint weights",
      );
      // Full footprint = FETCH (weights + companions).
      expect(w.find("[data-test=tile-footprint]").text()).toBe("24.2 GB");
    });

    it("uses the model kind in the weights heading", () => {
      mockDetail.value = catalogDetail(makeEntry({ kind: "lora" }));
      const w = mount(ModelDetailDrawer);
      expect(w.get("[data-test=tile-model-weights-label]").text()).toBe(
        "LoRA weights",
      );
      expect(w.text()).not.toContain("Checkpoint weights");
    });

    it("renders companion component chips", () => {
      mockDetail.value = catalogDetail(
        makeEntry({
          companion_details: [
            {
              name: "flux2-te-9b",
              kind: "text-encoder",
              repo: "x",
              size_bytes: 1,
            },
            { name: "flux2-vae", kind: "vae", repo: "y", size_bytes: 1 },
          ],
        }),
      );
      const w = mount(ModelDetailDrawer);
      const chips = w.find("[data-test=components]");
      expect(chips.text()).toContain("flux2-te-9b");
      expect(chips.text()).toContain("flux2-vae");
    });

    it("Pull calls startDownload with the entry id", async () => {
      mockDetail.value = catalogDetail();
      const w = mount(ModelDetailDrawer);
      await w.find("[data-test=pull-btn]").trigger("click");
      await flushPromises();
      expect(mockStartDownload).toHaveBeenCalledWith("hf:a");
      expect(mockCloseDetail).toHaveBeenCalled();
    });

    it("keeps the drawer open and reports a rejected Pull", async () => {
      mockDetail.value = catalogDetail();
      mockStartDownload.mockRejectedValueOnce(
        new Error("not a supported built-in model or LoRA"),
      );
      const w = mount(ModelDetailDrawer);
      await w.find("[data-test=pull-btn]").trigger("click");
      await flushPromises();

      expect(mockCloseDetail).not.toHaveBeenCalled();
      expect(mockToast).toHaveBeenCalledWith(
        "error",
        "not a supported built-in model or LoRA",
      );
    });

    it("selecting a variant changes the pull target", async () => {
      mockDetail.value = catalogDetail(makeEntry(), [
        { id: "hf:a", label: "safetensors" },
        { id: "hf:a@q4", label: "q4" },
      ]);
      const w = mount(ModelDetailDrawer);
      const variantBtns = w.find("[data-test=variants]").findAll("button");
      const q4 = variantBtns.find((b) => b.text() === "q4");
      expect(q4).toBeDefined();
      await q4!.trigger("click");
      await w.find("[data-test=pull-btn]").trigger("click");
      expect(mockStartDownload).toHaveBeenCalledWith("hf:a@q4");
    });

    it("shows Repair for an installed catalog entry", async () => {
      mockDetail.value = catalogDetail(makeEntry({ installed: true }));
      const w = mount(ModelDetailDrawer);
      const repair = w.find("[data-test=repair-btn]");
      expect(repair.exists()).toBe(true);
      await repair.trigger("click");
      expect(mockStartDownload).toHaveBeenCalledWith("hf:a");
    });

    it("keeps Pull for an installed entry a connected machine still lacks", async () => {
      const entry = addHost({
        url: "http://studio.local:7680",
        name: "Studio",
      });
      mockHosts.value = [host("origin"), host(entry.id)];
      mockOwners.value = { "hf:a": ["origin"] };
      mockDetail.value = catalogDetail(makeEntry({ installed: true }));

      const w = mount(ModelDetailDrawer);
      expect(w.find("[data-test=repair-btn]").exists()).toBe(false);
      const pull = w.find("[data-test=pull-btn]");
      expect(pull.exists()).toBe(true);

      await pull.trigger("click");
      await flushPromises();

      // Two machines, two different outcomes — the user picks.
      const targeting = useModelInstallTargets();
      expect(targeting.pending.value?.targets.map((t) => t.action)).toEqual([
        "install",
        "repair",
      ]);
      expect(mockStartDownload).not.toHaveBeenCalled();

      targeting.choose(targeting.pending.value!.targets[0]);
      await flushPromises();

      expect(mockHostCatalogDownload).toHaveBeenCalledWith(entry, "hf:a");
      expect(mockCloseDetail).toHaveBeenCalled();
    });

    it("disables Pull with an unsupported tooltip", () => {
      mockDetail.value = catalogDetail(makeEntry({ supported: false }));
      const w = mount(ModelDetailDrawer);
      const pull = w.find("[data-test=pull-btn]");
      expect((pull.element as HTMLButtonElement).disabled).toBe(true);
      expect(pull.attributes("title")).toMatch(/unsupported/i);
    });
  });

  describe("hero preview", () => {
    it("renders the thumbnail as a hero image", () => {
      mockDetail.value = catalogDetail(
        makeEntry({
          thumbnail_url:
            "https://image.civitai.com/abc/def/width=512/12345.jpeg",
        }),
      );
      const w = mount(ModelDetailDrawer);
      const img = w.find("[data-test=detail-hero] img");
      expect(img.exists()).toBe(true);
      expect(img.attributes("src")).toContain("12345.jpeg");
    });

    it("renders no hero block when there is no thumbnail", () => {
      mockDetail.value = catalogDetail(makeEntry({ thumbnail_url: null }));
      const w = mount(ModelDetailDrawer);
      expect(w.find("[data-test=detail-hero]").exists()).toBe(false);
    });

    it("drops the hero block when the image fails to load", async () => {
      mockDetail.value = catalogDetail(
        makeEntry({ thumbnail_url: "https://image.example/broken.jpeg" }),
      );
      const w = mount(ModelDetailDrawer);
      await w.get("[data-test=detail-hero] img").trigger("error");
      expect(w.find("[data-test=detail-hero]").exists()).toBe(false);
    });

    it("uses a video element for Civitai video previews", () => {
      mockDetail.value = catalogDetail(
        makeEntry({
          thumbnail_url:
            "https://image.civitai.com/abc/def/width=512/75044257.mp4",
        }),
      );
      const w = mount(ModelDetailDrawer);
      expect(w.find("[data-test=detail-hero] video").exists()).toBe(true);
      expect(w.find("[data-test=detail-hero] img").exists()).toBe(false);
    });
  });

  describe("catalog metadata", () => {
    it("prominently classifies modality, kind, and mature content", () => {
      mockDetail.value = catalogDetail(
        makeEntry({
          modality: "video",
          kind: "lora",
          nsfw: true,
        }),
      );
      const w = mount(ModelDetailDrawer);
      const classification = w.get("[data-test=model-classification]");
      expect(
        classification.get("[data-test=model-modality-badge]").text(),
      ).toBe("Video");
      expect(classification.get("[data-test=model-kind-badge]").text()).toBe(
        "LoRA",
      );
      expect(classification.get("[data-test=model-nsfw-badge]").text()).toBe(
        "18+ NSFW",
      );
    });

    it("omits the mature-content classification for safe models", () => {
      mockDetail.value = catalogDetail(makeEntry({ nsfw: false }));
      const w = mount(ModelDetailDrawer);
      expect(w.find("[data-test=model-nsfw-badge]").exists()).toBe(false);
    });

    it("renders downloads, likes, rating, format and updated date", () => {
      mockDetail.value = catalogDetail(
        makeEntry({
          kind: "lora",
          file_format: "safetensors",
          download_count: 1_579_037,
          likes: 4200,
          rating: 4.7,
          // 2025-06-01T00:00:00Z
          updated_at: 1_748_736_000,
        }),
      );
      const w = mount(ModelDetailDrawer);
      const rows = w.get("[data-test=meta-rows]").text();
      expect(rows).toContain("Downloads");
      expect(rows).toContain("1.6M");
      expect(rows).toContain("Likes");
      expect(rows).toContain("4.2k");
      expect(rows).toContain("Rating");
      expect(rows).toContain("4.7");
      expect(rows).toContain("safetensors");
      expect(rows).toContain("Updated");
      expect(rows).toContain("2025");
    });

    it("renders useful package metadata without duplicating the kind badge", () => {
      mockDetail.value = catalogDetail(
        makeEntry({
          kind: "control-net",
          family_role: "finetune",
          sub_family: "canny",
          bundling: "single-file",
        }),
      );
      const w = mount(ModelDetailDrawer);
      const rows = w.get("[data-test=meta-rows]").text();
      expect(rows).toContain("Role");
      expect(rows).toContain("Fine-tune");
      expect(rows).toContain("Variant");
      expect(rows).toContain("canny");
      expect(rows).toContain("Package");
      expect(rows).toContain("Single file");
      expect(rows).not.toContain("Kind");
      expect(w.get("[data-test=model-kind-badge]").text()).toBe("ControlNet");
    });

    it("renders available catalog tags and omits the section when empty", () => {
      mockDetail.value = catalogDetail(
        makeEntry({
          tags: ["portrait", "cinematic", " skin tones ", "portrait"],
        }),
      );
      const w = mount(ModelDetailDrawer);
      const tags = w.get("[data-test=model-tags]");
      expect(tags.text()).toContain("portrait");
      expect(tags.text()).toContain("cinematic");
      expect(tags.text()).toContain("skin tones");
      expect(tags.findAll(".md__chip")).toHaveLength(3);

      mockDetail.value = catalogDetail(makeEntry({ tags: [] }));
      const empty = mount(ModelDetailDrawer);
      expect(empty.find("[data-test=model-tags]").exists()).toBe(false);
    });

    it("skips null fields instead of printing an em dash for each", () => {
      mockDetail.value = catalogDetail(
        makeEntry({
          author: null,
          license: null,
          rating: null,
          likes: 0,
          download_count: 0,
          updated_at: null,
        }),
      );
      const w = mount(ModelDetailDrawer);
      const rows = w.get("[data-test=meta-rows]").text();
      expect(rows).not.toContain("—");
      expect(rows).not.toContain("Rating");
      expect(rows).not.toContain("Likes");
      expect(rows).not.toContain("License");
      // Source is always known for a catalog row.
      expect(rows).toContain("Hugging Face");
    });

    it("renders trained words as chips", () => {
      mockDetail.value = catalogDetail(
        makeEntry({ trained_words: ["ohwx woman", "cinematic glow"] }),
      );
      const w = mount(ModelDetailDrawer);
      const chips = w.get("[data-test=trained-words]");
      expect(chips.text()).toContain("ohwx woman");
      expect(chips.text()).toContain("cinematic glow");
    });

    it("omits the trained-words section when there are none", () => {
      mockDetail.value = catalogDetail(makeEntry({ trained_words: [] }));
      const w = mount(ModelDetailDrawer);
      expect(w.find("[data-test=trained-words]").exists()).toBe(false);
    });

    it("links out to the upstream model page", () => {
      mockDetail.value = catalogDetail(
        makeEntry({
          source: "civitai",
          page_url: "https://civitai.com/models/133005",
        }),
      );
      const w = mount(ModelDetailDrawer);
      const link = w.get("[data-test=page-link]");
      expect(link.attributes("href")).toBe("https://civitai.com/models/133005");
      expect(link.attributes("target")).toBe("_blank");
      expect(link.attributes("rel")).toContain("noopener");
      expect(link.text()).toContain("Civitai");
    });

    it("omits the page link when the entry has no page url", () => {
      mockDetail.value = catalogDetail(makeEntry({ page_url: null }));
      const w = mount(ModelDetailDrawer);
      expect(w.find("[data-test=page-link]").exists()).toBe(false);
    });
  });

  describe("installed model", () => {
    it.each(["Studio Model", "Studio Model by Catalog Author"])(
      "suppresses legacy synthetic description %j",
      (description) => {
        mockDetail.value = {
          kind: "installed",
          model: makeModel({
            display_name: "Studio Model",
            description,
          }),
          components: [],
        };
        const w = mount(ModelDetailDrawer);
        expect(w.get(".md__name").text()).toBe("Studio Model");
        expect(w.find(".md__desc").exists()).toBe(false);
      },
    );

    it("keeps a meaningful installed description", () => {
      mockDetail.value = {
        kind: "installed",
        model: makeModel({
          display_name: "Studio Model",
          description: "A fast portrait model tuned for natural light.",
        }),
        components: [],
      };
      const w = mount(ModelDetailDrawer);
      expect(w.get(".md__desc").text()).toContain(
        "A fast portrait model tuned for natural light.",
      );
    });

    it("renders retained catalog sidecar metadata for an installed model", () => {
      mockDetail.value = {
        kind: "installed",
        model: makeModel({
          name: "cv:4242",
          display_name: "Studio Adapter",
          hf_repo: "",
          description: "",
        }),
        components: [],
        catalogEntry: makeEntry({
          id: "cv:4242",
          source: "civitai",
          source_id: "4242",
          name: "Studio Adapter",
          author: "Catalog author",
          description: "Retained sidecar description.",
          license: "apache-2.0",
          tags: ["portrait", "cinematic"],
          page_url: "https://civitai.com/models/42?modelVersionId=4242",
        }),
      };
      const w = mount(ModelDetailDrawer);

      expect(w.get(".md__desc").text()).toContain(
        "Retained sidecar description.",
      );
      expect(w.get("[data-test=model-tags]").text()).toContain("portrait");
      expect(w.get("[data-test=meta-rows]").text()).toContain("Catalog author");
      expect(w.get("[data-test=meta-rows]").text()).toContain("Civitai");
      expect(w.get("[data-test=meta-rows]").text()).toContain("apache-2.0");
      expect(w.get("[data-test=page-link]").attributes("href")).toContain(
        "modelVersionId=4242",
      );
      expect(w.find("[data-test=load-btn]").exists()).toBe(true);
    });

    it("classifies installed models from their family", () => {
      mockDetail.value = {
        kind: "installed",
        model: makeModel({ family: "upscaler" }),
        components: [],
      };
      const w = mount(ModelDetailDrawer);
      expect(w.get("[data-test=model-modality-badge]").text()).toBe("Image");
      expect(w.get("[data-test=model-kind-badge]").text()).toBe("Upscaler");
    });

    it("prefers exact installed catalog metadata over family inference", () => {
      mockDetail.value = {
        kind: "installed",
        model: makeModel({
          family: "flux",
          kind: "lora",
          modality: "video",
          nsfw: true,
        }),
        components: [],
      };
      const w = mount(ModelDetailDrawer);
      expect(w.get("[data-test=model-modality-badge]").text()).toBe("Video");
      expect(w.get("[data-test=model-kind-badge]").text()).toBe("LoRA");
      expect(w.get("[data-test=model-nsfw-badge]").text()).toBe("18+ NSFW");
    });

    it("offers to install an installed model on a machine that lacks it", async () => {
      const entry = addHost({
        url: "http://studio.local:7680",
        name: "Studio",
      });
      mockHosts.value = [host("origin"), host(entry.id)];
      mockOwners.value = { "flux-schnell:q8": ["origin"] };
      mockDetail.value = {
        kind: "installed",
        model: makeModel(),
        components: [],
      };

      const w = mount(ModelDetailDrawer);
      const install = w.find("[data-test=install-elsewhere-btn]");
      expect(install.exists()).toBe(true);

      await install.trigger("click");
      await flushPromises();

      const targeting = useModelInstallTargets();
      targeting.choose(targeting.pending.value!.targets[0]);
      await flushPromises();

      expect(mockHostCatalogDownload).toHaveBeenCalledWith(
        entry,
        "flux-schnell:q8",
      );
    });

    it("hides the install action once every reachable machine owns it", () => {
      const entry = addHost({
        url: "http://studio.local:7680",
        name: "Studio",
      });
      mockHosts.value = [host("origin"), host(entry.id)];
      mockOwners.value = { "flux-schnell:q8": ["origin", entry.id] };
      mockDetail.value = {
        kind: "installed",
        model: makeModel(),
        components: [],
      };

      const w = mount(ModelDetailDrawer);
      expect(w.find("[data-test=install-elsewhere-btn]").exists()).toBe(false);
    });

    it("shows no install action for a single-host registry", () => {
      mockDetail.value = {
        kind: "installed",
        model: makeModel(),
        components: [],
      };
      const w = mount(ModelDetailDrawer);
      expect(w.find("[data-test=install-elsewhere-btn]").exists()).toBe(false);
    });

    it("renders Load, Unload and Delete actions", () => {
      mockDetail.value = {
        kind: "installed",
        model: makeModel(),
        components: [],
      };
      const w = mount(ModelDetailDrawer);
      expect(w.find("[data-test=load-btn]").exists()).toBe(true);
      expect(w.find("[data-test=unload-btn]").exists()).toBe(true);
      expect(w.find("[data-test=delete-btn]").exists()).toBe(true);
    });

    it("renders fetched component chips", () => {
      mockDetail.value = {
        kind: "installed",
        model: makeModel(),
        components: [
          { kind: "transformer", name: "transformer", present: true },
          { kind: "vae", name: "flux-vae", present: true },
        ],
      };
      const w = mount(ModelDetailDrawer);
      const chips = w.find("[data-test=components]");
      expect(chips.text()).toContain("transformer");
      expect(chips.text()).toContain("flux-vae");
    });

    it("does not call never-downloaded Discover companions missing", () => {
      mockDetail.value = catalogDetail(
        makeEntry({
          installed: false,
          companion_details: [
            {
              name: "t5xxl",
              kind: "text-encoder",
              repo: "org/t5xxl",
              size_bytes: 1,
            },
          ],
        }),
      );
      const w = mount(ModelDetailDrawer);
      expect(w.get("[data-test=component-included]").text()).toBe("t5xxl");
      expect(w.find("[data-test=component-missing]").exists()).toBe(false);
    });

    it("labels missing components and repairs the exact advertised target", async () => {
      mockDetail.value = {
        kind: "installed",
        model: makeModel(),
        components: [
          {
            kind: "text-encoder",
            name: "t5xxl",
            present: false,
            repair_model: "t5xxl:fp16",
          },
        ],
      };
      const w = mount(ModelDetailDrawer);
      expect(w.get("[data-test=component-missing]").text()).toContain(
        "missing",
      );
      await w.get("[data-test=component-repair]").trigger("click");
      expect(mockStartDownload).toHaveBeenCalledWith("t5xxl:fp16");
    });

    it("Load calls loadInstalled with the model name", async () => {
      mockDetail.value = {
        kind: "installed",
        model: makeModel(),
        components: [],
      };
      const w = mount(ModelDetailDrawer);
      await w.find("[data-test=load-btn]").trigger("click");
      await flushPromises();
      expect(mockLoad).toHaveBeenCalledWith("flux-schnell:q8");
    });

    it("disables Load and enables Unload when the model is loaded", () => {
      mockDetail.value = {
        kind: "installed",
        model: makeModel({ is_loaded: true }),
        components: [],
      };
      const w = mount(ModelDetailDrawer);
      expect(
        (w.find("[data-test=load-btn]").element as HTMLButtonElement).disabled,
      ).toBe(true);
      expect(
        (w.find("[data-test=unload-btn]").element as HTMLButtonElement)
          .disabled,
      ).toBe(false);
      // Deleting a loaded model would 409 server-side — keep it disabled.
      expect(
        (w.find("[data-test=delete-btn]").element as HTMLButtonElement)
          .disabled,
      ).toBe(true);
    });

    it("Unload calls unloadInstalled with the model name", async () => {
      mockDetail.value = {
        kind: "installed",
        model: makeModel({ is_loaded: true }),
        components: [],
      };
      const w = mount(ModelDetailDrawer);
      await w.find("[data-test=unload-btn]").trigger("click");
      await flushPromises();
      expect(mockUnload).toHaveBeenCalledWith("flux-schnell:q8");
    });

    it("hides Load/Unload and shows the server's reason for a download-only (runtime_available: false) row", () => {
      mockDetail.value = {
        kind: "installed",
        model: makeModel({
          runtime_available: false,
          runtime_unavailable_reason:
            "MiniMax H3 has no runtime for this model's weight layout in this build.",
        }),
        components: [],
      };
      const w = mount(ModelDetailDrawer);
      expect(w.find("[data-test=load-btn]").exists()).toBe(false);
      expect(w.find("[data-test=unload-btn]").exists()).toBe(false);
      expect(w.get("[data-test=runtime-unavailable-note]").text()).toContain(
        "no runtime for this model's weight layout",
      );
      // Pull/Repair/Remove stay reachable — only the runtime actions hide.
      expect(w.find("[data-test=delete-btn]").exists()).toBe(true);
    });

    it("falls back to a cause-free note when an older server sends no reason", () => {
      mockDetail.value = {
        kind: "installed",
        model: makeModel({ runtime_available: false }),
        components: [],
      };
      const w = mount(ModelDetailDrawer);
      expect(w.get("[data-test=runtime-unavailable-note]").text()).toContain(
        "cannot run this model",
      );
    });

    it("Delete confirms first, then calls deleteInstalled", async () => {
      mockDetail.value = {
        kind: "installed",
        model: makeModel(),
        components: [],
      };
      const w = mount(ModelDetailDrawer);
      await w.find("[data-test=delete-btn]").trigger("click");
      await flushPromises();
      expect(mockConfirm).toHaveBeenCalled();
      expect(mockDelete).toHaveBeenCalledWith("flux-schnell:q8");
    });

    it("does not delete when the confirm is dismissed", async () => {
      mockConfirm.mockResolvedValue(false);
      mockDetail.value = {
        kind: "installed",
        model: makeModel(),
        components: [],
      };
      const w = mount(ModelDetailDrawer);
      await w.find("[data-test=delete-btn]").trigger("click");
      await flushPromises();
      expect(mockConfirm).toHaveBeenCalled();
      expect(mockDelete).not.toHaveBeenCalled();
    });
  });

  it("the drawer close control calls closeDetail", async () => {
    mockDetail.value = catalogDetail();
    const w = mount(ModelDetailDrawer);
    await w.find("button[aria-label=Close]").trigger("click");
    expect(mockCloseDetail).toHaveBeenCalled();
  });
});

describe("viewport anchoring", () => {
  it("mounts the panel in a viewport-fixed host so scroll position can't hide it", async () => {
    // DrawerPanel/SheetPanel are `position: absolute; inset: 0` — they fill
    // their nearest positioned ancestor. Mounted inline on the long Models
    // page that ancestor is the scrolling column, so opening a card far down
    // the list drew the drawer at the top of the DOCUMENT, off-screen. The
    // fixed host re-anchors it to the viewport.
    mockDetail.value = {
      kind: "catalog",
      entry: null,
      variants: [],
    } as unknown as ModelDetail;
    const w = mount(ModelDetailDrawer);
    await flushPromises();

    const host = w.find("[data-test='detail-drawer-host']");
    expect(host.exists()).toBe(true);
    expect(host.classes()).toContain("fixed");
    expect(host.classes()).toContain("inset-0");
  });

  it("renders no host at all while closed", () => {
    mockDetail.value = null;
    mockDetailLoadingId.value = null;
    mockDetailError.value = null;
    const w = mount(ModelDetailDrawer);
    expect(w.find("[data-test='detail-drawer-host']").exists()).toBe(false);
  });
});
