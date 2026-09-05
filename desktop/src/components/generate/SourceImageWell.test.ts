import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { mount, flushPromises } from "@vue/test-utils";
import { reactive } from "vue";
import SourceImageWell from "./SourceImageWell.vue";
import ImagePickerModal from "./ImagePickerModal.vue";
import MaskEditorModal from "./MaskEditorModal.vue";
import ReferenceCropEditor from "@studio/components/ReferenceCropEditor.vue";
import SliderRow from "@ui/components/SliderRow.vue";
import { newGenerateForm, type GenerateForm } from "../../lib/generateForm";

vi.mock("../../lib/api/client", () => ({
  apiJson: vi.fn(() => Promise.resolve([])),
  apiFetch: vi.fn(),
  apiJsonTo: vi.fn(() => Promise.resolve([])),
  apiFetchTo: vi.fn(),
  currentTarget: vi.fn(() => ({ baseUrl: "http://127.0.0.1:49152", apiKey: null })),
}));

const { fetchCatalogInstalled } = vi.hoisted(() => ({
  fetchCatalogInstalled: vi.fn(() =>
    Promise.resolve({ entries: [], page: 1, page_size: 0, total: 0 }),
  ),
}));
vi.mock("../../lib/api/catalog", () => ({ fetchCatalogInstalled }));

function formFor(family: string): GenerateForm {
  return reactive({ ...newGenerateForm(), family, model: "m" });
}

describe("SourceImageWell", () => {
  beforeEach(() => setActivePinia(createPinia()));
  afterEach(() => (document.body.innerHTML = ""));

  it("sets the source image from an ImagePickerModal pick", async () => {
    const form = formFor("sd15");
    const wrapper = mount(SourceImageWell, { props: { form }, attachTo: document.body });

    await wrapper.get("[data-test='source-gallery']").trigger("click");
    wrapper
      .findComponent(ImagePickerModal)
      .vm.$emit("pick", [{ filename: "pick.png", base64: "PICKED" }]);
    await flushPromises();

    expect(form.sourceImage).toBe("PICKED");
  });

  it("previews a loaded source with a working remove and gallery action", async () => {
    const form = formFor("sd15");
    form.sourceImage = "SRC";
    const wrapper = mount(SourceImageWell, { props: { form }, attachTo: document.body });

    const mediaControls = wrapper.get("[data-test='source-media-controls']");
    expect(mediaControls.find("img").exists()).toBe(true);
    expect(mediaControls.find("[data-test='source-gallery']").exists()).toBe(true);
    await mediaControls.get("[data-test='source-remove']").trigger("click");
    expect(form.sourceImage).toBeNull();
  });

  it("gates the mask on a source image and applies the painted one", async () => {
    const form = formFor("sd15");
    const wrapper = mount(SourceImageWell, { props: { form }, attachTo: document.body });

    // No source yet → the well answers no, so the group renders no door and
    // asking for the editor anyway does nothing.
    expect(wrapper.vm.maskAvailable).toBe(false);
    expect(wrapper.find("[data-test='mask-well']").exists()).toBe(false);
    wrapper.vm.openMaskEditor();
    await flushPromises();
    expect(wrapper.findComponent(MaskEditorModal).props("open")).toBe(false);

    form.sourceImage = "SRC";
    await flushPromises();
    expect(wrapper.vm.maskAvailable).toBe(true);
    expect(wrapper.find("[data-test='mask-well']").exists()).toBe(true);
    wrapper.vm.openMaskEditor();
    await flushPromises();
    wrapper.findComponent(MaskEditorModal).vm.$emit("apply", "MASKB64");
    await flushPromises();

    expect(form.maskImage).toBe("MASKB64");
  });

  it("hides the mask controls for families without inpaint support", () => {
    // qwen-image-edit uses qwen-edit source mode → no mask.
    const form = formFor("qwen-image-edit");
    form.sourceImage = "SRC";
    const wrapper = mount(SourceImageWell, { props: { form }, attachTo: document.body });
    expect(wrapper.vm.maskAvailable).toBe(false);
    expect(wrapper.find("[data-test='mask-well']").exists()).toBe(false);
  });

  describe("source-fit selector", () => {
    it("appears only once a source image is attached", async () => {
      const form = formFor("sd15");
      const wrapper = mount(SourceImageWell, { props: { form }, attachTo: document.body });
      expect(wrapper.find("[data-test='source-fit-policy']").exists()).toBe(false);

      form.sourceImage = "SRC";
      await flushPromises();
      const select = wrapper.get("[data-test='source-fit-policy']").element as HTMLSelectElement;
      expect(select.value).toBe("crop-fill");
      expect(Array.from(select.options).map((o) => o.value)).toEqual([
        "pad-repaint",
        "crop-fill",
        "pad-fit",
        "lanczos-resize",
        "upscale-then-fit",
      ]);
    });

    it("stays visible alongside an existing mask (web parity: the mask is refit)", async () => {
      const form = formFor("sd15");
      form.sourceImage = "SRC";
      form.maskImage = "MASK";
      const wrapper = mount(SourceImageWell, { props: { form }, attachTo: document.body });
      await flushPromises();
      expect(wrapper.find("[data-test='source-fit-policy']").exists()).toBe(true);
    });

    it("maps crop-fill to a centered policy", async () => {
      const form = formFor("sd15");
      form.sourceImage = "SRC";
      const wrapper = mount(SourceImageWell, { props: { form }, attachTo: document.body });
      await wrapper.get("[data-test='source-fit-policy']").setValue("crop-fill");
      expect(form.sourceFit).toEqual({ mode: "crop-fill", alignX: "center", alignY: "center" });

      await wrapper.get("[data-test='source-fit-policy']").setValue("lanczos-resize");
      expect(form.sourceFit).toEqual({ mode: "lanczos-resize" });

      await wrapper.get("[data-test='source-fit-policy']").setValue("pad-fit");
      expect(form.sourceFit).toEqual({ mode: "pad-fit" });
    });

    it("upscale-then-fit seeds the upscaler from the form's post-generate pick", async () => {
      const form = formFor("sd15");
      form.sourceImage = "SRC";
      form.upscaleModel = "real-esrgan-x2plus:fp16";
      const wrapper = mount(SourceImageWell, { props: { form }, attachTo: document.body });
      await wrapper.get("[data-test='source-fit-policy']").setValue("upscale-then-fit");
      expect(form.sourceFit).toEqual({
        mode: "upscale-then-fit",
        upscalerModel: "real-esrgan-x2plus:fp16",
        fit: { mode: "crop-fill", alignX: "center", alignY: "center" },
      });
    });

    it("upscale-then-fit falls back to the first known upscaler model", async () => {
      const { useModelStore } = await import("../../stores/models");
      const models = useModelStore();
      models.all = [
        { name: "real-esrgan-x4plus:fp16", family: "upscaler" },
      ] as (typeof models.all)[number][];
      const form = formFor("sd15");
      form.sourceImage = "SRC";
      const wrapper = mount(SourceImageWell, { props: { form }, attachTo: document.body });
      await wrapper.get("[data-test='source-fit-policy']").setValue("upscale-then-fit");
      expect(form.sourceFit).toEqual({
        mode: "upscale-then-fit",
        upscalerModel: "real-esrgan-x4plus:fp16",
        fit: { mode: "crop-fill", alignX: "center", alignY: "center" },
      });
    });
  });

  describe("qwen-edit Target + Reference strip", () => {
    it("renders the shared Target well, fit choices, and ordered picture strip", () => {
      const form = formFor("qwen-image-edit");
      form.imageAttachments = ["T", "R"];
      const wrapper = mount(SourceImageWell, { props: { form }, attachTo: document.body });
      expect(wrapper.find("[data-test='attachment-strip']").exists()).toBe(true);
      expect(wrapper.text()).toContain("Target");
      expect(wrapper.find("[data-test='source-remove']").exists()).toBe(true);
      expect(wrapper.find("[data-test='source-fit-policy']").exists()).toBe(true);
    });

    it("keeps the single well for every other family (no strip)", () => {
      const form = formFor("sd15");
      form.sourceImage = "SRC";
      const wrapper = mount(SourceImageWell, { props: { form }, attachTo: document.body });
      expect(wrapper.find("[data-test='attachment-strip']").exists()).toBe(false);
    });

    it("renders the single well for ltx2 (image-to-video) without mask or pad-repaint", () => {
      const form = formFor("ltx2");
      form.sourceImage = "SRC";
      const wrapper = mount(SourceImageWell, { props: { form }, attachTo: document.body });
      const fitOptions = wrapper
        .get("[data-test='source-fit-policy']")
        .findAll("option")
        .map((o) => o.attributes("value"));
      // pad-repaint needs a repaint mask the family can't ship — not offered.
      expect(fitOptions).not.toContain("pad-repaint");
      expect(fitOptions).toContain("crop-fill");
      expect(wrapper.vm.maskAvailable).toBe(false);
    });

    it("never renders the well for plain ltx-video (engine has no img2vid path)", () => {
      const form = formFor("ltx-video");
      form.sourceImage = "SRC";
      const wrapper = mount(SourceImageWell, { props: { form }, attachTo: document.body });
      expect(wrapper.find("[data-test='source-fit-policy']").exists()).toBe(false);
      expect(wrapper.find("[data-test='source-choose-gallery']").exists()).toBe(false);
    });

    it("labels the first tile Target and the rest Reference, titled Picture N", () => {
      const form = formFor("qwen-image-edit");
      form.imageAttachments = ["T", "R1", "R2"];
      const wrapper = mount(SourceImageWell, { props: { form }, attachTo: document.body });
      expect(wrapper.get("[data-test='attachment-role-0']").text()).toBe("Target");
      expect(wrapper.get("[data-test='attachment-role-1']").text()).toBe("Reference");
      expect(wrapper.get("[data-test='attachment-role-2']").text()).toBe("Reference");
      expect(wrapper.get("[data-test='attachment-title-0']").text()).toBe("Picture 1");
      expect(wrapper.get("[data-test='attachment-title-2']").text()).toBe("Picture 3");
    });

    it("appends picks from the multi-select picker in order", async () => {
      const form = formFor("qwen-image-edit");
      form.sourceFit = { mode: "pad-fit" };
      const wrapper = mount(SourceImageWell, { props: { form }, attachTo: document.body });
      await wrapper.get("[data-test='add-edit-image']").trigger("click");
      const picker = wrapper
        .findAllComponents(ImagePickerModal)
        .find((candidate) => candidate.props("multiple") === true)!;
      expect(picker.props("multiple")).toBe(true);
      picker.vm.$emit("pick", [
        { filename: "a.png", base64: "A" },
        { filename: "b.png", base64: "B" },
      ]);
      await flushPromises();
      expect(form.imageAttachments).toEqual(["A", "B"]);
      expect(form.sourceFit).toEqual({ mode: "crop-fill" });
    });

    it("removes a tile", async () => {
      const form = formFor("qwen-image-edit");
      form.imageAttachments = ["T", "R1", "R2"];
      const wrapper = mount(SourceImageWell, { props: { form }, attachTo: document.body });
      await wrapper.get("[data-test='remove-attachment-1']").trigger("click");
      expect(form.imageAttachments).toEqual(["T", "R2"]);
    });

    it("reorders with the move buttons — a Reference promoted to slot 0 becomes the Target", async () => {
      const form = formFor("qwen-image-edit");
      form.imageAttachments = ["T", "R1", "R2"];
      const wrapper = mount(SourceImageWell, { props: { form }, attachTo: document.body });
      await wrapper.get("[data-test='move-attachment-up-1']").trigger("click");
      expect(form.imageAttachments).toEqual(["R1", "T", "R2"]);
      await wrapper.get("[data-test='move-attachment-down-1']").trigger("click");
      expect(form.imageAttachments).toEqual(["R1", "R2", "T"]);
    });
  });

  describe("controlnet picker", () => {
    async function mountSd15WithControl(catalogEntries: unknown[] = []) {
      fetchCatalogInstalled.mockResolvedValue({
        entries: catalogEntries,
        page: 1,
        page_size: catalogEntries.length,
        total: catalogEntries.length,
      } as never);
      const { useModelStore } = await import("../../stores/models");
      const models = useModelStore();
      models.all = [
        { name: "controlnet-canny-sd15", family: "controlnet", downloaded: true },
        { name: "controlnet-openpose-sd15", family: "controlnet", downloaded: false },
      ] as (typeof models.all)[number][];
      const form = formFor("sd15");
      form.controlImage = "CTRL";
      const wrapper = mount(SourceImageWell, { props: { form }, attachTo: document.body });
      await flushPromises();
      return { wrapper, form };
    }

    it("renders a select of installed + catalog options, disabling missing downloads", async () => {
      const { wrapper } = await mountSd15WithControl([
        {
          id: "cv:1",
          name: "cv-control",
          installed: true,
          primary_path: "/m/cv-control.safetensors",
        },
      ]);
      const select = wrapper.get("[data-test='controlnet-select']").element as HTMLSelectElement;
      const options = Array.from(select.options).map((o) => ({
        value: o.value,
        disabled: o.disabled,
      }));
      expect(options).toContainEqual({ value: "controlnet-canny-sd15", disabled: false });
      expect(options).toContainEqual({ value: "controlnet-openpose-sd15", disabled: true });
      expect(options).toContainEqual({ value: "/m/cv-control.safetensors", disabled: false });
      // No raw free-text input until Custom… is chosen.
      expect(wrapper.find("[data-test='controlnet-custom-input']").exists()).toBe(false);
    });

    it("selecting an option sets form.controlModel", async () => {
      const { wrapper, form } = await mountSd15WithControl();
      await wrapper.get("[data-test='controlnet-select']").setValue("controlnet-canny-sd15");
      expect(form.controlModel).toBe("controlnet-canny-sd15");
    });

    it("Custom… reveals the free-text escape hatch and keeps typed values", async () => {
      const { wrapper, form } = await mountSd15WithControl();
      await wrapper.get("[data-test='controlnet-select']").setValue("__custom__");
      const input = wrapper.get("[data-test='controlnet-custom-input']");
      await input.setValue("my-handmade-controlnet");
      expect(form.controlModel).toBe("my-handmade-controlnet");
    });

    it("scopes to a pinned host that is fetched but has no models (empty ≠ unknown)", async () => {
      // A sticky non-primary host whose /api/models came back empty must NOT
      // fall back to the primary's inventory — its inventory is *known* (just
      // empty), so the picker offers no installed controlnet models.
      const { useModelStore } = await import("../../stores/models");
      const models = useModelStore();
      models.all = [
        { name: "controlnet-canny-sd15", family: "controlnet", downloaded: true },
      ] as (typeof models.all)[number][];
      const { useHostsStore } = await import("../../stores/hosts");
      useHostsStore().extras = [
        {
          id: "h1",
          label: "h1",
          url: "http://h1",
          apiKey: null,
          status: "ready",
          error: null,
          instanceId: null,
        },
      ];
      const { useAppPrefsStore } = await import("../../stores/appPrefs");
      useAppPrefsStore().settings = { generateTargetHost: "h1" } as never;
      const { useHostModelsStore } = await import("../../stores/hostModels");
      useHostModelsStore().byHost = {
        h1: { entries: [], fetchedAt: Date.now(), error: null },
      };
      fetchCatalogInstalled.mockResolvedValue({
        entries: [],
        page: 1,
        page_size: 0,
        total: 0,
      } as never);

      const form = formFor("sd15");
      form.controlImage = "CTRL";
      const wrapper = mount(SourceImageWell, { props: { form }, attachTo: document.body });
      await flushPromises();
      const select = wrapper.get("[data-test='controlnet-select']").element as HTMLSelectElement;
      const values = Array.from(select.options).map((o) => o.value);
      expect(values).not.toContain("controlnet-canny-sd15");
    });

    it("gives the Custom… free-text input an accessible name", async () => {
      const { wrapper } = await mountSd15WithControl();
      await wrapper.get("[data-test='controlnet-select']").setValue("__custom__");
      const input = wrapper.get("[data-test='controlnet-custom-input']");
      expect(input.attributes("aria-label")?.trim()).toBeTruthy();
    });

    it("shows an unknown restored value as a selectable option (metadata reuse)", async () => {
      const { useModelStore } = await import("../../stores/models");
      useModelStore().all = [];
      fetchCatalogInstalled.mockResolvedValue({
        entries: [],
        page: 1,
        page_size: 0,
        total: 0,
      } as never);
      const form = formFor("sd15");
      form.controlImage = "CTRL";
      form.controlModel = "restored-from-metadata";
      const wrapper = mount(SourceImageWell, { props: { form }, attachTo: document.body });
      await flushPromises();
      const select = wrapper.get("[data-test='controlnet-select']").element as HTMLSelectElement;
      expect(Array.from(select.options).map((o) => o.value)).toContain("restored-from-metadata");
      expect(select.value).toBe("restored-from-metadata");
    });
  });
});

describe("SourceImageWell — per-model source conditioning (#772, #779)", () => {
  beforeEach(() => setActivePinia(createPinia()));
  afterEach(() => (document.body.innerHTML = ""));

  /** A wan form with the frame budget a first/last-frame render needs. */
  function wanForm(sourceImage: string | null | undefined): GenerateForm {
    return reactive({
      ...newGenerateForm(),
      family: "wan",
      model: "wan22-t2v-a14b",
      frames: 81,
      sourceImageCapability: sourceImage,
    });
  }

  it("exposes Ref2VA ordered references through the multi-image Library picker", async () => {
    const form = reactive({
      ...newGenerateForm(),
      family: "minimax-h3",
      model: "minimax-h3-ref2va:comfy-pruned-int8",
    });
    const wrapper = mount(SourceImageWell, { props: { form }, attachTo: document.body });

    expect(wrapper.find("[data-test='h3-reference-controls']").exists()).toBe(true);
    expect(wrapper.find("[data-test='h3-reference-files']").exists()).toBe(true);
    await wrapper.get("[data-test='h3-reference-library']").trigger("click");
    const picker = wrapper.getComponent(ImagePickerModal);
    expect(picker.props("open")).toBe(true);
    expect(picker.props("multiple")).toBe(true);
  });

  it("hosts the shared crop editor for a Ref2VA image and records the applied crop", async () => {
    const form = reactive({
      ...newGenerateForm(),
      family: "minimax-h3",
      model: "minimax-h3-ref2va:comfy-pruned-int8",
    });
    form.h3Authoring = {
      firstFrame: null,
      lastFrame: null,
      references: [
        {
          reference: {
            kind: "image",
            media: { authority: "inline", data: "SU1BR0U=" },
            provenance: { name: "subject.png", sha256: "a".repeat(64) },
            mime_type: "image/png",
            width: 1024,
            height: 768,
          },
        },
      ],
    };
    const wrapper = mount(SourceImageWell, { props: { form }, attachTo: document.body });
    expect(wrapper.findComponent(ReferenceCropEditor).exists()).toBe(false);

    await wrapper.get("[data-test='h3-reference-crop-0']").trigger("click");
    const editor = wrapper.findComponent(ReferenceCropEditor);
    expect(editor.exists()).toBe(true);
    expect(editor.props("image")).toMatchObject({ width: 1024, height: 768 });

    editor.vm.$emit("apply", { x: 256, y: 0, width: 512, height: 768 });
    await flushPromises();
    expect(form.h3Authoring.references[0]?.crop).toEqual({
      x: 256,
      y: 0,
      width: 512,
      height: 768,
    });
    expect(wrapper.findComponent(ReferenceCropEditor).exists()).toBe(false);
    wrapper.unmount();
  });

  it("keeps today's well and offers no end frame when the server advertises nothing", () => {
    const form = wanForm(undefined);
    const wrapper = mount(SourceImageWell, { props: { form }, attachTo: document.body });

    expect(wrapper.find("[data-test='source-media-controls']").exists()).toBe(true);
    expect(wrapper.find("[data-test='end-frame-media-controls']").exists()).toBe(false);
    expect(wrapper.find("[data-test='source-required-badge']").exists()).toBe(false);
    expect(wrapper.find("[data-test='source-conditioning-error']").exists()).toBe(false);
  });

  it("renders no source well at all for an advertised text-to-video checkpoint", () => {
    const form = wanForm("unsupported");
    const wrapper = mount(SourceImageWell, { props: { form }, attachTo: document.body });

    expect(wrapper.find("[data-test='source-media-controls']").exists()).toBe(false);
    expect(wrapper.find("[data-test='source-choose-gallery']").exists()).toBe(false);
    expect(wrapper.find("[data-test='end-frame-media-controls']").exists()).toBe(false);
  });

  it("marks the well required and names why until an image is attached", async () => {
    const form = wanForm("required");
    const wrapper = mount(SourceImageWell, { props: { form }, attachTo: document.body });

    expect(wrapper.find("[data-test='source-required-badge']").exists()).toBe(true);
    expect(wrapper.get("[data-test='source-conditioning-error']").text()).toContain(
      "Attach a source image",
    );

    form.sourceImage = "SRC";
    await flushPromises();
    expect(wrapper.find("[data-test='source-conditioning-error']").exists()).toBe(false);
  });

  it("offers the end-frame well on an advertised optional wan checkpoint", async () => {
    const form = wanForm("optional");
    const wrapper = mount(SourceImageWell, { props: { form }, attachTo: document.body });

    expect(wrapper.find("[data-test='end-frame-well']").exists()).toBe(true);

    // Same picker affordance as the source well; the end frame keeps its name.
    await wrapper.get("[data-test='end-frame-gallery']").trigger("click");
    wrapper
      .findAllComponents(ImagePickerModal)[1]!
      .vm.$emit("pick", [{ filename: "close.png", base64: "ENDB64" }]);
    await flushPromises();
    expect(form.endFrame).toEqual({ filename: "close.png", base64: "ENDB64" });

    // An end frame with no first frame is refused with the shared message.
    expect(wrapper.get("[data-test='source-conditioning-error']").text()).toContain(
      "An end frame needs a first frame",
    );

    form.sourceImage = "SRC";
    await flushPromises();
    expect(wrapper.find("[data-test='source-conditioning-error']").exists()).toBe(false);
  });

  it("clears the staged end frame from its own well", async () => {
    const form = wanForm("optional");
    form.sourceImage = "SRC";
    form.endFrame = { filename: "close.png", base64: "ENDB64" };
    const wrapper = mount(SourceImageWell, { props: { form }, attachTo: document.body });

    await wrapper.get("[data-test='end-frame-remove']").trigger("click");
    expect(form.endFrame).toBeNull();
  });
});

describe("SourceImageWell strength direction", () => {
  beforeEach(() => setActivePinia(createPinia()));
  afterEach(() => (document.body.innerHTML = ""));

  function strengthEnds(family: string) {
    const form = formFor(family);
    form.sourceImage = "SRC";
    const wrapper = mount(SourceImageWell, { props: { form }, attachTo: document.body });
    const row = wrapper
      .findAllComponents(SliderRow)
      .find((slider) => slider.props("label") === "How much to change it");
    return { row, wrapper };
  }

  it("names the ends of the track the way the family actually reads it", () => {
    const sd = strengthEnds("sd15");
    expect(sd.row?.props("low")).toBe("Keep the photo");
    expect(sd.row?.props("high")).toBe("Start fresh");
    sd.wrapper.unmount();

    // LTX-2 inverts it: 1.0 pins the opening frame, so the HIGH end is the
    // one that keeps the photo. Fixed end labels lied on this family.
    const ltx = strengthEnds("ltx2");
    expect(ltx.row?.props("low")).toBe("Start fresh");
    expect(ltx.row?.props("high")).toBe("Keep the photo");
    ltx.wrapper.unmount();
  });
});
