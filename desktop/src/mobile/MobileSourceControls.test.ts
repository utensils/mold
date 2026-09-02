import { flushPromises, mount } from "@vue/test-utils";
import { reactive } from "vue";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { hunyuan3dRecipe } from "@studio/lib/generationProfile.testFixtures";
import type { ModelEntry } from "../lib/api/types";
import { MAX_MOBILE_GENERATION_REQUEST_MEDIA_BYTES } from "../lib/generateValidation";
import { newGenerateForm, type GenerateForm } from "../lib/generateForm";
import MaskEditorModal from "../components/generate/MaskEditorModal.vue";
import MobileImagePickerSheet from "./MobileImagePickerSheet.vue";
import MobileReferenceCropSheet from "./MobileReferenceCropSheet.vue";
import ReferenceCropEditor from "@studio/components/ReferenceCropEditor.vue";

const { fetchCatalogInstalled } = vi.hoisted(() => ({
  fetchCatalogInstalled: vi.fn(),
}));
vi.mock("../lib/api/catalog", () => ({ fetchCatalogInstalled }));

import MobileSourceControls from "./MobileSourceControls.vue";

function formFor(family: string): GenerateForm {
  return reactive({ ...newGenerateForm(), family, model: `${family}:test` });
}

function model(name: string, family: string, downloaded = true): ModelEntry {
  return {
    name,
    family,
    downloaded,
    size_gb: 1,
    is_loaded: false,
    hf_repo: "example/model",
    default_steps: 20,
    default_guidance: 3.5,
    default_width: 1024,
    default_height: 1024,
    description: "",
  };
}

async function chooseFiles(
  wrapper: ReturnType<typeof mount>,
  selector: string,
  files: File[],
): Promise<void> {
  const input = wrapper.get<HTMLInputElement>(selector);
  Object.defineProperty(input.element, "files", { configurable: true, value: files });
  await input.trigger("change");
  await flushPromises();
  // happy-dom dispatches FileReader completion as a task rather than a
  // promise continuation; allow that task to run before asserting form state.
  await new Promise((resolve) => setTimeout(resolve, 0));
  await flushPromises();
}

describe("MobileSourceControls", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    fetchCatalogInstalled.mockResolvedValue({
      entries: [],
      page: 1,
      page_size: 0,
      total: 0,
    });
  });

  afterEach(() => {
    document.body.innerHTML = "";
  });

  it("does not render for a family without image conditioning", () => {
    const wrapper = mount(MobileSourceControls, { props: { form: formFor("ltx-video") } });
    expect(wrapper.find("[data-test='mobile-source-controls']").exists()).toBe(false);
  });

  it("keeps today's wan source well and offers no end frame when the host advertises nothing", () => {
    const form = formFor("wan");
    expect(form.sourceImageCapability).toBeNull();
    const wrapper = mount(MobileSourceControls, { props: { form } });

    expect(wrapper.find("[data-test='mobile-source-controls']").exists()).toBe(true);
    expect(wrapper.find("[data-test='mobile-source-well']").exists()).toBe(true);
    expect(wrapper.find("[data-test='source-required-badge']").exists()).toBe(false);
    // An older server rejects wan keyframes outright, so an absent contract
    // must never surface the End frame well.
    expect(wrapper.find("[data-test='mobile-end-frame-controls']").exists()).toBe(false);
    expect(wrapper.find("[data-test='mobile-source-conditioning-error']").exists()).toBe(false);
  });

  it("hides the whole source well for an advertised text-to-video checkpoint", () => {
    const form = formFor("wan");
    form.sourceImageCapability = "unsupported";
    const wrapper = mount(MobileSourceControls, { props: { form } });

    expect(wrapper.find("[data-test='mobile-source-controls']").exists()).toBe(false);
    expect(wrapper.find("[data-test='mobile-end-frame-controls']").exists()).toBe(false);
  });

  it("marks the well required and gates until an image-to-video source is attached", async () => {
    const form = formFor("wan");
    form.sourceImageCapability = "required";
    const wrapper = mount(MobileSourceControls, {
      props: { form },
      global: { stubs: { MobileImagePickerSheet: true } },
    });

    expect(wrapper.get("[data-test='source-required-badge']").text()).toContain("Required");
    expect(wrapper.get("[data-test='mobile-source-well']").attributes("aria-required")).toBe(
      "true",
    );
    expect(wrapper.get("[data-test='source-conditioning-error']").text()).toContain(
      "image-to-video only",
    );
    expect(wrapper.emitted("validity-change")?.at(-1)).toEqual([false]);

    form.sourceImage = "T1BFTklORw==";
    form.sourceImageName = "opening.png";
    await flushPromises();

    expect(wrapper.find("[data-test='mobile-source-conditioning-error']").exists()).toBe(false);
    expect(wrapper.emitted("validity-change")?.at(-1)).toEqual([true]);
  });

  // #783: a continuation supplies its own first frames from the tail of the
  // clip it continues, exactly as admission reads it through
  // `mold_core::validation::request_carries_source_frames`. Without that, the
  // Continue-a-video control iPhone now offers for a Wan I2V checkpoint kept
  // Develop disabled with "attach a source image".
  it("accepts a Wan I2V continuation as its own source frames", async () => {
    const form = formFor("wan");
    form.sourceImageCapability = "required";
    const wrapper = mount(MobileSourceControls, {
      props: { form },
      global: { stubs: { MobileImagePickerSheet: true } },
    });

    expect(wrapper.emitted("validity-change")?.at(-1)).toEqual([false]);

    form.extendVideo = { filename: "clip.mp4", base64: "Q0xJUA==" };
    await flushPromises();

    expect(wrapper.find("[data-test='mobile-source-conditioning-error']").exists()).toBe(false);
    expect(wrapper.emitted("validity-change")?.at(-1)).toEqual([true]);

    // Removing the clip restores the requirement — this is not a permanent
    // opt-out of the contract.
    form.extendVideo = null;
    await flushPromises();
    expect(wrapper.get("[data-test='source-conditioning-error']").text()).toContain(
      "image-to-video only",
    );
    expect(wrapper.emitted("validity-change")?.at(-1)).toEqual([false]);
  });

  it("offers an optional end frame on wan and refuses an end-frame-only draft", async () => {
    const form = formFor("wan");
    form.sourceImageCapability = "optional";
    const target = { baseUrl: "http://halcyon:7680", apiKey: "remote-key" };
    const wrapper = mount(MobileSourceControls, { props: { form, target } });

    expect(wrapper.find("[data-test='source-required-badge']").exists()).toBe(false);
    const endWell = wrapper.get("[data-test='source-media-wells']");
    expect(endWell.text()).toContain("Optional");

    await wrapper.get("[data-test='mobile-end-frame-gallery']").trigger("click");
    const endPicker = wrapper
      .findAllComponents(MobileImagePickerSheet)
      .find((sheet) => sheet.props("title") === "End frame");
    expect(endPicker).toBeDefined();
    expect(endPicker!.props()).toMatchObject({
      open: true,
      target,
      maxBytes: MAX_MOBILE_GENERATION_REQUEST_MEDIA_BYTES,
    });

    endPicker!.vm.$emit("pick", { filename: "closing.png", base64: "Q0xPU0lORw==" });
    await flushPromises();
    expect(form.endFrame).toEqual({ filename: "closing.png", base64: "Q0xPU0lORw==" });
    expect(endPicker!.props("open")).toBe(false);
    expect(wrapper.get("[data-test='mobile-end-frame-preview']").attributes("src")).toContain(
      "base64,Q0xPU0lORw==",
    );

    // A closing still with nothing to open the clip is refused, not silently
    // dropped or shipped as a lone keyframe.
    expect(wrapper.get("[data-test='source-conditioning-error']").text()).toContain(
      "needs a first frame",
    );
    expect(wrapper.emitted("validity-change")?.at(-1)).toEqual([false]);

    form.sourceImage = "T1BFTklORw==";
    await flushPromises();
    expect(wrapper.find("[data-test='source-conditioning-error']").exists()).toBe(false);
    expect(wrapper.emitted("validity-change")?.at(-1)).toEqual([true]);

    await wrapper.get("[data-test='mobile-end-frame-remove']").trigger("click");
    expect(form.endFrame).toBeNull();
  });

  it("exposes preview, strength, fit, replace, and remove for a source image", async () => {
    const form = formFor("sdxl");
    form.sourceImage = "cGhvdG8=";
    form.sourceImageName = "photo.jpg";
    const wrapper = mount(MobileSourceControls, { props: { form } });

    expect(form.sourceImage).toBe("cGhvdG8=");
    expect(form.sourceImageName).toBe("photo.jpg");
    expect(wrapper.get("[data-test='mobile-source-preview']").attributes("src")).toContain(
      "base64,cGhvdG8=",
    );
    expect(wrapper.get("[data-test='mobile-source-replace']").text()).toContain("Replace");
    expect(wrapper.get("[data-test='mobile-source-remove']").text()).toContain("Remove");
    expect(wrapper.find("[data-test='mobile-source-strength']").exists()).toBe(true);
    expect(wrapper.find("[data-test='mobile-source-fit']").exists()).toBe(true);

    await wrapper.get("[data-test='mobile-source-strength']").setValue("0.55");
    expect(form.strength).toBe(0.55);
    await wrapper.get("[data-test='mobile-source-fit']").setValue("crop-fill");
    expect(form.sourceFit).toEqual({ mode: "crop-fill", alignX: "center", alignY: "center" });

    await wrapper.get("[data-test='mobile-source-remove']").trigger("click");
    expect(form.sourceImage).toBeNull();
    expect(form.sourceImageName).toBeNull();
  });

  it("keeps the generation target separate from every available source gallery", async () => {
    const form = formFor("sd15");
    form.controlImage = btoa("control");
    const target = { baseUrl: "http://halcyon:7680", apiKey: "remote-key" };
    const gallerySources = [
      { id: "halcyon", label: "Halcyon", target },
      {
        id: "peer",
        label: "Peer",
        target: { baseUrl: "http://peer:7680", apiKey: "peer-key" },
      },
    ];
    const wrapper = mount(MobileSourceControls, {
      props: { form, target, gallerySources },
      global: { stubs: { MobileImagePickerSheet: true } },
    });

    await chooseFiles(wrapper, "[data-test='mobile-control-input']", [
      new File(["not-an-image"], "control.txt", { type: "text/plain" }),
    ]);
    expect(wrapper.get("[data-test='mobile-source-error']").text()).toContain("Only PNG or JPEG");

    await wrapper.get("[data-test='mobile-source-gallery']").trigger("click");
    const picker = wrapper.getComponent(MobileImagePickerSheet);
    expect(picker.props()).toMatchObject({
      open: true,
      target,
      gallerySources,
      title: "Source image",
      maxBytes: MAX_MOBILE_GENERATION_REQUEST_MEDIA_BYTES - 7,
    });

    picker.vm.$emit("pick", { filename: "gallery-print.png", base64: "R0FMTEVSWQ==" });
    await flushPromises();

    expect(form.sourceImage).toBe("R0FMTEVSWQ==");
    expect(form.sourceImageName).toBe("gallery-print.png");
    expect(form.sourceFit).toEqual({ mode: "crop-fill" });
    expect(picker.props("open")).toBe(false);
    expect(wrapper.find("[data-test='mobile-source-error']").exists()).toBe(false);
  });

  it("uses the first upscaler for upscale-then-fit and omits repaint for maskless video", async () => {
    const form = formFor("ltx2");
    form.sourceImage = "SRC";
    const wrapper = mount(MobileSourceControls, {
      props: { form, upscalers: [model("real-esrgan-x4plus", "upscaler")] },
    });

    const values = wrapper
      .get("[data-test='mobile-source-fit']")
      .findAll("option")
      .map((option) => option.attributes("value"));
    expect(values).not.toContain("pad-repaint");

    await wrapper.get("[data-test='mobile-source-fit']").setValue("upscale-then-fit");
    expect(form.sourceFit).toEqual({
      mode: "upscale-then-fit",
      upscalerModel: "real-esrgan-x4plus",
      fit: { mode: "crop-fill", alignX: "center", alignY: "center" },
    });
  });

  it("uploads a mask and applies a painted mask through the shared full-screen editor", async () => {
    const form = formFor("sd15");
    form.sourceImage = "SRC";
    form.sourceImageName = "source.png";
    const wrapper = mount(MobileSourceControls, {
      props: { form },
      attachTo: document.body,
    });

    await chooseFiles(wrapper, "[data-test='mobile-mask-input']", [
      new File(["mask"], "mask.png", { type: "image/png" }),
    ]);
    await vi.waitFor(() => expect(form.maskImage).not.toBeNull());
    expect(form.maskImage).toBe("bWFzaw==");

    await wrapper.get("[data-test='mobile-mask-edit']").trigger("click");
    const editor = wrapper.findComponent(MaskEditorModal);
    expect(editor.props("open")).toBe(true);
    expect(editor.props("filename")).toBe("source.png");
    editor.vm.$emit("apply", "PAINTED");
    await flushPromises();
    expect(form.maskImage).toBe("PAINTED");
    expect(wrapper.findComponent(MaskEditorModal).props("open")).toBe(false);
  });

  it("renders ordered, touch-operable target/reference tiles for qwen edit", async () => {
    const form = formFor("qwen-image-edit");
    form.imageAttachments = ["TARGET", "REFERENCE-1", "REFERENCE-2"];
    const wrapper = mount(MobileSourceControls, { props: { form } });

    expect(wrapper.find("[data-test='mobile-source-input']").exists()).toBe(false);
    expect(wrapper.text()).toContain("Target");
    expect(wrapper.find("[data-test='source-remove']").exists()).toBe(true);
    expect(wrapper.find("[data-test='mobile-source-fit']").exists()).toBe(true);
    expect(wrapper.get("[data-test='mobile-source-fit-help']").text()).toContain(
      "conditioning limit: 1 MP from this model",
    );
    expect(wrapper.get("[data-test='mobile-edit-role-0']").text()).toBe("Target");
    expect(wrapper.get("[data-test='mobile-edit-role-1']").text()).toBe("Reference");
    expect(wrapper.get("[data-test='mobile-edit-title-2']").text()).toBe("Picture 3");

    const moveEarlier = wrapper.get("[data-test='mobile-edit-earlier-1']");
    const remove = wrapper.get("[data-test='mobile-edit-remove-1']");
    expect(moveEarlier.classes()).toContain("mobile-media-tile-action");
    expect(remove.classes()).toContain("mobile-media-tile-action");

    await moveEarlier.trigger("click");
    expect(form.imageAttachments).toEqual(["REFERENCE-1", "TARGET", "REFERENCE-2"]);
    await wrapper.get("[data-test='mobile-edit-later-1']").trigger("click");
    expect(form.imageAttachments).toEqual(["REFERENCE-1", "REFERENCE-2", "TARGET"]);
    await wrapper.get("[data-test='mobile-edit-remove-1']").trigger("click");
    expect(form.imageAttachments).toEqual(["REFERENCE-1", "TARGET"]);
  });

  it("requires a Target photo for qwen edit", async () => {
    const form = formFor("qwen-image-edit");
    const wrapper = mount(MobileSourceControls, { props: { form } });

    expect(wrapper.get("[data-test='mobile-source-validation']").text()).toContain("Target photo");
    expect(wrapper.emitted("validity-change")?.at(-1)).toEqual([false]);

    form.imageAttachments = ["TARGET"];
    await flushPromises();
    expect(wrapper.find("[data-test='mobile-source-validation']").exists()).toBe(false);
    expect(wrapper.emitted("validity-change")?.at(-1)).toEqual([true]);
  });

  it("appends multiple PNG/JPEG edit pictures in file order", async () => {
    const form = formFor("qwen-image-edit");
    form.imageAttachments = ["EXISTING"];
    const wrapper = mount(MobileSourceControls, { props: { form } });

    const input = wrapper.get("[data-test='mobile-edit-input']");
    expect(input.attributes("multiple")).toBeDefined();
    await chooseFiles(wrapper, "[data-test='mobile-edit-input']", [
      new File(["a"], "a.png", { type: "image/png" }),
      new File(["b"], "b.jpg", { type: "image/jpeg" }),
    ]);
    await vi.waitFor(() => expect(form.imageAttachments).toHaveLength(3));
    expect(form.imageAttachments).toEqual(["EXISTING", "YQ==", "Yg=="]);
  });

  it("defaults the first Qwen edit target to crop fill", async () => {
    const form = formFor("qwen-image-edit");
    form.sourceFit = { mode: "lanczos-resize" };
    const wrapper = mount(MobileSourceControls, { props: { form } });

    await chooseFiles(wrapper, "[data-test='mobile-edit-input']", [
      new File(["target"], "target.png", { type: "image/png" }),
    ]);

    await vi.waitFor(() => expect(form.imageAttachments).toEqual(["dGFyZ2V0"]));
    expect(form.sourceFit).toEqual({ mode: "crop-fill" });
  });

  it("supports an SD1.5 ControlNet photo, known model, custom id, and scale", async () => {
    const form = formFor("sd15");
    form.sourceImage = "SRC";
    const wrapper = mount(MobileSourceControls, {
      props: {
        form,
        controlModels: [
          model("controlnet-canny-sd15", "controlnet"),
          model("controlnet-openpose-sd15", "controlnet", false),
        ],
      },
    });

    await chooseFiles(wrapper, "[data-test='mobile-control-input']", [
      new File(["control"], "control.png", { type: "image/png" }),
    ]);
    await vi.waitFor(() => expect(form.controlImage).not.toBeNull());
    expect(form.controlImage).toBe("Y29udHJvbA==");
    expect(form.controlModel).toBe("controlnet-canny-sd15");

    const select = wrapper.get("[data-test='mobile-control-model']");
    const options = select.findAll("option").map((option) => ({
      value: option.attributes("value"),
      disabled: option.attributes("disabled") !== undefined,
    }));
    expect(options).toContainEqual({ value: "controlnet-canny-sd15", disabled: false });
    expect(options).toContainEqual({ value: "controlnet-openpose-sd15", disabled: true });

    await select.setValue("controlnet-canny-sd15");
    expect(form.controlModel).toBe("controlnet-canny-sd15");
    await select.setValue("__custom__");
    await wrapper.get("[data-test='mobile-control-custom']").setValue("my-controlnet");
    expect(form.controlModel).toBe("my-controlnet");
    await wrapper.get("[data-test='mobile-control-scale']").setValue("1.25");
    expect(form.controlScale).toBe(1.25);
  });

  it("offers ControlNet independently when no source photo is selected", async () => {
    const form = formFor("sd15");
    expect(form.sourceImage).toBeNull();
    const wrapper = mount(MobileSourceControls, {
      props: {
        form,
        controlModels: [model("controlnet-canny-sd15", "controlnet")],
      },
    });

    expect(wrapper.find("[data-test='mobile-source-preview']").exists()).toBe(false);
    expect(wrapper.find("[data-test='mobile-control-add']").exists()).toBe(true);
    await chooseFiles(wrapper, "[data-test='mobile-control-input']", [
      new File(["control"], "control.png", { type: "image/png" }),
    ]);

    await vi.waitFor(() => expect(form.controlImage).toBe("Y29udHJvbA=="));
    expect(form.sourceImage).toBeNull();
    expect(wrapper.find("[data-test='mobile-control-model']").exists()).toBe(true);
  });

  it("blocks a ControlNet photo when no installed control model is available", async () => {
    const form = formFor("sd15");
    const wrapper = mount(MobileSourceControls, { props: { form } });

    await chooseFiles(wrapper, "[data-test='mobile-control-input']", [
      new File(["control"], "control.png", { type: "image/png" }),
    ]);
    await vi.waitFor(() => expect(form.controlImage).toBe("Y29udHJvbA=="));

    expect(form.controlModel).toBe("");
    expect(wrapper.get("[data-test='mobile-source-validation']").text()).toContain(
      "installed ControlNet model",
    );
    expect(wrapper.emitted("validity-change")?.at(-1)).toEqual([false]);
  });

  it("merges catalog-installed ControlNet paths from the selected remote host", async () => {
    const form = formFor("sd15");
    form.sourceImage = "SRC";
    form.controlImage = "CTRL";
    const target = { baseUrl: "http://halcyon:7680", apiKey: "remote-key" };
    fetchCatalogInstalled.mockResolvedValue({
      entries: [
        {
          id: "cv:1",
          name: "catalog-canny",
          installed: true,
          primary_path: "/models/catalog-canny.safetensors",
        },
        {
          id: "cv:2",
          name: "incomplete-control",
          installed: false,
          primary_path: null,
        },
      ],
      page: 1,
      page_size: 2,
      total: 2,
    });

    const wrapper = mount(MobileSourceControls, {
      props: {
        form,
        target,
        controlModels: [
          model("controlnet-canny-sd15", "controlnet"),
          model("controlnet-openpose-sd15", "controlnet", false),
        ],
      },
    });
    await flushPromises();

    expect(fetchCatalogInstalled).toHaveBeenCalledWith(
      { family: "sd15", kind: "control-net" },
      target,
    );
    const options = wrapper
      .get("[data-test='mobile-control-model']")
      .findAll("option")
      .map((option) => ({
        value: option.attributes("value"),
        disabled: option.attributes("disabled") !== undefined,
      }));
    expect(options).toContainEqual({ value: "controlnet-canny-sd15", disabled: false });
    expect(options).toContainEqual({ value: "controlnet-openpose-sd15", disabled: true });
    expect(options).toContainEqual({
      value: "/models/catalog-canny.safetensors",
      disabled: false,
    });
    expect(options.map((option) => option.value)).not.toContain("incomplete-control");
  });

  it("ignores stale catalog results after rapid host and model changes", async () => {
    type CatalogResponse = Awaited<ReturnType<typeof fetchCatalogInstalled>>;
    function deferred() {
      let resolve!: (value: CatalogResponse) => void;
      const promise = new Promise<CatalogResponse>((done) => {
        resolve = done;
      });
      return { promise, resolve };
    }
    function response(name: string, path: string): CatalogResponse {
      return {
        entries: [{ id: `cv:${name}`, name, installed: true, primary_path: path }],
        page: 1,
        page_size: 1,
        total: 1,
      } as CatalogResponse;
    }

    const first = deferred();
    const second = deferred();
    const latest = deferred();
    fetchCatalogInstalled
      .mockImplementationOnce(() => first.promise)
      .mockImplementationOnce(() => second.promise)
      .mockImplementationOnce(() => latest.promise);

    const form = formFor("sd15");
    form.sourceImage = "SRC";
    form.controlImage = "CTRL";
    const wrapper = mount(MobileSourceControls, {
      props: {
        form,
        target: { baseUrl: "http://host-a:7680", apiKey: null },
      },
    });
    await flushPromises();

    await wrapper.setProps({ target: { baseUrl: "http://host-b:7680", apiKey: "b-key" } });
    form.model = "sd15:replacement";
    await flushPromises();
    expect(fetchCatalogInstalled).toHaveBeenCalledTimes(3);

    latest.resolve(response("latest-control", "/host-b/latest.safetensors"));
    await flushPromises();
    second.resolve(response("stale-host", "/host-b/stale.safetensors"));
    first.resolve(response("stale-original", "/host-a/stale.safetensors"));
    await flushPromises();

    const values = wrapper
      .get("[data-test='mobile-control-model']")
      .findAll("option")
      .map((option) => option.attributes("value"));
    expect(values).toContain("/host-b/latest.safetensors");
    expect(values).not.toContain("/host-b/stale.safetensors");
    expect(values).not.toContain("/host-a/stale.safetensors");
    expect(fetchCatalogInstalled.mock.calls.at(-1)).toEqual([
      { family: "sd15", kind: "control-net" },
      { baseUrl: "http://host-b:7680", apiKey: "b-key" },
    ]);
  });

  it("does not query a fallback host and clears catalog options when ControlNet is unsupported", async () => {
    const form = formFor("sd15");
    form.sourceImage = "SRC";
    form.controlImage = "CTRL";
    const wrapper = mount(MobileSourceControls, { props: { form } });
    await flushPromises();
    expect(fetchCatalogInstalled).not.toHaveBeenCalled();

    await wrapper.setProps({
      target: { baseUrl: "http://halcyon:7680", apiKey: null },
    });
    await flushPromises();
    expect(fetchCatalogInstalled).toHaveBeenCalledTimes(1);

    form.family = "sdxl";
    form.model = "sdxl:test";
    await flushPromises();
    expect(fetchCatalogInstalled).toHaveBeenCalledTimes(1);
    expect(wrapper.find("[data-test='mobile-control-model']").exists()).toBe(false);
  });
});

describe("MobileSourceControls — MiniMax H3 FL2VA boundaries", () => {
  it("exposes Ref2VA ordered references through the multi-image Library picker", async () => {
    const form = formFor("minimax-h3");
    form.model = "minimax-h3-ref2va:comfy-pruned-int8";
    const wrapper = mount(MobileSourceControls, {
      props: { form, model: model(form.model, form.family) },
    });

    expect(wrapper.find("[data-test='mobile-h3-authoring']").exists()).toBe(true);
    expect(wrapper.find("[data-test='h3-reference-files']").exists()).toBe(true);
    expect(wrapper.text()).toContain("Add at least one image or video reference.");
    await wrapper.get("[data-test='h3-reference-library']").trigger("click");
    const picker = wrapper.getComponent(MobileImagePickerSheet);
    expect(picker.props("open")).toBe(true);
    expect(picker.props("multiple")).toBe(true);
  });

  it("opens the crop sheet for a Ref2VA image and stores the applied crop on the draft", async () => {
    const form = formFor("minimax-h3");
    form.model = "minimax-h3-ref2va:comfy-pruned-int8";
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
    const wrapper = mount(MobileSourceControls, {
      props: { form, model: model(form.model, form.family) },
    });
    const sheet = wrapper.getComponent(MobileReferenceCropSheet);
    expect(sheet.props("open")).toBe(false);

    await wrapper.get("[data-test='h3-reference-crop-0']").trigger("click");
    expect(sheet.props("open")).toBe(true);
    const editor = wrapper.getComponent(ReferenceCropEditor);
    expect(editor.props("large")).toBe(true);
    editor.vm.$emit("apply", { x: 256, y: 0, width: 512, height: 768 });
    await flushPromises();
    expect(form.h3Authoring?.references[0]?.crop).toEqual({
      x: 256,
      y: 0,
      width: 512,
      height: 768,
    });
    expect(sheet.props("open")).toBe(false);
  });

  it("renders the shared wells and applies a gallery pick to the first frame", async () => {
    const form = formFor("minimax-h3");
    form.model = "minimax-h3-fl2va:comfy-pruned-int8";
    const model = {
      name: form.model,
      family: "minimax-h3",
      source_image: "required",
    } as ModelEntry;
    const wrapper = mount(MobileSourceControls, { props: { form, model } });

    expect(wrapper.find("[data-test='mobile-h3-boundaries']").exists()).toBe(true);
    expect(wrapper.find("[data-test='source-well']").exists()).toBe(true);
    expect(wrapper.find("[data-test='source-required-badge']").exists()).toBe(true);
    // Reviewed first-frame-only runtime: no empty last-frame well.
    expect(wrapper.find("[data-test='end-frame-well']").exists()).toBe(false);

    await wrapper.get("[data-test='source-gallery']").trigger("click");
    const sheet = wrapper.getComponent(MobileImagePickerSheet);
    expect(sheet.props("open")).toBe(true);
    expect(sheet.props("title")).toBe("First frame");
    sheet.vm.$emit("pick", {
      filename: "opening.png",
      base64: "iVBORw0KGgoAAAANSUhEUgAAAAcAAAAECAIAAAAmkwkpAAAAAElFTkSuQmCC",
    });
    await flushPromises();
    expect(form.h3Authoring?.firstFrame).toMatchObject({
      filename: "opening.png",
      width: 7,
      height: 4,
    });
  });

  it("offers both boundary wells when no endpoint is required", () => {
    const form = formFor("minimax-h3");
    form.model = "minimax-h3-fl2va:comfy-pruned-int8";
    const model = { name: form.model, family: "minimax-h3" } as ModelEntry;
    const wrapper = mount(MobileSourceControls, { props: { form, model } });
    expect(wrapper.find("[data-test='source-well']").exists()).toBe(true);
    expect(wrapper.find("[data-test='end-frame-well']").exists()).toBe(true);
  });
});

/**
 * A canvasless (3-D) recipe fits its source to nothing: there is no canvas to
 * crop, pad or upscale toward, so the phone must not offer a fit policy —
 * `buildRequest` records none for a mesh print either. Strength and the mask
 * follow the same recipe, which refuses both.
 */
describe("MobileSourceControls on a canvasless recipe", () => {
  function meshForm(): GenerateForm {
    const form = formFor("hunyuan3d");
    form.model = "hunyuan3d-mini-turbo:fp16";
    form.sourceImage = "c291cmNl";
    form.sourceImageName = "armchair.png";
    return form;
  }

  const meshModel = {
    name: "hunyuan3d-mini-turbo:fp16",
    family: "hunyuan3d",
    source_image: "required",
    generation_profile: {
      schema_version: 1,
      profile_id: "hunyuan3d.mini",
      profile_hash: "hunyuan3d-mini-hash",
      default_recipe_id: "default",
      recipes: [hunyuan3dRecipe()],
    },
  } as ModelEntry;

  it("offers the source well but no fit policy, strength, or mask", () => {
    const form = meshForm();
    const wrapper = mount(MobileSourceControls, { props: { form, model: meshModel } });

    expect(wrapper.find("[data-test='mobile-source-preview']").exists()).toBe(true);
    expect(wrapper.find("[data-test='mobile-source-fit']").exists()).toBe(false);
    expect(wrapper.find("[data-test='mobile-source-strength']").exists()).toBe(false);
    expect(wrapper.find("[data-test='mobile-mask-well']").exists()).toBe(false);
    expect(wrapper.text()).not.toContain("Source fit");
  });

  it("still offers the fit policy on a raster recipe", () => {
    const form = formFor("sdxl");
    form.sourceImage = "c291cmNl";
    const wrapper = mount(MobileSourceControls, { props: { form } });

    expect(wrapper.find("[data-test='mobile-source-fit']").exists()).toBe(true);
  });
});

describe("MobileSourceControls — H3 boundary media budget", () => {
  it("refuses a direct file past the 45 MiB request budget before reading it", async () => {
    const form = formFor("minimax-h3");
    form.model = "minimax-h3-fl2va:comfy-pruned-int8";
    const model = { name: form.model, family: "minimax-h3" } as ModelEntry;
    const wrapper = mount(MobileSourceControls, { props: { form, model } });

    const oversized = new File([new Uint8Array(8)], "huge.png", {
      type: "image/png",
    });
    Object.defineProperty(oversized, "size", {
      value: MAX_MOBILE_GENERATION_REQUEST_MEDIA_BYTES + 1,
    });
    await wrapper
      .get("[data-test='source-well']")
      .trigger("drop", { dataTransfer: { files: [oversized] } });
    await flushPromises();

    expect(form.h3Authoring?.firstFrame ?? null).toBeNull();
    expect(wrapper.text()).toContain("45 MiB");
  });
});
