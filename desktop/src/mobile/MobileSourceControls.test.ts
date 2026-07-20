import { flushPromises, mount } from "@vue/test-utils";
import { reactive } from "vue";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ModelEntry } from "../lib/api/types";
import { newGenerateForm, type GenerateForm } from "../lib/generateForm";
import MaskEditorModal from "../components/generate/MaskEditorModal.vue";

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

  it("accepts a PNG source and exposes preview, strength, fit, replace, and remove", async () => {
    const form = formFor("sdxl");
    const wrapper = mount(MobileSourceControls, { props: { form } });

    const sourceInput = wrapper.get("[data-test='mobile-source-input']");
    expect(sourceInput.attributes("accept")).toBe("image/png,image/jpeg");
    await chooseFiles(wrapper, "[data-test='mobile-source-input']", [
      new File(["photo"], "photo.jpg", { type: "image/jpeg" }),
    ]);

    await vi.waitFor(() => expect(form.sourceImage).not.toBeNull());
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

  it("rejects non-PNG/JPEG photos with an associated alert", async () => {
    const form = formFor("sdxl");
    const wrapper = mount(MobileSourceControls, { props: { form } });

    await chooseFiles(wrapper, "[data-test='mobile-source-input']", [
      new File(["photo"], "photo.webp", { type: "image/webp" }),
    ]);

    expect(form.sourceImage).toBeNull();
    const alert = wrapper.get("[data-test='mobile-source-error']");
    expect(alert.attributes("role")).toBe("alert");
    expect(alert.text()).toContain("PNG or JPEG");
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
