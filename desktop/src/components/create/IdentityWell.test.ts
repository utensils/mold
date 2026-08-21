import { afterEach, describe, expect, it } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { reactive } from "vue";
import IdentityWell from "./IdentityWell.vue";
import ImageDropWell from "@studio/components/ImageDropWell.vue";
import { newGenerateForm, type GenerateForm } from "../../lib/generateForm";
import { IDENTITY_PHOTO_UNAVAILABLE } from "@studio/lib/identityConditioning";

afterEach(() => (document.body.innerHTML = ""));

/** Bytes whose PNG header declares `width × height` — every identity
 * pre-check reads the header alone, so the payload is never real pixels. */
function pngBytes(width: number, height: number): ArrayBuffer {
  const u32 = (value: number) => [
    (value >>> 24) & 0xff,
    (value >>> 16) & 0xff,
    (value >>> 8) & 0xff,
    value & 0xff,
  ];
  const bytes = new Uint8Array([
    0x89,
    0x50,
    0x4e,
    0x47,
    0x0d,
    0x0a,
    0x1a,
    0x0a,
    ...u32(13),
    0x49,
    0x48,
    0x44,
    0x52,
    ...u32(width),
    ...u32(height),
    8,
    6,
    0,
    0,
    0,
  ]);
  return bytes.buffer as ArrayBuffer;
}

function identityForm(): GenerateForm {
  return reactive({
    ...newGenerateForm(),
    family: "flux",
    model: "flux-dev:q8",
    steps: 20,
    identitySupported: true,
  });
}

function mountWell(form: GenerateForm) {
  return mount(IdentityWell, { props: { form }, attachTo: document.body });
}

describe("IdentityWell", () => {
  it("attaches a dropped PNG with its provenance label", async () => {
    const form = identityForm();
    const wrapper = mountWell(form);
    wrapper
      .get("[data-test='identity-photo-well']")
      .findComponent(ImageDropWell)
      .vm.$emit("file", new File([pngBytes(512, 512)], "face.png", { type: "image/png" }));
    await flushPromises();

    expect(form.identityImage?.filename).toBe("face.png");
    expect(form.identityImage?.base64).toBeTruthy();
    expect(wrapper.find("[data-test='identity-conditioning-error']").exists()).toBe(false);
  });

  it("refuses a non-PNG/JPEG file inline and stages nothing", async () => {
    const form = identityForm();
    const wrapper = mountWell(form);
    wrapper
      .findComponent(ImageDropWell)
      .vm.$emit("file", new File([new Uint8Array([1, 2, 3])], "clip.mp4", { type: "video/mp4" }));
    await flushPromises();

    expect(form.identityImage).toBeNull();
    expect(wrapper.get("[data-test='identity-conditioning-error']").text()).toContain(
      "PNG or JPEG",
    );
  });

  it("refuses a photo beyond the per-axis limit before it is staged", async () => {
    const form = identityForm();
    const wrapper = mountWell(form);
    wrapper
      .findComponent(ImageDropWell)
      .vm.$emit("file", new File([pngBytes(9000, 4000)], "huge.png", { type: "image/png" }));
    await flushPromises();

    expect(form.identityImage).toBeNull();
    expect(wrapper.get("[data-test='identity-conditioning-error']").text()).toContain("8192");
  });

  it("clears the staged photo and its inline reason", async () => {
    const form = identityForm();
    form.identityImage = { filename: "face.png", base64: "aWQ=" };
    const wrapper = mountWell(form);
    await wrapper.get("[data-test='identity-remove']").trigger("click");
    expect(form.identityImage).toBeNull();
  });

  it("discloses a reused photo that is no longer on this device", () => {
    const form = identityForm();
    // What Reuse settings leaves behind when the stash lookup misses: recorded
    // provenance with no bytes. Rendering a different face silently would be
    // worse than saying so.
    form.identityImage = { filename: "face.png", base64: "" };
    const wrapper = mountWell(form);
    expect(wrapper.get("[data-test='identity-conditioning-error']").text()).toBe(
      IDENTITY_PHOTO_UNAVAILABLE,
    );
  });

  it("renders the shared admission refusal inline, never as a toast", () => {
    const form = identityForm();
    form.identityImage = { filename: "face.png", base64: btoa("nope") };
    form.loras = [{ path: "style.safetensors", name: "style", scale: 1, trainedWords: [] }];
    const wrapper = mountWell(form);
    expect(wrapper.get("[data-test='identity-conditioning-error']").text()).toContain("LoRA");
  });
});
