import { mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import IdentityPanel from "./IdentityPanel.vue";
import {
  useGenerateForm,
  __testing__,
} from "../../composables/useGenerateForm";
import type { GenerateFormState, ModelInfoExtended } from "../../types";

const PNG_7x4 = "iVBORw0KGgoAAAANSUhEUgAAAAcAAAAECAIAAAAmkwkpAAAAAElFTkSuQmCC";

function baseForm(
  overrides: Partial<GenerateFormState> = {},
): GenerateFormState {
  __testing__.resetForTest();
  const state = useGenerateForm().state.value;
  return { ...state, model: "flux-dev-pulid:bf16", ...overrides };
}

/** A server-authored v1 recipe whose only interesting bit is the gate. */
function recipeModel(supportsIdentity: boolean): ModelInfoExtended {
  return {
    name: "flux-dev-pulid:bf16",
    family: "flux",
    downloaded: true,
    default_width: 1024,
    default_height: 1024,
    default_steps: 20,
    default_guidance: 3.5,
    generation_profile: {
      schema_version: 1,
      profile_id: "flux.v1",
      profile_hash: "hash",
      default_recipe_id: "default",
      recipes: [
        {
          id: "default",
          label: "Default",
          request_selector: {},
          defaults: { width: 1024, height: 1024, steps: 20, guidance: 3.5 },
          resolution: {
            domain: "dynamic",
            alignment: 64,
            min_width: 64,
            min_height: 64,
            max_pixels: 1_048_576,
            aspect_groups: [],
          },
          steps: { default: 20, min: 1, max: 100, step: 1, mode: "adjustable" },
          guidance: {
            default: 3.5,
            min: 0,
            max: 20,
            step: 0.1,
            mode: "adjustable",
          },
          capabilities: {
            guidance: { adjustable: true, supports_negative_prompt: false },
            negative_prompt: { mode: "hidden", required: false },
            supports_lora: true,
            supports_controlnet: false,
            supports_identity: supportsIdentity,
            supports_sequence: false,
            supports_extend: false,
            supports_audio: false,
            source_video: { mode: "hidden", required: false },
            mask: { mode: "hidden", required: false },
            keyframes: { mode: "hidden", required: false },
            audio: { mode: "hidden", required: false },
            lora: { mode: "adjustable", max_count: 4 },
            controlnet: { mode: "hidden", max_count: 0 },
            output: {
              default_format: "png",
              formats: ["png"],
              audio_requires_mp4: false,
            },
            wan_recipe: {
              mode: "hidden",
              supports_distill_strength: false,
              supports_first_last_frame: false,
            },
            schedulers: [],
          },
          provenance: [],
        },
      ],
    },
  } as unknown as ModelInfoExtended;
}

/** Older server: no generation profile at all, only the additive row flag. */
function rowOnlyModel(supports_identity?: boolean): ModelInfoExtended {
  return {
    name: "flux-dev-pulid:bf16",
    family: "flux",
    downloaded: true,
    default_width: 1024,
    default_height: 1024,
    default_steps: 20,
    default_guidance: 3.5,
    ...(supports_identity === undefined ? {} : { supports_identity }),
  } as ModelInfoExtended;
}

function factory(
  models: ModelInfoExtended[],
  overrides: Partial<GenerateFormState> = {},
) {
  return mount(IdentityPanel, {
    props: { modelValue: baseForm(overrides), models },
  });
}

const photo = () => ({
  kind: "upload" as const,
  filename: "ada.png",
  base64: PNG_7x4,
});

beforeEach(() => localStorage.clear());
afterEach(() => __testing__.resetForTest());

describe("IdentityPanel — the capability gate", () => {
  it("hides everything when the server-authored recipe says no", () => {
    const wrapper = factory([recipeModel(false)]);
    expect(wrapper.find("[data-test='identity-panel']").exists()).toBe(false);
    expect(wrapper.find("[data-test='identity-photo-well']").exists()).toBe(
      false,
    );
  });

  it("hides everything when neither the recipe nor the row advertises it", () => {
    // An older server omits the additive field entirely; absence reads as
    // "no", which is what keeps the control off a host that would refuse it.
    const wrapper = factory([rowOnlyModel()]);
    expect(wrapper.find("[data-test='identity-panel']").exists()).toBe(false);
  });

  it("renders the shared well when the recipe advertises identity", () => {
    const wrapper = factory([recipeModel(true)]);
    expect(wrapper.find("[data-test='identity-panel']").exists()).toBe(true);
    expect(wrapper.find("[data-test='identity-photo-well']").exists()).toBe(
      true,
    );
    expect(wrapper.text()).toContain("Identity");
  });

  it("renders when only the /api/models row advertises it", () => {
    const wrapper = factory([rowOnlyModel(true)]);
    expect(wrapper.find("[data-test='identity-photo-well']").exists()).toBe(
      true,
    );
  });

  it("keeps a staged photo reachable on a checkpoint that lost the capability", () => {
    // `toRequest` already keeps it off the wire, but hiding the well outright
    // would leave the inline refusal pointing at a control the user cannot
    // see — and no way to remove the photo that is blocking Generate.
    const wrapper = factory([recipeModel(false)], { identityImage: photo() });
    expect(wrapper.find("[data-test='identity-photo-well']").exists()).toBe(
      true,
    );
    expect(
      wrapper.get("[data-test='identity-conditioning-error']").text(),
    ).toContain("does not support identity photos");
  });
});

describe("IdentityPanel — attaching a photo", () => {
  it("decodes a dropped PNG onto the form untouched", async () => {
    const wrapper = factory([recipeModel(true)]);
    const bytes = Uint8Array.from(atob(PNG_7x4), (c) => c.charCodeAt(0));
    const file = new File([bytes], "ada.png", { type: "image/png" });
    await wrapper
      .get("[data-test='identity-well']")
      .trigger("drop", { dataTransfer: { files: [file] } });
    await new Promise((resolve) => setTimeout(resolve, 0));

    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    expect(next.identityImage).toMatchObject({
      filename: "ada.png",
      base64: PNG_7x4,
      width: 7,
      height: 4,
      mime: "image/png",
    });
  });

  it("refuses a non-PNG/JPEG file inline and stages nothing", async () => {
    const wrapper = factory([recipeModel(true)]);
    const file = new File(["webp"], "ada.webp", { type: "image/webp" });
    await wrapper
      .get("[data-test='identity-well']")
      .trigger("drop", { dataTransfer: { files: [file] } });
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(wrapper.emitted("update:modelValue")).toBeUndefined();
    expect(
      wrapper.get("[data-test='identity-conditioning-error']").text(),
    ).toContain("PNG or JPEG");
  });

  it("clears the staged photo through the well's remove control", async () => {
    const wrapper = factory([recipeModel(true)], { identityImage: photo() });
    await wrapper.get("[data-test='identity-remove']").trigger("click");
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    expect(next.identityImage).toBeNull();
  });

  it("offers no gallery picker in this pass", () => {
    const wrapper = factory([recipeModel(true)]);
    expect(wrapper.find("[data-test='identity-gallery']").exists()).toBe(false);
  });
});

describe("IdentityPanel — the combinations admission refuses", () => {
  it("names the LoRA conflict inline, never as an event", () => {
    const wrapper = factory([recipeModel(true)], {
      identityImage: photo(),
      loras: [{ path: "style.safetensors", scale: 1 }],
    });
    expect(
      wrapper.get("[data-test='identity-conditioning-error']").text(),
    ).toContain("cannot be combined with a LoRA");
  });

  it("names the source-image conflict inline", () => {
    const wrapper = factory([recipeModel(true)], {
      identityImage: photo(),
      imageAttachments: [
        { kind: "upload", filename: "scene.png", base64: PNG_7x4 },
      ],
    });
    expect(
      wrapper.get("[data-test='identity-conditioning-error']").text(),
    ).toContain("cannot be combined with a source image");
  });

  it("asks for the photo when only the knobs are set", () => {
    const wrapper = factory([recipeModel(true)], { identityWeight: 1.5 });
    expect(
      wrapper.get("[data-test='identity-conditioning-error']").text(),
    ).toContain("Attach an identity photo");
  });

  it("says nothing at all while identity is simply unused", () => {
    const wrapper = factory([recipeModel(true)]);
    expect(
      wrapper.find("[data-test='identity-conditioning-error']").exists(),
    ).toBe(false);
    expect(wrapper.find("[data-test='identity-hint']").exists()).toBe(true);
  });

  it("discloses a reuse whose photo is no longer on this device", () => {
    const wrapper = mount(IdentityPanel, {
      props: {
        modelValue: baseForm({
          identityImage: { kind: "upload", filename: "ada.png", base64: "" },
        }),
        models: [recipeModel(true)],
        notice: "gone",
      },
    });
    expect(wrapper.get("[data-test='identity-notice']").text()).toBe("gone");
  });
});
