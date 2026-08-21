import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import IdentityPhotoWell from "./IdentityPhotoWell.vue";
import { ID_IMAGE_ACCEPT } from "../lib/identityConditioning";

function factory(props: Record<string, unknown> = {}) {
  return mount(IdentityPhotoWell, { props });
}

describe("IdentityPhotoWell", () => {
  it("renders the shared Identity heading and hint while empty", () => {
    const wrapper = factory();
    expect(wrapper.find("[data-test='identity-photo-well']").exists()).toBe(
      true,
    );
    expect(wrapper.text()).toContain("Identity");
    expect(wrapper.find("[data-test='identity-hint']").exists()).toBe(true);
    expect(wrapper.find("[data-test='identity-well']").exists()).toBe(true);
  });

  it("accepts only the formats the engine reads", () => {
    const input = factory().find("[data-test='identity-file']");
    expect(input.attributes("accept")).toBe(ID_IMAGE_ACCEPT);
    expect(ID_IMAGE_ACCEPT).toBe("image/png,image/jpeg");
  });

  it("shows the attached photo with a remove control instead of the hint", () => {
    const wrapper = factory({
      image: "AAAA",
      mimeType: "image/png",
      filename: "ada.png",
    });
    expect(wrapper.find("[data-test='identity-remove']").exists()).toBe(true);
    expect(wrapper.find("[data-test='identity-hint']").exists()).toBe(false);
  });

  it("renders one inline refusal, never a toast, and hides the hint", () => {
    const wrapper = factory({ error: "Remove the LoRA or the photo." });
    const error = wrapper.find("[data-test='identity-conditioning-error']");
    expect(error.exists()).toBe(true);
    expect(error.attributes("role")).toBe("alert");
    expect(wrapper.find("[data-test='identity-hint']").exists()).toBe(false);
  });

  it("offers the gallery pick only when the surface has a picker", async () => {
    expect(factory().find("[data-test='identity-gallery']").exists()).toBe(
      false,
    );
    const wrapper = factory({ gallery: true });
    await wrapper.find("[data-test='identity-gallery']").trigger("click");
    expect(wrapper.emitted("gallery")).toHaveLength(1);
  });

  it("reports a cleared photo to the surface", async () => {
    const wrapper = factory({ image: "AAAA", filename: "ada.png" });
    await wrapper.find("[data-test='identity-remove']").trigger("click");
    expect(wrapper.emitted("clear")).toHaveLength(1);
  });
});
