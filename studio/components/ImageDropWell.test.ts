import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import ImageDropWell from "./ImageDropWell.vue";

describe("ImageDropWell", () => {
  it("opens the file picker from the drop zone and emits the chosen file", async () => {
    const wrapper = mount(ImageDropWell, { props: { testId: "t" } });
    const input = wrapper.get("[data-test='t-file']");
    let clicked = 0;
    (input.element as HTMLInputElement).click = () => {
      clicked += 1;
    };
    await wrapper.get("[data-test='t-well']").trigger("click");
    await wrapper.get("[data-test='t-well']").trigger("keydown.enter");
    expect(clicked).toBe(2);

    const file = new File(["png"], "still.png", { type: "image/png" });
    Object.defineProperty(input.element, "files", {
      configurable: true,
      value: [file],
    });
    await input.trigger("change");
    expect(wrapper.emitted("file")).toEqual([[file]]);
  });

  it("accepts a dropped file", async () => {
    const wrapper = mount(ImageDropWell, { props: { testId: "t" } });
    const file = new File(["png"], "dropped.png", { type: "image/png" });
    await wrapper
      .get("[data-test='t-well']")
      .trigger("drop", { dataTransfer: { files: [file] } });
    expect(wrapper.emitted("file")).toEqual([[file]]);
  });

  it("renders a preview with a working remove control once attached", async () => {
    const wrapper = mount(ImageDropWell, {
      props: {
        testId: "t",
        image: "QUJD",
        mimeType: "image/jpeg",
        alt: "First frame",
      },
    });
    expect(wrapper.find("[data-test='t-well']").exists()).toBe(false);
    expect(wrapper.get("img").attributes("src")).toBe(
      "data:image/jpeg;base64,QUJD",
    );
    await wrapper.get("[data-test='t-remove']").trigger("click");
    expect(wrapper.emitted("clear")).toHaveLength(1);
  });

  it("emits gallery intent without touching the file input", async () => {
    const wrapper = mount(ImageDropWell, {
      props: { testId: "t", gallery: true },
    });
    await wrapper.get("[data-test='t-gallery']").trigger("click");
    expect(wrapper.emitted("gallery")).toHaveLength(1);
    expect(wrapper.emitted("file")).toBeUndefined();
  });

  it("keeps removal available when only picking is disabled", async () => {
    const wrapper = mount(ImageDropWell, {
      props: { testId: "t", image: "QUJD", pickDisabled: true, gallery: true },
    });
    const remove = wrapper.get("[data-test='t-remove']");
    expect(remove.attributes("disabled")).toBeUndefined();
    await remove.trigger("click");
    expect(wrapper.emitted("clear")).toHaveLength(1);
  });

  it("blocks drop and pick while inert", async () => {
    const wrapper = mount(ImageDropWell, {
      props: { testId: "t", disabled: true },
    });
    const zone = wrapper.get("[data-test='t-well']");
    expect(zone.attributes("aria-disabled")).toBe("true");
    const file = new File(["png"], "blocked.png", { type: "image/png" });
    await zone.trigger("drop", { dataTransfer: { files: [file] } });
    expect(wrapper.emitted("file")).toBeUndefined();
  });

  it("offers reattach-with-provenance when the bytes were stripped", () => {
    const wrapper = mount(ImageDropWell, {
      props: { testId: "t", filename: "opening.png" },
    });
    const zone = wrapper.get("[data-test='t-well']");
    expect(zone.text()).toContain("opening.png");
    expect(zone.text()).toContain("Reattach original media");
    expect(wrapper.find("[data-test='t-remove']").exists()).toBe(true);
  });
});
