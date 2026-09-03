import { mount } from "@vue/test-utils";
import { describe, expect, it, vi } from "vitest";
import ImageDropWell from "./ImageDropWell.vue";

describe("ImageDropWell", () => {
  it("delegates a pick to a native shell without opening the file input", async () => {
    const wrapper = mount(ImageDropWell, {
      props: { testId: "native", nativePicker: true },
    });
    const input = wrapper.get("[data-test='native-file']")
      .element as HTMLInputElement;
    const click = vi.spyOn(input, "click");

    await wrapper.get("[data-test='native-well']").trigger("click");

    expect(wrapper.emitted("pick")).toHaveLength(1);
    expect(click).not.toHaveBeenCalled();
  });

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

  it("renders one shared preview with replace, filename, and remove actions", async () => {
    const wrapper = mount(ImageDropWell, {
      props: {
        testId: "t",
        image: "QUJD",
        mimeType: "image/jpeg",
        filename: "first-frame.jpg",
        alt: "First frame",
        gallery: true,
      },
    });
    expect(wrapper.find("[data-test='t-well']").exists()).toBe(false);
    expect(wrapper.get("img").attributes("src")).toBe(
      "data:image/jpeg;base64,QUJD",
    );
    expect(wrapper.get("figcaption").text()).toBe("first-frame.jpg");
    await wrapper.get("[data-test='t-replace']").trigger("click");
    expect(wrapper.emitted("gallery")).toHaveLength(1);
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

  it("exposes Android's 48dp touch target to every attached action", () => {
    const wrapper = mount(ImageDropWell, {
      props: {
        testId: "t",
        image: "cG5n",
        touchFriendly: true,
        touchTargetSize: 48,
      },
    });

    expect(wrapper.attributes("style")).toContain(
      "--image-well-touch-target: 48px",
    );
    expect(
      wrapper.get("[data-test='t-remove']").attributes("aria-label"),
    ).toContain("Remove");
    expect(wrapper.get("[data-test='t-replace']").text()).toBe("Replace photo");
  });
});

describe("ImageDropWell drop-target identity", () => {
  it("names itself for the shared drop router, filled or empty", () => {
    // The desktop shell resolves the well under an OS drag with
    // `elementFromPoint(...).closest("[data-drop-target]")`, so the attribute
    // has to sit on the ROOT — a filled well renders no drop zone at all.
    const empty = mount(ImageDropWell, {
      props: { testId: "t", dropTarget: "identity" },
    });
    expect(empty.attributes("data-drop-target")).toBe("identity");

    const filled = mount(ImageDropWell, {
      props: { testId: "t", image: "cG5n", dropTarget: "source" },
    });
    expect(filled.attributes("data-drop-target")).toBe("source");
  });

  it("renders no attribute when the surface names no target", () => {
    const wrapper = mount(ImageDropWell, { props: { testId: "t" } });
    expect(wrapper.attributes("data-drop-target")).toBeUndefined();
  });
});
