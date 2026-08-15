import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import MediaTile from "./MediaTile.vue";

function make(
  extra: Record<string, unknown> = {},
  slots: Record<string, string> = {},
) {
  return mount(MediaTile, {
    props: { src: "/prints/frog.png", alt: "A frog on bark", ...extra },
    slots,
  });
}

describe("MediaTile", () => {
  it("renders a button wrapping the image", () => {
    const wrapper = make();
    expect(wrapper.element.tagName).toBe("BUTTON");
    expect(wrapper.attributes("type")).toBe("button");
    expect(wrapper.find("img").exists()).toBe(true);
  });

  it("applies src and alt to the image", () => {
    const img = make().find("img");
    expect(img.attributes("src")).toBe("/prints/frog.png");
    expect(img.attributes("alt")).toBe("A frog on bark");
  });

  it("emits open on click", async () => {
    const wrapper = make();
    await wrapper.trigger("click");
    expect(wrapper.emitted("open")).toHaveLength(1);
  });

  it("renders the NEW badge only when fresh", () => {
    expect(make().find(".ms-tile__fresh").exists()).toBe(false);

    const fresh = make({ fresh: true });
    expect(fresh.find(".ms-tile__fresh").text()).toBe("New");
  });

  it("renders the overlay slot in the corner container", () => {
    const wrapper = make({}, { overlay: '<span class="duration">0:07</span>' });
    expect(wrapper.find(".ms-tile__overlay .duration").text()).toBe("0:07");
  });

  it("omits the overlay container when no slot is given", () => {
    expect(make().find(".ms-tile__overlay").exists()).toBe(false);
  });

  it("shows a shimmer and hides the image until its bytes land", async () => {
    const wrapper = make();
    expect(wrapper.attributes("data-loaded")).toBe("false");
    expect(wrapper.find(".ms-tile__ghost").exists()).toBe(true);
    await wrapper.find("img").trigger("load");
    expect(wrapper.attributes("data-loaded")).toBe("true");
    expect(wrapper.find(".ms-tile__ghost").exists()).toBe(false);
  });

  it("renders no img element at all for an empty src", () => {
    const wrapper = make({ src: "" });
    expect(wrapper.find("img").exists()).toBe(false);
    expect(wrapper.find(".ms-tile__ghost").exists()).toBe(true);
  });
});
