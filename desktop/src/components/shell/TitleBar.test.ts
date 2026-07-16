import { createPinia, setActivePinia } from "pinia";
import { mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it } from "vitest";
import TitleBar from "./TitleBar.vue";

describe("TitleBar", () => {
  beforeEach(() => setActivePinia(createPinia()));

  it("renders the Mold wordmark in its lowercase brand form", () => {
    const wrapper = mount(TitleBar);
    expect(wrapper.get(".brand-wordmark").text()).toBe("mold");
  });
});
