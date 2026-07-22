import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AboutSection from "./AboutSection.vue";

vi.mock("../../lib/api/client", () => ({
  apiJson: vi.fn().mockRejectedValue(new Error("offline")),
}));

vi.mock("../../lib/ipc", () => ({
  inTauri: () => false,
  ipc: { openLogsDir: vi.fn() },
}));

describe("AboutSection", () => {
  beforeEach(() => setActivePinia(createPinia()));

  it("credits both core contributors", () => {
    const wrapper = mount(AboutSection);

    expect(wrapper.text()).toContain("James Brink");
    expect(wrapper.text()).toContain("Jeffrey Dilley");
  });
});
