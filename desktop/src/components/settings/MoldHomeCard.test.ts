import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";

const getMoldHome = vi.fn();
const pickDirectory = vi.fn();
const changeMoldHome = vi.fn();

vi.mock("../../lib/ipc", () => ({
  ipc: {
    getMoldHome: (...args: unknown[]) => getMoldHome(...args),
    pickDirectory: (...args: unknown[]) => pickDirectory(...args),
    changeMoldHome: (...args: unknown[]) => changeMoldHome(...args),
  },
}));

import MoldHomeCard from "./MoldHomeCard.vue";

beforeEach(() => {
  vi.clearAllMocks();
  getMoldHome.mockResolvedValue({
    path: "/Users/test/.mold",
    source: "default",
    exists: true,
  });
  pickDirectory.mockResolvedValue("/Volumes/Studio/Mold");
  // Never resolve: success restarts the native app.
  changeMoldHome.mockReturnValue(new Promise(() => {}));
});

describe("MoldHomeCard", () => {
  it("shows the effective shared root and its provenance", async () => {
    const wrapper = mount(MoldHomeCard);
    await flushPromises();
    expect(wrapper.text()).toContain("/Users/test/.mold");
    expect(wrapper.text()).toContain("Default location");
  });

  it("supports the native folder picker and recommends migration", async () => {
    const wrapper = mount(MoldHomeCard);
    await flushPromises();
    await wrapper.get('[data-test="change-mold-home"]').trigger("click");
    expect(wrapper.get('input[type="radio"]:checked').attributes("value")).toBe("true");

    await wrapper.get('[data-test="browse-mold-home"]').trigger("click");
    await flushPromises();
    expect(wrapper.get<HTMLInputElement>('[data-test="mold-home-path"]').element.value).toBe(
      "/Volumes/Studio/Mold",
    );
  });

  it("passes a typed path and migration choice to the restart boundary", async () => {
    const wrapper = mount(MoldHomeCard);
    await flushPromises();
    await wrapper.get('[data-test="change-mold-home"]').trigger("click");
    await wrapper.get('[data-test="mold-home-path"]').setValue("/mnt/models/mold");
    await wrapper.findAll('input[type="radio"]')[1]!.setValue(true);
    await wrapper.get("form").trigger("submit");

    expect(changeMoldHome).toHaveBeenCalledWith("/mnt/models/mold", false);
    expect(wrapper.get('[data-test="apply-mold-home"]').text()).toBe("Switching…");
  });

  it("keeps validation failures in context", async () => {
    changeMoldHome.mockRejectedValueOnce(new Error("The destination is not writable."));
    const wrapper = mount(MoldHomeCard);
    await flushPromises();
    await wrapper.get('[data-test="change-mold-home"]').trigger("click");
    await wrapper.get('[data-test="mold-home-path"]').setValue("/locked/mold");
    await wrapper.get("form").trigger("submit");
    await flushPromises();

    expect(wrapper.get('[role="alert"]').text()).toContain("not writable");
    expect(wrapper.get('[data-test="apply-mold-home"]').attributes("disabled")).toBeUndefined();
  });

  it("explains why an environment-owned home cannot be changed in the app", async () => {
    getMoldHome.mockResolvedValueOnce({
      path: "/managed/mold",
      source: "environment",
      exists: true,
    });
    const wrapper = mount(MoldHomeCard);
    await flushPromises();

    expect(wrapper.get('[data-test="change-mold-home"]').attributes("disabled")).toBeDefined();
    expect(wrapper.text()).toContain("Unset MOLD_HOME");
  });

  it("surfaces an unavailable external-drive home with recovery guidance", async () => {
    getMoldHome.mockResolvedValueOnce({
      path: "/Volumes/Offline/Mold",
      source: "saved",
      exists: false,
    });
    const wrapper = mount(MoldHomeCard);
    await flushPromises();

    expect(wrapper.get('[role="alert"]').text()).toContain("Reconnect its drive");
    expect(wrapper.get('[data-test="change-mold-home"]').attributes("disabled")).toBeUndefined();
    await wrapper.get('[data-test="change-mold-home"]').trigger("click");
    const radios = wrapper.findAll('input[name="mold-home-action"]');
    expect(radios[0]!.attributes("disabled")).toBeDefined();
    expect((radios[1]!.element as HTMLInputElement).checked).toBe(true);
  });

  it("offers a repair state for a corrupt saved pointer", async () => {
    getMoldHome.mockResolvedValueOnce({ path: "", source: "invalid", exists: false });
    const wrapper = mount(MoldHomeCard);
    await flushPromises();

    expect(wrapper.text()).toContain("No valid saved location");
    expect(wrapper.get('[role="alert"]').text()).toContain("Choose a folder to repair it");
  });
});
