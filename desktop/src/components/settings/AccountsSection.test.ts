import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";

const secrets = new Map<string, string>();
const secretSet = vi.fn((name: string, value: string) => {
  secrets.set(name, value);
  return Promise.resolve();
});
const secretClear = vi.fn((name: string) => {
  secrets.delete(name);
  return Promise.resolve();
});
vi.mock("../../lib/ipc", () => ({
  inTauri: () => true,
  ipc: {
    secretGet: (name: string) => Promise.resolve(secrets.get(name) ?? null),
    secretSet: (name: string, value: string) => secretSet(name, value),
    secretClear: (name: string) => secretClear(name),
  },
}));

import AccountsSection from "./AccountsSection.vue";

beforeEach(() => {
  setActivePinia(createPinia());
  secrets.clear();
  secretSet.mockClear();
  secretClear.mockClear();
});

/*
 * These two tokens are FILE-backed on this device, not engine-config rows, so
 * this section owns the reads and writes and the shared control is told only
 * whether something is stored. It must never be handed, or render, the secret.
 */
describe("Settings ▸ Accounts & tokens", () => {
  it("says nothing is stored until something is", async () => {
    const wrapper = mount(AccountsSection);
    await flushPromises();
    expect(wrapper.text()).toContain("Hugging Face token");
    expect(wrapper.text()).toContain("Civitai token");
    expect(wrapper.findAll("[data-test='secret-clear']")).toHaveLength(0);
    expect(wrapper.findAll("[data-test='secret-edit']").map((b) => b.text())).toEqual([
      "Set…",
      "Set…",
    ]);
  });

  it("reads a stored token as present without ever rendering it", async () => {
    secrets.set("hf-token", "hf_supersecret");
    const wrapper = mount(AccountsSection);
    await flushPromises();
    expect(wrapper.text()).not.toContain("hf_supersecret");
    expect(wrapper.findAll("[data-test='secret-edit']").map((b) => b.text())).toEqual([
      "Replace…",
      "Set…",
    ]);
  });

  it("stores what is typed against the right secret, and removes it again", async () => {
    const wrapper = mount(AccountsSection);
    await flushPromises();

    // The second row is Civitai — each row must write its OWN name.
    await wrapper.findAll("[data-test='secret-edit']")[1]!.trigger("click");
    await wrapper.get("input[type='password']").setValue("civitai-key");
    await wrapper.get("[data-test='secret-save']").trigger("click");
    await flushPromises();
    expect(secretSet).toHaveBeenCalledWith("civitai-token", "civitai-key");

    await wrapper.get("[data-test='secret-clear']").trigger("click");
    await flushPromises();
    expect(secretClear).toHaveBeenCalledWith("civitai-token");
    expect(wrapper.findAll("[data-test='secret-clear']")).toHaveLength(0);
  });
});
