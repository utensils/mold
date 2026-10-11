import { mount } from "@vue/test-utils";
import { describe, it, expect, vi } from "vitest";
import OriginalMediaHost from "./OriginalMediaHost.vue";
import OriginalMediaButton from "./OriginalMediaButton.vue";

describe("OriginalMediaButton", () => {
  it("opens an original without mutating attachment and restores focus on Escape", async () => {
    const create = vi
      .spyOn(URL, "createObjectURL")
      .mockReturnValue("blob:original");
    const revoke = vi
      .spyOn(URL, "revokeObjectURL")
      .mockImplementation(() => {});
    const host = mount(OriginalMediaHost, { attachTo: document.body });
    const w = mount(OriginalMediaButton, {
      props: { data: "aGVsbG8=", mimeType: "image/png", label: "Source" },
      global: { stubs: { Teleport: true } },
      attachTo: document.body,
    });
    const button = w.get("button");
    (button.element as HTMLElement).focus();
    await button.trigger("click");
    await vi.waitFor(() =>
      expect(host.find('[role="dialog"]').exists()).toBe(true),
    );
    expect(host.get("img").attributes("src")).toBe("blob:original");
    await host.get('[data-test="original-actual-size"]').trigger("click");
    expect(host.get("img").classes()).toContain("original-media__actual");
    document.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Escape", bubbles: true }),
    );
    await w.vm.$nextTick();
    expect(host.find('[role="dialog"]').exists()).toBe(false);
    expect(document.activeElement).toBe(button.element);
    expect(revoke).toHaveBeenCalledWith("blob:original");
    expect(w.emitted("file")).toBeUndefined();
    expect(w.emitted("clear")).toBeUndefined();
    w.unmount();
    host.unmount();
    create.mockRestore();
    revoke.mockRestore();
  });
  it("does not publish a late original after closing", async () => {
    let resolve!: (blob: Blob) => void;
    const create = vi.spyOn(URL, "createObjectURL");
    const load = vi.fn(
      () =>
        new Promise<Blob>((r) => {
          resolve = r;
        }),
    );
    const host = mount(OriginalMediaHost, { attachTo: document.body });
    const w = mount(OriginalMediaButton, {
      props: { load, label: "Queue source" },
      global: { stubs: { Teleport: true } },
    });
    await w.get("button").trigger("click");
    await host.get('[data-test="original-close"]').trigger("click");
    resolve(new Blob(["late"], { type: "image/png" }));
    await w.vm.$nextTick();
    await Promise.resolve();
    expect(create).not.toHaveBeenCalled();
    w.unmount();
    host.unmount();
    create.mockRestore();
  });
});

it("revokes a selected preview when its authority changes, and refuses a late error", async () => {
  let reject!: (reason: Error) => void;
  const host = mount(OriginalMediaHost);
  const w = mount(OriginalMediaButton, {
    props: {
      authority: "machine:job:0",
      load: () =>
        new Promise<Blob>((_, r) => {
          reject = r;
        }),
    },
  });
  await w.get("button").trigger("click");
  expect(host.find('[role="dialog"]').exists()).toBe(true);
  await w.setProps({ authority: "machine:job:1" });
  reject(new Error("obsolete read"));
  await Promise.resolve();
  await host.vm.$nextTick();
  expect(host.find('[role="dialog"]').exists()).toBe(false);
  w.unmount();
  host.unmount();
});

it("uses the original rather than the displayed thumbnail, even for a parked well", async () => {
  const { default: ImageDropWell } = await import("./ImageDropWell.vue");
  const create = vi
    .spyOn(URL, "createObjectURL")
    .mockReturnValue("blob:original");
  const host = mount(OriginalMediaHost);
  const w = mount(ImageDropWell, {
    props: {
      image: btoa("thumbnail"),
      originalImage: btoa("original"),
      disabled: true,
    },
  });
  await w
    .findAll("button")
    .find((button) => button.text() === "View original")!
    .trigger("click");
  expect(await (create.mock.calls[0]![0] as Blob).text()).toBe("original");
  w.unmount();
  host.unmount();
  create.mockRestore();
});
