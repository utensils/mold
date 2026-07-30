import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { defineComponent, h, nextTick, ref } from "vue";
import ImagePickerModal from "./ImagePickerModal.vue";
import * as api from "../api";

vi.mock("../api", () => ({
  listGallery: vi.fn(async () => []),
  thumbnailUrl: vi.fn((filename: string) => `/thumb/${filename}`),
  imageUrl: vi.fn((filename: string) => `/image/${filename}`),
}));

vi.mock("../lib/base64", () => ({
  blobToBase64: vi.fn(async (blob: Blob & { name?: string }) =>
    blob.name ? `b64:${blob.name}` : "b64:blob",
  ),
}));

/**
 * Pictographs, dingbats, arrows and the variation selector. The Mold Studio
 * voice is icons-only, so any hit in rendered chrome is a regression.
 */
const EMOJI =
  /[\u{1F000}-\u{1FAFF}\u{2190}-\u{21FF}\u{2300}-\u{27BF}\u{2B00}-\u{2BFF}\u{FE0F}]/u;

function dialog(): HTMLElement {
  const el = document.body.querySelector<HTMLElement>("[role=dialog]");
  expect(el).not.toBeNull();
  return el as HTMLElement;
}

describe("ImagePickerModal", () => {
  afterEach(() => {
    document.body.innerHTML = "";
  });

  it("emits uploaded files in selected order", async () => {
    const w = mount(ImagePickerModal, {
      props: { open: true },
      attachTo: document.body,
    });

    const input = document.body.querySelector(
      "input[type='file']",
    ) as HTMLInputElement;
    const files = [
      new File(["a"], "target.png", { type: "image/png" }),
      new File(["b"], "ref.png", { type: "image/png" }),
    ];
    Object.defineProperty(input, "files", { value: files });
    input.dispatchEvent(new Event("change", { bubbles: true }));
    await flushPromises();

    const pick = w.emitted("pick")?.[0]?.[0];
    expect(pick).toEqual([
      { kind: "upload", filename: "target.png", base64: "b64:target.png" },
      { kind: "upload", filename: "ref.png", base64: "b64:ref.png" },
    ]);
    expect(w.emitted("close")).toHaveLength(1);
  });

  it("can be configured as a single-image mask picker", async () => {
    const w = mount(ImagePickerModal, {
      props: { open: true, title: "Mask base image", multiple: false },
      attachTo: document.body,
    });

    expect(document.body.textContent).toContain("Mask base image");

    const input = document.body.querySelector(
      "input[type='file']",
    ) as HTMLInputElement;
    const files = [
      new File(["a"], "mask-base.png", { type: "image/png" }),
      new File(["b"], "ignored.png", { type: "image/png" }),
    ];
    Object.defineProperty(input, "files", { value: files });
    input.dispatchEvent(new Event("change", { bubbles: true }));
    await flushPromises();

    expect(w.emitted("pick")?.[0]?.[0]).toEqual([
      {
        kind: "upload",
        filename: "mask-base.png",
        base64: "b64:mask-base.png",
      },
    ]);
  });

  it("filters local files and gallery rows to PNG and JPEG", async () => {
    vi.mocked(api.listGallery).mockResolvedValueOnce([
      {
        filename: "still.png",
        metadata: {} as never,
        timestamp: 3,
      },
      {
        filename: "clip.mp4",
        metadata: {} as never,
        timestamp: 2,
      },
      {
        filename: "loop.webp",
        metadata: {} as never,
        timestamp: 1,
      },
    ]);
    const w = mount(ImagePickerModal, {
      props: { open: true, multiple: false },
      attachTo: document.body,
    });
    await flushPromises();

    const input = document.body.querySelector(
      "input[type='file']",
    ) as HTMLInputElement;
    expect(input.accept).toBe("image/png,image/jpeg");
    Object.defineProperty(input, "files", {
      value: [new File(["x"], "clip.webp", { type: "image/webp" })],
    });
    Object.defineProperty(input, "value", {
      value: "C:\\fakepath\\clip.webp",
      writable: true,
    });
    input.dispatchEvent(new Event("change", { bubbles: true }));
    await flushPromises();
    expect(w.emitted("pick")).toBeUndefined();
    expect(document.body.textContent).toContain("Only PNG or JPEG");
    expect(input.value).toBe("");

    await w.setProps({ open: false });
    await w.setProps({ open: true });
    expect(document.body.textContent).not.toContain("Only PNG or JPEG");

    await w
      .get("[aria-label='Image source']")
      .findAll("button")[1]!
      .trigger("click");
    await flushPromises();
    expect(document.body.querySelectorAll(".ip__tile")).toHaveLength(1);
    expect(
      document.body.querySelector(".ip__tile")?.getAttribute("title"),
    ).toBe("still.png");
  });

  it("explains when the gallery has no compatible still images", async () => {
    vi.mocked(api.listGallery).mockResolvedValueOnce([
      {
        filename: "clip.mp4",
        metadata: {} as never,
        timestamp: 1,
      },
    ]);
    const w = mount(ImagePickerModal, {
      props: { open: true, multiple: false },
      attachTo: document.body,
    });
    await flushPromises();

    await w
      .get("[aria-label='Image source']")
      .findAll("button")[1]!
      .trigger("click");
    await flushPromises();

    expect(document.body.textContent).toContain(
      "no PNG or JPEG images available",
    );
  });

  it("is a labelled modal dialog", () => {
    mount(ImagePickerModal, {
      props: { open: true },
      attachTo: document.body,
    });

    const el = dialog();
    expect(el.getAttribute("aria-modal")).toBe("true");
    expect(el.getAttribute("aria-label")).toBe("Source image");
  });

  it("closes on Escape", async () => {
    const w = mount(ImagePickerModal, {
      props: { open: true },
      attachTo: document.body,
    });

    dialog().dispatchEvent(
      new KeyboardEvent("keydown", { key: "Escape", bubbles: true }),
    );
    await nextTick();

    expect(w.emitted("close")).toHaveLength(1);
  });

  it("closes on backdrop click but not on a click inside the panel", async () => {
    const w = mount(ImagePickerModal, {
      props: { open: true },
      attachTo: document.body,
    });

    const panel = document.body.querySelector(
      ".ms-modal__panel",
    ) as HTMLElement;
    panel.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    await nextTick();
    expect(w.emitted("close")).toBeUndefined();

    dialog().dispatchEvent(new MouseEvent("click", { bubbles: true }));
    await nextTick();
    expect(w.emitted("close")).toHaveLength(1);
  });

  it("moves focus into the panel on open and restores it to the opener on close", async () => {
    const Harness = defineComponent({
      setup() {
        const open = ref(false);
        return () => [
          h("button", {
            id: "opener",
            onClick: () => (open.value = true),
          }),
          h(ImagePickerModal, {
            open: open.value,
            onClose: () => (open.value = false),
          }),
        ];
      },
    });

    const w = mount(Harness, { attachTo: document.body });
    const opener = document.body.querySelector("#opener") as HTMLButtonElement;
    opener.focus();
    expect(document.activeElement).toBe(opener);

    await w.get("#opener").trigger("click");
    await flushPromises();
    await new Promise((resolve) => setTimeout(resolve, 0));

    const el = dialog();
    expect(el.contains(document.activeElement)).toBe(true);

    el.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Escape", bubbles: true }),
    );
    await flushPromises();

    expect(document.activeElement).toBe(opener);
    w.unmount();
  });

  it("traps Tab inside the panel", async () => {
    mount(ImagePickerModal, {
      props: { open: true },
      attachTo: document.body,
    });
    await flushPromises();

    const el = dialog();
    const tabbable = Array.from(
      el.querySelectorAll<HTMLElement>(
        'a[href],button:not([disabled]):not([tabindex="-1"]),input:not([disabled]):not([tabindex="-1"])',
      ),
    );
    expect(tabbable.length).toBeGreaterThan(1);
    const first = tabbable[0] as HTMLElement;
    const last = tabbable[tabbable.length - 1] as HTMLElement;

    last.focus();
    last.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Tab", bubbles: true }),
    );
    expect(document.activeElement).toBe(first);

    first.dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "Tab",
        bubbles: true,
        shiftKey: true,
      }),
    );
    expect(document.activeElement).toBe(last);
  });

  it("renders no emoji and no hard-coded hex colors", async () => {
    const w = mount(ImagePickerModal, {
      props: { open: true },
      attachTo: document.body,
    });
    await flushPromises();

    const html = w.html();
    expect(html).not.toMatch(EMOJI);
    expect(html).not.toMatch(/#[0-9a-fA-F]{3,8}\b/);
  });

  it("marks the drop target while a file is dragged over it", async () => {
    const w = mount(ImagePickerModal, {
      props: { open: true },
      attachTo: document.body,
    });

    const drop = w.get("[data-test='image-picker-drop']");
    expect(drop.attributes("data-dragging")).toBeUndefined();

    await drop.trigger("dragenter");
    expect(drop.attributes("data-dragging")).toBe("true");

    await drop.trigger("dragleave", { relatedTarget: document.body });
    expect(drop.attributes("data-dragging")).toBeUndefined();
  });
});
