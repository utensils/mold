import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import ImagePickerModal from "./ImagePickerModal.vue";

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
});
