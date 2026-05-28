import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { DOMWrapper, mount } from "@vue/test-utils";
import { nextTick } from "vue";
import MaskEditorModal from "./MaskEditorModal.vue";
import type { SourceImageState } from "../types";

const sourceImage: SourceImageState = {
  kind: "upload",
  filename: "source.png",
  base64: "SOURCE_BYTES",
};

function makeContext() {
  return {
    canvas: null,
    clearRect: vi.fn(),
    drawImage: vi.fn(),
    getImageData: vi.fn(() => ({ data: new Uint8ClampedArray(4) })),
    putImageData: vi.fn(),
    beginPath: vi.fn(),
    arc: vi.fn(),
    fill: vi.fn(),
    save: vi.fn(),
    restore: vi.fn(),
    globalCompositeOperation: "source-over",
    fillStyle: "",
  };
}

describe("MaskEditorModal", () => {
  let context: ReturnType<typeof makeContext>;

  function bodyGet<T extends Element>(selector: string): DOMWrapper<T> {
    const element = document.body.querySelector<T>(selector);
    expect(element).not.toBeNull();
    return new DOMWrapper(element as T);
  }

  beforeEach(() => {
    context = makeContext();
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(
      context as unknown as CanvasRenderingContext2D,
    );
    vi.spyOn(HTMLCanvasElement.prototype, "toDataURL").mockReturnValue(
      "data:image/png;base64,EDITED_MASK",
    );
    vi.spyOn(
      HTMLCanvasElement.prototype,
      "getBoundingClientRect",
    ).mockReturnValue({
      left: 0,
      top: 0,
      width: 512,
      height: 512,
      right: 512,
      bottom: 512,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
    document.body.innerHTML = "";
  });

  it("draws on the mask canvas and applies a base64 mask image", async () => {
    const wrapper = mount(MaskEditorModal, {
      props: { open: true, sourceImage },
      attachTo: document.body,
    });
    await nextTick();

    const canvas = bodyGet<HTMLCanvasElement>("[data-test='mask-canvas']");
    await canvas.trigger("pointerdown", { clientX: 64, clientY: 80 });
    await canvas.trigger("pointermove", { clientX: 96, clientY: 112 });
    await canvas.trigger("pointerup");
    await bodyGet<HTMLButtonElement>("[data-test='mask-apply']").trigger(
      "click",
    );

    expect(context.arc).toHaveBeenCalled();
    expect(wrapper.emitted("apply")?.[0]?.[0]).toEqual({
      kind: "upload",
      filename: "source-mask.png",
      base64: "EDITED_MASK",
    });
  });

  it("clears the mask canvas and supports undo and redo controls", async () => {
    const wrapper = mount(MaskEditorModal, {
      props: { open: true, sourceImage },
      attachTo: document.body,
    });
    await nextTick();

    await bodyGet<HTMLButtonElement>("[data-test='mask-clear']").trigger(
      "click",
    );
    await nextTick();
    await bodyGet<HTMLButtonElement>("[data-test='mask-undo']").trigger(
      "click",
    );
    await nextTick();
    await bodyGet<HTMLButtonElement>("[data-test='mask-redo']").trigger(
      "click",
    );

    expect(context.clearRect).toHaveBeenCalled();
    expect(context.getImageData).toHaveBeenCalled();
    expect(context.putImageData).toHaveBeenCalled();
  });

  it("switches between brush and erase modes and updates brush size", async () => {
    const wrapper = mount(MaskEditorModal, {
      props: { open: true, sourceImage },
      attachTo: document.body,
    });
    await nextTick();

    await bodyGet<HTMLButtonElement>("[data-test='mask-mode-erase']").trigger(
      "click",
    );
    await bodyGet<HTMLInputElement>("[data-test='mask-brush-size']").setValue(
      "48",
    );
    await bodyGet<HTMLCanvasElement>("[data-test='mask-canvas']").trigger(
      "pointerdown",
      { clientX: 32, clientY: 32 },
    );

    expect(context.globalCompositeOperation).toBe("destination-out");
    expect(context.arc).toHaveBeenCalledWith(
      expect.any(Number),
      expect.any(Number),
      24,
      0,
      Math.PI * 2,
    );
  });
});
