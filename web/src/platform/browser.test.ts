import { afterEach, describe, expect, it, vi } from "vitest";
import { browserPlatform } from "./browser";

describe("browserPlatform.saveMedia", () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("mounts the download anchor and defers object URL revocation", async () => {
    vi.useFakeTimers();
    const createObjectURL = vi.fn(() => "blob:mold-print");
    const revokeObjectURL = vi.fn();
    vi.stubGlobal("URL", { createObjectURL, revokeObjectURL });
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => undefined);

    await browserPlatform.saveMedia("print.png", new Uint8Array([1, 2, 3]));

    expect(click).toHaveBeenCalledOnce();
    expect(document.querySelector('a[download="print.png"]')).toBeNull();
    expect(revokeObjectURL).not.toHaveBeenCalled();

    await vi.runAllTimersAsync();
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:mold-print");
  });
});
