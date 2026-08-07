import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { open, invoke } = vi.hoisted(() => ({ open: vi.fn(), invoke: vi.fn() }));
vi.mock("@tauri-apps/plugin-dialog", () => ({ open }));

import { ipc } from "./ipc";

describe("ipc.pickSourceImages", () => {
  beforeEach(() => {
    Object.defineProperty(window, "__TAURI_INTERNALS__", {
      value: { invoke },
      configurable: true,
    });
    open.mockReset();
    invoke.mockReset();
  });

  afterEach(() => {
    delete (window as Window & { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__;
  });

  it("opens the native chooser with only supported still-image filters", async () => {
    open.mockResolvedValue(["/tmp/source.png", "/tmp/reference.jpeg"]);
    invoke
      .mockResolvedValueOnce({
        filename: "source.png",
        base64: "aA==",
        width: 64,
        height: 64,
        metadata: null,
      })
      .mockResolvedValueOnce({
        filename: "reference.jpeg",
        base64: "aQ==",
        width: 80,
        height: 96,
        metadata: null,
      });

    await expect(ipc.pickSourceImages(true)).resolves.toEqual([
      {
        filename: "source.png",
        base64: "aA==",
        width: 64,
        height: 64,
        metadata: null,
      },
      {
        filename: "reference.jpeg",
        base64: "aQ==",
        width: 80,
        height: 96,
        metadata: null,
      },
    ]);
    expect(open).toHaveBeenCalledWith({
      title: "Choose image",
      multiple: true,
      filters: [{ name: "PNG or JPEG images", extensions: ["png", "jpg", "jpeg"] }],
    });
    expect(invoke).toHaveBeenNthCalledWith(
      1,
      "import_source_image",
      { path: "/tmp/source.png" },
      undefined,
    );
    expect(invoke).toHaveBeenNthCalledWith(
      2,
      "import_source_image",
      {
        path: "/tmp/reference.jpeg",
      },
      undefined,
    );
  });

  it("does not invoke the importer when the native chooser is cancelled", async () => {
    open.mockResolvedValue(null);

    await expect(ipc.pickSourceImages(false)).resolves.toBeNull();
    expect(invoke).not.toHaveBeenCalled();
  });
});

describe("ipc.saveMediaBytes browser fallback", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    Object.defineProperty(URL, "createObjectURL", {
      value: vi.fn(() => "blob:rendered-image"),
      configurable: true,
    });
    Object.defineProperty(URL, "revokeObjectURL", {
      value: vi.fn(),
      configurable: true,
    });
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("rejects malformed base64 instead of throwing before returning a promise", async () => {
    const result = ipc.saveMediaBytes("broken.png", "%%%");
    expect(result).toBeInstanceOf(Promise);
    await expect(result).rejects.toThrow();
  });

  it("defers object URL cleanup until after the download click", async () => {
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});

    await expect(ipc.saveMediaBytes("render.png", "aA==")).resolves.toEqual({
      filename: "render.png",
      path: "render.png",
      directory: "Downloads",
    });
    expect(click).toHaveBeenCalledOnce();
    expect(URL.revokeObjectURL).not.toHaveBeenCalled();

    vi.runAllTimers();
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:rendered-image");
  });
});
