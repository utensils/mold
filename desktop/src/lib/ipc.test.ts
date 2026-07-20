import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const open = vi.fn();
const invoke = vi.fn();
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
      .mockResolvedValueOnce({ filename: "source.png", base64: "aA==", metadata: null })
      .mockResolvedValueOnce({ filename: "reference.jpeg", base64: "aQ==", metadata: null });

    await expect(ipc.pickSourceImages(true)).resolves.toEqual([
      { filename: "source.png", base64: "aA==", metadata: null },
      { filename: "reference.jpeg", base64: "aQ==", metadata: null },
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
