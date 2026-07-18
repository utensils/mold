import { beforeEach, describe, expect, it, vi } from "vitest";

const sseStream = vi.fn();
vi.mock("./sse", () => ({
  sseStream: (...args: unknown[]) => sseStream(...args),
}));

import { upscaleImage } from "./upscale";

interface CapturedOptions {
  method?: string;
  body?: unknown;
  target?: { baseUrl: string; apiKey: string | null };
  signal: AbortSignal;
  onEvent: (event: string, data: string) => void;
  onClose?: (error: Error | null) => void;
}

// Mirror the real sseStream: emit some frames, then keep the connection open
// until the caller aborts (or the signal was already aborted), at which point
// close cleanly with no error — exactly what fetchEventSource does on abort.
function openUntilAborted(
  emit: (options: CapturedOptions) => void,
): (path: string, options: CapturedOptions) => Promise<void> {
  return (_path, options) => {
    emit(options);
    return new Promise<void>((resolve) => {
      const finish = () => {
        options.onClose?.(null);
        resolve();
      };
      if (options.signal.aborted) finish();
      else options.signal.addEventListener("abort", finish, { once: true });
    });
  };
}

describe("upscaleImage", () => {
  // Block body on purpose: mockReset() returns the mock (a function), and a
  // function returned from beforeEach is invoked by vitest as teardown.
  beforeEach(() => {
    sseStream.mockReset();
  });

  it("POSTs /api/upscale/stream on the given host and resolves the completed image", async () => {
    sseStream.mockImplementation((_path: string, options: CapturedOptions) => {
      options.onEvent("progress", JSON.stringify({ type: "stage_start", name: "upscale" }));
      options.onEvent("complete", JSON.stringify({ image: "UPSCALED", format: "png" }));
      options.onClose?.(null);
      return Promise.resolve();
    });

    const target = { baseUrl: "http://gpu-box:7680", apiKey: "k" };
    const image = await upscaleImage({ model: "real-esrgan-x2plus:fp16", image: "SRC", target });

    expect(image).toBe("UPSCALED");
    const [path, options] = sseStream.mock.calls[0] as [string, CapturedOptions];
    expect(path).toBe("/api/upscale/stream");
    expect(options.method).toBe("POST");
    expect(options.target).toEqual(target);
    expect(options.body).toEqual({
      model: "real-esrgan-x2plus:fp16",
      image: "SRC",
      output_format: "png",
    });
  });

  it("rejects on a server error event", async () => {
    sseStream.mockImplementation((_path: string, options: CapturedOptions) => {
      options.onEvent("error", JSON.stringify({ error: "unknown upscaler model" }));
      options.onClose?.(null);
      return Promise.resolve();
    });

    await expect(upscaleImage({ model: "nope", image: "SRC" })).rejects.toThrow(
      "unknown upscaler model",
    );
  });

  it("rejects when the stream closes without a complete event", async () => {
    sseStream.mockImplementation((_path: string, options: CapturedOptions) => {
      options.onClose?.(new Error("connection reset"));
      return Promise.resolve();
    });

    await expect(upscaleImage({ model: "m", image: "SRC" })).rejects.toThrow("connection reset");
  });

  it("rejects promptly on an error frame even when the stream stays open", async () => {
    // Server reports an error but never closes the SSE stream. Without wiring
    // the AbortController, upscaleImage would hang until connection close.
    sseStream.mockImplementation(
      openUntilAborted((options) => {
        options.onEvent("error", JSON.stringify({ error: "boom" }));
      }),
    );

    await expect(upscaleImage({ model: "m", image: "SRC" })).rejects.toThrow("boom");
  });

  it("resolves on a complete frame even when the stream stays open", async () => {
    // The abort we fire after the terminal frame must not become a spurious
    // rejection — a clean close on an aborted stream still resolves the image.
    sseStream.mockImplementation(
      openUntilAborted((options) => {
        options.onEvent("complete", JSON.stringify({ image: "OK", format: "png" }));
      }),
    );

    await expect(upscaleImage({ model: "m", image: "SRC" })).resolves.toBe("OK");
  });
});
