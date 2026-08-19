import { afterEach, describe, expect, it, vi } from "vitest";
import {
  copyTextToClipboard,
  notificationClipboardText,
} from "./notificationClipboard";

const entry = {
  text: "Generation failed",
  description: "Server error: out of memory on device 0",
  hostLabel: "plato",
};

afterEach(() => {
  vi.unstubAllGlobals();
  document.body.innerHTML = "";
});

describe("notificationClipboardText", () => {
  it("yields the message, the full body, and the origin line", () => {
    expect(notificationClipboardText(entry, "09:19 AM")).toBe(
      "Generation failed\nServer error: out of memory on device 0\nplato · 09:19 AM",
    );
  });

  it("omits parts the entry does not carry", () => {
    expect(
      notificationClipboardText(
        { text: "Queued", description: null, hostLabel: null },
        null,
      ),
    ).toBe("Queued");
    expect(
      notificationClipboardText({ ...entry, hostLabel: null }, "09:19 AM"),
    ).toBe(
      "Generation failed\nServer error: out of memory on device 0\n09:19 AM",
    );
  });
});

describe("copyTextToClipboard", () => {
  it("uses the async clipboard when it is available", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    vi.stubGlobal("navigator", { clipboard: { writeText } });
    expect(await copyTextToClipboard("hello")).toBe(true);
    expect(writeText).toHaveBeenCalledWith("hello");
  });

  it("falls back to execCommand when the clipboard API is missing or rejects", async () => {
    // An http LAN origin is an insecure context: navigator.clipboard is absent.
    vi.stubGlobal("navigator", {});
    const execCommand = vi.fn().mockReturnValue(true);
    Object.assign(document, { execCommand });
    expect(await copyTextToClipboard("hello")).toBe(true);
    expect(execCommand).toHaveBeenCalledWith("copy");
    // The staging textarea is always removed again.
    expect(document.querySelector("textarea")).toBeNull();

    vi.stubGlobal("navigator", {
      clipboard: { writeText: vi.fn().mockRejectedValue(new Error("denied")) },
    });
    expect(await copyTextToClipboard("hello")).toBe(true);
  });

  it("reports failure rather than pretending, and never copies nothing", async () => {
    vi.stubGlobal("navigator", {});
    Object.assign(document, { execCommand: vi.fn().mockReturnValue(false) });
    expect(await copyTextToClipboard("hello")).toBe(false);
    expect(await copyTextToClipboard("")).toBe(false);
  });
});
