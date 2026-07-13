import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { openExternal } from "./openExternal";
import { inTauri } from "./ipc";

vi.mock("./ipc", () => ({ inTauri: vi.fn(() => false) }));

const { openUrlMock } = vi.hoisted(() => ({ openUrlMock: vi.fn() }));
vi.mock("@tauri-apps/plugin-opener", () => ({ openUrl: openUrlMock }));

const inTauriMock = vi.mocked(inTauri);

let openSpy: ReturnType<typeof vi.spyOn>;

beforeEach(() => {
  inTauriMock.mockReturnValue(false);
  openUrlMock.mockReset();
  openSpy = vi.spyOn(window, "open").mockImplementation(() => null);
});

afterEach(() => {
  openSpy.mockRestore();
});

describe("openExternal", () => {
  it("opens http and https URLs in a browser tab outside Tauri", async () => {
    await openExternal("https://huggingface.co/black-forest-labs/FLUX.1-dev");
    await openExternal("http://example.com/page");
    expect(openSpy).toHaveBeenCalledTimes(2);
    expect(openSpy).toHaveBeenCalledWith(
      "https://huggingface.co/black-forest-labs/FLUX.1-dev",
      "_blank",
      "noopener",
    );
  });

  it("routes https URLs through the Tauri opener inside the app", async () => {
    inTauriMock.mockReturnValue(true);
    await openExternal("https://civitai.com/models/9001?modelVersionId=8001");
    expect(openUrlMock).toHaveBeenCalledWith("https://civitai.com/models/9001?modelVersionId=8001");
    expect(openSpy).not.toHaveBeenCalled();
  });

  it.each([
    "file:///etc/passwd",
    // eslint-disable-next-line no-script-url
    "javascript:alert(1)",
    "vscode://malicious/payload",
    "mold-custom://boom",
  ])("refuses non-web scheme %s — server-supplied URLs are untrusted", async (url) => {
    await openExternal(url);
    inTauriMock.mockReturnValue(true);
    await openExternal(url);
    expect(openSpy).not.toHaveBeenCalled();
    expect(openUrlMock).not.toHaveBeenCalled();
  });

  it.each(["", "not a url", "//protocol-relative.example", "huggingface.co/no-scheme"])(
    "refuses unparseable or scheme-less input %j",
    async (url) => {
      await openExternal(url);
      expect(openSpy).not.toHaveBeenCalled();
      expect(openUrlMock).not.toHaveBeenCalled();
    },
  );
});
