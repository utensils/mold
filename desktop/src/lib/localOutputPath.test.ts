import { beforeEach, describe, expect, it, vi } from "vitest";
import { ipc } from "./ipc";
import { copyLocalOutputPath } from "./localOutputPath";

vi.mock("./ipc", () => ({
  ipc: {
    localOutputFilePath: vi.fn(),
  },
}));

describe("copyLocalOutputPath", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("reports unavailable local files without assuming the print is remote", async () => {
    vi.mocked(ipc.localOutputFilePath).mockResolvedValue(null);

    await expect(copyLocalOutputPath("missing.png")).rejects.toThrow(
      "This print is not available as a local file.",
    );
  });
});
