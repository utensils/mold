import { beforeEach, describe, expect, it, vi } from "vitest";
import { apiJson, apiJsonTo } from "./client";
import { fetchLoras } from "./loras";

vi.mock("./client", () => ({
  apiJson: vi.fn(),
  apiJsonTo: vi.fn(),
}));

describe("fetchLoras", () => {
  beforeEach(() => {
    vi.mocked(apiJson).mockReset().mockResolvedValue([]);
    vi.mocked(apiJsonTo).mockReset().mockResolvedValue([]);
  });

  it("keeps the primary-host path for desktop callers", async () => {
    await fetchLoras("flux:fp16");

    expect(apiJson).toHaveBeenCalledWith("/api/loras?model=flux%3Afp16");
    expect(apiJsonTo).not.toHaveBeenCalled();
  });

  it("directs the lookup to the selected remote host", async () => {
    const target = { baseUrl: "https://studio.tailnet.ts.net", apiKey: "secret" };

    await fetchLoras("flux:fp16", target);

    expect(apiJsonTo).toHaveBeenCalledWith(target, "/api/loras?model=flux%3Afp16");
    expect(apiJson).not.toHaveBeenCalled();
  });
});
