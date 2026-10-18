import { expect, it, vi } from "vitest";
const api = vi.hoisted(() => ({ fetch: vi.fn() }));
vi.mock("./client", async (original) => ({
  ...(await original<typeof import("./client")>()),
  apiFetchTo: api.fetch,
}));
import { offer } from "./galleryCopyWire";
const machine = {
  id: "source",
  label: "Source",
  instanceId: "s1",
  target: { baseUrl: "https://source", apiKey: null },
};
it.each([64, 512])(
  "names the %i MiB copy limit before downloading an oversized output",
  async (mib) => {
    api.fetch.mockReset();
    api.fetch.mockResolvedValue(
      new Response(
        JSON.stringify({
          archive_identity_sha256: "a".repeat(64),
          output_sha256: "b".repeat(64),
          output_size_bytes: (mib + 1) * 1024 * 1024,
          metadata: {},
          members: [],
        }),
      ),
    );
    await expect(
      offer(machine, "large.png", undefined, mib * 1024 * 1024),
    ).rejects.toThrow(`${mib} MiB`);
    expect(api.fetch).toHaveBeenCalledTimes(1);
    expect(api.fetch.mock.calls[0]![1]).toBe(
      "/api/gallery/source-media/large.png/transfer",
    );
  },
);
