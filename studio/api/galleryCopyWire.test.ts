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
const chainSlots = [
  ["stage_source:0", "authored-source"],
  ["stage_source:4294967295", "authored-source"],
  ["stage_reference:2:0", "authored-reference"],
  ["stage_keyframe:0:1", "authored-keyframe"],
  ["chain_continuation_tail:1", "continuation-tail"],
];
function retainedOffer(role: string, position: string, sink = "memory") {
  return {
    archive_identity_sha256: "a".repeat(64),
    output_sha256: "b".repeat(64),
    output_size_bytes: 3,
    metadata: {},
    members: [
      {
        member_id: "member",
        role,
        position,
        sink,
        size_bytes: 1,
        sha256: "c".repeat(64),
      },
    ],
  };
}
it.each(chainSlots)(
  "accepts the server-owned %s role with exact %s position",
  async (role, position) => {
    api.fetch.mockReset();
    api.fetch.mockResolvedValue(
      new Response(JSON.stringify(retainedOffer(role!, position!))),
    );
    expect((await offer(machine, "movie.mp4")).members).toHaveLength(1);
  },
);
it.each([
  ["stage_source:01", "authored-source"],
  ["stage_source:4294967296", "authored-source"],
  ["stage_source:-1", "authored-source"],
  ["stage_reference:0:01", "authored-reference"],
  ["stage_reference:0:4294967296", "authored-reference"],
  ["stage_reference:1:2:3", "authored-reference"],
  ["stage_keyframe:0", "authored-keyframe"],
  ["chain_continuation_tail:01", "continuation-tail"],
  ["stage_source:0", "scalar"],
  ["stage_reference:0:0", "scalar"],
  ["stage_keyframe:0:0", "authored-reference"],
  ["chain_continuation_tail:0", "authored-source"],
])("refuses noncanonical or mismatched %s/%s", async (role, position) => {
  api.fetch.mockReset();
  api.fetch.mockResolvedValue(
    new Response(JSON.stringify(retainedOffer(role!, position!))),
  );
  await expect(offer(machine, "movie.mp4")).rejects.toThrow("invalid");
});
it("refuses duplicate authored slots and unsupported sinks", async () => {
  const value = retainedOffer("stage_reference:0:0", "authored-reference");
  value.members.push({ ...value.members[0]!, member_id: "different" });
  api.fetch.mockReset();
  api.fetch.mockResolvedValue(new Response(JSON.stringify(value)));
  await expect(offer(machine, "movie.mp4")).rejects.toThrow("invalid");
  api.fetch.mockResolvedValue(
    new Response(
      JSON.stringify(
        retainedOffer("stage_source:0", "authored-source", "/private/path"),
      ),
    ),
  );
  await expect(offer(machine, "movie.mp4")).rejects.toThrow("invalid");
});
