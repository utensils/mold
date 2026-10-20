import { beforeEach, describe, expect, it, vi } from "vitest";
const json = vi.hoisted(() => vi.fn());
vi.mock("./client", async (original) => ({
  ...(await original<typeof import("./client")>()),
  apiJsonTo: json,
}));
import {
  admitClips,
  sameChainOwner,
  validateClips,
  definiteChainRefusal,
  mutateChain,
} from "./chainJobs";
import { ApiError } from "./client";
const owner = {
  hostId: "plato",
  label: "Plato",
  instanceId: "original",
  target: { baseUrl: "http://plato", apiKey: "secret" },
};
const request = {
  model: "test",
  width: 384,
  height: 256,
  steps: 8,
  guidance: 1,
  stages: [{ prompt: "test", frames: 97 }],
};
beforeEach(() => {
  json.mockReset();
});
describe("durable clip admission", () => {
  it("keeps ambiguous timeout, conflict and rate-limit responses recoverable", () => {
    for (const status of [408, 409, 425, 429, 499, 500])
      expect(definiteChainRefusal(new ApiError("uncertain", status))).toBe(
        false,
      );
    expect(definiteChainRefusal(new ApiError("invalid clip", 422))).toBe(true);
  });
  it("rechecks the route after asynchronous instance verification before resuming", async () => {
    json.mockResolvedValue({ instance_id: "original" });
    let current = true;
    const operation = mutateChain(owner, "same-job", "resume", () => current);
    current = false;
    await expect(operation).rejects.toThrow("machine changed");
    expect(json).toHaveBeenCalledTimes(1);
  });
  it("persists the exact operation and safe owner before POST, retaining recovery after a lost reply", async () => {
    const order: string[] = [];
    json.mockImplementation(async (_, path) => {
      if (path === "/api/status") return { instance_id: "original" };
      order.push("post");
      throw new Error("lost reply");
    });
    let saved: unknown;
    await expect(
      admitClips(
        owner,
        request,
        (record) => {
          order.push("persist");
          saved = record;
        },
        "operation",
      ),
    ).rejects.toThrow("lost reply");
    expect(order).toEqual(["persist", "post"]);
    expect(JSON.stringify(saved)).not.toContain("secret");
    expect(saved).toMatchObject({
      operationId: "operation",
      instanceId: "original",
    });
    expect(json.mock.calls[1]?.[2].headers["x-mold-operation-id"]).toBe(
      "operation",
    );
  });
  it("never dispatches if the installation or journal changed", async () => {
    json.mockResolvedValue({ instance_id: "replacement" });
    await expect(admitClips(owner, request, vi.fn())).rejects.toThrow(
      "installation changed",
    );
    expect(json).toHaveBeenCalledTimes(1);
    json.mockClear().mockResolvedValue({ instance_id: "original" });
    await expect(
      admitClips(owner, request, () => {
        throw new Error("storage full");
      }),
    ).rejects.toThrow("storage full");
    expect(json).toHaveBeenCalledTimes(1);
  });
  it("does not adopt another endpoint or installation on recovery", () => {
    const record = {
      operationId: "id",
      hostId: "plato",
      instanceId: "original",
      baseUrl: "http://plato",
      createdAt: 1,
    };
    expect(sameChainOwner(record, owner)).toBe(true);
    expect(
      sameChainOwner(record, { ...owner, instanceId: "replacement" }),
    ).toBe(false);
    expect(
      sameChainOwner(record, {
        ...owner,
        target: { ...owner.target, baseUrl: "http://hal" },
      }),
    ).toBe(false);
  });
  it("protects literal backslashes on validate and admission wire copies without changing the draft", async () => {
    json.mockImplementation(async (_, path) =>
      path === "/api/status"
        ? { instance_id: "original" }
        : { job_id: "operation" },
    );
    const literal = String.raw`Show C:\new\render and literal \n`;
    const body = {
      ...request,
      original_prompt: literal,
      stages: [{ prompt: literal, negative_prompt: literal, frames: 97 }],
    };
    await validateClips(owner, body);
    await admitClips(owner, body, () => {}, "operation");
    const writes = json.mock.calls.filter((call) => call[2]?.method === "POST");
    for (const call of writes) {
      const wire = JSON.parse(call[2].body);
      expect(wire.stages[0].prompt).toBe(literal.replaceAll("\\", "\\\\"));
      expect(wire.stages[0].negative_prompt).toBe(
        literal.replaceAll("\\", "\\\\"),
      );
      expect(wire.original_prompt).toBe(literal.replaceAll("\\", "\\\\"));
    }
    expect(body.stages[0]?.prompt).toBe(literal);
  });
});
