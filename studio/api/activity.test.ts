import { describe, expect, it } from "vitest";
import {
  mergeFleetActivity,
  parseActiveWorkSnapshot,
  reconcileActivityHost,
} from "./activity";

const route = {
  hostId: "host-a",
  hostLabel: "Studio",
  target: { baseUrl: "http://studio:7680", apiKey: "key" },
};

describe("active work reconciliation", () => {
  it("parses a server snapshot and defaults additive cancellation", () => {
    expect(
      parseActiveWorkSnapshot({
        instance_id: "instance-a",
        observed_at_unix_ms: 10,
        items: [
          {
            id: "job-1",
            kind: "generation",
            phase: "running",
            created_at_unix_ms: 1,
            updated_at_unix_ms: 9,
          },
        ],
      }).items[0]?.can_cancel,
    ).toBe(false);
  });

  it("replaces successful snapshots so terminal jobs cannot linger or resurrect", () => {
    const first = reconcileActivityHost(route, undefined, {
      instance_id: "instance-a",
      observed_at_unix_ms: 10,
      items: [
        {
          id: "job-1",
          kind: "generation",
          phase: "running",
          created_at_unix_ms: 1,
          updated_at_unix_ms: 9,
          can_cancel: false,
        },
      ],
    });
    const terminal = reconcileActivityHost(route, first, {
      instance_id: "instance-a",
      observed_at_unix_ms: 11,
      items: [],
    });
    expect(terminal.items).toEqual([]);
  });

  it("accepts clock rollback and fences a changed server instance", () => {
    const previous = reconcileActivityHost(route, undefined, {
      instance_id: "instance-a",
      observed_at_unix_ms: 20,
      items: [
        {
          id: "old-work",
          kind: "generation",
          phase: "running",
          created_at_unix_ms: 1,
          updated_at_unix_ms: 19,
          can_cancel: false,
        },
      ],
    });
    const replacement = reconcileActivityHost(
      {
        ...route,
        target: { ...route.target, baseUrl: "http://replacement:7680" },
      },
      previous,
      {
        instance_id: "instance-b",
        observed_at_unix_ms: 1,
        items: [],
      },
    );
    expect(replacement).toMatchObject({
      instanceId: "instance-b",
      observedAtUnixMs: 1,
      items: [],
    });
  });

  it("retains the last verified snapshot while a host is offline", () => {
    const previous = reconcileActivityHost(route, undefined, {
      instance_id: "instance-a",
      observed_at_unix_ms: 10,
      items: [
        {
          id: "job-1",
          kind: "generation",
          phase: "queued",
          created_at_unix_ms: 1,
          updated_at_unix_ms: 9,
          can_cancel: true,
        },
      ],
    });
    const stale = reconcileActivityHost(route, previous, new Error("offline"));
    expect(stale.items).toEqual(previous.items);
    expect(stale.stale).toBe(true);
  });

  it("does not attach cached work to a changed route that is offline", () => {
    const previous = reconcileActivityHost(route, undefined, {
      instance_id: "instance-a",
      observed_at_unix_ms: 10,
      items: [
        {
          id: "old-work",
          kind: "generation",
          phase: "running",
          created_at_unix_ms: 1,
          updated_at_unix_ms: 9,
          can_cancel: false,
        },
      ],
    });
    const changed = reconcileActivityHost(
      {
        ...route,
        target: { ...route.target, baseUrl: "http://new-host:7680" },
      },
      previous,
      new Error("unreachable"),
    );
    expect(changed.items).toEqual([]);
    expect(changed.routeUrl).toBe("http://new-host:7680");
  });

  it("refreshes healthy kinds while retaining only an unavailable kind", () => {
    const previous = reconcileActivityHost(route, undefined, {
      instance_id: "instance-a",
      observed_at_unix_ms: 10,
      items: [
        {
          id: "old-generation",
          kind: "generation",
          phase: "running",
          created_at_unix_ms: 1,
          updated_at_unix_ms: 9,
          can_cancel: false,
        },
        {
          id: "sequence-a",
          kind: "sequence",
          phase: "running",
          created_at_unix_ms: 1,
          updated_at_unix_ms: 9,
          can_cancel: true,
        },
      ],
    });
    const partial = reconcileActivityHost(route, previous, {
      instance_id: "instance-a",
      observed_at_unix_ms: 11,
      items: [
        {
          id: "download-a",
          kind: "download",
          phase: "downloading",
          created_at_unix_ms: 2,
          updated_at_unix_ms: 11,
          can_cancel: true,
        },
      ],
      unavailable_kinds: ["sequence"],
    });
    expect(partial.items.map((item) => item.id)).toEqual([
      "download-a",
      "sequence-a",
    ]);
    const rows = mergeFleetActivity([partial]);
    expect(rows.find((row) => row.id === "sequence-a")).toMatchObject({
      stale: true,
      hostError: "sequence status is temporarily unavailable",
    });
    expect(rows.find((row) => row.id === "download-a")?.stale).toBe(false);
  });

  it("does not retain unavailable work across route or instance fences", () => {
    const previous = reconcileActivityHost(route, undefined, {
      instance_id: "instance-a",
      observed_at_unix_ms: 10,
      items: [
        {
          id: "sequence-a",
          kind: "sequence",
          phase: "running",
          created_at_unix_ms: 1,
          updated_at_unix_ms: 9,
          can_cancel: true,
        },
      ],
    });
    const partial = {
      instance_id: "instance-b",
      observed_at_unix_ms: 11,
      items: [],
      unavailable_kinds: ["sequence"],
    };

    expect(reconcileActivityHost(route, previous, partial).items).toEqual([]);
    expect(
      reconcileActivityHost(
        {
          ...route,
          target: { ...route.target, baseUrl: "http://replacement:7680" },
        },
        previous,
        { ...partial, instance_id: "instance-a" },
      ).items,
    ).toEqual([]);
  });

  it("keeps identical ids from different hosts as separate attributed rows", () => {
    const a = reconcileActivityHost(route, undefined, {
      instance_id: "instance-a",
      observed_at_unix_ms: 10,
      items: [
        {
          id: "same",
          kind: "generation",
          phase: "queued",
          created_at_unix_ms: 1,
          updated_at_unix_ms: 1,
          can_cancel: false,
        },
      ],
    });
    const b = reconcileActivityHost(
      { ...route, hostId: "host-b", hostLabel: "Render box" },
      undefined,
      {
        instance_id: "instance-b",
        observed_at_unix_ms: 10,
        items: [{ ...a.items[0]!, id: "same" }],
      },
    );
    expect(mergeFleetActivity([a, b]).map((row) => row.key)).toEqual([
      "host-a:generation:same",
      "host-b:generation:same",
    ]);
  });
});
