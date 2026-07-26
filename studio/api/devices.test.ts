import { afterEach, describe, expect, it, vi } from "vitest";
import { IncompatibleHostError } from "./client";
import { listDevices, setDeviceEnabled } from "./devices";

afterEach(() => vi.unstubAllGlobals());

describe("device API", () => {
  it("fetches every device through the explicit target without leaking the API key", async () => {
    let captured: [RequestInfo | URL, RequestInit | undefined] | null = null;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        captured = [input, init];
        return Response.json({
          devices: [
            {
              id: "cuda:0123456789abcdef0123456789abcdef",
              backend: "cuda",
              ordinal: 0,
              device_kind: "full_gpu",
              nvml_uuid: "GPU-first",
              physical_uuid: "GPU-first",
              mig_uuid: null,
              mig_parent_uuid: null,
              mig_profile: null,
              name: "RTX 3090",
              pci_bus_id: "00000000:01:00.0",
              compute_capability: "8.6",
              memory: {
                total_bytes: 24_576_000_000,
                used_bytes: 1_024,
                mold_used_bytes: null,
                other_used_bytes: null,
              },
              telemetry: {
                utilization_percent: 50,
                temperature_c: null,
                power_w: null,
              },
              desired_enabled: true,
              admin_state: "enabled",
              health: "healthy",
              activity: "generating",
              schedulable: false,
              unschedulable_reason: "active_lease",
              loaded_models: ["flux-dev:q8"],
              active_work_id: "job-1",
              planned_work_ids: ["job-2"],
            },
            {
              id: "cuda:fedcba9876543210fedcba9876543210",
              backend: "cuda",
              ordinal: 1,
              device_kind: "unknown_cuda",
              nvml_uuid: null,
              physical_uuid: null,
              mig_uuid: null,
              mig_parent_uuid: null,
              mig_profile: null,
              name: "CUDA device",
              pci_bus_id: null,
              compute_capability: null,
              memory: {
                total_bytes: null,
                used_bytes: null,
                mold_used_bytes: null,
                other_used_bytes: null,
              },
              telemetry: {
                utilization_percent: null,
                temperature_c: null,
                power_w: null,
              },
              desired_enabled: true,
              admin_state: "enabled",
              health: "healthy",
              activity: "idle",
              schedulable: true,
              unschedulable_reason: null,
              loaded_models: [],
              active_work_id: null,
              planned_work_ids: [],
            },
          ],
          plan_version: 42,
        });
      }),
    );

    const result = await listDevices({
      baseUrl: "https://beast.tailnet.example",
      apiKey: "durable-secret",
    });

    expect(result.devices).toHaveLength(2);
    expect(result.devices.map((device) => device.id)).toEqual([
      "cuda:0123456789abcdef0123456789abcdef",
      "cuda:fedcba9876543210fedcba9876543210",
    ]);
    expect(result.plan_version).toBe(42);
    const [url, init] = captured!;
    expect(url).toBe("https://beast.tailnet.example/api/devices");
    expect(String(url)).not.toContain("durable-secret");
    expect((init?.headers as Headers).get("x-api-key")).toBe("durable-secret");
  });

  it("rejects a host whose devices response lacks the additive contract", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => Response.json({ devices: [{ ordinal: 0 }] })),
    );

    await expect(
      listDevices({ baseUrl: "http://old-host:7680", apiKey: null }),
    ).rejects.toBeInstanceOf(IncompatibleHostError);
  });

  it("patches the URL-encoded stable ID through the explicit target and API key", async () => {
    let captured: [RequestInfo | URL, RequestInit | undefined] | null = null;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        captured = [input, init];
        return Response.json({
          id: "cuda:0123456789abcdef0123456789abcdef",
          backend: "cuda",
          ordinal: 0,
          device_kind: "full_gpu",
          nvml_uuid: null,
          physical_uuid: null,
          mig_uuid: null,
          mig_parent_uuid: null,
          mig_profile: null,
          name: "RTX 3090",
          pci_bus_id: null,
          compute_capability: "8.6",
          memory: {
            total_bytes: 24_576_000_000,
            used_bytes: 0,
            mold_used_bytes: null,
            other_used_bytes: null,
          },
          telemetry: {
            utilization_percent: 0,
            temperature_c: null,
            power_w: null,
          },
          desired_enabled: false,
          admin_state: "draining",
          health: "healthy",
          activity: "generating",
          schedulable: false,
          unschedulable_reason: "device_draining",
          loaded_models: [],
          active_work_id: "job-1",
          planned_work_ids: [],
        });
      }),
    );

    const updated = await setDeviceEnabled(
      {
        baseUrl: "https://beast.tailnet.example",
        apiKey: "durable-secret",
      },
      "cuda:0123456789abcdef0123456789abcdef",
      false,
    );

    expect(updated.admin_state).toBe("draining");
    const [url, init] = captured!;
    expect(url).toBe(
      "https://beast.tailnet.example/api/devices/cuda%3A0123456789abcdef0123456789abcdef",
    );
    expect(init?.method).toBe("PATCH");
    expect((init?.headers as Headers).get("x-api-key")).toBe("durable-secret");
    expect(init?.body).toBe(JSON.stringify({ enabled: false }));
  });
});
