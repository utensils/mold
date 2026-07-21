import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import ConnectModal from "./ConnectModal.vue";
import { listStoredHosts } from "../../lib/hostRegistry";
import type { HostStatus } from "./hostClient";

const hostStatus = vi.fn();
const hostCapabilities = vi.fn();
const hostDiscoveryPeers = vi.fn();

vi.mock("./hostClient", () => ({
  hostStatus: (...args: unknown[]) => hostStatus(...args),
  hostCapabilities: (...args: unknown[]) => hostCapabilities(...args),
  hostDiscoveryPeers: (...args: unknown[]) => hostDiscoveryPeers(...args),
}));

function okStatus(over: Partial<HostStatus> = {}): HostStatus {
  return {
    version: "0.16.0",
    models_loaded: [],
    busy: false,
    uptime_secs: 10,
    instance_id: "uuid-remote",
    ...over,
  };
}

const mountModal = () => mount(ConnectModal, { props: { open: true } });

async function advanceToDetails(w: ReturnType<typeof mountModal>) {
  await w.get('[data-test="connect-continue"]').trigger("click");
}

afterEach(() => {
  vi.useRealTimers();
});

beforeEach(() => {
  localStorage.clear();
  hostStatus.mockReset();
  hostCapabilities.mockReset();
  hostCapabilities.mockResolvedValue({ discovery: { can_browse: false } });
  hostDiscoveryPeers.mockReset();
  hostDiscoveryPeers.mockResolvedValue([]);
});

describe("ConnectModal", () => {
  it("keeps Local network disabled on older servers without the capability and never probes", async () => {
    const w = mountModal();
    await flushPromises();
    expect(w.find('[data-test="type-remote"]').exists()).toBe(true);
    const lan = w.get('[data-test="type-lan"]');
    expect((lan.element as HTMLButtonElement).disabled).toBe(true);
    expect(lan.text()).toContain("Browsers can't discover LAN hosts");
    expect(hostDiscoveryPeers).not.toHaveBeenCalled();
  });

  it("shows discovered peers, filters remembered instance UUIDs, and connects directly", async () => {
    localStorage.setItem(
      "mold.web.hosts.v1",
      JSON.stringify([
        {
          id: "remembered",
          name: "Remembered",
          url: "http://old-name.local:7680",
          instanceId: "existing-instance",
        },
      ]),
    );
    hostCapabilities.mockResolvedValue({ discovery: { can_browse: true } });
    hostDiscoveryPeers.mockResolvedValue([
      {
        name: "already-added-7680",
        url: "http://192.168.1.15:7680",
        host: "192.168.1.15",
        port: 7680,
        version: "0.20.2",
        auth_required: false,
        instance_id: "existing-instance",
        is_this_machine: false,
      },
      {
        name: "studio-7680",
        url: "http://192.168.1.20:7680",
        host: "192.168.1.20",
        port: 7680,
        version: "0.20.2",
        auth_required: false,
        instance_id: "studio-instance",
        is_this_machine: false,
      },
    ]);
    hostStatus.mockResolvedValue(okStatus({ instance_id: "studio-instance" }));

    const w = mountModal();
    await flushPromises();
    const lan = w.get('[data-test="type-lan"]');
    expect((lan.element as HTMLButtonElement).disabled).toBe(false);
    await lan.trigger("click");
    await flushPromises();

    expect(w.findAll('[data-test="discovery-peer"]')).toHaveLength(1);
    expect(w.text()).toContain("studio-7680");
    expect(w.text()).not.toContain("already-added-7680");
    expect(w.text()).toContain("this server's local network");

    await w.get('[data-test="discovery-peer-connect"]').trigger("click");
    await flushPromises();
    expect(hostStatus).toHaveBeenCalledWith(
      expect.objectContaining({ url: "http://192.168.1.20:7680" }),
    );
    expect(listStoredHosts().map((host) => host.instanceId)).toEqual([
      "existing-instance",
      "studio-instance",
    ]);
    expect(w.find('[data-test="connect-confirm"]').exists()).toBe(true);
  });

  it("prompts for a per-host key before connecting an authenticated discovery result", async () => {
    hostCapabilities.mockResolvedValue({ discovery: { can_browse: true } });
    hostDiscoveryPeers.mockResolvedValue([
      {
        name: "locked-7680",
        url: "http://locked.local:7680",
        host: "locked.local",
        port: 7680,
        version: null,
        auth_required: true,
        instance_id: "locked-instance",
        is_this_machine: false,
      },
    ]);
    hostStatus.mockResolvedValue(okStatus({ instance_id: "locked-instance" }));

    const w = mountModal();
    await flushPromises();
    await w.get('[data-test="type-lan"]').trigger("click");
    await flushPromises();
    await w.get('[data-test="discovery-peer-connect"]').trigger("click");
    expect(w.find('[data-test="discovery-key"]').exists()).toBe(true);
    expect(hostStatus).not.toHaveBeenCalled();

    await w.get('[data-test="discovery-key"]').setValue("peer-secret");
    await w.get('[data-test="connect-submit"]').trigger("click");
    await flushPromises();
    expect(hostStatus).toHaveBeenCalledWith(
      expect.objectContaining({ apiKey: "peer-secret" }),
    );
    expect(listStoredHosts()[0]?.url).not.toContain("peer-secret");
  });

  it("refreshes the server cache so a peer appears within a few seconds", async () => {
    vi.useFakeTimers();
    hostCapabilities.mockResolvedValue({ discovery: { can_browse: true } });
    hostDiscoveryPeers.mockResolvedValueOnce([]).mockResolvedValueOnce([
      {
        name: "late-peer-7680",
        url: "http://192.168.1.30:7680",
        host: "192.168.1.30",
        port: 7680,
        version: "0.20.2",
        auth_required: false,
        instance_id: "late-peer",
        is_this_machine: false,
      },
    ]);

    const w = mountModal();
    await flushPromises();
    await w.get('[data-test="type-lan"]').trigger("click");
    await flushPromises();
    expect(w.find('[data-test="discovery-peer"]').exists()).toBe(false);

    await vi.advanceTimersByTimeAsync(2000);
    await flushPromises();
    expect(w.get('[data-test="discovery-peer"]').text()).toContain(
      "late-peer-7680",
    );
    w.unmount();
    vi.useRealTimers();
  });

  it("shows an on-voice empty result after scanning", async () => {
    hostCapabilities.mockResolvedValue({ discovery: { can_browse: true } });
    const w = mountModal();
    await flushPromises();
    await w.get('[data-test="type-lan"]').trigger("click");
    await flushPromises();
    expect(w.get('[data-test="discovery-empty"]').text()).toContain(
      "No other mold machines found",
    );
  });

  it("advances type → details → back", async () => {
    const w = mountModal();
    await advanceToDetails(w);
    expect(w.find('[data-test="connect-address"]').exists()).toBe(true);
    await w.get('[data-test="connect-back"]').trigger("click");
    expect(w.find('[data-test="type-remote"]').exists()).toBe(true);
  });

  it("keeps the typed values and shows a blunt error when the probe fails", async () => {
    hostStatus.mockRejectedValue(new Error("network down"));
    const w = mountModal();
    await advanceToDetails(w);
    await w.get('[data-test="connect-address"]').setValue("192.168.1.20:7680");
    await w.get('[data-test="connect-name"]').setValue("Studio");
    await w.get('[data-test="connect-submit"]').trigger("click");
    await flushPromises();

    expect(w.find('[data-test="connect-error"]').exists()).toBe(true);
    // Input is preserved for a retry, and we stay on the details step.
    expect(
      (w.get('[data-test="connect-address"]').element as HTMLInputElement)
        .value,
    ).toBe("192.168.1.20:7680");
    expect(listStoredHosts()).toHaveLength(0);
  });

  it("surfaces an auth-specific error on 401", async () => {
    hostStatus.mockRejectedValue(new Error("GET /api/status failed: 401"));
    const w = mountModal();
    await advanceToDetails(w);
    await w.get('[data-test="connect-address"]').setValue("box.local");
    await w.get('[data-test="connect-submit"]').trigger("click");
    await flushPromises();
    expect(w.get('[data-test="connect-error"]').text()).toContain(
      "Authentication failed",
    );
  });

  it("probes, dedupes by instance id, stores the host, and confirms", async () => {
    hostStatus.mockResolvedValue(okStatus());
    const w = mountModal();
    await advanceToDetails(w);
    await w.get('[data-test="connect-address"]').setValue("192.168.1.20:7680");
    await w.get('[data-test="connect-name"]').setValue("Studio");
    await w.get('[data-test="connect-key"]').setValue("sekret");
    await w.get('[data-test="connect-submit"]').trigger("click");
    await flushPromises();

    // Probe used the normalized origin URL and carried the key on the entry.
    const probeArg = hostStatus.mock.calls[0]?.[0] as {
      url: string;
      apiKey?: string;
    };
    expect(probeArg.url).toBe("http://192.168.1.20:7680");
    expect(probeArg.apiKey).toBe("sekret");

    const stored = listStoredHosts();
    expect(stored).toHaveLength(1);
    expect(stored[0]?.name).toBe("Studio");
    expect(stored[0]?.instanceId).toBe("uuid-remote");
    expect(stored[0]?.url).not.toContain("sekret");

    // Confirmation step, then Done emits the added host.
    expect(w.find('[data-test="connect-confirm"]').exists()).toBe(true);
    await w.get('[data-test="connect-done"]').trigger("click");
    expect(w.emitted("added")).toBeTruthy();
    expect(w.emitted("close")).toBeTruthy();
  });

  it("rejects an empty address without probing", async () => {
    const w = mountModal();
    await advanceToDetails(w);
    await w.get('[data-test="connect-submit"]').trigger("click");
    await flushPromises();
    expect(w.find('[data-test="connect-error"]').exists()).toBe(true);
    expect(hostStatus).not.toHaveBeenCalled();
  });
});
