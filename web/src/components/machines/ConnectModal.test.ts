import { enableAutoUnmount, flushPromises, mount } from "@vue/test-utils";
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

enableAutoUnmount(afterEach);

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
    expect(lan.text()).toContain(
      "This server does not advertise network discovery",
    );
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

  it("does not overlap slow discovery refreshes", async () => {
    vi.useFakeTimers();
    hostCapabilities.mockResolvedValue({ discovery: { can_browse: true } });
    let finishRefresh: ((peers: []) => void) | undefined;
    const slowRefresh = new Promise<[]>((resolve) => {
      finishRefresh = resolve;
    });
    hostDiscoveryPeers
      .mockResolvedValueOnce([])
      .mockReturnValueOnce(slowRefresh);

    const w = mountModal();
    await flushPromises();
    await w.get('[data-test="type-lan"]').trigger("click");
    await flushPromises();

    await vi.advanceTimersByTimeAsync(4000);
    expect(hostDiscoveryPeers).toHaveBeenCalledTimes(2);

    finishRefresh?.([]);
    await flushPromises();
    await vi.advanceTimersByTimeAsync(2000);
    expect(hostDiscoveryPeers).toHaveBeenCalledTimes(3);
    w.unmount();
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

  it("probes a bare IP through the default protocol and port", async () => {
    hostStatus.mockResolvedValue(okStatus());
    const w = mountModal();
    await advanceToDetails(w);
    await w.get('[data-test="connect-address"]').setValue("100.123.198.98");
    await w.get('[data-test="connect-submit"]').trigger("click");
    await flushPromises();

    expect(hostStatus.mock.calls[0]?.[0]).toMatchObject({
      url: "http://100.123.198.98:7680",
    });
  });

  it("labels an unnamed host with the server's hostname, never the raw URL", async () => {
    hostStatus.mockResolvedValue(okStatus({ hostname: "plato" }));
    const w = mountModal();
    await advanceToDetails(w);
    await w.get('[data-test="connect-address"]').setValue("192.168.1.20:7680");
    await w.get('[data-test="connect-submit"]').trigger("click");
    await flushPromises();

    expect(listStoredHosts()[0]?.name).toBe("plato");
  });

  it("rejects an empty address without probing", async () => {
    const w = mountModal();
    await advanceToDetails(w);
    await w.get('[data-test="connect-submit"]').trigger("click");
    await flushPromises();
    expect(w.find('[data-test="connect-error"]').exists()).toBe(true);
    expect(hostStatus).not.toHaveBeenCalled();
  });
  it.each(["back", "close", "unmount"])(
    "refuses a late successful probe after %s",
    async (exit) => {
      let finish!: (value: HostStatus) => void;
      hostStatus.mockReturnValue(
        new Promise<HostStatus>((resolve) => {
          finish = resolve;
        }),
      );
      const w = mountModal();
      await advanceToDetails(w);
      await w
        .get('[data-test="connect-address"]')
        .setValue("192.168.1.20:7680");
      await w.get('[data-test="connect-submit"]').trigger("click");
      if (exit === "back")
        await w.get('[data-test="connect-back"]').trigger("click");
      else if (exit === "close") await w.setProps({ open: false });
      else w.unmount();
      finish(okStatus());
      await flushPromises();
      expect(listStoredHosts()).toEqual([]);
      expect(w.emitted("added")).toBeUndefined();
    },
  );

  it("rejects duplicate Enter submissions and saves exactly the tested key", async () => {
    let finish!: (value: HostStatus) => void;
    hostStatus.mockReturnValue(
      new Promise<HostStatus>((resolve) => {
        finish = resolve;
      }),
    );
    const w = mountModal();
    await advanceToDetails(w);
    await w.get('[data-test="connect-address"]').setValue("192.168.1.20:7680");
    const key = w.get('[data-test="connect-key"]');
    await key.setValue("tested-key");
    await key.trigger("keydown", { key: "Enter" });
    await key.trigger("keydown", { key: "Enter" });
    expect(hostStatus).toHaveBeenCalledTimes(1);
    expect(key.attributes("disabled")).toBeDefined();
    // A programmatic edit must not change the credentials committed by this probe.
    await key.setValue("changed-key");
    finish(okStatus());
    await flushPromises();
    expect(listStoredHosts()[0]?.apiKey).toBe("tested-key");
  });

  it("does not repopulate or restart discovery after Back", async () => {
    vi.useFakeTimers();
    hostCapabilities.mockResolvedValue({ discovery: { can_browse: true } });
    let finish!: (value: unknown[]) => void;
    hostDiscoveryPeers.mockReturnValue(
      new Promise((resolve) => {
        finish = resolve;
      }),
    );
    const w = mountModal();
    await flushPromises();
    await w.get('[data-test="type-lan"]').trigger("click");
    expect(w.find('[data-test="discovery-scanning"]').exists()).toBe(true);
    expect(w.find('[data-test="connect-done"]').exists()).toBe(false);
    await w.get('[data-test="connect-back"]').trigger("click");
    finish([]);
    await flushPromises();
    await vi.advanceTimersByTimeAsync(6000);
    expect(hostDiscoveryPeers).toHaveBeenCalledTimes(1);
    expect(w.find('[data-test="discovery-empty"]').exists()).toBe(false);
  });

  it("retains authentication errors when discovery refresh succeeds", async () => {
    vi.useFakeTimers();
    hostCapabilities.mockResolvedValue({ discovery: { can_browse: true } });
    hostDiscoveryPeers.mockResolvedValue([
      {
        name: "Studio",
        url: "http://studio:7680",
        host: "studio",
        port: 7680,
        auth_required: true,
        instance_id: "studio",
        is_this_machine: false,
      },
    ]);
    hostStatus.mockRejectedValue(new Error("GET /api/status failed: 401"));
    const w = mountModal();
    await flushPromises();
    await w.get('[data-test="type-lan"]').trigger("click");
    await flushPromises();
    await w.get('[data-test="discovery-peer-connect"]').trigger("click");
    await w.get('[data-test="discovery-key"]').setValue("wrong");
    await w.get('[data-test="connect-submit"]').trigger("click");
    await flushPromises();
    await vi.advanceTimersByTimeAsync(2000);
    expect(w.get('[data-test="connect-error"]').text()).toContain(
      "Authentication failed",
    );
  });
  it("offers explicit discovery retry without presenting failure as empty", async () => {
    hostCapabilities.mockResolvedValue({ discovery: { can_browse: true } });
    hostDiscoveryPeers
      .mockRejectedValueOnce(new Error("unreachable"))
      .mockResolvedValue([]);
    const w = mountModal();
    await flushPromises();
    await w.get('[data-test="type-lan"]').trigger("click");
    await flushPromises();
    expect(w.get('[data-test="discovery-error"]').text()).toContain(
      "Couldn't scan",
    );
    expect(w.find('[data-test="discovery-empty"]').exists()).toBe(false);
    await w.get('[data-test="discovery-retry"]').trigger("click");
    await flushPromises();
    expect(w.find('[data-test="discovery-error"]').exists()).toBe(false);
    expect(w.find('[data-test="discovery-empty"]').exists()).toBe(true);
    expect(w.find('[data-test="connect-done"]').exists()).toBe(false);
  });

  it("keeps a newer probe busy when an obsolete probe completes", async () => {
    let finishOld!: (value: HostStatus) => void;
    let finishNew!: (value: HostStatus) => void;
    hostStatus
      .mockReturnValueOnce(
        new Promise<HostStatus>((resolve) => {
          finishOld = resolve;
        }),
      )
      .mockReturnValueOnce(
        new Promise<HostStatus>((resolve) => {
          finishNew = resolve;
        }),
      );
    const w = mountModal();
    await advanceToDetails(w);
    await w.get('[data-test="connect-address"]').setValue("old.test");
    await w.get('[data-test="connect-submit"]').trigger("click");
    await w.get('[data-test="connect-back"]').trigger("click");
    await advanceToDetails(w);
    await w.get('[data-test="connect-address"]').setValue("new.test");
    await w.get('[data-test="connect-submit"]').trigger("click");
    finishOld(okStatus());
    await flushPromises();
    expect(
      w.get('[data-test="connect-submit"]').attributes("disabled"),
    ).toBeDefined();
    expect(listStoredHosts()).toEqual([]);
    finishNew(okStatus());
    await flushPromises();
    expect(listStoredHosts()[0]?.url).toBe("http://new.test:7680");
  });
});
