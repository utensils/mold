import { flushPromises, mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ConnectModal from "./ConnectModal.vue";
import { listStoredHosts } from "../../lib/hostRegistry";
import type { HostStatus } from "./hostClient";

const hostStatus = vi.fn();

vi.mock("./hostClient", () => ({
  hostStatus: (...args: unknown[]) => hostStatus(...args),
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

beforeEach(() => {
  localStorage.clear();
  hostStatus.mockReset();
});

describe("ConnectModal", () => {
  it("starts on the type step with a disabled local-network card", () => {
    const w = mountModal();
    expect(w.find('[data-test="type-remote"]').exists()).toBe(true);
    const lan = w.get('[data-test="type-lan"]');
    expect((lan.element as HTMLButtonElement).disabled).toBe(true);
    expect(lan.text()).toContain("Browsers can't discover LAN hosts");
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
