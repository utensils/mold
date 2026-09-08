import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ref } from "vue";
import type { HostRouting } from "./useHostRouting";

const listActiveWork = vi.hoisted(() => vi.fn());
vi.mock("@studio/api/activity", async (original) => ({
  ...(await original<typeof import("@studio/api/activity")>()),
  listActiveWork,
}));

function routing(): HostRouting {
  return {
    hosts: ref([
      { id: "box", label: "Box", url: "http://box:7680", status: "ready" },
    ]),
  } as unknown as HostRouting;
}
function snapshot(id = "work") {
  return {
    instance_id: "instance",
    observed_at_unix_ms: 10,
    items: [
      {
        id,
        kind: "generation",
        phase: "running",
        created_at_unix_ms: 1,
        updated_at_unix_ms: 10,
        can_cancel: true,
      },
    ],
  };
}

describe("shell-owned live activity", () => {
  beforeEach(() => {
    vi.resetModules();
    vi.useFakeTimers();
    localStorage.clear();
    listActiveWork.mockReset().mockResolvedValue(snapshot());
  });
  afterEach(() => {
    vi.clearAllTimers();
    vi.useRealTimers();
  });

  it("keeps one poll alive as page consumers come and go", async () => {
    const { useLiveActivity } = await import("./useLiveActivity");
    const hosts = routing();
    const shell = useLiveActivity(hosts);
    shell.start();
    shell.start();
    await vi.advanceTimersByTimeAsync(0);
    expect(listActiveWork).toHaveBeenCalledTimes(1);
    for (let index = 0; index < 3; index += 1) {
      const page = useLiveActivity(hosts);
      page.start();
      expect(page.rows.value).toEqual(shell.rows.value);
      page.stop();
      page.stop();
    }
    listActiveWork.mockResolvedValue(snapshot("next"));
    await vi.advanceTimersByTimeAsync(5_000);
    expect(listActiveWork).toHaveBeenCalledTimes(2);
    expect(shell.rows.value[0]?.id).toBe("next");
    shell.stop();
    await vi.advanceTimersByTimeAsync(10_000);
    expect(listActiveWork).toHaveBeenCalledTimes(2);
  });

  it("does not revive an old polling loop after stop and restart during a request", async () => {
    const { useLiveActivity } = await import("./useLiveActivity");
    let finish!: (value: ReturnType<typeof snapshot>) => void;
    listActiveWork.mockImplementationOnce(
      () => new Promise((resolve) => (finish = resolve)),
    );
    const shell = useLiveActivity(routing());
    shell.start();
    await vi.advanceTimersByTimeAsync(0);
    shell.stop();
    shell.start();
    await vi.advanceTimersByTimeAsync(0);
    finish(snapshot());
    await vi.advanceTimersByTimeAsync(0);
    await vi.advanceTimersByTimeAsync(5_000);
    expect(listActiveWork).toHaveBeenCalledTimes(2);
    shell.stop();
  });
});
