import { afterEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { useHostsStore } from "./hosts";
import { useLiveActivityStore } from "./liveActivity";

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => (resolve = done));
  return { promise, resolve };
}

function activityResponse(items: unknown[]) {
  return new Response(
    JSON.stringify({
      instance_id: "render-instance",
      observed_at_unix_ms: Date.now(),
      items,
    }),
    { status: 200, headers: { "content-type": "application/json" } },
  );
}

describe("live activity refresh", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("guarantees a fresh pass after a refresh requested during an in-flight poll", async () => {
    setActivePinia(createPinia());
    useHostsStore().extras.push({
      id: "render",
      label: "Render box",
      url: "http://render:7680",
      apiKey: "secret",
      status: "ready",
      error: null,
      instanceId: "render-instance",
    });
    const first = deferred<Response>();
    const second = deferred<Response>();
    const fetchMock = vi
      .fn()
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise);
    vi.stubGlobal("fetch", fetchMock);
    const activity = useLiveActivityStore();

    const preCancelRefresh = activity.refresh();
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    const postCancelRefresh = activity.refresh();
    expect(fetchMock).toHaveBeenCalledTimes(1);

    first.resolve(
      activityResponse([
        {
          id: "job-1",
          kind: "generation",
          phase: "running",
          created_at_unix_ms: 1,
          updated_at_unix_ms: 2,
          can_cancel: true,
        },
      ]),
    );
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    second.resolve(activityResponse([]));

    await Promise.all([preCancelRefresh, postCancelRefresh]);
    expect(activity.hosts.render?.items).toEqual([]);
    expect(activity.refreshing).toBe(false);
  });
});
