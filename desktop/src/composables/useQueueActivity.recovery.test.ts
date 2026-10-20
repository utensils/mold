import { expect, it, vi } from "vitest";
const rows = vi.hoisted(() => ({
  value: [
    {
      key: "failed",
      id: "failed",
      hostId: "plato",
      kind: "sequence",
      phase: "failed",
      created_at_unix_ms: 1,
    },
    {
      key: "interrupted",
      id: "interrupted",
      hostId: "plato",
      kind: "sequence",
      phase: "interrupted",
      created_at_unix_ms: 2,
    },
    {
      key: "queued",
      id: "queued",
      hostId: "plato",
      kind: "sequence",
      phase: "queued",
      created_at_unix_ms: 3,
    },
  ],
}));
vi.mock("../stores/generation", () => ({
  GENERATION_HISTORY_LIMIT: 20,
  railOrder: (jobs: unknown[]) => jobs,
  useGenerationStore: () => ({ jobs: [] }),
}));
vi.mock("../stores/hosts", () => ({ useHostsStore: () => ({ primaryHost: { id: "local" } }) }));
vi.mock("../stores/liveActivity", () => ({ useLiveActivityStore: () => ({ rows: rows.value }) }));
import { useQueueActivity } from "./useQueueActivity";
it("retains terminal recovery rows without counting them as rendering or waiting", () => {
  const queue = useQueueActivity();
  expect(queue.rows.value).toHaveLength(3);
  expect(queue.activeCount.value).toBe(0);
  expect(queue.waitingCount.value).toBe(1);
  expect(queue.liveCount.value).toBe(1);
});
