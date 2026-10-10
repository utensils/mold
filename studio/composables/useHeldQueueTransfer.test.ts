import { computed } from "vue";
import { describe, expect, it } from "vitest";
import { createHeldQueueTransfer } from "./useHeldQueueTransfer";
describe("queue transfer destinations", () => {
  it("excludes aliases of the same durable destination owner", () => {
    const controller = createHeldQueueTransfer(
      computed(() => [
        {
          id: "source",
          label: "Source",
          instanceId: "a",
          transferIdentity: "owner",
          ready: true,
          target: { baseUrl: "http://source", apiKey: "" },
        },
        {
          id: "alias",
          label: "Alias",
          instanceId: "b",
          transferIdentity: "owner",
          ready: true,
          target: { baseUrl: "http://alias", apiKey: "" },
        },
      ]),
    );
    expect(controller.canSend("source")).toBe(false);
  });
  it("hides Move to when every distinct ready peer cannot generate", () => {
    const controller = createHeldQueueTransfer(
      computed(() => [
        {
          id: "source",
          label: "Source",
          instanceId: "source",
          ready: true,
          preRenderTransfer: true,
          target: { baseUrl: "http://source", apiKey: "" },
        },
        {
          id: "peer",
          label: "Peer",
          instanceId: "peer",
          ready: true,
          generates: false,
          target: { baseUrl: "http://peer", apiKey: "" },
        },
      ]),
    );
    expect(controller.canSend("source")).toBe(false);
    expect(controller.canSendState("source", "queued")).toBe(false);
    controller.open("source", "job");
    expect(controller.selection.value).toBeNull();
  });
});
