import {
  computed,
  inject,
  provide,
  ref,
  type ComputedRef,
  type InjectionKey,
} from "vue";
import type { QueueTransferHost } from "../api/queueTransfer";

export function createHeldQueueTransfer(
  hosts: ComputedRef<QueueTransferHost[]>,
) {
  const busy = ref(false);
  const selection = ref<{ source: QueueTransferHost; jobId: string } | null>(
    null,
  );
  const canSend = (hostId: string) => {
    const source = hosts.value.find((host) => host.id === hostId && host.ready);
    return (
      !!source &&
      hosts.value.some(
        (host) => host.ready && host.instanceId !== source.instanceId,
      )
    );
  };
  return {
    hosts,
    selection,
    busy,
    close() {
      if (!busy.value) selection.value = null;
    },
    canSend,
    destinations: computed(() =>
      hosts.value.filter(
        (host) =>
          host.ready && host.instanceId !== selection.value?.source.instanceId,
      ),
    ),
    open(hostId: string, jobId: string) {
      if (busy.value || !canSend(hostId)) return;
      const source = hosts.value.find((host) => host.id === hostId)!;
      selection.value = {
        source: { ...source, target: { ...source.target } },
        jobId,
      };
    },
  };
}
export type HeldQueueTransferController = ReturnType<
  typeof createHeldQueueTransfer
>;
const key: InjectionKey<HeldQueueTransferController> = Symbol(
  "held-queue-transfer",
);
export function provideHeldQueueTransfer(
  hosts: ComputedRef<QueueTransferHost[]>,
) {
  const controller = createHeldQueueTransfer(hosts);
  provide(key, controller);
  return controller;
}
export function useHeldQueueTransfer() {
  return inject(key, null);
}
