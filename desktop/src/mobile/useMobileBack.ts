import { onBeforeUnmount, watch, type Ref } from "vue";
import { isNativeAndroidRuntime } from "./platform";

const HISTORY_KEY = "mold.mobile.transient";
interface Entry {
  id: string;
  close: () => void;
}
const entries: Entry[] = [];
// Retain dismissed slots until traversal consumes them. A parent and child can
// close together, or a new sheet can open while a history traversal is pending.
let historyIds: string[] = [];
let serial = 0;
let installed = false;
let navigating = false;
let scheduled = false;

function reconcile(): void {
  scheduled = false;
  if (navigating) return;
  let current = historyIds.indexOf(window.history.state?.[HISTORY_KEY]);
  for (const entry of entries) {
    if (historyIds.includes(entry.id)) continue;
    historyIds = historyIds.slice(0, current + 1);
    window.history.pushState({ ...(window.history.state ?? {}), [HISTORY_KEY]: entry.id }, "");
    historyIds.push(entry.id);
    current++;
  }
  const top = entries.at(-1);
  const target = top ? historyIds.indexOf(top.id) : -1;
  if (target !== current) {
    navigating = true;
    window.history.go(target - current);
  }
}

function scheduleReconcile(): void {
  if (scheduled) return;
  scheduled = true;
  queueMicrotask(reconcile);
}

function onPop(event: PopStateEvent): void {
  if (navigating) {
    navigating = false;
    event.stopImmediatePropagation();
    scheduleReconcile();
    return;
  }
  const top = entries.at(-1);
  if (!top) return;
  event.stopImmediatePropagation();
  top.close();
  scheduleReconcile();
}

/** Android Back dismisses one temporary surface. Closing nested surfaces in
 * one render consumes their history together without closing the next screen. */
export function useMobileBack(open: Ref<boolean>, close: () => void): void {
  if (!isNativeAndroidRuntime()) return;
  if (!installed) {
    window.addEventListener("popstate", onPop, true);
    installed = true;
  }
  const entry: Entry = { id: `mobile-${++serial}`, close };
  function remove(): void {
    const index = entries.indexOf(entry);
    if (index < 0) return;
    entries.splice(index, 1);
    scheduleReconcile();
  }
  watch(
    open,
    (active) => {
      if (active) {
        // Each opening gets a fresh history identity, including a sheet that
        // reopens before the previous close has finished traversing history.
        entry.id = `mobile-${++serial}`;
        entries.push(entry);
        scheduleReconcile();
      } else remove();
    },
    { immediate: true, flush: "sync" },
  );
  onBeforeUnmount(remove);
}
