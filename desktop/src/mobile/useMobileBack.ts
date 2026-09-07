import { onBeforeUnmount, watch, type Ref } from "vue";
import { isNativeAndroidRuntime } from "./platform";

const HISTORY_KEY = "mold.mobile.transient";
interface Entry {
  id: string;
  close: () => void;
}
const entries: Entry[] = [];
let serial = 0;
let installed = false;
let closingFromBack = false;
let ignoreNextPop = false;

function onPop(event: PopStateEvent): void {
  if (ignoreNextPop) {
    ignoreNextPop = false;
    event.stopImmediatePropagation();
    return;
  }
  const top = entries.at(-1);
  if (!top) return;
  event.stopImmediatePropagation();
  closingFromBack = true;
  try {
    top.close();
  } finally {
    closingFromBack = false;
  }
}

/** Android Back dismisses one temporary surface. Closing with Done consumes
 * its history entry too, without closing a viewer underneath it. */
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
    if (!closingFromBack && window.history.state?.[HISTORY_KEY] === entry.id) {
      ignoreNextPop = true;
      window.history.back();
    }
  }
  watch(
    open,
    (active) => {
      if (active) {
        entries.push(entry);
        window.history.pushState({ ...(window.history.state ?? {}), [HISTORY_KEY]: entry.id }, "");
      } else remove();
    },
    { immediate: true, flush: "sync" },
  );
  onBeforeUnmount(remove);
}
