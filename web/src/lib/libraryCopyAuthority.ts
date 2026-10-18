import { HOSTS_CHANGED_EVENT } from "./hostRegistry";
import { ORIGIN_AUTH_CHANGED_EVENT } from "./originAuth";
import { STUDIO_PRIMARY_CHANGED } from "./secureStudio";

/** Catch authority changes even when a route or key is changed back mid-read. */
export function observeLibraryCopyAuthority(
  invalidate: () => void,
): () => void {
  const events = [
    HOSTS_CHANGED_EVENT,
    ORIGIN_AUTH_CHANGED_EVENT,
    STUDIO_PRIMARY_CHANGED,
  ];
  const pairingChanged = (event: Event) => {
    const authority = (event as CustomEvent<{ authority?: string }>).detail
      ?.authority;
    if (authority === "removed" || authority === "replaced") invalidate();
  };
  for (const event of events) window.addEventListener(event, invalidate);
  window.addEventListener("mold:secure-pairings-changed", pairingChanged);
  return () => {
    for (const event of events) window.removeEventListener(event, invalidate);
    window.removeEventListener("mold:secure-pairings-changed", pairingChanged);
  };
}
