/**
 * Pure helpers for the LAN server-discovery list. Kept free of Tauri/IPC so the
 * ordering, de-duplication, and label logic are unit-testable in isolation.
 */

import type { DiscoveredHost } from "./ipc";

/**
 * Order discovered hosts for display: this machine first, then alphabetically
 * by name (case-insensitive), with URL as a stable tiebreaker.
 */
export function sortHosts(hosts: DiscoveredHost[]): DiscoveredHost[] {
  return [...hosts].sort((a, b) => {
    if (a.isThisMachine !== b.isThisMachine) return a.isThisMachine ? -1 : 1;
    const byName = a.name.localeCompare(b.name, undefined, { sensitivity: "base" });
    if (byName !== 0) return byName;
    return a.url.localeCompare(b.url);
  });
}

/**
 * Drop duplicate advertisements that resolve to the same URL, keeping the first
 * occurrence. mDNS can surface a host twice (IPv4 + IPv6 records) before they
 * converge; the URL is the identity the UI acts on.
 */
export function dedupeHosts(hosts: DiscoveredHost[]): DiscoveredHost[] {
  const seen = new Set<string>();
  const out: DiscoveredHost[] = [];
  for (const h of hosts) {
    if (seen.has(h.url)) continue;
    seen.add(h.url);
    out.push(h);
  }
  return out;
}

/** Sorted + de-duplicated hosts, ready to render. */
export function prepareHosts(hosts: DiscoveredHost[]): DiscoveredHost[] {
  return sortHosts(dedupeHosts(hosts));
}

/** `host:port` label for the mono address chip. */
export function addressLabel(host: DiscoveredHost): string {
  return `${host.host}:${host.port}`;
}

/** Version label (`mold 0.14.0`), or a neutral fallback when unknown. */
export function versionLabel(host: DiscoveredHost): string {
  return host.version ? `mold ${host.version}` : "mold";
}
