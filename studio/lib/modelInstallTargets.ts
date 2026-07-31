/*
 * Which machines a model can be sent to, and what sending it there means.
 *
 * Every surface merges its installed models across hosts, so a row can be
 * installed on one machine and absent from the next. Treating "installed
 * somewhere" as "installed everywhere" is what removed the install action
 * from a model the user does not actually have locally — the whole point of
 * this module is that an install stays on offer until every reachable machine
 * has the model, at which point the only remaining action is a repair.
 */

/** What sending a model to one machine actually does there. */
export type ModelInstallAction = "install" | "repair";

export interface ModelInstallTarget<H> {
  host: H;
  action: ModelInstallAction;
}

export interface ModelInstallOptions<H> {
  /**
   * False while a machine's inventory has not been read yet. Unknown machines
   * are left out entirely: we never claim one is missing a model we have not
   * looked for. Ownership always wins over this — it is positive knowledge.
   */
  inventoryKnown?: (host: H) => boolean;
}

export interface ModelInstallPlan<H> {
  /** Install targets first (that is the action the user came for), then repairs. */
  targets: ModelInstallTarget<H>[];
  /** True when at least one reachable machine is known to lack the model. */
  canInstall: boolean;
  /** What the row's action control should say. */
  label: "Pull" | "Repair";
}

/**
 * `hosts` must already be narrowed to machines that can accept a download
 * (reachable, addressable). `ownerIds` are the machines known to have the
 * model on disk.
 */
export function planModelInstall<H extends { id: string }>(
  hosts: readonly H[],
  ownerIds: Iterable<string> | null | undefined,
  options: ModelInstallOptions<H> = {},
): ModelInstallPlan<H> {
  const owners = new Set(ownerIds ?? []);
  const known = options.inventoryKnown ?? (() => true);

  const install: ModelInstallTarget<H>[] = [];
  const repair: ModelInstallTarget<H>[] = [];
  for (const host of hosts) {
    if (owners.has(host.id)) repair.push({ host, action: "repair" });
    else if (known(host)) install.push({ host, action: "install" });
  }

  const canInstall = install.length > 0;
  return {
    targets: [...install, ...repair],
    canInstall,
    // A model nobody owns is always a pull, even with nowhere to send it yet —
    // calling that a "repair" would describe files that do not exist.
    label: canInstall || owners.size === 0 ? "Pull" : "Repair",
  };
}
