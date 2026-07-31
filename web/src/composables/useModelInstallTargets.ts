/*
 * Which machine a model gets installed on (spec §08 multi-host, shared rule in
 * `@studio/lib/modelInstallTargets`).
 *
 * The Models workspace used to be origin-scoped: `/api/models` and every
 * `/api/catalog/...` call went to the serving host, so a model installed there
 * read as "installed" everywhere and the browser had no way to put it on any
 * other connected machine. Web is a multi-host client — the host registry and
 * `useHostRouting`'s per-host `/api/models` poll already know exactly who has
 * what — so the action stays an install until EVERY reachable machine owns the
 * model, and only then degrades to Repair.
 *
 * The pending picker lives at module scope (like `lib/toasts`) so the one
 * `ModelInstallTargetDialog` mounted by the Models page serves the Discover
 * cards, the installed rows, and the detail drawer alike.
 */
import { ref, type Ref } from "vue";
import {
  planModelInstall,
  type ModelInstallAction,
  type ModelInstallPlan,
  type ModelInstallTarget,
} from "@studio/lib/modelInstallTargets";
import { getHost, ORIGIN_HOST_ID } from "../lib/hostRegistry";
import { hostModelDownload } from "../components/machines/hostClient";
import { useCatalog } from "./useCatalog";
import { useHostRouting } from "./useHostRouting";
import type { RoutableHost } from "../lib/hostRouting";

export type InstallTarget = ModelInstallTarget<RoutableHost>;

/** The open machine picker. `resolve` stays private to this module. */
export interface PendingInstallChoice {
  modelId: string;
  /** What the dialog calls the model. */
  displayName: string;
  targets: InstallTarget[];
}

export interface InstallChoiceRequest {
  modelId: string;
  displayName: string;
  /** True when the serving host is known to already hold it (a Discover row's
   * `installed` flag, or a row from the origin's installed shelf) — positive
   * knowledge that lands before the per-host poll does. */
  ownedByOrigin?: boolean;
}

export type InstallChoice =
  /** `target: null` means "no machine is reachable yet" — act on the origin,
   * exactly as this surface did before it was multi-host. */
  { kind: "target"; target: InstallTarget | null } | { kind: "cancelled" };

const pending = ref<PendingInstallChoice | null>(null);
let resolvePending: ((target: InstallTarget | null) => void) | null = null;

function settle(target: InstallTarget | null): void {
  const resolve = resolvePending;
  resolvePending = null;
  pending.value = null;
  resolve?.(target);
}

export interface ModelInstallTargets {
  /** The open machine picker, or null. */
  pending: Ref<PendingInstallChoice | null>;
  planFor: (
    modelId: string,
    ownedByOrigin?: boolean,
  ) => ModelInstallPlan<RoutableHost>;
  /** Resolve where an install goes, asking the user only when it is ambiguous. */
  chooseInstallTarget: (req: InstallChoiceRequest) => Promise<InstallChoice>;
  choose: (target: InstallTarget) => void;
  cancel: () => void;
  startDownloadOn: (
    target: InstallTarget | null,
    modelId: string,
  ) => Promise<void>;
  queuedMessage: (
    target: InstallTarget | null,
    fallbackAction?: ModelInstallAction,
  ) => string;
}

export function useModelInstallTargets(): ModelInstallTargets {
  const routing = useHostRouting();
  const cat = useCatalog();

  function planFor(
    modelId: string,
    ownedByOrigin = false,
  ): ModelInstallPlan<RoutableHost> {
    // Only a machine that answered its status poll can accept a download. A
    // connecting or errored one is neither an install target nor evidence that
    // the model is missing there.
    const reachable = routing.hosts.value.filter((h) => h.status === "ready");
    const owners = new Set(routing.modelOwnerIds(modelId));
    if (ownedByOrigin) owners.add(ORIGIN_HOST_ID);
    return planModelInstall(reachable, owners, {
      inventoryKnown: (host) => routing.inventoryKnown(host.id),
    });
  }

  async function chooseInstallTarget(
    req: InstallChoiceRequest,
  ): Promise<InstallChoice> {
    const plan = planFor(req.modelId, req.ownedByOrigin);
    if (plan.targets.length === 0) return { kind: "target", target: null };
    if (plan.targets.length === 1) {
      return { kind: "target", target: plan.targets[0] };
    }
    // Supersede any picker still on screen rather than stacking two.
    settle(null);
    const chosen = await new Promise<InstallTarget | null>((resolve) => {
      resolvePending = resolve;
      pending.value = {
        modelId: req.modelId,
        displayName: req.displayName,
        targets: plan.targets,
      };
    });
    return chosen ? { kind: "target", target: chosen } : { kind: "cancelled" };
  }

  async function startDownloadOn(
    target: InstallTarget | null,
    modelId: string,
  ): Promise<void> {
    if (!target || target.host.id === ORIGIN_HOST_ID) {
      // Keep the origin on the catalog composable: it routes by id shape and
      // repaints the downloads centre immediately instead of waiting on the
      // SSE `enqueued` event.
      await cat.startDownload(modelId);
      return;
    }
    const entry = getHost(target.host.id);
    if (!entry) {
      throw new Error(`${target.host.label} is no longer a connected machine`);
    }
    await hostModelDownload(entry, modelId);
  }

  function queuedMessage(
    target: InstallTarget | null,
    fallbackAction: ModelInstallAction = "install",
  ): string {
    const action = target?.action ?? fallbackAction;
    const verb = action === "repair" ? "Repair queued" : "Download queued";
    // One machine, one possible answer — say exactly what this surface has
    // always said rather than adding noise for single-host users.
    if (!target || routing.hosts.value.length < 2) return verb;
    return `${verb} on ${target.host.label}`;
  }

  return {
    pending,
    planFor,
    chooseInstallTarget,
    choose: (target: InstallTarget) => settle(target),
    cancel: () => settle(null),
    startDownloadOn,
    queuedMessage,
  };
}
