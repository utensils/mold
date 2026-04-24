import { computed, ref, type ComputedRef, type Ref } from "vue";

/*
 * Global "hide everything" state.
 *
 * One privacy knob, shared by Gallery and Generate. When the shroud is
 * on, every preview surface (gallery feed, running-job tiles) renders
 * behind a heavy blur. Users can peek a single item via its per-card
 * Reveal button without flipping the global preference.
 *
 * Toggle semantics are user-visible-state driven, not flag-driven:
 *   - Anything currently visible (shroud off, or at least one peek)
 *     → hide everything: shroud on, peek set cleared.
 *   - Everything shrouded (shroud on, no peeks)
 *     → reveal everything: shroud off.
 *
 * This makes "click the button to return to full privacy" work whether
 * the user arrived at unshrouded content via the toggle itself or via
 * per-item peeks, which is the behavior users expect once they've
 * opted into hide mode.
 *
 * `hideMode` is persisted so the preference survives reloads. The
 * `revealed` set is intentionally not — reloads re-apply the shroud.
 */
const STORAGE_KEY = "mold.gallery.hide";

function load(): boolean {
  try {
    return localStorage.getItem(STORAGE_KEY) === "true";
  } catch {
    return false;
  }
}

function persist(v: boolean): void {
  try {
    localStorage.setItem(STORAGE_KEY, String(v));
  } catch {
    /* localStorage may be blocked (private mode, sandboxed iframe) */
  }
}

export interface UseHideMode {
  hideMode: Ref<boolean>;
  revealed: Ref<Set<string>>;
  /** True when at least one tile is visible to the user right now. */
  anyVisible: ComputedRef<boolean>;
  /** Cycle between fully-hidden and fully-visible states. */
  toggle: () => void;
  /** Add a single filename to the peek set (idempotent). */
  revealOne: (filename: string) => void;
}

let singleton: UseHideMode | null = null;

export function useHideMode(): UseHideMode {
  if (singleton) return singleton;

  const hideMode = ref<boolean>(load());
  const revealed = ref<Set<string>>(new Set());

  const anyVisible = computed(() => !hideMode.value || revealed.value.size > 0);

  function toggle(): void {
    if (anyVisible.value) {
      hideMode.value = true;
      revealed.value = new Set();
      persist(true);
    } else {
      hideMode.value = false;
      persist(false);
    }
  }

  function revealOne(filename: string): void {
    if (revealed.value.has(filename)) return;
    const next = new Set(revealed.value);
    next.add(filename);
    revealed.value = next;
  }

  singleton = { hideMode, revealed, anyVisible, toggle, revealOne };
  return singleton;
}

/**
 * Test-only hook to reset the singleton between cases. Unused in the
 * app — exported so vitest can start each scenario from a clean slate.
 */
export function __resetHideModeForTests(): void {
  singleton = null;
}
