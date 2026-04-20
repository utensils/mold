import { ref, watch } from "vue";

/*
 * Client-side "starred" set for gallery items, keyed by filename.
 * Purely local preference state — the backend doesn't track stars — so
 * we round-trip through localStorage and expose a singleton Set<string>.
 */

const STORAGE_KEY = "mold.gallery.starred";

function load(): Set<string> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return new Set();
    const arr = JSON.parse(raw) as unknown;
    if (Array.isArray(arr))
      return new Set(arr.filter((x) => typeof x === "string"));
    return new Set();
  } catch {
    return new Set();
  }
}

const starred = ref<Set<string>>(load());

watch(
  starred,
  (s) => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify([...s]));
    } catch {
      /* ignore */
    }
  },
  { deep: true },
);

export function useStarred() {
  function toggle(filename: string) {
    const next = new Set(starred.value);
    if (next.has(filename)) next.delete(filename);
    else next.add(filename);
    starred.value = next;
  }
  function has(filename: string) {
    return starred.value.has(filename);
  }
  return { starred, toggle, has };
}
