import type { Ref } from "vue";
import type { ChainJobDetail } from "../types";

export function useChainJobStream(_jobId: Ref<string | null>): {
  detail: Ref<ChainJobDetail | null>;
  connected: Ref<boolean>;
} {
  throw new Error(
    "TODO(BR55 Phase 6): implement chain job stream composable with snapshot folding and reconnect fallback",
  );
}
