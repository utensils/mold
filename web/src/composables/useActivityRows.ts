import { computed, type Ref } from "vue";
import type { FleetActiveWork } from "@studio/api/activity";
import type { Job } from "./useGenerateStream";
import {
  localRowHiddenFromStrip,
  sharedRowIsLocallyOwned,
} from "../lib/activityDedup";
import { ORIGIN_HOST_ID } from "../lib/hostRegistry";

/** Create and Queue resolve local/fleet ownership identically. No new store. */
export function useActivityRows(
  jobs: Ref<Job[]>,
  rows: Ref<FleetActiveWork[]>,
) {
  return {
    sharedActivityRows: computed(() =>
      rows.value.filter(
        (row) => !sharedRowIsLocallyOwned(row, jobs.value, ORIGIN_HOST_ID),
      ),
    ),
    localActivityJobs: computed(() =>
      jobs.value.filter(
        (job) => !localRowHiddenFromStrip(job, rows.value, ORIGIN_HOST_ID),
      ),
    ),
  };
}
