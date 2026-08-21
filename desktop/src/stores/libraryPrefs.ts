import { defineStore } from "pinia";
import { AUTO_TAG_TITLE_DEFAULT, loadAutoTagTitle, saveAutoTagTitle } from "../lib/libraryPrefs";
import { useGenerateFormStore } from "./generateForm";

/**
 * Client-side Library preferences (Settings ▸ Library). Today that is one
 * toggle: whether a titled print picks up its own slug as a ghost tag.
 *
 * The store also owns the MIRROR onto the Create form. `buildRequest` is a
 * pure function of the form, so the preference has to ride along on it; doing
 * that here rather than in a view means the mirror is correct no matter which
 * workspace is mounted when the setting changes.
 */
export const useLibraryPrefsStore = defineStore("libraryPrefs", {
  state: () => ({ autoTagTitle: AUTO_TAG_TITLE_DEFAULT }),
  actions: {
    /** Boot: read the persisted value and mirror it onto the form. */
    init() {
      this.autoTagTitle = loadAutoTagTitle();
      this.mirror();
    },
    setAutoTagTitle(value: boolean) {
      this.autoTagTitle = value;
      saveAutoTagTitle(value);
      this.mirror();
    },
    /** Push the preference onto the Create form's `fileUnderAutoTag`. The
     * form defaults it OFF so a surface that never wired the group up cannot
     * file a ghost tag invisibly; desktop opts in here. */
    mirror() {
      useGenerateFormStore().form.fileUnderAutoTag = this.autoTagTitle;
    },
  },
});
