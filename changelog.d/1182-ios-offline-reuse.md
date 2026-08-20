- **iPhone Library can reuse saved print settings while a machine is offline.**
  Library now caches the non-secret model and capability snapshot needed to
  interpret each server instance's print metadata, restores settings from that
  snapshot immediately, and refreshes it in the background. A missing snapshot
  gets a bounded Retry-able load instead of an endless spinner. Print preview's
  Close control remains available during reuse and source restoration, and a
  dismissed attempt cannot later alter Create or navigate when its host finally
  responds ([#1182](https://github.com/utensils/mold/issues/1182)).
