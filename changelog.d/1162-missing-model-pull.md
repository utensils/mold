- **Auto and Most capable no longer dead-end on a model nobody has.** When
  every machine can run the print but none has the checkpoint, desktop and web
  now offer to download it — naming the machine, or asking which one when
  several could take it — and run the print there once the pull lands, instead
  of reporting "Nothing was queued". A machine that simply cannot fit the print
  still says so; only a missing model is answered with a download
  ([#1162](https://github.com/utensils/mold/issues/1162)).
- **A slow machine under Auto is no longer reported as one that refused.**
  While no machine has produced a plan yet, Auto extends its response window
  once rather than describing a check that is still running as "did not answer"
  ([#1162](https://github.com/utensils/mold/issues/1162)).
- **Restoring a print whose model is gone keeps the model.** Dropping a print
  into Create, or Reuse settings, now shows the recorded model in the selector
  with a *Not installed* tag and offers the same download, on desktop and web.
  Web no longer silently swaps in a different model — along with its size,
  steps, guidance and LoRAs — behind your back
  ([#1162](https://github.com/utensils/mold/issues/1162)).
