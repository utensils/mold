# iPhone usability follow-up

Scope: native app, iPhone only; exploratory UAT against existing media and
installed models, with no generation submissions. Simulator cannot establish
physical speaker output, camera pairing, or device-only extension delivery.

## Findings and implementation plan

| Finding | Reproduction | Improvement | Verification |
| --- | --- | --- | --- |
| First-run Machines clips its explanation and action at AX5 | Open Machines with no saved hosts, largest text and nearby discovery | Keep discovery count in the scrolling explanation; wrap Add; reserve opaque space above tabs for the action | First-run iPhone UAT |
| More Options becomes unreadable at AX5 | Select an installed picture model, set largest accessibility text, open Options | Give Shape, Steps, Batch and Length separate form rows; wrap values; remove the inert More Options button inside its own sheet | Populated iPhone UI regression and exploratory UAT |
| Seed choice ignores large text and loses its label | In the same sheet, scroll to Look | Use a labeled system menu for Repeat this look | AX5 UAT |
| Clip and 3-D drafts reopen as Still picture | Choose either kind, background, terminate, relaunch before model profiles arrive | Restore kind from the saved recipe once profiles arrive, preserving valid authored options and reconciling audio capabilities | Failing-then-passing cold-start controller tests and relaunch UAT |
| Empty model search gives no feedback | Search installed models for an unmatched word | Explain that no models match and allow clearing the search | Populated UI regression and UAT |
| Models pane chooser ignores large text | Open Installed/Discover at AX5 | Use a labeled system menu that scales with the text | iPhone UAT |
| Offline inventory falsely claims zero installed | Add an unreachable local test machine, open its detail and Models | Distinguish an unread inventory from a confirmed empty one; omit unknown counts | Model-store regression and UAT |
| Removing a machine leaves a dead detail page | Remove the temporary machine from its detail | Return to Machines after removal | iPhone UAT |
| Info has no explicit dismissal action | Open Info at its large detent | Add a standard Done button | Viewer UAT |
| Shared video has an extensionless temporary filename | Share an existing clip | Preserve each original filename in an owned export directory; clean it after sharing/copying/saving | Failing-then-passing sharing test and system share-sheet UAT |
| Slow machine responses replace the saved model | Restore a clip while another machine answers first | Keep the saved choice pending; explicit kind/model choice cancels restoration | Two-host failing-then-passing controller regressions |
| Offline Generate and Queue claim missing models or an empty queue | Open both tabs with only unreachable machines | Explain unavailable data and direct the user to Machines; invalidate failed queue reads | Controller/queue tests and SE UAT |
| A removed machine silently drops a selected export | Remove a source machine before an export begins | Abort the entire export and clean staged files, preserving entry/file alignment | Export regression |
| Viewer actions are covered by the main tabs | Open an existing print, try Info at the bottom | Hide the main tab bar while the viewer is open, preserving its own controls and back navigation | Viewer UAT on iPhone |

## Verification record

- Baseline exploratory screenshots reproduced the Options and viewer failures.
- Cold-start regression failed for both clip and mesh with `.picture` and no
  recipe. A second assertion exposed sound being disabled on restored clip
  drafts until recipe capabilities were reconciled.
- Populated UI tests use a loopback fixture with generated model profiles.
  The fixture cannot generate, download, or modify remote data.
- Baseline Pro exploration covered Generate long text and model/kind choices,
  Library browsing/search/selection and existing video playback, Models
  installed/discover/search, Machines add/edit/cancel/offline/remove, invalid
  address/pairing and scanner fallback, and Settings in light/default and
  dark/AX5. The Library selection toolbar was correctly positioned.
- After draft/audio reconciliation and offline inventory changes, all 100
  native unit tests passed.
- Populated SE AX5 Options/search regression passed, including test-machine
  removal. The updated viewer's actions are visible and Info opens in ordinary
  pushed navigation; the agent restored the original favourite state after
  testing a neutral existing print.
- Post-fix exploratory UAT, peer review and exact-head CI are in progress.
