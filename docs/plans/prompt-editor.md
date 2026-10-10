# Prompt editor and native Library preferences

## Design

Keep Generate calm: a compact, bounded prompt field with an explicit **Edit prompt** action. Short prompts remain quick to type. A dedicated **Prompt** editor supplies the space for long text: native large sheet on iOS, roomy native macOS sheet, responsive shared modal on web/Tauri and full-height phone presentation. Text takes most of the surface; controls are clearly labeled, visually quiet, and keyboard accessible.

One authoritative draft is live-bound through existing authoring/provenance handlers. **Done** and ordinary dismissal retain edits; there is no ambiguous Cancel/Save split or hidden second buffer. Enter inserts a newline. Editor arrow keys always navigate text, never unexpectedly replace it with history. Prevent generation shortcuts from dispatching behind the editor. Preserve platform selection, clipboard, IME and native undo where available.

Editor actions: **Recent prompts** (searchable, selects prompt text only), **Clear** (immediate, disabled when empty, offers **Undo clear** until a later edit/replacement), and existing prompt rewrite affordances where supported. Reuse current history and rewrite authorities, preserve host fencing and offline/error states; never create generation history on every keystroke. Closing returns focus to the opener. Existing negative prompt and image/reference/model/settings semantics stay intact. Capability-ignored prompts must not gain a misleading editing affordance.

Date separators: native iOS Settings and native macOS Library Settings gain **Show date separators**, default on and persisted per client. Off builds one continuous section in existing sort order, retaining all filters, selections, unread identities and viewport behavior. Preferences affect all applicable native Library views, including search and shelves. Keep the toggle out of Library chrome.

## Delivery and ownership

Primary agent owns design, integration review, visual review and delivery. Sol agents implement bounded iOS, macOS and shared frontend tracks following this design, with failing contract/regression tests first. Native shared section projection is owned by the primary agent; agents must coordinate any shared changes. No real generations or production media mutations during validation.

Validate long multiline editing, selection and newline behavior, dismissal/reopen/relaunch persistence, clear/undo, history selection without changing other settings, rewrite/capability states, keyboard/focus/scroll behavior, small phone and wide layouts. Verify date grouping on/off and persisted defaults. Update affected docs, rules and changelog. Commit and push on feat/prompt-editor, obtain independent review, verify exact-head CI, merge, and verify that successful nightly artifacts include the merge commit. The user explicitly authorized end-to-end merge/nightly delivery for this task.

## Review and validation

The shared client suite passed 1,280 tests. Frontend validation passed 12,082 tests across Studio, web and desktop, plus production web, desktop and mobile builds, architecture checks, dead-code checks and Studio formatting. Browser fixture interaction checks exercised the real composer/editor, long multiline drafts, Clear/Undo, recent prompt selection, dismissal/reopen, focus restoration and nested rewrite dialogs at desktop and phone sizes. Visual review caught and corrected portal typography and a crowded phone toolbar.

Native macOS passed 1,040 tests, followed by 17 focused tests after the final history marker correction. Actual native sheets were rendered in light and dark mode and native multiline text editing exercised. Review removed competing compact/editor rewrite popovers and process-wide shortcut suppression, and retained print identity while date grouping changes. Independent review of the Mac and shared projection changes found no outstanding issue.

Native iOS passed 12 focused unit tests and both complete Simulator fixture flows. The exact result bundle confirms every selected case passed: large-text keyboard layout; date preference off/relaunch/continuous-grid/on; multiline editing; generation-shortcut protection; Clear/Undo; rewrite/Clear/Undo/Undo rewrite; alternative suggestion selection; recent history; and draft persistence after relaunch. No generation requests occurred. Runtime validation caught and corrected an unmounted environment-dependent SwiftUI menu view; independent review confirmed the final mounted-view fix and single presentation ownership. Screenshots were inspected at normal and largest accessibility text sizes. This is Simulator evidence, not physical-device testing.
