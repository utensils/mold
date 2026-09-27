---
paths:
  - "website/**"
---

# Public website analytics

Moved from the root CLAUDE.md; loaded only when working under the paths above.

`website/.vitepress/theme/analytics.mjs` owns the automatic GA4 integration
(`G-RG6PPTGX2T`), restricted to `https://utensils.io/mold/`. Never import it into
Studio or the apps. The theme setup starts analytics without a popup and sends one
explicit page view after VitePress navigation. Keep GA enhanced-measurement
browser-history page views OFF to avoid duplicates; search, form, and video
measurement are also off. `bun run verify` in `website/` runs automatic-start and
navigation tests. Update `website/privacy.md` when this behavior changes.
