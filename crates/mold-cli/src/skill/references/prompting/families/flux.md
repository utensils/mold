# FLUX.1 prompting

Manifest family: `flux`.

Applies to Schnell, Dev, Krea, and FLUX.1 fine-tunes. Put the subject and action
first, then composition, lighting, lens or material cues, and style. Prefer
natural sentences to long tag piles. Schnell is best for simple four-step
drafts; Dev, Krea, and quality fine-tunes reward precise spatial relationships.

```bash
mold run flux-dev:q4 \
  "A cozy Japanese tea house interior, two ceramic cups steaming on a low cedar table, warm paper lanterns, rain beyond the open shoji, intimate eye-level composition, delicate watercolor texture" \
  --seed 1337
```

With `--id-image`, let the reference own facial features. Describe the new role,
clothing, setting, pose, composition, and light. Start near `--id-weight 0.8`;
check `supports_identity` for the exact model before offering the workflow.
