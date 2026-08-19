# Changelog fragments

One file per pull request. Each PR that changes shipped source adds
`changelog.d/<short-slug>.md` containing its release note as a Keep-a-Changelog
bullet — the same prose that used to go under `## [Unreleased]` in
`CHANGELOG.md`:

```markdown
- **Short bold title.** What changed and why it matters to a user
  ([#1234](https://github.com/utensils/mold/issues/1234)).
```

Rules:

- Never edit the `[Unreleased]` section of `CHANGELOG.md` by hand. Two open PRs
  inserting at that same line is what made every PR conflict there. CI's
  `changelog` check refuses direct edits and asks for a fragment.
- Slug = your branch topic (`wan-metal-perf.md`, `fix-1059-metal-admission.md`).
  Names only need to be unique while the fragment exists.
- Multi-line bullets are fine; continuation lines are indented two spaces.
  Several bullets in one fragment are fine when one PR ships several notes.
- A PR that ships nothing user-visible (pure refactor, CI, tests) may skip the
  fragment with the `skip-changelog` label.
- `README.md` is documentation, never a fragment.

On the release PR, `scripts/release/sync-release-pr.sh` assembles every
fragment under the new version heading (newest first, by the commit that added
it), deletes the fragments, and refreshes the compare links. Until then,
`cat changelog.d/*.md` is the pending changelog.
