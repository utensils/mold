---
paths:
  - "crates/mold-cli/src/skill/**"
---

# Agent skill installation

Keep target discovery roots in `AGENTS` and its `fixtures/agent-bundles.json`
contract. Reuse the canonical portable renderer when an agent accepts standard
Agent Skills; add a profile only when its metadata or runtime needs differ.

Oh My Pi (`omp`) is distinct from Pi (`pi`). Its default user root is
`~/.omp/agent/skills`, and its project root is `.omp/skills`. Detect `.omp`,
include OMP in bulk installation, and retain the managed manifest lifecycle.
The installer targets the default user profile; project scope works with named
OMP profiles and custom agent directories. Discovery and required description
frontmatter follow upstream `packages/coding-agent/src/discovery/builtin.ts`:
https://github.com/can1357/oh-my-pi/blob/main/packages/coding-agent/src/discovery/builtin.ts
