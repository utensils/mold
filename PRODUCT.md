# Product

## Register

product

## Users

Mold serves developers, artists, and technical creators generating images and video locally or through a remote Mold engine. They work from a dense desktop studio or an iPhone remote companion where model choice, host health, GPU constraints, queue and download state, parameters, and output provenance must stay legible while they iterate.

## Product Purpose

Mold makes local AI image and video generation observable and controllable. The desktop app should let users configure an engine, compose and queue work, understand what the GPU is doing, inspect results, and reuse successful settings without leaving the task. The iPhone app should make the remote parts of that loop first-class without pretending the phone is an inference host: connect and inspect servers, queue generation, manage models, browse media, and reuse successful work from anywhere the host is reachable.

## Brand Personality

Technical, tactile, and confident. The interface can carry the physical character of a darkroom or the cyan-magenta energy of the Mold logo, but it should remain disciplined enough for long sessions and trustworthy enough for resource-intensive work.

## Anti-references

Avoid generic AI dashboards, decorative glass panels, neon-on-black cyberpunk shells, oversized pill controls, and card grids that separate related controls without purpose. Do not let a branded theme turn generation states into ambiguous decoration or alter the color of generated media.

## Design Principles

- Make engine state and resource pressure visible before they become errors.
- Treat generated media as the work, with chrome supporting it rather than competing with it.
- Use color semantically: latent, active, complete, and failed states must remain distinguishable in every theme.
- Keep expert workflows compact and familiar: keyboard-first on desktop,
  direct and touch-first on iPhone.
- Let themes change atmosphere without changing information hierarchy or control behavior.
- On iPhone, respect safe areas and 44pt touch targets, keep host/queue/pull
  state visible, and prevent accidental page zoom or rubber-band movement from
  making the interface feel unstable.
- Treat image export as an explicit action on iPhone: tapping generated work
  opens the full viewer, while Copy image and Save photo replace fragile
  long-press behavior. New and Upscaled state cues stay aligned with desktop.

## Accessibility & Inclusion

Target WCAG 2.2 AA contrast for text and controls. Support keyboard and switch navigation, clear focus treatment, screen-reader semantics, reduced motion, system-level magnification, and state cues that do not depend on hue alone. Generated images and video must remain color-accurate across themes.
