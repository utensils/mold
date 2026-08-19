- **iPhone Create can route generations automatically.** With two or more
  connected machines reachable, the Host control adds **Auto** (the least busy
  machine that already has the model) and **Most capable** (the strongest GPU
  that has it — CUDA before Metal, then VRAM), the model picker becomes the
  union across those machines with a per-model availability tag, and the chosen
  machine is frozen into the same recovery record a pinned machine uses, so a
  killed app re-attaches to the exact host. With one machine nothing changes.
  Desktop, web, and iPhone now share one routing policy in
  `studio/lib/hostRouting.ts`
  ([#1163](https://github.com/utensils/mold/issues/1163)).
