# Requested-duration video across surfaces

The owner authorized merge and deployment after independent review, completed
UAT and green checks. Temporary UAT builds are permitted. Final production
deployment and installer refresh follow a green main revision.

## Interaction

- Keep one prompt and the selected recipe's normal short duration as the default.
- Show **Duration**, the single-render limit, and the planned joined clip count.
- Offer **Define scenes** as an optional expansion with per-scene prompts/lengths.
- Keep a chosen duration when switching models; recompute the plan for that model.
- Offer **Extend video** on an existing Library movie: prompt plus seconds to add.
  Publish the original plus extension as a new movie and retain the original.
- **Use settings** restores an editable draft with the saved duration, literal
  prompts, exact seed and authenticated inputs. For extensions, restore added
  time and explain that the original video is not attached. Legacy authored
  clips retain their individual settings; unavailable provenance refuses before
  changing the current draft.
- Keep exact **Resume** separate from **Start again** and **Send to another
  machine**. New attempts preserve original inputs and validate today's recipe.

## Implementation contracts

The capability authority remains the server's generation profile. A durable
duration parent owns a sequence of private one-generation children. Each clip
uses the normal scheduler and releases its generation slot between clips.
Recipe-selected continuation or blended joins assemble one file without
accumulating all video frames in RAM or GPU memory. Continuous PCM is encoded
once into the final movie. There is no fixed overall clip-count/duration cap.

Admission seals media before accepting the known operation UUID. Lost replies
reconcile that UUID rather than creating duplicate work. Immutable child
ownership survives cleanup; public queue controls cannot dispatch or alter a
private stage independently. Durable checkpoints bind output and receipt hashes.
Only the finished parent appears in Library.

Restart exports preserve literal prompts, full-width seeds and original media.
Cross-host prefix transfer uses bounded chunks and verifies the complete source
hash at the destination. Resume retains frozen recipe authority; a new attempt
validates the destination's current recipe. Placement and progress never move.

## Coverage and acceptance

| Surface              | Required proof                                                                                                                            |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| CLI and Rust SDK     | One prompt/duration; Library extension; known-ID recovery; same-host restart and cross-host transfer; no duplicate POST after uncertainty |
| Web                  | Playwright generation, extension, scene planning and durable Queue actions                                                                |
| Tauri desktop/mobile | Shared composer and Library flows, shell build checks and browser-visible interaction proof                                               |
| Native macOS         | Composer, Library extension, durable Queue/restart controls; debug-app UAT                                                                |
| Native iOS           | Composer/keyboard dismissal, extension and durable Queue/restart controls; Simulator UAT identified separately from physical-device proof |
| Discord              | Duration and optional next prompt; Library extension; owner receipt and recovery command tests                                            |
| Agent skill          | Concise duration-first examples; independent Luna blind execution; correct published Library filename                                     |

All actual generation UAT targets Plato's active `/storage/mold`, publishing to
its normal output Library in **Video chaining UI UAT**. Receipts are retained
under `/storage/mold/uat/video-chaining-ui-20261020`. Preserve unrelated GPU
workloads. The final acceptance movie is one minute of H3 Fast text-to-video,
visible from the active Library, with duration, audio and ownership verified.

Independent reviews cover the shared contracts, browser/native behavior,
backend ownership/cancellation, output assembly and CLI recovery. After all
checks and UAT, merge the reviewed feature, deploy main and refresh the local
nightly CLI and skills.
