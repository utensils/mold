# H3 FL2VA endpoint admission

The native first+last request reached hal9000 with 30 prompt tokens and 2,032
presentation rows. The compiled FL2VA profile was copied from a first-only
qualification: it capped text at 2,048 rows, vision at 4,032 patches,
conditioning at 1,008 packed rows, and identity at one first endpoint. Merely
raising the text cap would expose the remaining identity/row refusals and
undercharge memory.

## Implementation

- Use the already validated FL2VA mode for both early admission and authenticated
  runtime qualification, including re-opening the frozen attempt.
- Preserve the first-only 2,048 text-row budget. Add each additional endpoint's
  1,008 maximum vision pads and presentation allowance; cap vision patches and
  conditioning latents by endpoint count. Last-only uses the same endpoint
  policy, without weakening
  duplicate/order checks. Text-only remains unavailable under the current
  qualification and is tracked separately in #1552.
- Scale Qwen and denoise grants using the additional conditioner and packed rows.
  Keep historical measurement denominators and the current canvas/frame limits.
  Condition-VAE peak remains per encode, not the sum of sequential encodes.
- Keep private UAT records bound to their authenticated ceilings. Do not silently
  truncate prompts or change preprocessing, sampling, or model weights.
- Leave native attachment UI intact: its capability contract is correct.

## Upstream evidence

ComfyUI commit b87fe48b0491425f682f7ffdaed56d0387cb6c5d,
`comfy_extras/nodes_minimax_h3.py:140-163` accepts optional first/last frames,
normalizes each, and encodes each endpoint. `comfy/text_encoders/minimax.py:193-197`
adds one numbered Picture label and vision block per image before prompt text.
The existing Mold pipeline already implements these modes; this fix aligns its
admission policy with them and does not change numerical inference.

## Verification and delivery

First reproduce the 2,062-row request with a failing contract test. Cover the
three nonempty FL2VA endpoint modes, endpoint identity/order, prompt headroom,
row ceilings, memory scaling, and authenticated qualification revalidation.
Run the H3 inference suite with public and private-UAT configurations and relevant
format/static gates. Obtain independent sub-agent review before PR creation,
address findings, then require green checks on the final PR head before merge.
Live GPU output quality is not proved by hermetic admission tests; no production
jobs are retried or changed as part of this source fix.
