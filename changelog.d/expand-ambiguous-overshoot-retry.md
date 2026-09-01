- **One rambling completion no longer kills a whole expansion batch.** When an
  expansion backend answered a chunk with more prompts than were requested —
  the shape a small local model reliably produces when a batch template asks it
  for exactly one variation — the request failed outright with
  `expansion backend returned 3 prompts when exactly 1 were requested`, even
  though the chunk had two unused retries left. That response now spends one
  attempt and asks again, and the failure after a chunk's whole budget says how
  many prompts it did assemble.
- **Expansion chunks are evened out instead of filled greedily.** A batch of
  five was asked for four prompts and then a lone one, which handed a batch
  instruction — "generate 1 distinct prompts, output as a JSON array of 1
  strings" — to the model; it is now asked for three and then two, at no extra
  cost in model calls.
- **A short run of one-prompt JSON arrays is no longer glued into one prompt.**
  A backend that emits `["a"]` per line and came up short had its lines joined
  into a single literal `["a"] ["b"]` prompt; those lines are now kept as the
  prompts they are and the remainder is retried.
- **The host records why an expansion failed.** A failed expansion left only a
  bare `500` in the server journal, so the reason existed nowhere but in the
  client's error toast. Backend URL credentials are redacted from that entry.
