- **The terminal now shows what a host adjusted or dropped.** Advisories about
  an accepted request — a lip-dub render retimed to its reference clip, or a
  filing a host could not apply — ride the `x-mold-request-warning` header, but
  no client read it. `mold run` and `mold chain` now print each one, and the TUI
  can read the same values off the completed generation. `GenerateResponse` and
  `ChainResponse` gain an additive `request_warnings` list for API consumers.
