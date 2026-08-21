- **Identity conditioning cannot be promised before it can run.**
  `supports_identity` is advertised, and `id_image` requests admitted, only on
  a build whose runtime adapter is present — the `pulid` feature alone no
  longer advertises the capability, and a request on a build that cannot
  execute it is refused with a distinct message instead of rendering a print
  with no face in it. Auto-materialized PuLID assets are verified against their
  manifest SHA-256 pins before the paths are frozen, and the InsightFace
  license is pinned to an immutable upstream commit so acceptance is bound to
  exact terms ([#1220](https://github.com/utensils/mold/issues/1220)).
