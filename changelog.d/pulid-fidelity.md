- **PuLID identity conditioning now masks the face crop the way upstream does.**
  The aligned 512 crop is segmented by facexlib's BiSeNet parser before the
  vision tower sees it — background painted white, face converted to greyscale
  — which is what PuLID conditions on. Mold's identity now reproduces
  upstream's to within 1.0e-5 of its own peak, against 1.5e-2 before, so a
  generated face follows the reference photograph much more closely. The parser
  arrives as a fifth file in the hidden `pulid-flux` bundle; run
  `mold pull pulid-flux` to repair an existing install
  ([#1225](https://github.com/utensils/mold/issues/1225)).
- **`mold pull pulid-flux` needs no extra licence acceptance for the parser.**
  facexlib is MIT, weights included, so only the two InsightFace antelopev2
  models keep their recorded non-commercial acceptance
  ([#1225](https://github.com/utensils/mold/issues/1225)).
