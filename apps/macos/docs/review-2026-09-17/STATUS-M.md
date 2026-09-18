# Lane M — one picture chooser for every well, and wells that say what they are

Two owner asks (2026-09-17, with screenshots), not review findings:

| id  | ask                                                                                    | status | commit | test                                                                                             |
| --- | -------------------------------------------------------------------------------------- | ------ | ------ | ------------------------------------------------------------------------------------------------ |
| M#1 | one selector -- file or Library -- behind EVERY picture well, identity included          | fixed  | (1)    | `PictureWellTests.everyPictureWellChoosesThroughTheOneComponent`, `.everyWellThatTakesAPictureOffersTheSameThreeDoors` |
| M#2 | the two anonymous squares in the prompt bar say which is which, parked included          | fixed  | (1)    | `PictureWellTests.aParkedWellSaysSoInWords`, `.everyCaptionFitsOnOneLineInTheColumnItIsDrawnIn`, `.aCaptionedWellIsOneLineTallerThanItsSquare`, `.theSourceWellAndTheStripsAddWellAreNotTheSameSquare` |
| M#3 | identity add well: click menu, Library, Paste, drop on the well AND the group            | fixed  | (1)    | `PictureWellTests.aPickedPictureIsStagedAsAnIdentityPhotograph`, `.replacingAStagedPhotographKeepsItsPlace` |
| M#4 | a Library print is conformed the way a file is (WebP is not identity-readable)           | fixed  | (1)    | `PictureWellTests.aLibraryPrintIsConformedToTheWellThatAskedForIt`                                |
| M#5 | README says it                                                                           | fixed  | (2)    | --                                                                                                 |

## What the chooser is

`PictureWell` (`PictureWell.swift` + `+Shape` + `+Import`), with `PictureIntake`
as the one import pipeline behind its three doors and `WellCaption` as the one
caption rule. Call sites: `SourceImageWell`, `ReferenceStrip` (add well AND
each staged reference), `ControlPictureWell`, `IdentityGroup` (add well AND
each staged photograph). `MediaWell` stays apart, and its doc comment's
argument still holds -- it takes a clip, audio or a keyframe still shown by its
FILENAME and draws no preview at all. `ControlPictureWell`'s doc comment used
to argue the same for itself; that argument was about the BINDING shape, and
the binding is now the only thing left in the file.

Deleted: `ReferenceWell.swift`, `IdentityPhotoWell.swift` (both were
"decode an encoded string into a rounded square", which is what the chooser
draws), the source well's and the strip's private import machinery, the strip's
second `LibraryPickerSheet`, and three of the four `NSOpenPanel`s.

## Judgement calls, for the integrator

- **`ImportedPicture`, not a new `PickedPicture`.** The brief sketched
  `(PickedPicture) -> Void`; `ImportedPicture` already IS that type and the
  wave-3 rule is to use the shared type rather than invent a parallel one.
- **The brief's item 4 was already true.** The strip's add well already drew
  `plus` against the source well's `photo.badge.plus`, and already carried a
  `.help` and an accessibility label naming it. Both are now pinned
  (`theSourceWellAndTheStripsAddWellAreNotTheSameSquare`) rather than left to
  drift; the caption is what actually answers the owner's complaint.
- **A staged picture's well does not open its menu on a plain click**
  (`opensOnClick: false`). Its square carries inline controls -- the ✕, the
  Target/order badge -- and a `Menu` label swallows their taps. It keeps the
  identical contextual menu and the drop.
- **The strip's BACKGROUND menu lost Add… and Paste** and is now Remove All
  alone. Both live on the add well, which is drawn whenever there is room, so
  they were two menus one square apart offering the same door; when the strip
  is FULL neither was offered anyway.
- **Two new `GenerateAction` cases**: `.replaceFromLibrary` (one case, shared
  by a reference and an identity photograph -- the WELL knows which slot it is
  replacing, the door does not) and `.removeControl` (the ControlNet well's ✕
  became a menu row, because a `Menu` label would have swallowed the button).
  `.replacePicture`/`.replacePhoto` are retitled "Replace from File…": both
  read "Choose File…", which says nothing about what happens to the picture
  already in the slot.
- **`KeyframeTable` now opens `PictureSource.choose()`** instead of its own
  panel -- a one-line change that also means an iPhone photograph can be a
  keyframe at all (its panel offered PNG and JPEG only, and `PictureImport`
  transcodes the rest).
- **`PicturePaste` names its bytes `.tiff`**, which is what the pasteboard
  hands over. `PictureImport.conform` renames only what it TRANSCODES, so
  calling them a PNG shipped TIFF under a PNG name to every well that reads
  TIFF.
- **One feature commit, not three.** The chooser, the identity doors and the
  captions land on the same wells in the same files; a commit that rebuilt the
  wells without their captions, or the reverse, would not compile.

## Cross-lane edits

None. Every file touched is under `Sources/Mold/Generate/**`, its tests, or
`README.md`. `GenerateController+Expand`, `GeneratePane.swift`, `RenderDraft*`
and `ReuseStore*` (Lane L's) are untouched.
