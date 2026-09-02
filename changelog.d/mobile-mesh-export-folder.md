- **Save 3-D exports to a Mold folder on iPhone and Android.** Every 3-D
  export in the phone's Library viewer — the stored GLB itself, the OBJ / STL
  / PLY transcodes, and the GIF / APNG / WebP turntables — now offers two
  actions: **Share…** (the native share sheet, as before) and **Save to Mold
  folder**, which writes the file into an on-device folder you can browse.
  On iPhone that is the app's own Documents folder, now exposed to the Files
  app as On My iPhone ▸ Mold (the entry appears after the first save); on
  Android it is the `Download/Mold` directory (Downloads ▸ Mold in the Files
  app), through MediaStore on Android 10 and later. A second export of the
  same print gets a numbered name (`chair (2).stl`) instead of overwriting,
  the turntable's options sheet carries the same Share / Save choice, and
  the status line names where the file went. The list stays whatever the
  host advertises on `capabilities.mesh.export_formats`.
- **Shared 3-D exports and turntables carry the print's own filename.** The
  phone's share sheet used to show a staged temp name
  (`mold-export-123-chair.stl`); each export is now staged under its real
  name, so the share sheet and a "Save to Files" from it show `chair.stl`.
