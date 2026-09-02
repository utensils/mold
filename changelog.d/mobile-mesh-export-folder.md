- **Save 3-D exports to a Mold folder on iPhone and Android.** Every 3-D
  export in the phone's Library viewer — the stored GLB itself, the OBJ / STL
  / PLY transcodes, and the GIF / APNG / WebP turntables — now offers two
  actions: **Share…** (the native share sheet, as before) and **Save to Mold
  folder**, which writes the file into an on-device folder you can browse.
  On iPhone that is Files ▸ On My iPhone ▸ Mold (the app's Documents folder,
  now exposed to the Files app); on Android it is Downloads/Mold through
  MediaStore, or the public Downloads directory on releases before Android 10. A second export of the same print gets a numbered name (`chair (2).stl`)
  instead of overwriting, the turntable's options sheet carries the same
  Share / Save choice, and the status line names where the file went. The
  list stays whatever the host advertises on
  `capabilities.mesh.export_formats`.
