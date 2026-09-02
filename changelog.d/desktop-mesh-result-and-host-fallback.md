- **Fixed: a finished 3-D print now renders on the desktop Create canvas.** A
  completed Hunyuan3D generation drew a broken-resource icon over black, with
  no viewer controls and no geometry caption, while the same print opened
  correctly from the Library. The result stage now mounts the mesh viewer on
  the print's own media URL and captions it with the geometry the viewer read
  ([#1534](https://github.com/utensils/mold/issues/1534)).
- **Fixed: the desktop settings panel keeps a 3-D model's controls when the
  target machine has to download it.** Aiming Create at a machine without the
  selected mesh checkpoint replaced the Mesh group with raster controls and a
  Resolution field showing `NaN×NaN px` under an uncorrectable validation
  error. The checkpoint's own advertised recipe answers for the settings
  wherever it is installed, and a 3-D family stays canvasless even when no
  recipe can be resolved at all
  ([#1534](https://github.com/utensils/mold/issues/1534)).
