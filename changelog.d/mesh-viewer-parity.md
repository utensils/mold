- **Desktop's 3-D view works again.** The Library lightbox and the Create
  canvas both fell back to "The 3-D view couldn't start, so here's the
  poster" for every Hunyuan3D print in a packaged desktop build, because the
  app's content-security policy refused the viewer's `blob:` fetch. The
  lightbox also now shows the print's poster while a mesh loads, instead of
  a black area.
- **One camera for the poster, the 3-D viewer, and a turntable.** Opening the
  interactive viewer now lands on exactly the thumbnail's view — orthographic,
  framed once to the mesh's own sweep extent — and "reset view" returns
  there. Posters of existing prints are re-rendered to match.
- **Turntables spin the way you'd drag them.** A rendered turntable now turns
  the same direction a rightward drag turns the mesh in the 3-D viewer (and
  the way auto-rotate tours it). A turntable exported before this change
  spins the opposite way from one exported after it.
