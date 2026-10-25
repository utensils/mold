- **One notification per finished video.** A Duration video made from several
  clips used to post a Mac notification as each clip finished, because every
  clip is briefly published before the finished video replaces it. The server
  now marks those clip publications as internal work (`gallery_added` gains an
  additive `internal` flag), and the Mac app and the desktop's new-prints badge
  skip them, so only the finished video is announced.
