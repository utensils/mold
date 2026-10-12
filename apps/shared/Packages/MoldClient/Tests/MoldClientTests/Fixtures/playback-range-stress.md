# Generated playback fixture

`playback-range-stress.mp4` is a three-second, silent H.264 720p test pattern,
generated locally without external media:

```sh
ffmpeg -v error -f lavfi -i testsrc2=size=1280x720:rate=24 -t 3 -an \
  -c:v libx264 -preset veryfast -crf 12 -pix_fmt yuv420p -movflags +faststart \
  playback-range-stress.mp4
```

`SecureMediaPlaybackTests` requires decoded frames and sustained AVPlayer clock
advancement through the production authenticated resource loader, then verifies
seeking, range requests, credentials and Trash routing. Run it on iOS Simulator:

```sh
xcodebuild test -scheme MoldClient-Package \
  -destination 'platform=iOS Simulator,id=<simulator-udid>' \
  -only-testing:MoldClientTests/SecureMediaPlaybackTests
```

The smaller `reference-video.mp4` did not cover an actual running player.
