# Tiny playback endpoint fixtures

All clips are deterministic 32×32 H.264/yuv420p. Their last displayed frame is blue; preceding frames are red. `endpoint-single` contains one blue frame. These files exercise actual AVFoundation local/secure decoding without requiring ffmpeg in tests.

| File | Final presentation timestamp (seconds) | Purpose |
| --- | ---: | --- |
| endpoint-bframes.mp4 | 2 | Nine frames at 4fps; final decode-order sample presents at 1.75, before the final displayed sample |
| endpoint-vfr.mp4 | 3 | Uneven 0.25/0.5-second presentation spacing |
| endpoint-audio-overhang.mp4 | 2 | Video ends at 2.25 seconds; AAC continues to approximately 2.955 |
| endpoint-silent-tail.mp4 | 2 | Silent AAC extends beyond video; native UI acceptance makes no host sound |
| endpoint-offset.mp4 | 3 | One-second initial empty edit, followed by the B-frame clip |
| endpoint-single.mp4 | 0 | One frame, duration 0.25 seconds |

The MP4 B-frame edit maps source time 0.5…2.75 to presentation time 0…2.25. Raw sample-cursor timestamps must be mapped through track segments before use as player timestamps. `ffprobe -show_packets` provides an independent presentation-order oracle; AVAssetImageGenerator.actualTime verifies the asset timeline.

Regenerate with ffmpeg/libx264 (commands below), then inspect packet PTS/DTS and the final blue image before updating golden expectations:

```sh
ffmpeg -f lavfi -i 'color=c=red:s=32x32:r=4:d=2.25' -vf 'drawbox=color=blue:t=fill:enable=gte(n\,8)' -c:v libx264 -pix_fmt yuv420p -bf 2 -g 12 -x264-params 'b-adapt=0:scenecut=0' -movflags +faststart endpoint-bframes.mp4
ffmpeg -i endpoint-bframes.mp4 -f lavfi -i 'sine=frequency=440:sample_rate=24000:duration=3' -map 0:v -map 1:a -c:v copy -c:a aac -movflags +faststart endpoint-audio-overhang.mp4
ffmpeg -i endpoint-bframes.mp4 -f lavfi -i anullsrc=r=24000:cl=mono -t 3 -map 0:v -map 1:a -c:v copy -c:a aac -movflags +faststart endpoint-silent-tail.mp4
ffmpeg -f lavfi -i 'color=c=red:s=32x32:r=4:d=2.25' -vf 'drawbox=color=blue:t=fill:enable=gte(n\,8),setpts=if(lt(N\,4)\,N/4/TB\,(N-4)/2/TB+1/TB)' -fps_mode vfr -c:v libx264 -pix_fmt yuv420p -bf 2 -g 12 -x264-params 'b-adapt=0:scenecut=0' -movflags +faststart endpoint-vfr.mp4
ffmpeg -i endpoint-bframes.mp4 -c copy -output_ts_offset 1 endpoint-offset.mp4
ffmpeg -f lavfi -i 'color=c=blue:s=32x32:r=4' -frames:v 1 -c:v libx264 -pix_fmt yuv420p endpoint-single.mp4
ffprobe -v error -select_streams v:0 -show_packets -show_entries packet=pts_time,dts_time,duration_time -of json endpoint-bframes.mp4
```
