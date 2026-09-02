//! Turntable animation of a stored mesh: the poster set spinning.
//!
//! Shared by the server's `POST /api/gallery/export/:filename` route and the
//! TUI's in-process export of a local print, so a `.glb` exports to the same
//! bytes whichever machine renders it. Frames come from
//! [`super::poster::render_sequence_frame_rgb`] over [`super::poster::turntable_cameras`]
//! and go into the animation encoders every other mold export uses
//! (`ltx_video::video_enc`), so a mesh GIF and a video GIF share one
//! quantizer, one loop/bounce contract and one repeat contract.
//!
//! See [`super::poster::turntable_cameras`] for why a loop is a full turn
//! stopping one step short and a bounce is a half turn played back.

use std::ops::RangeInclusive;

use anyhow::{bail, Result};
use image::RgbImage;

use crate::hunyuan3d::mesh::Mesh;
use crate::hunyuan3d::poster::{render_sequence_frame_rgb, turntable_cameras, MAX_POSTER_SIZE};
use crate::ltx_video::video_enc;

/// Frames in a turntable unless the request says otherwise: a 10° step, which
/// at [`DEFAULT_FPS`] is a 3.6 s turn — slow enough to read the shape.
pub const DEFAULT_FRAMES: usize = 36;
/// Accepted `frames`. Eight is the coarsest sweep that still reads as a
/// rotation rather than a slideshow; 180 is a 2° step, past which more frames
/// only cost bytes.
pub const FRAMES_RANGE: RangeInclusive<usize> = 8..=180;
/// Frame rate unless the request says otherwise.
pub const DEFAULT_FPS: u32 = 10;
/// Accepted `fps`. Thirty is already a 1.2 s turn at the default frame count;
/// the video export's 60 would spin the default sweep in half a second.
pub const FPS_RANGE: RangeInclusive<u32> = 1..=30;
/// Frame edge unless the request says otherwise. The poster's own size.
pub const DEFAULT_SIZE: u32 = 512;

/// Upper bound on the decoded frame buffer, the same figure the video export
/// holds itself to (`ltx2::media::MAX_ANIMATION_EXPORT_RGB_BYTES`): a
/// turntable is the same kind of object once rendered, and the GIF encoder
/// makes an RGBA copy of every frame on top of this.
pub const MAX_TURNTABLE_RGB_BYTES: u64 = 256 * 1024 * 1024;

/// How a turntable is rendered and played back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TurntableOptions {
    /// Rendered frames. See [`turntable_cameras`] for what a frame count
    /// means under each sweep.
    pub frames: usize,
    pub fps: u32,
    /// Frame edge in pixels; frames are square like the poster.
    pub size: u32,
    /// Sweep a half turn and play it back, instead of a full turn. GIF only.
    pub bounce: bool,
    /// `false` plays once and rests on the final frame. GIF only — APNG and
    /// WebP animations loop.
    pub repeat_forever: bool,
}

impl Default for TurntableOptions {
    fn default() -> Self {
        Self {
            frames: DEFAULT_FRAMES,
            fps: DEFAULT_FPS,
            size: DEFAULT_SIZE,
            bounce: false,
            repeat_forever: true,
        }
    }
}

/// Refuse a request whose frame buffer would exceed
/// [`MAX_TURNTABLE_RGB_BYTES`], before any frame renders.
///
/// The message names both knobs because either brings it under: 180 frames
/// fit at the default size, and the largest size fits at a short sweep.
pub fn check_frame_budget(options: &TurntableOptions) -> std::result::Result<(), String> {
    let bytes = options.frames as u64 * options.size as u64 * options.size as u64 * 3;
    if bytes > MAX_TURNTABLE_RGB_BYTES {
        return Err(format!(
            "{} frames at {} px need {} MiB of frame buffer, over the {} MiB export budget; lower frames or max_dimension",
            options.frames,
            options.size,
            bytes / (1024 * 1024),
            MAX_TURNTABLE_RGB_BYTES / (1024 * 1024)
        ));
    }
    Ok(())
}

/// Render every frame of the turntable, in playback order for a loop and in
/// sweep order for a bounce (the encoder appends the reversal).
pub fn render_turntable(mesh: &Mesh, options: &TurntableOptions) -> Result<Vec<RgbImage>> {
    if !FRAMES_RANGE.contains(&options.frames) {
        bail!(
            "frames must be between {} and {}",
            FRAMES_RANGE.start(),
            FRAMES_RANGE.end()
        );
    }
    if options.size == 0 || options.size > MAX_POSTER_SIZE {
        bail!(
            "frame size {} is outside 1..={MAX_POSTER_SIZE}",
            options.size
        );
    }
    if let Err(message) = check_frame_budget(options) {
        bail!("{message}");
    }
    turntable_cameras(options.frames, options.bounce)
        .iter()
        .map(|camera| render_sequence_frame_rgb(mesh, camera, options.size))
        .collect()
}

/// Render and encode a turntable of `mesh` as `format`.
///
/// `format` is one of the animation containers (`Gif`, `Apng`, `Webp`);
/// anything else is refused by name, because a raster format cannot hold a
/// sequence and a mesh format is not an animation. Bounce is refused outside
/// GIF exactly as the video export refuses it, and for the same reason: only
/// the GIF encoder writes a reversal.
pub fn export_turntable(
    mesh: &Mesh,
    format: mold_core::OutputFormat,
    options: &TurntableOptions,
) -> Result<Vec<u8>> {
    use mold_core::OutputFormat;
    if !matches!(
        format,
        OutputFormat::Gif | OutputFormat::Apng | OutputFormat::Webp
    ) {
        bail!(
            "'{}' is not an animation format; a turntable exports as gif, apng, or webp",
            format.extension()
        );
    }
    if options.bounce && format != OutputFormat::Gif {
        bail!("bounce playback is only supported for GIF exports");
    }
    if !FPS_RANGE.contains(&options.fps) {
        bail!(
            "fps must be between {} and {}",
            FPS_RANGE.start(),
            FPS_RANGE.end()
        );
    }
    let frames = render_turntable(mesh, options)?;
    match format {
        OutputFormat::Gif => video_enc::encode_gif_with_options(
            &frames,
            options.fps,
            options.bounce,
            options.repeat_forever,
        ),
        OutputFormat::Apng => video_enc::encode_apng(&frames, options.fps, None),
        OutputFormat::Webp => {
            #[cfg(feature = "webp")]
            {
                video_enc::encode_webp(&frames, options.fps)
            }
            #[cfg(not(feature = "webp"))]
            {
                bail!("WebP export requires a mold build with the webp feature")
            }
        }
        _ => unreachable!("refused above"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hunyuan3d::mesh::Mesh;
    use image::AnimationDecoder;

    fn cube() -> Mesh {
        let half = 0.5;
        let v = [
            [-half, -half, half],
            [half, -half, half],
            // One corner pulled out: a symmetric cube looks the same half a
            // turn later under camera-frame lighting, which would make a
            // turntable that never turned pass.
            [half * 1.8, half * 1.4, half],
            [-half, half, half],
            [-half, -half, -half],
            [half, -half, -half],
            [half, half, -half],
            [-half, half, -half],
        ];
        let quads = [
            [0, 1, 2, 3],
            [5, 4, 7, 6],
            [1, 5, 6, 2],
            [4, 0, 3, 7],
            [3, 2, 6, 7],
            [4, 5, 1, 0],
        ];
        let mut faces = Vec::new();
        for q in quads {
            faces.push([q[0] as u32, q[1] as u32, q[2] as u32]);
            faces.push([q[0] as u32, q[2] as u32, q[3] as u32]);
        }
        Mesh {
            vertices: v.to_vec(),
            faces,
            ..Default::default()
        }
    }

    fn gif_frame_count(bytes: &[u8]) -> usize {
        image::codecs::gif::GifDecoder::new(std::io::Cursor::new(bytes))
            .expect("decode GIF")
            .into_frames()
            .count()
    }

    #[test]
    fn defaults_and_bounds_are_the_advertised_contract() {
        let options = TurntableOptions::default();
        assert_eq!(options.frames, DEFAULT_FRAMES);
        assert_eq!(options.fps, DEFAULT_FPS);
        assert_eq!(options.size, DEFAULT_SIZE);
        assert!(!options.bounce);
        assert!(options.repeat_forever);
        assert_eq!(FRAMES_RANGE, 8..=180);
        assert_eq!(FPS_RANGE, 1..=30);
        assert_eq!(DEFAULT_SIZE, 512);
    }

    /// Every frame is the poster's size, and the mesh actually turns: the
    /// half-way frame of a loop looks at the far side and differs from the
    /// poster.
    #[test]
    fn render_turntable_produces_distinct_square_frames() {
        let frames = render_turntable(
            &cube(),
            &TurntableOptions {
                frames: 8,
                size: 48,
                ..TurntableOptions::default()
            },
        )
        .expect("render turntable");
        assert_eq!(frames.len(), 8);
        for frame in &frames {
            assert_eq!((frame.width(), frame.height()), (48, 48));
        }
        assert_ne!(frames[0], frames[4], "the mesh did not turn");
        assert_ne!(frames[0], frames[7], "a loop's last frame is not the first");
    }

    /// The GIF carries exactly the frames the sweep implies: N for a loop,
    /// 2N - 2 for a bounce that repeats (the encoder drops both turning
    /// points from the reversal so nothing is held twice), 2N - 1 for a
    /// bounce played once (the first frame is the final resting frame).
    #[test]
    fn export_gif_frame_counts_follow_the_playback() {
        let mesh = cube();
        let base = TurntableOptions {
            frames: 8,
            size: 32,
            ..TurntableOptions::default()
        };
        let looped = export_turntable(&mesh, mold_core::OutputFormat::Gif, &base).unwrap();
        assert_eq!(&looped[..6], b"GIF89a");
        assert_eq!(gif_frame_count(&looped), 8);

        let bounce_forever = export_turntable(
            &mesh,
            mold_core::OutputFormat::Gif,
            &TurntableOptions {
                bounce: true,
                ..base
            },
        )
        .unwrap();
        assert_eq!(gif_frame_count(&bounce_forever), 14);

        let bounce_once = export_turntable(
            &mesh,
            mold_core::OutputFormat::Gif,
            &TurntableOptions {
                bounce: true,
                repeat_forever: false,
                ..base
            },
        )
        .unwrap();
        assert_eq!(gif_frame_count(&bounce_once), 15);
    }

    /// A flat mesh passes through edge-on views on its way round, and those
    /// frames are blank by rights; the turntable must not fail on them.
    #[test]
    fn a_flat_mesh_turns_through_its_edge_on_views() {
        let plane = Mesh {
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            faces: vec![[0, 1, 2]],
            ..Default::default()
        };
        let frames = render_turntable(
            &plane,
            &TurntableOptions {
                frames: 36,
                size: 32,
                ..TurntableOptions::default()
            },
        )
        .expect("a plane still turns");
        assert_eq!(frames.len(), 36);
        // The poster view (frame 0) sees the face; the frame at azimuth 90°
        // (index 6, 30° + 6 * 10°) sees the edge and is background only.
        assert!(frames[0].pixels().any(|p| p[0] > 100));
        assert!(frames[6].pixels().all(|p| p[0] < 0x20));
    }

    #[test]
    fn export_apng_is_an_animated_png() {
        let bytes = export_turntable(
            &cube(),
            mold_core::OutputFormat::Apng,
            &TurntableOptions {
                frames: 8,
                size: 32,
                ..TurntableOptions::default()
            },
        )
        .unwrap();
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        assert!(
            bytes.windows(4).any(|chunk| chunk == b"acTL"),
            "no animation control chunk"
        );
        let decoder =
            image::codecs::png::PngDecoder::new(std::io::Cursor::new(bytes.as_slice())).unwrap();
        assert!(decoder.is_apng().unwrap());
        assert_eq!(decoder.apng().unwrap().into_frames().count(), 8);
    }

    #[cfg(feature = "webp")]
    #[test]
    fn export_webp_is_an_animated_webp() {
        let bytes = export_turntable(
            &cube(),
            mold_core::OutputFormat::Webp,
            &TurntableOptions {
                frames: 8,
                size: 32,
                ..TurntableOptions::default()
            },
        )
        .unwrap();
        assert_eq!(&bytes[..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WEBP");
    }

    /// Bounce is a GIF contract (APNG and WebP have no reversal in their
    /// encoders), exactly as the video export refuses it, and a raster
    /// container is never a turntable.
    #[test]
    fn export_refuses_bounce_outside_gif_and_non_animation_formats() {
        let mesh = cube();
        let bounce = TurntableOptions {
            frames: 8,
            size: 32,
            bounce: true,
            ..TurntableOptions::default()
        };
        let error = export_turntable(&mesh, mold_core::OutputFormat::Apng, &bounce)
            .unwrap_err()
            .to_string();
        assert!(error.contains("bounce"), "{error}");
        let error = export_turntable(
            &mesh,
            mold_core::OutputFormat::Png,
            &TurntableOptions::default(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("png"), "{error}");
    }

    /// The frame buffer is bounded the way the video export's is, and a
    /// request over budget is refused BEFORE any frame renders, naming the
    /// two knobs that bring it under.
    #[test]
    fn frame_budget_is_checked_before_rendering() {
        assert!(check_frame_budget(&TurntableOptions::default()).is_ok());
        assert!(check_frame_budget(&TurntableOptions {
            frames: 180,
            size: 512,
            ..TurntableOptions::default()
        })
        .is_ok());
        let error = check_frame_budget(&TurntableOptions {
            frames: 180,
            size: 2048,
            ..TurntableOptions::default()
        })
        .unwrap_err();
        assert!(
            error.contains("frames") && error.contains("max_dimension"),
            "{error}"
        );
        let error = render_turntable(
            &cube(),
            &TurntableOptions {
                frames: 180,
                size: 2048,
                ..TurntableOptions::default()
            },
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("frames"), "{error}");
    }
}
