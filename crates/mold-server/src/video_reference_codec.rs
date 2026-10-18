//! Bounded server-only fresh-video normalization. Inference never invokes
//! ffmpeg. See ffmpeg.org/ffmpeg.html (mapping, fps_mode, autorotate) and
//! ffmpeg.org/ffmpeg-protocols.html (protocol_whitelist).
use crate::{reference_uploads::MAX_REFERENCE_UPLOAD_FILE_BYTES, routes::ApiError};
use axum::http::StatusCode;
use mold_core::video_normalization::{
    VideoNormalizationCapabilities, CANDIDATE_DIGEST_HEADER, CANDIDATE_PATH, FINALIZE_PATH, VERSION,
};
use std::{path::Path, process::Stdio, time::Duration};
use tokio::{
    io::{AsyncRead, AsyncReadExt},
    process::Command,
};
use tokio_util::sync::CancellationToken;

const MAX_PROBE_BYTES: usize = 2 * 1024 * 1024;
const MAX_STDERR_BYTES: usize = 64 * 1024;
const MAX_FRAMES: usize = 3601;
const MAX_PIXELS: u64 = 16_777_216;
fn error(message: impl Into<String>) -> ApiError {
    ApiError::with_code(
        message,
        "REFERENCE_VIDEO_NORMALIZATION",
        StatusCode::UNPROCESSABLE_ENTITY,
    )
}
fn command(tool: &str) -> Command {
    let mut command: Command = crate::video_upscale::codec_command(tool).into();
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    #[cfg(unix)]
    // SAFETY: the child hook calls only async-signal-safe setrlimit. It neither
    // allocates nor acquires a lock. The hard file bound applies before writes.
    unsafe {
        command.pre_exec(|| {
            let limit = libc::rlimit {
                rlim_cur: MAX_REFERENCE_UPLOAD_FILE_BYTES as libc::rlim_t,
                rlim_max: MAX_REFERENCE_UPLOAD_FILE_BYTES as libc::rlim_t,
            };
            if libc::setrlimit(libc::RLIMIT_FSIZE, &limit) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            #[cfg(target_os = "linux")]
            {
                let memory = libc::rlimit {
                    rlim_cur: 2 * 1024 * 1024 * 1024,
                    rlim_max: 2 * 1024 * 1024 * 1024,
                };
                if libc::setrlimit(libc::RLIMIT_AS, &memory) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
            }
            Ok(())
        });
    }
    command
}
async fn read_bounded(
    reader: impl AsyncRead + Unpin,
    limit: usize,
) -> Result<Vec<u8>, std::io::Error> {
    let mut bytes = Vec::new();
    reader
        .take(limit as u64 + 1)
        .read_to_end(&mut bytes)
        .await?;
    Ok(bytes)
}
async fn run(
    mut command: Command,
    cancel: &CancellationToken,
    timeout: Duration,
    limit: usize,
) -> Result<Vec<u8>, ApiError> {
    let mut child = command
        .spawn()
        .map_err(|_| error("video codec tool could not start"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| error("video codec output unavailable"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| error("video codec diagnostics unavailable"))?;
    let out = tokio::spawn(read_bounded(stdout, limit));
    let err = tokio::spawn(read_bounded(stderr, MAX_STDERR_BYTES));
    let status = tokio::select! {
        value = child.wait() => value.map_err(|_| error("video codec process could not be reaped")),
        _ = cancel.cancelled() => { let _ = child.kill().await; let _ = child.wait().await; Err(error("video reference conversion cancelled")) },
        _ = tokio::time::sleep(timeout) => { let _ = child.kill().await; let _ = child.wait().await; Err(error("video reference conversion exceeded its time limit")) },
    };
    let output = out
        .await
        .map_err(|_| error("video codec output reader failed"))?
        .map_err(|_| error("video codec output read failed"))?;
    let diagnostics = err
        .await
        .map_err(|_| error("video codec diagnostic reader failed"))?
        .map_err(|_| error("video codec diagnostic read failed"))?;
    let status = status?;
    if output.len() > limit || diagnostics.len() > MAX_STDERR_BYTES {
        return Err(error("video codec exceeded its bounded output limit"));
    }
    if !status.success() {
        return Err(error("video reference codec/probe failed; the file may be corrupt, truncated, unsupported or exceed conversion limits"));
    }
    Ok(output)
}

/// One async, cached runtime inspection. Probe decoders, demuxers, encoders
/// and the exact canonical probe's supported audio path; version alone cannot
/// authorize this capability.
#[derive(Default)]
struct CodecCache {
    value: Option<VideoNormalizationCapabilities>,
    checked_at: Option<std::time::Instant>,
    probing: bool,
}
fn codec_cache() -> &'static std::sync::Mutex<CodecCache> {
    static CACHE: std::sync::OnceLock<std::sync::Mutex<CodecCache>> = std::sync::OnceLock::new();
    CACHE.get_or_init(Default::default)
}
fn refresh_due(cache: &CodecCache) -> bool {
    !cache.probing
        && cache.checked_at.is_none_or(|checked| {
            checked.elapsed()
                >= if cache.value.is_some() {
                    Duration::from_secs(300)
                } else {
                    Duration::from_secs(5)
                }
        })
}
/// Warm independently of HTTP requests. Failed/transient probes retry after a
/// short cooldown; they never disable normalization for the process lifetime.
pub(crate) fn warm_up() {
    let Ok(runtime) = tokio::runtime::Handle::try_current() else {
        return;
    };
    {
        let mut cache = codec_cache()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !refresh_due(&cache) {
            return;
        }
        cache.probing = true;
    }
    struct ResetProbe;
    impl Drop for ResetProbe {
        fn drop(&mut self) {
            codec_cache()
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .probing = false;
        }
    }
    let reset = ResetProbe;
    runtime.spawn(async move {
        let _reset = reset;
        let value = inspect_runtime().await;
        let mut cache = codec_cache()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        cache.value = value;
        cache.checked_at = Some(std::time::Instant::now());
    });
}
/// Discovery reads the latest completed proof without waiting on subprocesses.
/// Startup and later capability reads trigger bounded background refreshes.
pub(crate) async fn capabilities() -> Option<VideoNormalizationCapabilities> {
    warm_up();
    codec_cache()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .value
        .clone()
}

async fn inspect_runtime() -> Option<VideoNormalizationCapabilities> {
    // Normalization is optional on each host. Do not advertise conversion
    // where this implementation cannot enforce the child output-file
    // hard limit; canonical V2 remains available on those hosts.
    if !cfg!(unix) {
        return None;
    }
    let cancel = CancellationToken::new();
    let mut supported = Vec::new();
    for (tool, argument) in [
        ("ffmpeg", "-decoders"),
        ("ffmpeg", "-demuxers"),
        ("ffmpeg", "-encoders"),
        ("ffprobe", "-version"),
    ] {
        let mut cmd = command(tool);
        cmd.args(["-hide_banner", argument]);
        supported.push(
            String::from_utf8(
                run(cmd, &cancel, Duration::from_secs(5), MAX_PROBE_BYTES)
                    .await
                    .ok()?,
            )
            .ok()?,
        );
    }
    let has = |text: &str, name: &str| {
        text.lines()
            .any(|line| line.split_whitespace().nth(1) == Some(name))
    };
    if !has(&supported[2], "libx264")
        || !has(&supported[2], "aac")
        || !has(&supported[0], "aac")
        || !supported[1].lines().any(|line| {
            line.split_whitespace()
                .nth(1)
                .is_some_and(|names| names.split(',').any(|name| name == "mov"))
        })
        || !has(&supported[1], "matroska,webm")
    {
        return None;
    }
    let codecs = ["h264", "hevc", "vp9", "av1", "prores"]
        .into_iter()
        .filter(|codec| {
            has(&supported[0], codec) || (*codec == "av1" && has(&supported[0], "libdav1d"))
        })
        .map(str::to_string)
        .collect::<Vec<_>>();
    if !codecs.iter().any(|codec| codec == "h264") {
        return None;
    }
    if !runtime_smoke_test(&cancel).await {
        return None;
    }
    Some(VideoNormalizationCapabilities {
        available: true,
        protocol_version: VERSION,
        input_roles: vec!["reference_video".into()],
        candidate_session_path: CANDIDATE_PATH.into(),
        finalize_path: FINALIZE_PATH.into(),
        candidate_digest_header: CANDIDATE_DIGEST_HEADER.into(),
        max_original_file_bytes: MAX_REFERENCE_UPLOAD_FILE_BYTES,
        max_canonical_file_bytes: MAX_REFERENCE_UPLOAD_FILE_BYTES,
        max_session_bytes: crate::reference_uploads::MAX_REFERENCE_UPLOAD_SESSION_BYTES,
        accepted_mime_types: [
            "video/mp4",
            "video/quicktime",
            "video/x-m4v",
            "video/webm",
            "video/x-matroska",
        ]
        .map(str::to_string)
        .to_vec(),
        accepted_extensions: ["mp4", "mov", "m4v", "webm", "mkv"]
            .map(str::to_string)
            .to_vec(),
        accepted_video_codecs: codecs,
        canonical_mime_type: "video/mp4".into(),
    })
}

async fn input_format(path: &Path) -> Result<&'static str, ApiError> {
    let mut file = tokio::fs::File::open(path)
        .await
        .map_err(|_| error("video input is unavailable"))?;
    let mut header = [0; 12];
    file.read_exact(&mut header)
        .await
        .map_err(|_| error("video container header is truncated"))?;
    if header[..4] == [0x1a, 0x45, 0xdf, 0xa3] {
        return Ok("matroska");
    }
    if matches!(
        &header[4..8],
        b"ftyp" | b"moov" | b"wide" | b"free" | b"mdat" | b"skip"
    ) {
        return Ok("mov");
    }
    Err(error(
        "only MP4/MOV/M4V/WebM/MKV video containers are supported",
    ))
}
fn input_options(command: &mut Command, format: &str) {
    command.args([
        "-protocol_whitelist",
        "file",
        "-format_whitelist",
        "mov,matroska,webm",
        "-f",
        format,
        "-threads",
        "2",
    ]);
    if format == "mov" {
        command.args(["-enable_drefs", "0"]);
    }
}
fn number(value: &serde_json::Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_str()?.parse::<f64>().ok())
        .filter(|n| n.is_finite())
}
fn rate(value: &serde_json::Value) -> Option<f64> {
    let value = value.as_str()?;
    let (numerator, denominator) = value.split_once('/')?;
    let rate = numerator.parse::<f64>().ok()? / denominator.parse::<f64>().ok()?;
    (rate.is_finite() && rate > 0.0 && rate <= 240.0).then_some(rate)
}
struct VideoFacts {
    duration: f64,
    fps: u32,
    canonical: bool,
}
fn inspect(json: &serde_json::Value, codecs: &[String]) -> Result<VideoFacts, ApiError> {
    let streams = json["streams"]
        .as_array()
        .ok_or_else(|| error("video probe omitted streams"))?;
    if streams.len() > 32 {
        return Err(error("video has too many streams"));
    }
    let video = streams
        .iter()
        .find(|stream| stream["codec_type"] == "video")
        .ok_or_else(|| error("reference contains no video stream"))?;
    let codec = video["codec_name"]
        .as_str()
        .ok_or_else(|| error("video codec is unavailable"))?;
    if !codecs.iter().any(|accepted| accepted == codec) {
        return Err(error("video codec is not supported by this server"));
    }
    let width = video["width"].as_u64().unwrap_or(0);
    let height = video["height"].as_u64().unwrap_or(0);
    if width == 0
        || height == 0
        || width > 8192
        || height > 8192
        || width.saturating_mul(height) > MAX_PIXELS
    {
        return Err(error(
            "video normalization is limited to 8192 pixels per axis and 16 megapixels per frame",
        ));
    }
    let frames = json["frames"]
        .as_array()
        .ok_or_else(|| error("video timing could not be decoded"))?;
    let times = frames
        .iter()
        .filter(|frame| frame["media_type"] == "video" && frame["stream_index"] == video["index"])
        .map(|frame| number(&frame["best_effort_timestamp_time"]))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| error("video presentation timestamps are unavailable"))?;
    if times.is_empty()
        || times.len() > MAX_FRAMES
        || times.windows(2).any(|pair| pair[1] <= pair[0])
    {
        return Err(error("video presentation timing or frame count is invalid"));
    }
    let average = rate(&video["avg_frame_rate"]);
    let target_fps = average.unwrap_or(24.0).round().clamp(1.0, 240.0) as u32;
    let frame_duration = frames
        .iter()
        .rev()
        .find(|frame| frame["media_type"] == "video" && frame["stream_index"] == video["index"])
        .and_then(|frame| number(&frame["pkt_duration_time"]))
        .filter(|duration| *duration > 0.0)
        .unwrap_or(1.0 / average.unwrap_or(24.0));
    let observed_duration = times[times.len() - 1] - times[0] + frame_duration;
    let duration = number(&video["duration"])
        .filter(|duration| *duration > 0.0)
        .unwrap_or(observed_duration)
        .max(observed_duration);
    if !(0.001..=15.001).contains(&duration) {
        return Err(error("reference video duration must be at most 15 seconds"));
    }
    let cfr = average.is_some_and(|fps| {
        times
            .windows(2)
            .all(|pair| ((pair[1] - pair[0]) - 1.0 / fps).abs() <= 0.0001)
    });
    let rotation = video["tags"]["rotate"]
        .as_str()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(0.0)
        != 0.0
        || video["side_data_list"].as_array().is_some_and(|side| {
            side.iter()
                .any(|item| number(&item["rotation"]).unwrap_or(0.0) != 0.0)
        });
    let audio = streams
        .iter()
        .filter(|stream| stream["codec_type"] == "audio")
        .collect::<Vec<_>>();
    let canonical_audio = audio.len() <= 1
        && audio.iter().all(|stream| {
            stream["codec_name"] == "aac"
                && stream["channels"]
                    .as_u64()
                    .is_some_and(|channels| matches!(channels, 1 | 2))
        });
    let canonical = codec == "h264"
        && video["pix_fmt"] == "yuv420p"
        && cfr
        && !rotation
        && canonical_audio
        && streams
            .iter()
            .all(|stream| stream["codec_type"] == "video" || stream["codec_type"] == "audio")
        && streams
            .iter()
            .filter(|stream| stream["codec_type"] == "video")
            .count()
            == 1;
    Ok(VideoFacts {
        duration,
        fps: target_fps,
        canonical,
    })
}
async fn probe(
    path: &Path,
    format: &str,
    cancel: &CancellationToken,
) -> Result<serde_json::Value, ApiError> {
    let mut cmd = command("ffprobe");
    cmd.args(["-v", "error", "-max_alloc", "67108864"]);
    input_options(&mut cmd, format);
    cmd.arg("-i").arg(path).args(["-read_intervals", "%+16", "-show_streams", "-show_frames", "-show_entries", "stream=index,codec_type,codec_name,width,height,pix_fmt,avg_frame_rate,r_frame_rate,duration,nb_frames,channels,sample_rate:stream_tags=rotate:stream_side_data=rotation:frame=media_type,stream_index,best_effort_timestamp_time,pkt_duration_time", "-of", "json"]);
    serde_json::from_slice(&run(cmd, cancel, Duration::from_secs(30), MAX_PROBE_BYTES).await?)
        .map_err(|_| error("invalid video probe response"))
}

pub(crate) async fn normalize(
    input: &Path,
    output: &Path,
    cancel: &CancellationToken,
) -> Result<bool, ApiError> {
    let caps = capabilities()
        .await
        .ok_or_else(|| error("video normalization is unavailable"))?;
    let format = input_format(input).await?;
    // Header-only bounds are checked BEFORE decoding any frames. A hostile
    // compressed header must not allocate an enormous frame merely to learn
    // that its dimensions exceed the ingress envelope.
    let mut header = command("ffprobe");
    header.args(["-v", "error", "-max_alloc", "67108864"]);
    input_options(&mut header, format);
    header
        .arg("-i")
        .arg(input)
        .args(["-show_streams", "-show_format", "-of", "json"]);
    let headers: serde_json::Value = serde_json::from_slice(
        &run(header, cancel, Duration::from_secs(10), MAX_PROBE_BYTES).await?,
    )
    .map_err(|_| error("invalid video header probe"))?;
    let streams = headers["streams"]
        .as_array()
        .ok_or_else(|| error("video header has no streams"))?;
    let video = streams
        .iter()
        .find(|stream| stream["codec_type"] == "video")
        .ok_or_else(|| error("reference has no video stream"))?;
    let width = video["width"].as_u64().unwrap_or(0);
    let height = video["height"].as_u64().unwrap_or(0);
    if streams.len() > 32
        || width == 0
        || height == 0
        || width > 8192
        || height > 8192
        || width.saturating_mul(height) > MAX_PIXELS
        || number(&video["duration"]).is_some_and(|duration| duration > 15.001)
        || number(&headers["format"]["duration"]).is_some_and(|duration| duration > 30.0)
        || number(&video["nb_frames"]).is_some_and(|frames| frames > MAX_FRAMES as f64)
    {
        return Err(error(
            "video header exceeds normalization dimensions, frame count or duration limits",
        ));
    }
    if !video["codec_name"].as_str().is_some_and(|codec| {
        caps.accepted_video_codecs
            .iter()
            .any(|supported| supported == codec)
    }) {
        return Err(error("video codec is not supported by this server"));
    }
    let facts = inspect(
        &probe(input, format, cancel).await?,
        &caps.accepted_video_codecs,
    )?;
    if facts.canonical && format == "mov" {
        // Probe the exact existing canonical reader as well; unsupported MP4
        // variants must convert rather than acquiring a false unchanged receipt.
        let path = input.to_path_buf();
        let accepted = tokio::task::spawn_blocking(move || {
            mold_inference::ltx2::media::probe_video(&path).is_ok()
        })
        .await
        .map_err(|_| error("canonical video probe failed"))?;
        if accepted {
            tokio::fs::rename(input, output)
                .await
                .map_err(|_| error("could not stage unchanged canonical video"))?;
            return Ok(false);
        }
    }
    let private_output = output.to_path_buf();
    tokio::task::spawn_blocking(move || {
        crate::batch_transaction::create_private_file_no_follow(&private_output)
    })
    .await
    .map_err(|_| error("private canonical staging failed"))?
    .map_err(|_| error("private canonical staging failed"))?;
    let mut cmd = command("ffmpeg");
    cmd.args([
        "-hide_banner",
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-filter_threads",
        "2",
    ]);
    input_options(&mut cmd, format);
    cmd.arg("-i")
        .arg(input)
        .args([
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-map_metadata",
            "-1",
            "-map_chapters",
            "-1",
            "-sn",
            "-dn",
            "-c:v",
            "libx264",
            "-threads",
            "2",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-vf",
        ])
        .arg(format!("scale=ceil(iw/2)*2:ceil(ih/2)*2,fps={}", facts.fps))
        .args([
            "-fps_mode",
            "cfr",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ac",
            "2",
            "-ar",
            "48000",
            "-movflags",
            "+faststart",
            "-f",
            "mp4",
        ])
        .arg(output);
    run(cmd, cancel, Duration::from_secs(120), 1024).await?;
    let canonical = inspect(
        &probe(output, "mov", cancel).await?,
        &caps.accepted_video_codecs,
    )?;
    if !canonical.canonical
        || (canonical.duration - facts.duration).abs() > 1.0 / f64::from(facts.fps) + 0.002
    {
        return Err(error(
            "normalized video timing differs by more than one output frame",
        ));
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn invalid_rates_do_not_become_conversion_facts() {
        assert_eq!(rate(&serde_json::json!("0/0")), None);
        assert_eq!(
            rate(&serde_json::json!("30000/1001")),
            Some(30000.0 / 1001.0)
        );
    }
}

async fn runtime_smoke_test(cancel: &CancellationToken) -> bool {
    struct Scratch(std::path::PathBuf);
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let path =
        std::env::temp_dir().join(format!("mold-video-codec-check-{}", uuid::Uuid::new_v4()));
    let mut builder = std::fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    if builder.create(&path).is_err() {
        return false;
    }
    let scratch = Scratch(path);
    let output = scratch.0.join("codec.mp4");
    if crate::batch_transaction::create_private_file_no_follow(&output).is_err() {
        return false;
    }
    let mut cmd = command("ffmpeg");
    cmd.args([
        "-hide_banner",
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=blue:s=16x16:r=24:d=0.1",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=48000:cl=stereo",
        "-t",
        "0.1",
        "-c:v",
        "libx264",
        "-threads",
        "1",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
    ])
    .arg(&output);
    if run(cmd, cancel, Duration::from_secs(10), 1024)
        .await
        .is_err()
    {
        return false;
    }
    if probe(&output, "mov", cancel)
        .await
        .and_then(|value| inspect(&value, &["h264".into()]))
        .is_err()
    {
        return false;
    }
    tokio::task::spawn_blocking(move || {
        let _scratch = scratch;
        mold_inference::ltx2::media::probe_video(&output).is_ok()
            && mold_inference::ltx2::media::probe_decoded_mp4_audio_file(&output)
                .is_ok_and(|audio| audio.is_some())
    })
    .await
    .unwrap_or(false)
}

#[cfg(test)]
mod fixture_tests {
    use super::*;
    use base64::Engine;
    fn bytes(value: &str) -> Vec<u8> {
        base64::engine::general_purpose::STANDARD
            .decode(value.trim())
            .unwrap()
    }
    #[tokio::test]
    async fn advertised_runtime_normalizes_real_codecs_and_preserves_canonical_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let canonical = bytes(include_str!("../tests/fixtures/reference-video.mp4.b64"));
        let input = dir.path().join("original");
        let output = dir.path().join("canonical.mp4");
        std::fs::write(&input, &canonical).unwrap();
        let cancel = CancellationToken::new();
        let inspected = inspect_runtime().await;
        {
            let mut cache = codec_cache()
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            cache.value = inspected.clone();
            cache.checked_at = Some(std::time::Instant::now());
        }
        let Some(capability) = inspected else {
            assert!(normalize(&input, &output, &cancel).await.is_err());
            assert!(!output.exists(), "unavailable runtime must publish nothing");
            println!("codec_runtime=unavailable; normalization correctly refused");
            return;
        };
        assert!(!normalize(&input, &output, &cancel).await.unwrap());
        assert_eq!(std::fs::read(&output).unwrap(), canonical);
        println!("codec_fixture=canonical_mp4 converted=false bytes_preserved=true");
        let rational = bytes(include_str!(
            "../tests/fixtures/reference-rational-cfr.mp4.b64"
        ));
        let input = dir.path().join("rational-input");
        let output = dir.path().join("rational.mp4");
        std::fs::write(&input, &rational).unwrap();
        assert!(!normalize(&input, &output, &cancel).await.unwrap());
        assert_eq!(std::fs::read(&output).unwrap(), rational);
        // The existing canonical reader exposes integer fps; do not
        // rewrite it to invent a new canonical metadata contract here.
        assert!(
            mold_inference::ltx2::media::probe_video(&output)
                .unwrap()
                .fps
                > 0
        );
        let encoded = probe(&output, "mov", &cancel).await.unwrap();
        let encoded_rate = rate(&encoded["streams"][0]["avg_frame_rate"]).unwrap();
        assert!((encoded_rate - 24_000.0 / 1001.0).abs() < 0.001);
        println!("codec_fixture=rational_cfr converted=false exact_bytes_and_rate_preserved=true");
        let fixtures = [
            (
                "hevc_mov",
                "hevc",
                include_str!("../tests/fixtures/reference-hevc.mov.b64"),
            ),
            (
                "vp9_webm",
                "vp9",
                include_str!("../tests/fixtures/reference-vp9.webm.b64"),
            ),
            (
                "av1_mkv",
                "av1",
                include_str!("../tests/fixtures/reference-av1.mkv.b64"),
            ),
            (
                "prores_mov",
                "prores",
                include_str!("../tests/fixtures/reference-prores.mov.b64"),
            ),
            (
                "vfr_mp4",
                "h264",
                include_str!("../tests/fixtures/reference-vfr.mp4.b64"),
            ),
            (
                "rotation_mp4",
                "h264",
                include_str!("../tests/fixtures/reference-rotation.mp4.b64"),
            ),
        ];
        for (name, codec, fixture) in fixtures {
            if !capability
                .accepted_video_codecs
                .iter()
                .any(|value| value == codec)
            {
                continue;
            }
            let input = dir.path().join(name);
            let output = dir.path().join(format!("{name}.mp4"));
            std::fs::write(&input, bytes(fixture)).unwrap();
            if name == "rotation_mp4" {
                let source = probe(&input, "mov", &cancel).await.unwrap();
                assert_eq!(source["streams"][0]["side_data_list"][0]["rotation"], 90);
            }
            assert!(
                normalize(&input, &output, &cancel).await.unwrap(),
                "{name} must convert"
            );
            let probe = mold_inference::ltx2::media::probe_video(&output).unwrap();
            assert!(probe.duration_ms.unwrap().abs_diff(2000) <= 60, "{name}");
            assert!(probe.frames.unwrap() > 0);
            assert!(!probe.has_audio);
            if name == "rotation_mp4" {
                assert_eq!((probe.width, probe.height), (48, 32));
            }
            println!(
                "codec_fixture={name} converted=true canonical_probe=true duration_ms={}",
                probe.duration_ms.unwrap()
            );
        }
        let input = dir.path().join("overhang");
        let output = dir.path().join("overhang.mp4");
        std::fs::write(
            &input,
            bytes(include_str!(
                "../tests/fixtures/reference-audio-overhang.mp4.b64"
            )),
        )
        .unwrap();
        normalize(&input, &output, &cancel).await.unwrap();
        let video = mold_inference::ltx2::media::probe_video(&output).unwrap();
        let audio = mold_inference::ltx2::media::probe_decoded_mp4_audio_file(&output)
            .unwrap()
            .unwrap();
        assert!(video.duration_ms.unwrap().abs_diff(2000) <= 60);
        assert!(audio.samples_per_channel as f64 / f64::from(audio.sample_rate) > 2.9);
        println!(
            "codec_fixture=audio_overhang video_duration_ms={} audio_duration_gt_2_9s=true",
            video.duration_ms.unwrap()
        );
    }
}

#[cfg(test)]
pub(crate) async fn completed_runtime_probe_for_test() -> Option<VideoNormalizationCapabilities> {
    let value = inspect_runtime().await;
    let mut cache = codec_cache()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    cache.value = value.clone();
    cache.checked_at = Some(std::time::Instant::now());
    value
}
